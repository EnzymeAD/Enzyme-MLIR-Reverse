//===- RemoveUnusedEnzymeOps.cpp - Remove unnecessary or unused gradient and
// cache ops
//------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Rewrite/PatternApplicator.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace enzyme;
using llvm::errs;
namespace {

SmallVector<mlir::Block *> getDominatorToposort(Region &region) {
  SmallVector<mlir::Block *> dominatorToposortBlocks;
  if (region.hasOneBlock()) {
    dominatorToposortBlocks.push_back(&*(region.begin()));
  } else {
    auto dInfo = mlir::detail::DominanceInfoBase<false>(nullptr);
    llvm::DominatorTreeBase<Block, false> &dt = dInfo.getDomTree(&region);
    auto root = dt.getNode(&*(region.begin()));

    for (llvm::DomTreeNodeBase<mlir::Block> *node : llvm::breadth_first(root)) {
      dominatorToposortBlocks.push_back(node->getBlock());
    }
  }
  return dominatorToposortBlocks;
}

SmallVector<Value> Enzyme2Reg(Region &region, bool ignoreFirstBlock) {
  if (region.begin() != region.end()) {
    SmallVector<mlir::Block *> blocks =
        getDominatorToposort(region);
    llvm::SmallPtrSet<Value, 4> gradientsSet;

    // Collect all gradient values
    for (Block *block : blocks) {
      block->walk([&](Operation *op) {
        if (auto initOp = dyn_cast<enzyme::InitOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(initOp.getType())) {
            gradientsSet.insert(initOp);
          }
        }
        else if (auto getOp = dyn_cast<enzyme::GetOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(getOp.getGradient().getType())) {
            gradientsSet.insert(getOp.getGradient());
          }
        }
        else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(setOp.getGradient().getType())) {
            gradientsSet.insert(setOp.getGradient());
          }
        }
      });
    }

    SmallVector<Value> gradients(gradientsSet.begin(), gradientsSet.end());

    // Write all gradient values to block parameters
    // TODO : Do we always ignore the first block?
    int start = (int) ignoreFirstBlock;
    for (int i = start; i < blocks.size(); i++) {
      Block *block = blocks[i];
      for (Value grad : gradients) {
        enzyme::GradientType type = grad.getType().cast<enzyme::GradientType>();
        block->addArgument(type.getBasetype(), grad.getLoc());
      }
    }

    // Replace all gradient values with block parameters
    for (int i = 0; i < blocks.size(); i++) {
      Block *block = blocks[i];
      BlockAndValueMapping mapping;
      if (i != 0) {
        for (int j = 0; j < gradients.size(); j++) {
          Value grad = gradients[j];
          mapping.map(grad, block->getArgument(block->getNumArguments() -
                                               gradients.size() + j));
        }
      }
      block->walk([&](Operation *op) {
        // Handle all gradient ops
        if (auto getOp = dyn_cast<enzyme::GetOp>(op)) {
          getOp.replaceAllUsesWith(mapping.lookup(getOp.getGradient()));
        } else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
          mapping.map(setOp.getGradient(), setOp.getValue());
        }
      });

      // Handle Terminators
      Operation *term = block->getTerminator();
      if (mlir::BranchOpInterface brOp =
              dyn_cast<mlir::BranchOpInterface>(term)) {
        for (int i = 0; i < (int)term->getNumSuccessors(); i++) {
          SuccessorOperands sOps = brOp.getSuccessorOperands(i);
          for (int j = 0; j < gradients.size(); j++) {
            Value grad = mapping.lookup(gradients[j]);
            sOps.append(grad);
          }
        }
      }
    }

    // Remove all gradient ops
    for (Block *block : blocks) {
      block->walk([&](Operation *op) {
        if (auto getOp = dyn_cast<enzyme::GetOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(
                  getOp.getGradient().getType())) {
            op->erase();
          }
        } else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(
                  setOp.getGradient().getType())) {
            op->erase();
          }
        }
      });
    }

    // Delete all init ops
    for (Block *block : blocks) {
      block->walk([&](Operation *op) {
        if (auto initOp = dyn_cast<enzyme::InitOp>(op)) {
          if (auto type = dyn_cast<enzyme::GradientType>(initOp.getType())) {
            op->erase();
          }
        }
      });
    }

    return gradients;
  }
  return SmallVector<Value>();
}

struct Enzyme2RegPass : public enzyme::Enzyme2RegPassBase<Enzyme2RegPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    Region *region = getOperation()->getParentRegion();
    getOperation()->walk([&](FunctionOpInterface op) { Enzyme2Reg(op.getFunctionBody(), true); });
  };
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEnzyme2RegPass() {
  return std::make_unique<Enzyme2RegPass>();
}
} // namespace enzyme
} // namespace mlir
