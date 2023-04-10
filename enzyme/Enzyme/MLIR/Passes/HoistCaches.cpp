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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace enzyme;
using llvm::errs;
namespace {



void HoistCaches(Region &region) {
  region.walk(
    [&](Operation * op){
      //Quietly assume that each cache is only used once
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)){
        //llvm::errs() << "LinalgOp\n";
        //linalgOp.dump();

        SmallPtrSet<Value,1> caches;
        linalgOp.getBlock()->walk(
          [&](Operation * op2){
            if (auto pushOp = dyn_cast<PushOp>(op2)){
              caches.insert(pushOp.getCache());
            }
          }
        );

        for (auto cache : caches){
          Type cacheType = CacheType::get(cache->getContext(), t);
          linalgOp.getInputs()
        }
      }
    }
  );
}

struct HoistCachesPass : public enzyme::HoistCachesPassBase<HoistCachesPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    Region *region = getOperation()->getParentRegion();
    getOperation()->walk([&](FunctionOpInterface op) { HoistCaches(op.getFunctionBody()); });
  };
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {

std::unique_ptr<Pass> createHoistCachesPass() {
  return std::make_unique<HoistCachesPass>();
}

} // namespace enzyme
} // namespace mlir
