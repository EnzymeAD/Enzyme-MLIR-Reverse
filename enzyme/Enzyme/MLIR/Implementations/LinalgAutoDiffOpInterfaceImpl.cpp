//===- LinalgAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

using namespace mlir;
using namespace mlir::enzyme;

namespace {


// TODO: Try to find a way to register autodiff interface for all LinalgOps
struct GenericOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GenericOpInterfaceReverse, linalg::GenericOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    // TODO: currently in progress, this doesn't work
    auto linalgOp = cast<linalg::LinalgOp>(op);
    assert(linalgOp.hasBufferSemantics() &&
           "Linalg op with tensor semantics not yet supported");

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<StringRef> iteratorTypes{linalgOp.getNumLoops(),
                                         getParallelIteratorTypeName()};

    for (OpOperand *output : linalgOp.getOutputOperands()) {
      if(gutils->hasInvertPointer(output->get())){
        indexingMaps.push_back(linalgOp.getMatchingIndexingMap(output));
        outputs.push_back(gutils->invertPointerM(output->get(), builder));
      }
    }

    for (OpOperand *input : linalgOp.getInputOperands()) {
      if(gutils->hasInvertPointer(input->get())){
        indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
        inputs.push_back(gutils->invertPointerM(input->get(), builder));
      }
    }

    auto adjoint = builder.create<linalg::GenericOp>(
        op->getLoc(), outputs, inputs, indexingMaps, iteratorTypes);
  
    int numInputs = inputs.size();
    std::function<buildReturnFunction> buildFuncReturnOp = [numInputs](OpBuilder &builder, Location loc,
                                               SmallVector<Value> retargs) {
      builder.create<linalg::YieldOp>(loc, ValueRange{retargs}.take_front(numInputs));
      return;
    };

    gutils->Logic.differentiate(gutils, *linalgOp.getBlock()->getParent(),
                                adjoint.getBodyRegion(),
                                /*parentRegion=*/false, buildFuncReturnOp);

    Block * body = &(adjoint.getBodyRegion().front());
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());

    for(auto opOperand : yieldOp.getOperands()){
      body->addArgument(opOperand.getType(), opOperand.getLoc());
    }
    OpBuilder builderAdd(yieldOp);
    for (auto it : llvm::enumerate(yieldOp.getOperands())) {
      Value arg = body->getArgument(outputs.size() + it.index());
      auto diffeType = cast<AutoDiffTypeInterface>(arg.getType());
      Value grad = diffeType.createAddOp(builderAdd, it.value().getLoc(), arg, it.value());
      yieldOp.setOperand(it.index(), grad);
    }
      
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};
} // namespace

void mlir::enzyme::registerLinalgDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    linalg::GenericOp::attachInterface<GenericOpInterfaceReverse>(*context);
  });
}
