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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

#include "mlir/Dialect/Shape/IR/ShapeOpsTypes.h.inc"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Value invertMemref(Value inp, OpBuilder &builder, Location loc){
  MemRefType iType = dyn_cast<MemRefType>(inp.getType());
  SmallVector<Value> dims;
  SmallVector<Value> dimSubOnes;
  SmallVector<Value> strides;
  Value negOne = builder.create<arith::ConstantIndexOp>(loc, -1);
  for (int i = 0; i < iType.getShape().size(); i++){
    Value dim = builder.create<memref::DimOp>(loc, inp, i);
    dims.push_back(dim);
    auto dimSubOne = builder.create<arith::AddIOp>(loc, dim, negOne);
    dimSubOnes.push_back(dimSubOne);
    strides.push_back(negOne);
  }
  Value view = builder.create<memref::SubViewOp>(
    loc, inp, ValueRange(dimSubOnes), ValueRange(dims),
    ValueRange(strides));
  return view;
}
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
    linalg::LinalgOp newOp = cast<linalg::LinalgOp>(gutils->getNewFromOriginal(linalgOp));

    auto cacheBuilder = OpBuilder(newOp);
    
    //get iteration domain
    AffineMap aMap = newOp.getShapesToLoopsMap();
    SmallVector<Value> dims;
    for (OpOperand *input : newOp.getInputOperands()) {
      auto shape = cast<MemRefType>(input->get().getType()).getShape();
      for (int i = 0; i < shape.size(); i++){
        auto dimI = cacheBuilder.create<arith::ConstantIndexOp>(op->getLoc(), i);
        //input->get().getParentBlock()->dump();
        auto dim = cacheBuilder.create<memref::DimOp>(op->getLoc(), input->get(), dimI);
        dims.push_back(dim);
      }
    }
  
    SmallVector<Value> iterationDomains;
    std::vector<int64_t> shapes;
    for (unsigned int i = 0; i < aMap.getNumResults(); i++){
      AffineMap subMap = aMap.getSubMap({i});
      Value domain = cacheBuilder.create<AffineApplyOp>(op->getLoc(), subMap, ValueRange(dims));
      iterationDomains.push_back(domain);
      shapes.push_back(-1);
    }

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<StringRef> iteratorTypes{linalgOp.getNumLoops(),
                                         getParallelIteratorTypeName()};

    for (OpOperand *output : linalgOp.getOutputOperands()) {
      if(gutils->hasInvertPointer(output->get())){
        indexingMaps.push_back(linalgOp.getMatchingIndexingMap(output));
        Value out = gutils->invertPointerM(output->get(), builder);
        Value view = invertMemref(out, builder, op->getLoc());
        outputs.push_back(view);
      }
    }

    for (OpOperand *input : linalgOp.getInputOperands()) {
      if(gutils->hasInvertPointer(input->get())){
        indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
        Value inp = gutils->invertPointerM(input->get(), builder);
        Value view = invertMemref(inp, builder, op->getLoc());
        inputs.push_back(view);
      }
    }

    linalg::GenericOp adjoint = builder.create<linalg::GenericOp>(
        op->getLoc(), outputs, inputs, indexingMaps, iteratorTypes);

    int numInputs = inputs.size();
    std::function<buildReturnFunction> buildFuncReturnOp = [numInputs](OpBuilder &builder, Location loc,
                                               SmallVector<Value> retargs) {
      builder.create<linalg::YieldOp>(loc, ValueRange{retargs}.take_front(numInputs));
      return;
    };

    Region * newOpRegion = newOp.getBlock()->getParent();
    int numInputsNewOp = cast<linalg::GenericOp>(newOp).getInputs().size();
    Region * adjointRegion = &adjoint.getBodyRegion();
    int numInputsAdjoint = adjoint.getInputs().size();
    Location loc = op->getLoc();
    int numCaches = 0;


    //std::function<std::pair<Value, Value>(Type)> hook = nullptr;
    std::function<std::pair<Value, Value>(Type)> hook = [newOpRegion, adjointRegion, loc, &numCaches = numCaches, numInputsNewOp, numInputsAdjoint](Type t) {
      Value pushCache = newOpRegion->insertArgument(numInputsNewOp + numCaches, t, loc);
      adjointRegion->front().dump();
      Value popCache = adjointRegion->insertArgument(numInputsAdjoint + numCaches, t, loc);
      adjointRegion->front().dump();
      numCaches++;
      return std::pair<Value, Value>(pushCache, popCache);
    };

    gutils->Logic.differentiate(gutils, *linalgOp.getBlock()->getParent(),
                                adjoint.getBodyRegion(),
                                /*parentRegion=*/false, buildFuncReturnOp, hook);

    Block * body = &(adjoint.getBodyRegion().front());
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());

    for(auto opOperand : yieldOp.getOperands()){
      body->addArgument(opOperand.getType(), opOperand.getLoc());
    }
    OpBuilder builderAdd(yieldOp);
    for (auto it : llvm::enumerate(yieldOp.getOperands())) {
      Value arg = body->getArgument(outputs.size() + numCaches + it.index());
      auto diffeType = cast<AutoDiffTypeInterface>(arg.getType());
      Value grad = diffeType.createAddOp(builderAdd, it.value().getLoc(), arg, it.value());
      yieldOp.setOperand(it.index(), grad);
    }

    auto indexing_maps = newOp.getIndexingMapsArray();
    auto indexing_maps_adjoint = adjoint.getIndexingMapsArray();
    for (int i = 0; i < numCaches; i++){
      Value cacheArg = body->getArgument(outputs.size() + i);
      
      Type ct = cacheArg.getType();
      Type type = MemRefType::get(ArrayRef(shapes), ct);
      Value alloc = cacheBuilder.create<memref::AllocOp>(op->getLoc(), type, ValueRange(iterationDomains));
      Value cache = gutils->initAndPushCache(alloc, cacheBuilder);
      alloc.getDefiningOp()->setAttr("operand_segment_sizes", cacheBuilder.getDenseI32ArrayAttr({1,0}));
      
      cast<linalg::GenericOp>(newOp).getInputsMutable().append(ValueRange({alloc}));
      indexing_maps.insert(indexing_maps.begin() + numInputsNewOp + i, AffineMap::getMultiDimIdentityMap(iterationDomains.size(), cacheBuilder.getContext()));
      
      OpBuilder builder2(adjoint);
      Value retrievedValue = gutils->popCache(cache, builder2);
      retrievedValue = invertMemref(retrievedValue, builder2, op->getLoc());
      adjoint.getInputsMutable().append(ValueRange({retrievedValue}));
      indexing_maps_adjoint.insert(indexing_maps_adjoint.begin() + numInputsAdjoint + i, AffineMap::getMultiDimIdentityMap(iterationDomains.size(), builder2.getContext()));
    }
    SmallVector<Attribute> indexing_maps_attr;
    SmallVector<Attribute> indexing_maps_attr_adjoint;
    for (auto map : indexing_maps){
      indexing_maps_attr.push_back(AffineMapAttr::get(map));
    }
    for (auto map : indexing_maps_adjoint){
      indexing_maps_attr_adjoint.push_back(AffineMapAttr::get(map));
    }
    newOp->setAttr("indexing_maps", cacheBuilder.getArrayAttr(indexing_maps_attr));
    adjoint->setAttr("indexing_maps", builder.getArrayAttr(indexing_maps_attr_adjoint));

    //aMap.dump();
    //llvm::errs() << numCaches << "\n";
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
