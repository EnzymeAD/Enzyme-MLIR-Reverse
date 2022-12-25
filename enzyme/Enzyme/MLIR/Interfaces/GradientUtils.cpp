//===- GradientUtils.cpp - Utilities for gradient interfaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/GradientUtils.h"
#include "Interfaces/CloneFunction.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::enzyme::MGradientUtils::MGradientUtils(
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_, MTypeResults TR_,
    BlockAndValueMapping &invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_, DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode, unsigned width, bool omp)
    : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_), TA(TA_),
      TR(TR_), omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      invertedPointers(invertedPointers_) {
  
  auto valueMap = invertedPointers.getValueMap();
  assert(invertedPointers.getBlockMap().begin() == invertedPointers.getBlockMap().end());
  for(auto it = valueMap.begin(); it != valueMap.end(); it++){
    mapInvertPointer(it->first, it->second);
  }

  /*
  for (BasicBlock &BB : *oldFunc) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        originalCalls.push_back(CI);
      }
    }
  }
  */

  /*
  for (BasicBlock &oBB : *oldFunc) {
    for (Instruction &oI : oBB) {
      newToOriginalFn[originalToNewFn[&oI]] = &oI;
    }
    newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
  }
  for (Argument &oArg : oldFunc->args()) {
    newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
  }
  */
  /*
  for (BasicBlock &BB : *newFunc) {
    originalBlocks.emplace_back(&BB);
  }
  tape = nullptr;
  tapeidx = 0;
  assert(originalBlocks.size() > 0);

  SmallVector<BasicBlock *, 4> ReturningBlocks;
  for (BasicBlock &BB : *oldFunc) {
    if (isa<ReturnInst>(BB.getTerminator()))
      ReturningBlocks.push_back(&BB);
  }
  for (BasicBlock &BB : *oldFunc) {
    bool legal = true;
    for (auto BRet : ReturningBlocks) {
      if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
        legal = false;
        break;
      }
    }
    if (legal)
      BlocksDominatingAllReturns.insert(&BB);
  }
  */
}

Value mlir::enzyme::MGradientUtils::getNewFromOriginal(
    const mlir::Value originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new val from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Block *
mlir::enzyme::MGradientUtils::getNewFromOriginal(mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation *
mlir::enzyme::MGradientUtils::getNewFromOriginal(Operation *originst) const {
  auto found = originalToNewFnOps.find(originst);
  if (found == originalToNewFnOps.end()) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    for (auto &pair : originalToNewFnOps) {
      llvm::errs() << " map[" << pair.first << "] = " << pair.second << "\n";
      // llvm::errs() << " map[" << pair.first << "] = " << pair.second << "
      // -- " << *pair.first << " " << *pair.second << "\n";
    }
    llvm::errs() << originst << " - " << *originst << "\n";
    llvm_unreachable("Could not get new op from original");
  }
  return found->second;
}

Operation *mlir::enzyme::MGradientUtils::cloneWithNewOperands(OpBuilder &B,
                                                              Operation *op) {
  BlockAndValueMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

bool mlir::enzyme::MGradientUtils::isConstantValue(Value v) const {
  if (isa<mlir::IntegerType>(v.getType()))
    return true;
  if (isa<mlir::IndexType>(v.getType()))
    return true;

  if (matchPattern(v, m_Constant()))
    return true;

  // TODO
  return false;
}

Value mlir::enzyme::MGradientUtils::invertPointerM(Value v,
                                                   OpBuilder &Builder2) {
  // TODO
  if (invertedPointers.contains(v))
    return invertedPointers.lookupOrNull(v);

  if (isConstantValue(v)) {
    if (auto iface = v.getType().cast<AutoDiffTypeInterface>()) {
      OpBuilder::InsertionGuard guard(Builder2);
      Builder2.setInsertionPoint(getNewFromOriginal(v.getDefiningOp()));
      Value dv = iface.createNullValue(Builder2, v.getLoc());
      mapInvertPointer(v, dv);
      return dv;
    }
    return getNewFromOriginal(v);
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

Value mlir::enzyme::MGradientUtils::invertPointerReverseM(Value v, Block * askingOp) {
  if (invertedPointersReverse.count(v) != 0){
    auto values = (invertedPointersReverse.find(v))->second;
    for (auto it = values.rbegin(); it != values.rend(); it++){
      if(mlir::DominanceInfo().dominates(it->getParentBlock(), askingOp)){
        return *it;
      }
    }
    llvm::errs() << "could not find in vector " << v << "\n";
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

Optional<Value> mlir::enzyme::MGradientUtils::invertPointerReverseMOptional(Value v, Block * askingOp) {
  if (invertedPointersReverse.count(v) != 0){
    auto values = (invertedPointersReverse.find(v))->second;
    for (auto it = values.rbegin(); it != values.rend(); it++){
      if(mlir::DominanceInfo().dominates(it->getParentBlock(), askingOp)){
        return *it;
      }
    }
  }
  return Optional<Value>();
}

void mlir::enzyme::MGradientUtils::mapInvertPointer(mlir::Value v, mlir::Value invertValue){
  invertedPointers.map(v, invertValue);
  if (invertedPointersReverse.count(v) == 0){
    invertedPointersReverse[v] = SmallVector<mlir::Value, 4>();
  }
  invertedPointersReverse[v].push_back(invertValue);
}

bool mlir::enzyme::MGradientUtils::hasInvertPointer(mlir::Value v){
  return invertedPointersReverse.count(v) != 0;
}

void mlir::enzyme::MGradientUtils::setDiffe(mlir::Value val, mlir::Value toset,
                                            OpBuilder &BuilderM) {
  /*
 if (auto arg = dyn_cast<Argument>(val))
   assert(arg->getParent() == oldFunc);
 if (auto inst = dyn_cast<Instruction>(val))
   assert(inst->getParent()->getParent() == oldFunc);
   */
  if (isConstantValue(val)) {
    llvm::errs() << newFunc << "\n";
    llvm::errs() << val << "\n";
  }
  assert(!isConstantValue(val));
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit) {
    assert(getShadowType(val.getType()) == toset.getType());
    auto found = invertedPointers.lookupOrNull(val);
    assert(found != nullptr);
    auto placeholder = found.getDefiningOp<enzyme::PlaceholderOp>();
    invertedPointers.erase(val);
    // replaceAWithB(placeholder, toset);
    placeholder.replaceAllUsesWith(toset);
    erase(placeholder);
    mapInvertPointer(val, toset);
    return;
  } else if (mode == DerivativeMode::ReverseModeGradient) {
    assert(getShadowType(val.getType()) == toset.getType());
    auto found = invertedPointers.lookupOrNull(val);
    assert(found != nullptr);
    auto placeholder = found.getDefiningOp<enzyme::PlaceholderOp>();
    invertedPointers.erase(val);
    // replaceAWithB(placeholder, toset);
    placeholder.replaceAllUsesWith(toset);
    erase(placeholder);
    mapInvertPointer(val, toset);
    return;
  }
  /*
  Value *tostore = getDifferential(val);
  if (toset->getType() != tostore->getType()->getPointerElementType()) {
    llvm::errs() << "toset:" << *toset << "\n";
    llvm::errs() << "tostore:" << *tostore << "\n";
  }
  assert(toset->getType() == tostore->getType()->getPointerElementType());
  BuilderM.CreateStore(toset, tostore);
  */
}

void mlir::enzyme::MGradientUtils::forceAugmentedReturnsReverse() {
  assert(mode == DerivativeMode::ReverseModeGradient);

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      mapInvertPointer(val, dval);
    }
  });

  int index = oldFunc.getNumArguments() - 1;
  auto argument = oldFunc.getArgument(index);
  oldFunc.walk([&](Block *blk) {
    auto terminator = blk->getTerminator();
    if (terminator->hasTrait<OpTrait::ReturnLike>()) {
      auto nblk = getNewFromOriginal(blk);
      mapInvertPointer(terminator->getOperand(0), argument);
    }
  });
}

void mlir::enzyme::MGradientUtils::forceAugmentedReturns() {
  // TODO also block arguments
  // assert(TR.getFunction() == oldFunc);

  // Don't create derivatives for code that results in termination
  // if (notForAnalysis.find(&oBB) != notForAnalysis.end())
  //  continue;

  // LoopContext loopContext;
  // getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      mapInvertPointer(val, dval);
    }
  });

  oldFunc.walk([&](Operation *inst) {
    if (inst == oldFunc)
      return;
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      OpBuilder BuilderZ(getNewFromOriginal(inst));
      for (auto res : inst->getResults()) {
        if (!isConstantValue(res)) {
          mlir::Type antiTy = getShadowType(res.getType());
          auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(),
                                                             res.getType());
          mapInvertPointer(res, anti);
        }
      }
      return;
    }
    /*

    if (inst->getType()->isFPOrFPVectorTy())
      continue; //! op->getType()->isPointerTy() &&
                //! !op->getType()->isIntegerTy()) {

    if (!TR.query(inst)[{-1}].isPossiblePointer())
      continue;

    if (isa<LoadInst>(inst)) {
      IRBuilder<> BuilderZ(inst);
      getForwardBuilder(BuilderZ);
      Type *antiTy = getShadowType(inst->getType());
      PHINode *anti =
          BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'il_phi");
      invertedPointers.insert(std::make_pair(
          (const Value *)inst, InvertedPointerVH(this, anti)));
      continue;
    }

    if (!isa<CallInst>(inst)) {
      continue;
    }

    if (isa<IntrinsicInst>(inst)) {
      continue;
    }

    if (isConstantValue(inst)) {
      continue;
    }

    CallInst *op = cast<CallInst>(inst);
    Function *called = op->getCalledFunction();

    IRBuilder<> BuilderZ(inst);
    getForwardBuilder(BuilderZ);
    Type *antiTy = getShadowType(inst->getType());

    PHINode *anti =
        BuilderZ.CreatePHI(antiTy, 1, op->getName() + "'ip_phi");
    invertedPointers.insert(
        std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

    if (called && isAllocationFunction(called->getName(), TLI)) {
      anti->setName(op->getName() + "'mi");
    }
    */
  });
}

LogicalResult MGradientUtils::visitChildReverse(Operation *op,
                                                OpBuilder &builder) {
  if (mode == DerivativeMode::ReverseModeGradient) {
    if (auto binst = dyn_cast<BranchOpInterface>(op)) {
      
    }
    else if (auto binst = dyn_cast<func::ReturnOp>(op)) {
      
    }
    else if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      return iface.createReverseModeAdjoint(builder, this);
    }
  }
  return success();
}

LogicalResult MGradientUtils::visitChild(Operation *op) {
  if (mode == DerivativeMode::ForwardMode) {
    if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(getNewFromOriginal(op));
      return iface.createForwardModeAdjoint(builder, this);
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

void createTerminator(MDiffeGradientUtils *gutils, mlir::Block *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  MTypeResults &TR = gutils->TR;
  auto inst = oBB->getTerminator();

  mlir::Block *nBB = gutils->getNewFromOriginal(inst->getBlock());
  assert(nBB);
  auto newInst = nBB->getTerminator();

  OpBuilder nBuilder(inst);
  nBuilder.setInsertionPointToEnd(nBB);

  if (auto binst = dyn_cast<BranchOpInterface>(inst)) {
    // TODO generalize to cloneWithNewBlockArgs interface
    SmallVector<Value> newVals;

    llvm::errs() << oBB->getNumSuccessors() << "oBB->getNumSuccessors() \n";

    SmallVector<int32_t> segSizes;
    auto successors = binst.getSuccessorOperands(0);
    if (successors.size() > 0) {
      size_t len = successors.getForwardedOperands().getBeginOperandIndex();
      for (size_t i = 0; i < len; i++) {
        newVals.push_back(gutils->getNewFromOriginal(binst->getOperand(i)));
      }
    }
    segSizes.push_back(newVals.size());
    for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
      size_t cur = newVals.size();
      for (auto op : binst.getSuccessorOperands(i).getForwardedOperands()) {
        newVals.push_back(gutils->getNewFromOriginal(op));
        if (!gutils->isConstantValue(op)) {
          newVals.push_back(gutils->invertPointerM(op, nBuilder));
        }
      }
      cur = newVals.size() - cur;
      segSizes.push_back(cur);
    }

    SmallVector<NamedAttribute> attrs(newInst->getAttrs());
    for (auto &attr : attrs) {
      if (attr.getName() == "operand_segment_sizes")
        attr.setValue(nBuilder.getDenseI32ArrayAttr(segSizes));
    }

    nBB->push_back(newInst->create(
        newInst->getLoc(), newInst->getName(), TypeRange(), newVals, attrs,
        newInst->getSuccessors(), newInst->getNumRegions()));
    gutils->erase(newInst);
    return;
  }

  // In forward mode we only need to update the return value
  if (!inst->hasTrait<OpTrait::ReturnLike>())
    return;

  SmallVector<mlir::Value, 2> retargs;

  switch (retVal) {
  case ReturnType::Return: {
    auto ret = inst->getOperand(0);

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      Type retTy = ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
      toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(nBuilder,
                                                                  ret.getLoc());
    }
    retargs.push_back(toret);

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    auto ret = inst->getOperand(0);

    retargs.push_back(gutils->getNewFromOriginal(ret));

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      Type retTy = ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
      toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(nBuilder,
                                                                  ret.getLoc());
    }
    retargs.push_back(toret);
    break;
  }
  case ReturnType::Void: {
    break;
  }
  case ReturnType::Tape: {
    for (auto attribute : gutils->oldFunc.getBody().getArguments()) {
      auto attributeGradient = gutils->invertPointerM(attribute, nBuilder);
      retargs.push_back(attributeGradient);
    }
    break;
  }
  default: {
    llvm::errs() << "Invalid return type: " << to_string(retVal)
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  nBB->push_back(newInst->create(
      newInst->getLoc(), newInst->getName(), TypeRange(), retargs,
      newInst->getAttrs(), newInst->getSuccessors(), newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

FunctionOpInterface mlir::enzyme::MEnzymeLogic::CreateForwardDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented) {
  if (fn.getBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  MForwardCacheKey tup = {
      fn, retType, constants,
      // std::map<Argument *, bool>(_uncacheable_args.begin(),
      //                           _uncacheable_args.end()),
      returnUsed, mode, static_cast<unsigned>(width), addedType, type_args};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }
  bool retActive = retType != DIFFE_TYPE::CONSTANT;
  ReturnType returnValue =
      returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                 : (retActive ? ReturnType::Return : ReturnType::Void);
  auto gutils = MDiffeGradientUtils::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType,
      /*diffeReturnArg*/ false, constants, returnValue, addedType,
      /*omp*/ false);
  ForwardCachedFunctions[tup] = gutils->newFunc;

  insert_or_assign2<MForwardCacheKey, FunctionOpInterface>(
      ForwardCachedFunctions, tup, gutils->newFunc);

  // gutils->FreeMemory = freeMemory;

  const SmallPtrSet<mlir::Block *, 4> guaranteedUnreachable;
  // = getGuaranteedUnreachable(gutils->oldFunc);

  // gutils->forceActiveDetection();
  gutils->forceAugmentedReturns();
  /*

  // TODO populate with actual unnecessaryInstructions once the dependency
  // cycle with activity analysis is removed
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  if (mode == DerivativeMode::ForwardModeSplit)
    gutils->computeGuaranteedFrees();

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(
      *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
  returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
  gutils->unnecessaryValuesP = &unnecessaryValues;

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);
                                  */

  for (Block &oBB : gutils->oldFunc.getBody().getBlocks()) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = gutils->getNewFromOriginal(&oBB);

      SmallVector<Operation *, 4> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : llvm::reverse(toerase)) {
        gutils->eraseIfUnused(I, /*erase*/ true, /*check*/ false);
      }
      OpBuilder builder(gutils->oldFunc.getContext());
      builder.setInsertionPointToEnd(newBB);
      builder.create<LLVM::UnreachableOp>(gutils->oldFunc.getLoc());
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      // TODO: propagate errors.
      (void)gutils->visitChild(&*it);
    }

    createTerminator(gutils, &oBB, retType, returnValue);
  }

  // if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
  //  restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

  // gutils->eraseFictiousPHIs();

  mlir::Block *entry = &gutils->newFunc.getBody().front();

  // cleanupInversionAllocs(gutils, entry);
  // clearFunctionAttributes(gutils->newFunc);

  /*
  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }
  */

  auto nf = gutils->newFunc;
  delete gutils;

  // if (PostOpt)
  //  PPC.optimizeIntermediate(nf);
  // if (EnzymePrint) {
  //  llvm::errs() << nf << "\n";
  //}
  return nf;
}
