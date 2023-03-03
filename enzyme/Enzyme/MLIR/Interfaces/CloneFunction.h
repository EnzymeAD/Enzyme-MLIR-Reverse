#pragma once

#include "EnzymeLogic.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

mlir::Type getShadowType(mlir::Type type, unsigned width = 1);

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, mlir::enzyme::DerivativeModeMLIR mode,
    unsigned width, mlir::Type additionalArg,
    llvm::ArrayRef<mlir::enzyme::DIFFE_TYPE_MLIR> constant_args,
    bool diffeReturnArg, mlir::enzyme::ReturnTypeMLIR returnValue,
    mlir::enzyme::DIFFE_TYPE_MLIR ReturnTypeMLIR);

void cloneInto(mlir::Region *src, mlir::Region *dest,
               mlir::Region::iterator destPos,
               mlir::BlockAndValueMapping &mapper,
               std::map<mlir::Operation *, mlir::Operation *> &opMap);

void cloneInto(mlir::Region *src, mlir::Region *dest,
               mlir::BlockAndValueMapping &mapper,
               std::map<mlir::Operation *, mlir::Operation *> &opMap);

mlir::Operation *clone(mlir::Operation *src, mlir::BlockAndValueMapping &mapper,
                       mlir::Operation::CloneOptions options,
                       std::map<mlir::Operation *, mlir::Operation *> &opMap);

mlir::FunctionOpInterface CloneFunctionWithReturns(
    mlir::enzyme::DerivativeModeMLIR mode, unsigned width,
    mlir::FunctionOpInterface F, mlir::BlockAndValueMapping &ptrInputs,
    mlir::ArrayRef<mlir::enzyme::DIFFE_TYPE_MLIR> constant_args,
    mlir::SmallPtrSetImpl<mlir::Value> &constants,
    mlir::SmallPtrSetImpl<mlir::Value> &nonconstants,
    mlir::SmallPtrSetImpl<mlir::Value> &returnvals,
    mlir::enzyme::ReturnTypeMLIR returnValue,
    mlir::enzyme::DIFFE_TYPE_MLIR ReturnTypeMLIR, mlir::Twine name,
    mlir::BlockAndValueMapping &VMap,
    std::map<mlir::Operation *, mlir::Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg);