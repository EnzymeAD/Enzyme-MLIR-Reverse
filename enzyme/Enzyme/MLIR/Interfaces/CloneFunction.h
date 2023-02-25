#pragma once

#include "EnzymeLogic.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme;

Type getShadowType(Type type, unsigned width = 1);

mlir::FunctionType
getFunctionTypeForClone(mlir::FunctionType FTy, DerivativeModeMLIR mode,
                        unsigned width, mlir::Type additionalArg,
                        llvm::ArrayRef<DIFFE_TYPE_MLIR> constant_args,
                        bool diffeReturnArg, ReturnTypeMLIR returnValue,
                        DIFFE_TYPE_MLIR ReturnTypeMLIR);

void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap);

void cloneInto(Region *src, Region *dest, BlockAndValueMapping &mapper,
               std::map<mlir::Operation *, mlir::Operation *> &opMap);

Operation *clone(Operation *src, BlockAndValueMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap);

FunctionOpInterface CloneFunctionWithReturns(
    DerivativeModeMLIR mode, unsigned width, FunctionOpInterface F,
    BlockAndValueMapping &ptrInputs, ArrayRef<DIFFE_TYPE_MLIR> constant_args,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnTypeMLIR returnValue,
    DIFFE_TYPE_MLIR ReturnTypeMLIR, Twine name, BlockAndValueMapping &VMap,
    std::map<Operation *, Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg);