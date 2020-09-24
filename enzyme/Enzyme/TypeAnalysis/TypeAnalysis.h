//===- TypeAnalysis.h - Declaration of Type Analysis   ------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of Type Analysis, a utility for
// computing the underlying data type of LLVM values.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_H
#define ENZYME_TYPE_ANALYSIS_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstVisitor.h"

#include "llvm/IR/Dominators.h"

#include "TypeTree.h"

/// Struct containing all contextual type information for a
/// particular function call
struct FnTypeInfo {
  /// Function being analyzed
  llvm::Function *Function;
  
  FnTypeInfo(llvm::Function *fn) : Function(fn) {}
  FnTypeInfo(const FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &&) = default;

  /// Types of arguments
  std::map<llvm::Argument *, TypeTree> Arguments;

  /// Type of return
  TypeTree Return;

  /// The specific constant(s) known to represented by an argument, if constant
  std::map<llvm::Argument *, std::set<int64_t>> KnownValues;

  /// The set of known values val will take
  std::set<int64_t>
  knownIntegralValues(llvm::Value *val, const llvm::DominatorTree &DT,
                std::map<llvm::Value *, std::set<int64_t>> &intseen) const;
};

static inline bool operator<(const FnTypeInfo &lhs,
                             const FnTypeInfo &rhs) {

  if (lhs.Function < rhs.Function)
    return true;
  if (rhs.Function < lhs.Function)
    return false;

  if (lhs.Arguments < rhs.Arguments)
    return true;
  if (rhs.Arguments < lhs.Arguments)
    return false;
  if (lhs.Return < rhs.Return)
    return true;
  if (rhs.Return < lhs.Return)
    return false;
  return lhs.KnownValues < rhs.KnownValues;
}

class TypeAnalyzer;
class TypeAnalysis;

/// A holder class representing the results of running TypeAnalysis
/// on a given function
class TypeResults {
public:
  TypeAnalysis &analysis;
  const FnTypeInfo info;

public:
  TypeResults(TypeAnalysis &analysis, const FnTypeInfo &fn);
  ConcreteType intType(llvm::Value *val, bool errIfNotFound = true);

  /// Returns whether in the first num bytes there is pointer, int, float, or
  /// none If pointerIntSame is set to true, then consider either as the same
  /// (and thus mergable)
  ConcreteType firstPointer(size_t num, llvm::Value *val, bool errIfNotFound = true,
                        bool pointerIntSame = false);

  /// The TypeTree of a particular Value
  TypeTree query(llvm::Value *val);

  /// The TypeInfo calling convention
  FnTypeInfo getAnalyzedTypeInfo();

  /// The Type of the return
  TypeTree getReturnAnalysis();

  /// Prints all known information
  void dump();

  ///The set of values val will take on during this program
  std::set<int64_t> knownIntegralValues(llvm::Value *val) const;
};

/// Helper class that computes the fixed-point type results of a given function
class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
  /// List of value's which should be re-analyzed now with new information
  std::deque<llvm::Value *> workList;

private:
  /// Tell TypeAnalyzer to reanalyze this value
  void addToWorkList(llvm::Value *val);

  /// Map of Value to known integer constants that it will take on
  std::map<llvm::Value *, std::set<int64_t>> intseen;

public:
  /// Calling context
  const FnTypeInfo fntypeinfo;

  /// Calling TypeAnalysis to be used in the case of calls to other
  /// functions
  TypeAnalysis &interprocedural;

  /// Intermediate conservative, but correct Type analysis results
  std::map<llvm::Value *, TypeTree> analysis;

  llvm::DominatorTree DT;

  TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA);

  TypeTree getAnalysis(llvm::Value *val);

  /// Add additional information to the Type info of val, readding it to the
  /// work queue as necessary
  void updateAnalysis(llvm::Value *val, BaseType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, ConcreteType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, TypeTree data, llvm::Value *origin);

  /// Analyze type info given by the arguments, possibly adding to work queue
  void prepareArgs();

  /// Analyze type info given by the TBAA, possibly adding to work queue
  void considerTBAA();

  /// Run the interprocedural type analysis starting from this function
  void run();

  /// Set any unused values to a particular type
  bool runUnusedChecks();

  void visitValue(llvm::Value &val);

  void visitCmpInst(llvm::CmpInst &I);

  void visitAllocaInst(llvm::AllocaInst &I);

  void visitLoadInst(llvm::LoadInst &I);

  void visitStoreInst(llvm::StoreInst &I);

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep);

  void visitPHINode(llvm::PHINode &phi);

  void visitTruncInst(llvm::TruncInst &I);

  void visitZExtInst(llvm::ZExtInst &I);

  void visitSExtInst(llvm::SExtInst &I);

  void visitAddrSpaceCastInst(llvm::AddrSpaceCastInst &I);

  void visitFPTruncInst(llvm::FPTruncInst &I);

  void visitFPToUIInst(llvm::FPToUIInst &I);

  void visitFPToSIInst(llvm::FPToSIInst &I);

  void visitUIToFPInst(llvm::UIToFPInst &I);

  void visitSIToFPInst(llvm::SIToFPInst &I);

  void visitPtrToIntInst(llvm::PtrToIntInst &I);

  void visitIntToPtrInst(llvm::IntToPtrInst &I);

  void visitBitCastInst(llvm::BitCastInst &I);

  void visitSelectInst(llvm::SelectInst &I);

  void visitExtractElementInst(llvm::ExtractElementInst &I);

  void visitInsertElementInst(llvm::InsertElementInst &I);

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &I);

  void visitExtractValueInst(llvm::ExtractValueInst &I);

  void visitInsertValueInst(llvm::InsertValueInst &I);

  void visitBinaryOperator(llvm::BinaryOperator &I);

  void visitIPOCall(llvm::CallInst &call, llvm::Function &fn);

  void visitCallInst(llvm::CallInst &call);

  void visitMemTransferInst(llvm::MemTransferInst &MTI);

  void visitIntrinsicInst(llvm::IntrinsicInst &II);

  TypeTree getReturnAnalysis();

  void dump();

  std::set<int64_t> knownIntegralValues(llvm::Value *val);

  //TODO handle fneg on LLVM 10+
};

/// Full interprocedural TypeAnalysis
class TypeAnalysis {
public:
  /// Map of possible query states to TypeAnalyzer intermediate results
  std::map<FnTypeInfo, TypeAnalyzer> analyzedFunctions;

  /// Analyze a particular function, returning the results
  TypeResults analyzeFunction(const FnTypeInfo &fn);

  /// Get the TypeTree of a given value from a given function context
  TypeTree query(llvm::Value *val, const FnTypeInfo &fn);

  /// Get the underlying data type of value val given a particular context
  /// If the type is not known err if errIfNotFound
  ConcreteType intType(llvm::Value *val, const FnTypeInfo &fn,
                   bool errIfNotFound = true);

  /// Get the underlying data type of first num bytes of val given a particular context
  /// If the type is not known err if errIfNotFound. Consider ints and pointers
  /// the same if pointerIntSame.
  ConcreteType firstPointer(size_t num, llvm::Value *val, const FnTypeInfo &fn,
                        bool errIfNotFound = true, bool pointerIntSame = false);

  /// Get the TyeTree of the returned value of a given function and context
  inline TypeTree getReturnAnalysis(const FnTypeInfo &fn) {
    analyzeFunction(fn);
    return analyzedFunctions.find(fn)->second.getReturnAnalysis();
  }
};

#endif