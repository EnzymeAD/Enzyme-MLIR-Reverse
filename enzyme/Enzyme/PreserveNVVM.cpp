//===- PreserveNVVM.cpp - Mark NVVM attributes for preservation.  -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"

#include <map>

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "preserve-nvvm"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif

namespace {

class PreserveNVVM final : public FunctionPass {
public:
  static char ID;
  bool Begin;
  PreserveNVVM(bool Begin = true) : FunctionPass(ID), Begin(Begin) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    bool changed = false;
    std::map<std::string, std::pair<std::string, std::string>> Implements;
    for (std::string T : {"", "f"}) {
      // sincos, sinpi, cospi, sincospi, cyl_bessel_i1
      for (std::string name :
           {"sin",        "cos",     "tan",       "log2",   "exp",    "exp2",
            "exp10",      "cosh",    "sinh",      "tanh",   "atan2",  "atan",
            "asin",       "acos",    "log",       "log10",  "log1p",  "acosh",
            "asinh",      "atanh",   "expm1",     "hypot",  "rhypot", "norm3d",
            "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",  "cbrt",
            "rcbrt",      "j0",      "j1",        "y0",     "y1",     "yn",
            "jn",         "erf",     "erfinv",    "erfc",   "erfcx",  "erfcinv",
            "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
            "modf",       "fmod",    "remainder", "remquo", "powi",   "tgamma",
            "round",      "fdim",    "ilogb",     "logb",   "isinf",  "pow",
            "sqrt"}) {
        std::string nvname = "__nv_" + name;
        std::string llname = "llvm." + name + ".";
        std::string mathname = name;

        if (T == "f") {
          mathname += "f";
          nvname += "f";
          llname += "f32";
        } else {
          llname += "f64";
        }

        Implements[nvname] = std::make_pair(mathname, llname);
      }
    }
    auto found = Implements.find(F.getName().str());
    if (found != Implements.end()) {
      changed = true;
      if (Begin) {
        F.removeFnAttr(Attribute::AlwaysInline);
        F.addFnAttr(Attribute::NoInline);
        // As a side effect, enforces arguments
        // cannot be erased.
        F.setLinkage(Function::LinkageTypes::ExternalLinkage);
        F.addFnAttr("implements", found->second.second);
        F.addFnAttr("implements2", found->second.first);
        F.addFnAttr("enzyme_math", found->second.first);
      } else {
        F.addFnAttr(Attribute::AlwaysInline);
        F.removeFnAttr(Attribute::NoInline);
        F.setLinkage(Function::LinkageTypes::InternalLinkage);
      }
    }
    constexpr static const char gradient_handler_name[] =
        "__enzyme_register_gradient";
    constexpr static const char derivative_handler_name[] =
        "__enzyme_register_derivative";
    constexpr static const char splitderivative_handler_name[] =
        "__enzyme_register_splitderivative";
    for (GlobalVariable &g : F.getParent()->globals()) {
      if (g.getName().contains(gradient_handler_name) ||
          g.getName().contains(derivative_handler_name) ||
          g.getName().contains(splitderivative_handler_name) ||
          g.getName().contains("__enzyme_inactivefn") ||
          g.getName().contains("__enzyme_function_like") ||
          g.getName().contains("__enzyme_allocation_like")) {
        if (g.hasInitializer()) {
          Value *V = g.getInitializer();
          while (1) {
            if (auto CE = dyn_cast<ConstantExpr>(V)) {
              V = CE->getOperand(0);
              continue;
            }
            if (auto CA = dyn_cast<ConstantAggregate>(V)) {
              V = CA->getOperand(0);
              continue;
            }
            break;
          }
          if (V == &F) {
            changed = true;
            if (Begin && !F.hasFnAttribute("prev_fixup")) {
              F.addFnAttr("prev_fixup");
              if (F.hasFnAttribute(Attribute::AlwaysInline))
                F.addFnAttr("prev_always_inline");
              if (F.hasFnAttribute(Attribute::NoInline))
                F.addFnAttr("prev_no_inline");
              F.addFnAttr("prev_linkage", std::to_string(F.getLinkage()));
              F.setLinkage(Function::LinkageTypes::ExternalLinkage);
              F.addFnAttr(Attribute::NoInline);
              F.removeFnAttr(Attribute::AlwaysInline);
            }
            break;
          }
        }
      }
    }
    SmallVector<GlobalVariable *, 1> toErase;
    for (GlobalVariable &g : F.getParent()->globals()) {
      if (g.getName().contains("__enzyme_inactive_global")) {
        if (g.hasInitializer()) {
          Value *V = g.getInitializer();
          while (1) {
            if (auto CE = dyn_cast<ConstantExpr>(V)) {
              V = CE->getOperand(0);
              continue;
            }
            if (auto CA = dyn_cast<ConstantAggregate>(V)) {
              V = CA->getOperand(0);
              continue;
            }
            break;
          }
          if (auto GV = cast<GlobalVariable>(V)) {
            GV->setMetadata("enzyme_inactive", MDNode::get(g.getContext(), {}));
            toErase.push_back(&g);
            changed = true;
          } else {
            llvm::errs() << "Param of __enzyme_inactive_global must be a "
                            "global variable"
                         << g << "\n"
                         << *V << "\n";
            llvm_unreachable("__enzyme_inactivefn");
          }
        }
      }
      if (g.getName().contains("__enzyme_inactivefn")) {
        if (g.hasInitializer()) {
          Value *V = g.getInitializer();
          while (1) {
            if (auto CE = dyn_cast<ConstantExpr>(V)) {
              V = CE->getOperand(0);
              continue;
            }
            if (auto CA = dyn_cast<ConstantAggregate>(V)) {
              V = CA->getOperand(0);
              continue;
            }
            break;
          }
          if (auto F = cast<Function>(V)) {
            F->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(g.getContext(), "enzyme_inactive"));
            toErase.push_back(&g);
            changed = true;
          } else {
            llvm::errs() << "Param of __enzyme_inactivefn must be a "
                            "constant function"
                         << g << "\n"
                         << *V << "\n";
            llvm_unreachable("__enzyme_inactivefn");
          }
        }
      }
      if (g.getName().contains("__enzyme_function_like")) {
        if (g.hasInitializer()) {
          auto CA = dyn_cast<ConstantAggregate>(g.getInitializer());
          if (!CA || CA->getNumOperands() < 2) {
            llvm::errs() << "Use of "
                         << "enzyme_function_like"
                         << " must be a "
                            "constant of size at least "
                         << 2 << " " << g << "\n";
            llvm_unreachable("enzyme_function_like");
          }
          Value *V = CA->getOperand(0);
          Value *name = CA->getOperand(1);
          while (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
          }
          while (auto CE = dyn_cast<ConstantExpr>(name)) {
            name = CE->getOperand(0);
          }
          StringRef nameVal;
          if (auto GV = dyn_cast<GlobalVariable>(name))
            if (GV->isConstant())
              if (auto C = GV->getInitializer())
                if (auto CA = dyn_cast<ConstantDataArray>(C))
                  if (CA->getType()->getElementType()->isIntegerTy(8) &&
                      CA->isCString())
                    nameVal = CA->getAsCString();

          if (nameVal == "") {
            llvm::errs() << *name << "\n";
            llvm::errs() << "Use of "
                         << "enzyme_function_like"
                         << "requires a non-empty function name"
                         << "\n";
            llvm_unreachable("enzyme_function_like");
          }
          if (auto F = cast<Function>(V)) {
            F->addAttribute(
                AttributeList::FunctionIndex,
                Attribute::get(g.getContext(), "enzyme_math", nameVal));
            toErase.push_back(&g);
            changed = true;
          } else {
            llvm::errs() << "Param of __enzyme_function_like must be a "
                            "constant function"
                         << g << "\n"
                         << *V << "\n";
            llvm_unreachable("__enzyme_inactivefn");
          }
        }
      }
    }

    for (auto G : toErase) {
      if (auto V = F.getParent()->getGlobalVariable("llvm.used")) {
        auto C = cast<ConstantArray>(V->getInitializer());
        SmallVector<Constant *, 1> toKeep;
        bool found = false;
        for (unsigned i = 0; i < C->getNumOperands(); i++) {
          Value *Op = C->getOperand(i)->stripPointerCasts();
          if (Op == G)
            found = true;
          else
            toKeep.push_back(C->getOperand(i));
        }
        if (found) {
          if (toKeep.size()) {
            auto CA = ConstantArray::get(
                ArrayType::get(C->getType()->getElementType(), toKeep.size()),
                toKeep);
            GlobalVariable *NGV = new GlobalVariable(
                CA->getType(), V->isConstant(), V->getLinkage(), CA, "",
                V->getThreadLocalMode());
            V->getParent()->getGlobalList().insert(V->getIterator(), NGV);
            NGV->takeName(V);

            // Nuke the old list, replacing any uses with the new one.
            if (!V->use_empty()) {
              Constant *VV = NGV;
              if (VV->getType() != V->getType())
                VV = ConstantExpr::getBitCast(VV, V->getType());
              V->replaceAllUsesWith(VV);
            }
          }
          V->eraseFromParent();
        }
      }
      changed = true;
      G->replaceAllUsesWith(ConstantPointerNull::get(G->getType()));
      G->eraseFromParent();
    }

    if (!Begin && F.hasFnAttribute("prev_fixup")) {
      changed = true;
      F.removeFnAttr("prev_fixup");
      if (F.hasFnAttribute("prev_always_inline")) {
        F.addFnAttr(Attribute::AlwaysInline);
        F.removeFnAttr("prev_always_inline");
      }
      if (F.hasFnAttribute("prev_no_inline")) {
        F.removeFnAttr("prev_no_inline");
      } else {
        F.removeFnAttr(Attribute::NoInline);
      }
      int64_t val;
      F.getFnAttribute("prev_linkage").getValueAsString().getAsInteger(10, val);
      F.setLinkage((Function::LinkageTypes)val);
    }
    return changed;
  }
};

} // namespace

char PreserveNVVM::ID = 0;

static RegisterPass<PreserveNVVM> X("preserve-nvvm", "Preserve NVVM Pass");

FunctionPass *createPreserveNVVMPass(bool Begin) {
  return new PreserveNVVM(Begin);
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddPreserveNVVMPass(LLVMPassManagerRef PM, uint8_t Begin) {
  unwrap(PM)->add(createPreserveNVVMPass((bool)Begin));
}
