//===- TypeTree.cpp - Implementation of Type Analysis Type Trees-----------===//
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
// This file contains the implementation TypeTrees -- a class
// representing all of the underlying types of a particular LLVM value. This
// consists of a map of memory offsets to an underlying ConcreteType. This
// permits TypeTrees to represent distinct underlying types at different
// locations. Presently, TypeTree's have both a fixed depth of memory lookups
// and a maximum offset to ensure that Type Analysis eventually terminates.
// In the future this should be modified to better represent recursive types
// rather than limiting the depth.
//
//===----------------------------------------------------------------------===//
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"

#include "TypeTree.h"

using namespace llvm;

// TODO keep type information that is striated
// e.g. if you have an i8* [0:Int, 8:Int] => i64* [0:Int, 1:Int]
// After a depth len into the index tree, prune any lookups that are not {0} or
// {-1} Todo handle {double}** to double** where there is a 0 removed
TypeTree TypeTree::KeepForCast(const llvm::DataLayout &DL, llvm::Type *From,
                                 llvm::Type *To) const {
  assert(From);
  assert(To);

  bool FromOpaque = isa<StructType>(From) && cast<StructType>(From)->isOpaque();
  bool ToOpaque = isa<StructType>(To) && cast<StructType>(To)->isOpaque();

  TypeTree Result;

  for (auto &pair : mapping) {

    TypeTree SubResult;

    if (pair.first.size() == 0) {
      SubResult.insert(pair.first, pair.second);
      goto add;
    }

    // Only consider casts of non-opaque types
    // This requirement exists because we need the sizes
    // of types to ensure bounds are appropriately applied
    if (!FromOpaque && !ToOpaque) {
      uint64_t Fromsize = (DL.getTypeSizeInBits(From) + 7) / 8;
      if (Fromsize == 0) {
        llvm::errs() << "Found FromType of size 0 -- " << *From << "\n";
        assert(0 && "Found FromType of size 0");
        llvm_unreachable("Found FromType of size 0");
      }
      uint64_t Tosize = (DL.getTypeSizeInBits(To) + 7) / 8;
      if (Tosize == 0) {
        llvm::errs() << "Found ToType of size 0 -- " << *To << "\n";
        assert(0 && "Found ToType of size 0");
        llvm_unreachable("Found ToType of size 0");
      }

      // If the sizes are the same, whatever the original one is okay [ since
      // tomemory[ i*sizeof(from) ] indeed the start of an object of type to
      // since tomemory is "aligned" to type to
      if (Fromsize == Tosize) {
        SubResult.insert(pair.first, pair.second);
        goto add;
      }

      // If the offset is a fixed (non-repeating) value, it's to include
      // directly.
      if (pair.first[0] != -1) {
        SubResult.insert(pair.first, pair.second);
        goto add;
      } else {
        // Case where pair.first[0] == -1

        if (Fromsize < Tosize) {
          if (Tosize % Fromsize == 0) {
            // TODO should really be at each offset do a -1
            SubResult.insert(pair.first, pair.second);
            goto add;
          } else {
            auto tmp(pair.first);
            tmp[0] = 0;
            SubResult.insert(tmp, pair.second);
            goto add;
          }
        } else {
          // fromsize > tosize
          // TODO should really insert all indices which are multiples of
          // fromsize
          auto tmp(pair.first);
          tmp[0] = 0;
          SubResult.insert(tmp, pair.second);
          goto add;
        }
      }
    }

    continue;
  add:;
    Result |= SubResult;
  }
  return Result;
}