//===- AqedParser.h - .smt to AqedLIB dialect parser --------------*- C++ -*-===//
//
// Defines the interface to the .smt file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AQEDPARSER_H
#define CIRCT_DIALECT_AQEDPARSER_H

namespace llvm {
class SourceMgr;
}

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace circt {
namespace aqed {

struct AqedParserOptions {
  /// If this is set to true, the @info locators are ignored, and the locations
  /// are set to the location in the .smt file.
  bool ignoreInfoLocators = false;
};

mlir::OwningModuleRef importAqed(llvm::SourceMgr &sourceMgr,
                                   mlir::MLIRContext *context,
                                   AqedParserOptions options = {});
void registerAqedParserTranslation();

} // namespace aqed
} // namespace circt

#endif // CIRCT_DIALECT_AQEDPARSER_H