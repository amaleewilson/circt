//===- AqedParser.cpp - .smt to AqedLIB dialect parser ----------------------===//
//
// This implements a .smt file parser.
//
//===----------------------------------------------------------------------===//

#include "circt/AqedParser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "AqedLexer.h"
#include "mlir/IR/BuiltinOps.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Translation.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/IR/Dialect.h"

#include <iostream>

using namespace circt;
using namespace aqed;
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// GlobalAqedParserState
//===----------------------------------------------------------------------===//

namespace {
/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position.  This is separated out from the parser
/// so that individual subparsers can refer to the same state.
struct GlobalAqedParserState {
  GlobalAqedParserState(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                       AqedParserOptions options)
      : context(context), options(options), lex(sourceMgr, context),
        curToken(lex.lexToken()) {}

  /// The context we're parsing into.
  MLIRContext *const context;

  // Options that control the behavior of the parser.
  const AqedParserOptions options;

  /// The lexer for the source file we're parsing.
  AqedLexer lex;

  /// This is the next token that hasn't been consumed yet.
  AqedToken curToken;

private:
  GlobalAqedParserState(const GlobalAqedParserState &) = delete;
  void operator=(const GlobalAqedParserState &) = delete;
};
} // end anonymous namespace

namespace {
/// This class implements logic common to all levels of the parser, including
/// things like types and helper logic.
struct AqedParser {
  AqedParser(GlobalAqedParserState &state) : state(state) {}

  // Helper methods to get stuff from the parser-global state.
  GlobalAqedParserState &getState() const { return state; }
  MLIRContext *getContext() const { return state.context; }
  const llvm::SourceMgr &getSourceMgr() { return state.lex.getSourceMgr(); }

  /// Return the current token the parser is inspecting.
  const AqedToken &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(state.curToken.getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return state.lex.translateLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(AqedToken::Kind kind) {
    if (state.curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(AqedToken::eof, AqedToken::error) &&
           "shouldn't advance past EOF or errors");
    state.curToken = state.lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(AqedToken::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Capture the current token's location into the specified value.  This
  /// always succeeds.
  ParseResult parseGetLocation(SMLoc &loc);
  ParseResult parseGetLocation(Location &loc);

  /// Capture the current token's spelling into the specified value.  This
  /// always succeeds.
  ParseResult parseGetSpelling(StringRef &spelling) {
    spelling = getTokenSpelling();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(AqedToken::Kind expectedToken, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.
  ParseResult parseListUntil(AqedToken::Kind rightToken,
                             const std::function<ParseResult()> &parseElement);

private:
  AqedParser(const AqedParser &) = delete;
  void operator=(const AqedParser &) = delete;

  /// AqedParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to the GlobalAqedParserState class.
  GlobalAqedParserState &state;
};
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic AqedParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(AqedToken::error))
    diag.abandon();
  return diag;
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Capture the current token's location into the specified value.  This
/// always succeeds.
ParseResult AqedParser::parseGetLocation(SMLoc &loc) {
  loc = getToken().getLoc();
  return success();
}

ParseResult AqedParser::parseGetLocation(Location &loc) {
  loc = translateLocation(getToken().getLoc());
  return success();
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult AqedParser::parseToken(AqedToken::Kind expectedToken,
                                  const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a list of elements, terminated with an arbitrary token.
ParseResult
AqedParser::parseListUntil(AqedToken::Kind rightToken,
                          const std::function<ParseResult()> &parseElement) {

  while (!consumeIf(rightToken)) {
    if (parseElement())
      return failure();
  }
  return success();
}



//===----------------------------------------------------------------------===//
// AqedSpecParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the outer level of the parser, including things
/// like circuit and module.
struct AqedSpecParser : public AqedParser {
  explicit AqedSpecParser(GlobalAqedParserState &state, ModuleOp mlirModule)
      : AqedParser(state), mlirModule(mlirModule) {}

  ParseResult parseAqedSpec();

private:
  ModuleOp mlirModule;
};

} // end anonymous namespace

/// file ::= circuit
/// circuit ::= 'circuit' id ':' info? INDENT module* DEDENT EOF
///
ParseResult AqedSpecParser::parseAqedSpec() {

  SMLoc smloc = getToken().getLoc();
  Location loc = translateLocation(smloc);
  StringAttr name = StringAttr::get("aqed", getContext());

  // Ports that are always there
  SmallVector<rtl::ModulePortInfo, 10> ports;
  SmallVector<rtl::ModulePortInfo, 10> output_ports;
  int argNum = 0;
  rtl::ModulePortInfo clk;
  clk.name = StringAttr::get("clk", getContext());
  clk.type = IntegerType::get(getContext(), 1);
  clk.direction = rtl::PortDirection::INPUT;
  clk.argNum = argNum++;
  ports.push_back(clk);

  rtl::ModulePortInfo reset;
  reset.name =  StringAttr::get("reset", getContext());
  reset.type = IntegerType::get(getContext(), 1);
  reset.direction = rtl::PortDirection::INPUT;
  reset.argNum = argNum++;
  ports.push_back(reset);

  rtl::ModulePortInfo exec_dup;
  exec_dup.name =  StringAttr::get("exec_dup", getContext());
  exec_dup.type = IntegerType::get(getContext(), 1);
  exec_dup.direction = rtl::PortDirection::INPUT;
  exec_dup.argNum = argNum++;
  ports.push_back(exec_dup);

  rtl::ModulePortInfo data_in;
  data_in.name =  StringAttr::get("data_in", getContext());
  data_in.type = IntegerType::get(getContext(), 1);
  data_in.direction = rtl::PortDirection::INPUT;
  data_in.argNum = argNum++;
  ports.push_back(data_in);

  // TODO: get valid_out from the input! 

  rtl::ModulePortInfo data_out_in;
  data_out_in.name =  StringAttr::get("data_out_in", getContext());
  data_out_in.type = IntegerType::get(getContext(), 1);
  data_out_in.direction = rtl::PortDirection::INPUT;
  data_out_in.argNum = argNum++;
  ports.push_back(data_out_in);

  rtl::ModulePortInfo data_out;
  data_out.name =  StringAttr::get("data_out", getContext());
  data_out.type = IntegerType::get(getContext(), 1);
  data_out.direction = rtl::PortDirection::OUTPUT;
  ports.push_back(data_out);
  output_ports.push_back(data_out);

  rtl::ModulePortInfo qed_done;
  qed_done.name =  StringAttr::get("qed_done", getContext());
  qed_done.type = IntegerType::get(getContext(), 1);
  qed_done.direction = rtl::PortDirection::OUTPUT;
  ports.push_back(qed_done);

  // rtl::ModulePortInfo qed_check;
  // qed_check.name =  StringAttr::get("qed_check", getContext());
  // qed_check.type = IntegerType::get(getContext(), 1);
  // qed_check.direction = rtl::PortDirection::OUTPUT;
  // ports.push_back(qed_check);

  // Multiple outputs seems to break things? 

  // Create the top-level circuit op in the MLIR module.
  OpBuilder b(mlirModule.getBodyRegion());
  // Testing building RTL module.
  auto circuit = b.create<rtl::RTLModuleOp>(loc, name, ports);


  OpBuilder b2(circuit.getBodyRegion());
  
  // SmallVector<rtl::RTLModuleOp, 4> outputs;
  // outputs.push_back(data_out);
  // outputs.push_back(qed_done);
  // outputs.push_back(qed_check);
  // // outputs.push_back(circuit.getBodyBlock()->getArgument(0));
  // outputs.push_back(circuit.getBodyBlock()->getArgument(1));
  // outputs.push_back(circuit.getBodyBlock()->getArgument(2));
  
  // SmallVector<Value, 4> actual_outputs;
  // for (auto port : outputs) {
  //   Value newArg = b2.create<sv::WireOp>(port.type, port.getName().str() + ".output");
  //   actual_outputs.push_back(newArg);
  // }
  

  // auto outputOp = circuit.getBodyBlock()->getTerminator();
  // outputOp->setOperands(actual_outputs);

  // Parse the A-QED information
  // while (true) {
  //   switch (getToken().getKind()) {
  //   // If we got to the end of the file, then we're done.
  //   case AqedToken::eof:
  //     return success();

  //   // If we got an error token, then the lexer already emitted an error,
  //   // just stop.  We could introduce error recovery if there was demand for
  //   // it.
  //   case AqedToken::error:
  //     return failure(); 

  //   default:
  //     emitError("unexpected token in circuit");
  //     return failure();

  //   case AqedToken::kw_accvalid: {
  //     consumeToken(AqedToken::kw_accvalid);
  //     consumeToken(AqedToken::string);
      
  //   }
  //   case AqedToken::kw_accready: {
      
  //   }
  //   }
  // }

  

  // OpBuilder b2(circuit.getBodyRegion());
  // auto outputOp = circuit.getBodyBlock()->getTerminator();
  // // segfault
  // auto arg = circuit.getBodyBlock()->getArgument(0);
  // outputOp->setOperands(arg);


  // // Parse any contained modules.
  // while (true) {
  //   switch (getToken().getKind()) {
  //   // If we got to the end of the file, then we're done.
  //   case AqedToken::eof:
  //     return success();

  //   // If we got an error token, then the lexer already emitted an error,
  //   // just stop.  We could introduce error recovery if there was demand for
  //   // it.
  //   case AqedToken::error:
  //     return failure();

  //   default:
  //     emitError("unexpected token in circuit");
  //     return failure();

  //   case AqedToken::kw_module:
  //   case AqedToken::kw_extmodule: {
  //     auto indent = getIndentation();
  //     if (!indent.hasValue())
  //       return emitError("'module' must be first token on its line"), failure();
  //     unsigned moduleIndent = indent.getValue();

  //     if (moduleIndent <= circuitIndent)
  //       return emitError("module should be indented more"), failure();

  //     FIRModuleParser mp(getState(), circuit);
  //     if (getToken().is(AqedToken::kw_module) ? mp.parseModule(moduleIndent)
  //                                            : mp.parseExtModule(moduleIndent))
  //       return failure();
  //     break;
  //   }
  //   }
  // }
}


//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified aqed spec file into the specified MLIR context.
OwningModuleRef circt::aqed::importAqed(SourceMgr &sourceMgr,
                                            MLIRContext *context,
                                            AqedParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<circt::rtl::RTLDialect>();
  context->loadDialect<circt::sv::SVDialect>();

  // This is the result module we are parsing into.
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0, context)));

  GlobalAqedParserState state(sourceMgr, context, options);
  if (AqedSpecParser(state, *module).parseAqedSpec())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  if (failed(verify(*module)))
    return {};

  return module;
}

void circt::aqed::registerAqedParserTranslation() {
  static TranslateToMLIRRegistration fromAQED(
      "import-aqed", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return importAqed(sourceMgr, context);
      });
}
