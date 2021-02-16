//===- AqedLexer.h - .smt lexer and token definitions ------------*- C++ -*-===//
//
// Defines the a Lexer and Token interface for .smt files.
//
//===----------------------------------------------------------------------===//

#ifndef AQEDTOMLIR_AQEDLEXER_H
#define AQEDTOMLIR_AQEDLEXER_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class Location;
} // namespace mlir

namespace circt {
namespace aqed {

/// This represents a specific token for .smt files.
class AqedToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "AqedTokenKinds.def"
  };

  AqedToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_wire).
  bool isKeyword() const;

  /// Given a token containing a string literal, return its value, including
  /// removing the quote characters and unescaping the contents of the string.
  /// The lexer has already verified that this token is valid.
  std::string getStringValue() const;
  static std::string getStringValue(StringRef spelling);

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

/// This implements a lexer for .fir files.
class AqedLexer {
public:
  AqedLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  AqedToken lexToken();

  mlir::Location translateLocation(llvm::SMLoc loc);

  /// Return the indentation level of the specified token or None if this token
  /// is preceded by another token on the same line.
  Optional<unsigned> getIndentation(const AqedToken &tok) const;

private:
  // Helpers.
  AqedToken formToken(AqedToken::Kind kind, const char *tokStart) {
    return AqedToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  AqedToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  AqedToken lexFileInfo(const char *tokStart);
  AqedToken lexIdentifierOrKeyword(const char *tokStart);
  AqedToken lexNumber(const char *tokStart);
  AqedToken lexFloatingPoint(const char *tokStart);
  void skipComment();
  AqedToken lexString(const char *tokStart);

  const llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;

  StringRef curBuffer;
  const char *curPtr;

  AqedLexer(const AqedLexer &) = delete;
  void operator=(const AqedLexer &) = delete;
};

} // namespace aqed
} // namespace circt

#endif // AQEDTOMLIR_AQEDLEXER_H
