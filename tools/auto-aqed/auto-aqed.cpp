//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/AqedParser.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> disableOptimization("disable-opt",
                                         cl::desc("disable optimizations"));

enum OutputFormatKind { OutputMLIR, OutputVerilog, OutputDisabled };

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(clEnumValN(OutputMLIR, "mlir", "Emit MLIR dialect"),
               clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
               clEnumValN(OutputDisabled, "disable-output",
                          "Do not output anything")),
    cl::init(OutputMLIR));

/// Process a single buffer of the input.
static LogicalResult
processBuffer(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              raw_ostream &os) {
  MLIRContext context;

  // Register our dialects.
  context.loadDialect<rtl::RTLDialect, sv::SVDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Nothing in the parser is threaded.  Disable synchronization overhead.
  context.disableMultithreading();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(true);
  applyPassManagerCLOptions(pm);

  OwningModuleRef module;

  aqed::AqedParserOptions options;
  module = importAqed(sourceMgr, &context, options);

//   // If we have optimizations enabled, clean it up.
//   if (!disableOptimization) {
//     pm.addPass(createCSEPass());
//     pm.addPass(createCanonicalizerPass());
//   }

  if (!module)
    return failure();

  // Allow optimizations to run multithreaded.
  context.disableMultithreading(false);

    // If enabled, run the optimizer.
  if (!disableOptimization) {
    pm.addNestedPass<rtl::RTLModuleOp>(sv::createAlwaysFusionPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
  }


  if (failed(pm.run(module.get())))
    return failure();

  // Finally, emit the output.
  switch (outputFormat) {
  case OutputMLIR:
    module->print(os);
    return success();
  case OutputDisabled:
    return success();
  case OutputVerilog:
    return exportVerilog(module.get(), os);
  }
};

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "circt modular optimizer driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(processBuffer(std::move(input), output->os())))
    return 1;

  output->keep();
  return 0;
}
