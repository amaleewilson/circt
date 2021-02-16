set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(auto-aqed
 auto-aqed.cpp
)
llvm_update_compile_flags(auto-aqed)
target_link_libraries(auto-aqed PRIVATE
  CIRCTExportVerilog
  CIRCTAqedParser
  CIRCTSVTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
  )
