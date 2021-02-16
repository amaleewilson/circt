file(GLOB globbed *.cpp)

add_mlir_library(CIRCTAqedParser
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTRTL
  CIRCTSV
  MLIRTranslation
  )