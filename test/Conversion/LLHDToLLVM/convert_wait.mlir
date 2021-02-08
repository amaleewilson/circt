// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @llhdSuspend(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64)

// CHECK-LABEL:   llvm.func @convert_resume_timed(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i32>
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.icmp "eq" %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           llvm.cond_br %[[VAL_8]], ^bb4, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.icmp "eq" %[[VAL_6]], %[[VAL_9]] : i32
// CHECK:           llvm.cond_br %[[VAL_10]], ^bb3, ^bb5
// CHECK:         ^bb3:
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(dense<[0, 0, 1]> : vector<3xi64>) : !llvm.array<3 x i64>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_12]], %[[VAL_13]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_14]], %[[VAL_15]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<0 x i1>>>
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<ptr<array<0 x i1>>>
// CHECK:           %[[VAL_18:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_19:.*]] = llvm.extractvalue %[[VAL_11]][0 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_20:.*]] = llvm.extractvalue %[[VAL_11]][1 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_21:.*]] = llvm.extractvalue %[[VAL_11]][2 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_22:.*]] = llvm.call @llhdSuspend(%[[VAL_0]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         ^bb4:
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_23]], %[[VAL_24]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<0 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<0 x i1>>>
// CHECK:           %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr<ptr<array<0 x i1>>>
// CHECK:           llvm.return
// CHECK:         ^bb5:
// CHECK:           llvm.return
// CHECK:         }
llhd.proc @convert_resume_timed () -> () {
  %t = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  llhd.wait for %t, ^end
^end:
  llhd.halt
}

// CHECK-LABEL:   llvm.func @convert_resume_observe_partial(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: !llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>>,
// CHECK-SAME:                                              %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i32>
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "eq" %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:           llvm.cond_br %[[VAL_14]], ^bb4, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.icmp "eq" %[[VAL_12]], %[[VAL_15]] : i32
// CHECK:           llvm.cond_br %[[VAL_16]], ^bb3, ^bb5
// CHECK:         ^bb3:
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_17]], %[[VAL_18]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_19]], %[[VAL_20]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<3 x i1>>>
// CHECK:           %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<ptr<array<3 x i1>>>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_19]], %[[VAL_24]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_23]], %[[VAL_25]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_19]], %[[VAL_26]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_23]], %[[VAL_27]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_19]], %[[VAL_28]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_23]], %[[VAL_29]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_30:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_19]], %[[VAL_20]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_31:.*]] = llvm.load %[[VAL_30]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_33:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_19]], %[[VAL_31]]] : (!llvm.ptr<array<3 x i1>>, i32, i64) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_32]], %[[VAL_33]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_34:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_19]], %[[VAL_20]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_35:.*]] = llvm.load %[[VAL_34]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_37:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_19]], %[[VAL_35]]] : (!llvm.ptr<array<3 x i1>>, i32, i64) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_36]], %[[VAL_37]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_38:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>> to !llvm.ptr<i8>
// CHECK:           llvm.return
// CHECK:         ^bb4:
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_41:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_39]], %[[VAL_40]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<3 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<3 x i1>>>
// CHECK:           %[[VAL_42:.*]] = llvm.load %[[VAL_41]] : !llvm.ptr<ptr<array<3 x i1>>>
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_45:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_39]], %[[VAL_43]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_44]], %[[VAL_45]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_48:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_39]], %[[VAL_46]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_47]], %[[VAL_48]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_51:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_39]], %[[VAL_49]]] : (!llvm.ptr<array<3 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_50]], %[[VAL_51]] : !llvm.ptr<i1>
// CHECK:           llvm.return
// CHECK:         ^bb5:
// CHECK:           llvm.return
// CHECK:         }
llhd.proc @convert_resume_observe_partial (%in0 : !llhd.sig<i1>, %in1 : !llhd.sig<i32>) -> (%out0 : !llhd.sig<i20>) {
  llhd.wait (%in0, %out0 : !llhd.sig<i1>, !llhd.sig<i20>), ^end
^end:
  llhd.halt
}

// CHECK-LABEL:   llvm.func @convert_resume_timed_observe(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>>,
// CHECK-SAME:                                            %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<i32>
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.icmp "eq" %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           llvm.cond_br %[[VAL_10]], ^bb4, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.icmp "eq" %[[VAL_8]], %[[VAL_11]] : i32
// CHECK:           llvm.cond_br %[[VAL_12]], ^bb3, ^bb5
// CHECK:         ^bb3:
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(dense<[0, 0, 1]> : vector<3xi64>) : !llvm.array<3 x i64>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_14]], %[[VAL_15]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_16]], %[[VAL_17]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<1 x i1>>>
// CHECK:           %[[VAL_19:.*]] = llvm.load %[[VAL_18]] : !llvm.ptr<ptr<array<1 x i1>>>
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_16]], %[[VAL_17]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_21:.*]] = llvm.load %[[VAL_20]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_19]]{{\[}}%[[VAL_16]], %[[VAL_21]]] : (!llvm.ptr<array<1 x i1>>, i32, i64) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_22]], %[[VAL_23]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_24:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_25:.*]] = llvm.extractvalue %[[VAL_13]][0 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_26:.*]] = llvm.extractvalue %[[VAL_13]][1 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_27:.*]] = llvm.extractvalue %[[VAL_13]][2 : i32] : !llvm.array<3 x i64>
// CHECK:           %[[VAL_28:.*]] = llvm.call @llhdSuspend(%[[VAL_0]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         ^bb4:
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_31:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_29]], %[[VAL_30]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<1 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<1 x i1>>>
// CHECK:           %[[VAL_32:.*]] = llvm.load %[[VAL_31]] : !llvm.ptr<ptr<array<1 x i1>>>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[VAL_32]]{{\[}}%[[VAL_29]], %[[VAL_33]]] : (!llvm.ptr<array<1 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_34]], %[[VAL_35]] : !llvm.ptr<i1>
// CHECK:           llvm.return
// CHECK:         ^bb5:
// CHECK:           llvm.return
// CHECK:         }
llhd.proc @convert_resume_timed_observe (%in0 : !llhd.sig<i32>) -> () {
  %t = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  llhd.wait for %t, (%in0 : !llhd.sig<i32>), ^end
^end:
  llhd.halt
}

// CHECK-LABEL:   llvm.func @convert_resume_observe_all(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_5]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i32>
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.icmp "eq" %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           llvm.cond_br %[[VAL_12]], ^bb4, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "eq" %[[VAL_10]], %[[VAL_13]] : i32
// CHECK:           llvm.cond_br %[[VAL_14]], ^bb3, ^bb5
// CHECK:         ^bb3:
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i32>
// CHECK:           llvm.store %[[VAL_15]], %[[VAL_16]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<2 x i1>>>
// CHECK:           %[[VAL_20:.*]] = llvm.load %[[VAL_19]] : !llvm.ptr<ptr<array<2 x i1>>>
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_20]]{{\[}}%[[VAL_17]], %[[VAL_22]]] : (!llvm.ptr<array<2 x i1>>, i32, i64) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_23]], %[[VAL_24]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_28:.*]] = llvm.getelementptr %[[VAL_20]]{{\[}}%[[VAL_17]], %[[VAL_26]]] : (!llvm.ptr<array<2 x i1>>, i32, i64) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_27]], %[[VAL_28]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_29:.*]] = llvm.bitcast %[[VAL_1]] : !llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>> to !llvm.ptr<i8>
// CHECK:           llvm.return
// CHECK:         ^bb4:
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_32:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_30]], %[[VAL_31]]] : (!llvm.ptr<struct<(i32, i32, ptr<array<2 x i1>>, struct<()>)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<array<2 x i1>>>
// CHECK:           %[[VAL_33:.*]] = llvm.load %[[VAL_32]] : !llvm.ptr<ptr<array<2 x i1>>>
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_33]]{{\[}}%[[VAL_30]], %[[VAL_34]]] : (!llvm.ptr<array<2 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_35]], %[[VAL_36]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(0 : i32) : i1
// CHECK:           %[[VAL_39:.*]] = llvm.getelementptr %[[VAL_33]]{{\[}}%[[VAL_30]], %[[VAL_37]]] : (!llvm.ptr<array<2 x i1>>, i32, i32) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_38]], %[[VAL_39]] : !llvm.ptr<i1>
// CHECK:           llvm.return
// CHECK:         ^bb5:
// CHECK:           llvm.return
// CHECK:         }
llhd.proc @convert_resume_observe_all (%in0 : !llhd.sig<i1>) -> (%out0 : !llhd.sig<i20>) {
  llhd.wait (%out0, %in0: !llhd.sig<i20>, !llhd.sig<i1>), ^end
^end:
  llhd.halt
}
