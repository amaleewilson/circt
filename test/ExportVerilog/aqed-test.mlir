// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK-LABEL: module aqed(
rtl.module @aqed(%clk: i1, %clk_en: i1, %reset: i1, 
                 %flush: i1, %exec_dup: i1, %data_in: i16, 
                 %valid_out: i1, %ren_in: i1, %data_out_in: i16, 
                 %wen_in: i1) -> (%data_out: i16, %qed_done: i1, %qed_check: i1) {


    %orig_in_reg = sv.reg : !rtl.inout<i16>
    %orig_in = sv.read_inout %orig_in_reg : !rtl.inout<i16>
    %orig_out_reg = sv.reg : !rtl.inout<i16>
    %orig_out = sv.read_inout %orig_out_reg : !rtl.inout<i16>
    %dup_out_reg = sv.reg : !rtl.inout<i16>
    %dup_out = sv.read_inout %dup_out_reg : !rtl.inout<i16>

    %orig_val_reg = sv.reg : !rtl.inout<i32>
    %orig_val = sv.read_inout %orig_val_reg : !rtl.inout<i32>

    %dup_val_reg = sv.reg : !rtl.inout<i32>
    %dup_val = sv.read_inout %dup_val_reg : !rtl.inout<i32>


    %match_reg = sv.reg : !rtl.inout<i1>

    %in_count_reg = sv.reg : !rtl.inout<i32>
    %in_count = sv.read_inout %in_count_reg : !rtl.inout<i32>

    %out_count_reg = sv.reg : !rtl.inout<i32>
    %out_count = sv.read_inout %out_count_reg : !rtl.inout<i32>


    %orig_issued_reg = sv.reg : !rtl.inout<i1>
    %orig_issued = sv.read_inout %orig_issued_reg : !rtl.inout<i1>
    %dup_issued_reg = sv.reg : !rtl.inout<i1>
    %dup_issued = sv.read_inout %dup_issued_reg : !rtl.inout<i1>
    %dup_done_reg = sv.reg : !rtl.inout<i1>
    %dup_done = sv.read_inout %dup_done_reg : !rtl.inout<i1>

    %issue_orig_wire = sv.wire : !rtl.inout<i1>
    %issue_orig = sv.read_inout %issue_orig_wire : !rtl.inout<i1>
    %issue_dup_wire = sv.wire : !rtl.inout<i1>
    %issue_dup = sv.read_inout %issue_dup_wire : !rtl.inout<i1>
    %issue_other_wire = sv.wire : !rtl.inout<i1>
    %issue_other = sv.read_inout %issue_other_wire : !rtl.inout<i1>

    // Ugh, so annoying to do ~ like this. 
    %allone = rtl.constant (-1 : i1) : i1
    %false = rtl.constant (0 : i1) : i1
    %true = rtl.constant (1 : i1) : i1
    %not_reset = rtl.xor %reset, %allone : i1
    %not_orig_issued = rtl.xor %orig_issued, %allone : i1
    %not_flush = rtl.xor %flush, %allone : i1

    %big_and_issue_orig = rtl.and %not_reset, %exec_dup, %wen_in, %not_orig_issued, %not_flush : i1
    sv.connect %issue_orig_wire,  %big_and_issue_orig : i1

    %not_issue_orig = rtl.xor %big_and_issue_orig, %allone : i1
    %not_issue_dup = rtl.xor %issue_dup, %allone : i1

    %big_and_issue_other = rtl.and %not_reset, %not_issue_orig, %not_issue_dup, %wen_in, %not_flush : i1
    sv.connect %issue_other_wire,  %big_and_issue_other : i1
    
    sv.always posedge %clk {
        %condition1 = rtl.and %clk_en, %issue_orig : i1
        %condition2 = rtl.and %clk_en, %issue_dup : i1
        sv.if %reset {
            sv.passign %orig_issued_reg, %false : i1
            sv.passign %dup_issued_reg, %false : i1
        } else {
            sv.if %condition1 {
                sv.passign %orig_issued_reg, %true : i1
            } else {
                sv.if %condition2 {
                    sv.passign %dup_issued_reg, %true : i1
                }
            }
        }
    }

    %not_dup_issued = rtl.xor %dup_issued, %allone : i1
    %big_and_issue_dup = rtl.and %not_reset, %exec_dup, %orig_issued, %not_dup_issued, %not_flush : i1
    sv.connect %issue_dup_wire,  %big_and_issue_other : i1
//    assign issue_dup = (~reset)&(exec_dup)&(orig_issued)&(wen_in)&(~dup_issued)&(~flush);

    %two32 = rtl.constant (4294967295 : i32) : i32
    %false16 = rtl.constant (0 : i16) : i16
    %false32 = rtl.constant (0 : i32) : i32
    %one32 = rtl.constant (1 : i32) : i32

    sv.always posedge %clk {
        %condition1 = rtl.and %clk_en, %issue_orig : i1
        %condition2 = rtl.and %clk_en, %issue_dup : i1
        %condition3 = rtl.and %clk_en, %issue_other : i1
        sv.if %reset {
            sv.passign %orig_in_reg, %false16 : i16
            sv.passign %orig_val_reg, %two32 : i32
            sv.passign %in_count_reg, %false32 : i32
            sv.passign %dup_val_reg, %two32 : i32
        }
        sv.if %condition1 {
            sv.passign %orig_in_reg, %data_in : i16
            sv.passign %orig_val_reg, %in_count : i32
            %add_one = rtl.add %in_count, %one32 : i32
            sv.passign %in_count_reg, %add_one : i32
        }
        sv.if %condition2 {
            sv.passign %dup_val_reg, %in_count : i32
            %add_one = rtl.add %in_count, %one32 : i32
            sv.passign %in_count_reg, %add_one : i32
        }
        sv.if %condition3 {
            %add_one = rtl.add %in_count, %one32 : i32
            sv.passign %in_count_reg, %add_one : i32
        }
    }

    %tern_approx = rtl.mux %issue_dup, %orig_in, %data_in : i16
    %the_data_output = sv.reg : !rtl.inout<i16>
    sv.connect %the_data_output, %tern_approx : i16

    %the_data_output_read = sv.read_inout %the_data_output : !rtl.inout<i16>

    sv.always posedge %clk {
        %orig_equal = rtl.icmp eq %out_count, %orig_val : i32
        %dup_equal = rtl.icmp eq %out_count, %dup_val : i32
        %condition1 = rtl.and %clk_en, %wen_in, %valid_out, %orig_equal : i1
        %condition2 = rtl.and %clk_en, %wen_in, %valid_out, %dup_equal : i1
        %condition3 = rtl.and %clk_en, %issue_other : i1
        sv.if %reset {
            sv.passign %out_count_reg, %false32 : i32
            sv.passign %orig_out_reg, %false16 : i16
            sv.passign %dup_out_reg, %false16 : i16
            sv.passign %dup_done_reg, %false : i1
        }
        sv.if %condition1 {
            sv.passign %orig_out_reg, %data_out_in : i16
            %add_one_out = rtl.add %out_count, %one32 : i32
            sv.passign %out_count_reg, %add_one_out : i32
        }
        sv.if %condition2 {
            sv.passign %dup_out_reg, %data_out_in : i16
            %add_one_out_dup = rtl.add %out_count, %one32 : i32
            sv.passign %out_count_reg, %add_one_out_dup : i32
            sv.passign %dup_done_reg, %true : i1
        }
        sv.if %condition3 {
            %add_one_out_dup2 = rtl.add %out_count, %one32 : i32
            sv.passign %out_count_reg, %add_one_out_dup2 : i32
        }
    }

    %match = rtl.xor %orig_out, %dup_out : i16
    %allone16 = rtl.constant (-1 : i16) : i16
    %match_not = rtl.xor %match, %allone16 : i16
    %match_reduced = rtl.andr %match_not : i16

    rtl.output %the_data_output_read, %dup_done, %match_reduced: i16, i1, i1
}

// TODOS to make this work: 
// - need a bitwise negation op in SV or RTL
// - 