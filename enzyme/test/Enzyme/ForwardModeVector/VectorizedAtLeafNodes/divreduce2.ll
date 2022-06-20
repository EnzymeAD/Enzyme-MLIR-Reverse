; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -early-cse-memssa -instsimplify -correlated-propagation -adce -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(i8*, ...)

; TODO optimize this style reduction

; Function Attrs: norecurse nounwind readonly uwtable
define double @alldiv(double* nocapture readonly %A, i64 %N, double %start) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %reduce = phi double [ %start, %entry ], [ %div, %body ]
  %cmp = icmp ult i64 %i, %N
  br i1 %cmp, label %body, label %end

body:
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  br label %loop

end:                                                ; preds = %9, %3
  ret double %reduce
}

; Function Attrs: nounwind uwtable
define <3 x double> @main(double* %A, <3 x double>* %dA, i64 %N, double %start) {
  %r = call <3 x double> (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), metadata !"enzyme_width", i64 3, double* %A, <3 x double>* %dA, i64 %N, double %start, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %r
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20200809072545+ef32c611aa2-1~exp1~20200809173142.193"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}


; CHECK: define internal <3 x double> @fwddiffe3alldiv(double* nocapture readonly %A, <3 x double>* %"A'", i64 %N, double %start, <3 x double> %"start'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %body ], [ 0, %entry ]
; CHECK-NEXT:   %0 = phi fast <3 x double> [ %"start'", %entry ], [ %5, %body ]
; CHECK-NEXT:   %reduce = phi double [ %start, %entry ], [ %div, %body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %N
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: body:                                             ; preds = %loop
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds <3 x double>, <3 x double>* %"A'", i64 %iv
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:   %"ld'ipl" = load <3 x double>, <3 x double>* %"gep'ipg", align 8, !tbaa !2
; CHECK-NEXT:   %ld = load double, double* %gep, align 8, !tbaa !2
; CHECK-NEXT:   %div = fdiv double %reduce, %ld
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %reduce, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %ld, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %1 = fmul fast <3 x double> %0, %.splat2
; CHECK-NEXT:   %2 = fmul fast <3 x double> %.splat, %"ld'ipl"
; CHECK-NEXT:   %3 = fsub fast <3 x double> %1, %2
; CHECK-NEXT:   %4 = fmul fast double %ld, %ld
; CHECK-NEXT:   %.splatinsert3 = insertelement <3 x double> poison, double %4, i32 0
; CHECK-NEXT:   %.splat4 = shufflevector <3 x double> %.splatinsert3, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %5 = fdiv fast <3 x double> %3, %.splat4
; CHECK-NEXT:   br label %loop

; CHECK: end:                                              ; preds = %loop
; CHECK-NEXT:   ret <3 x double> %0
; CHECK-NEXT: }