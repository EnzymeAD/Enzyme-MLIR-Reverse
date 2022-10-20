; RUN: %opt < %s %loadEnzyme -enzyme -O1 -S | FileCheck %s

@.str = private unnamed_addr constant [11 x i8] c"predict, 0\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"m\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@enzyme_condition = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_interface = dso_local local_unnamed_addr global i32 0, align 4

define dso_local double @normal(double %mean, double %var) #0 {
entry:
  ret double 4.000000e+00 ; chosen by fair dice roll.
                          ; guaranteed to be random.
}

define dso_local double @normal_pdf(double noundef %mean, double noundef %var, double noundef %x) local_unnamed_addr #1 {
entry:
  %div = fdiv double 1.000000e+00, %var
  %mul = fmul double %div, 0x40040D931FF62705
  %div1 = fdiv double %mean, %var
  %sub = fsub double %x, %div1
  %square = fmul double %sub, %sub
  %mul3 = fmul double %square, -5.000000e-01
  %call4 = tail call double @exp(double noundef %mul3) #6
  %mul5 = fmul double %mul, %call4
  ret double %mul5
}

declare double @exp(double noundef) local_unnamed_addr #2

define dso_local double @normal_logpdf(double noundef %mean, double noundef %var, double noundef %x) #1 {
entry:
  %call = tail call double @normal_pdf(double noundef %mean, double noundef %var, double noundef %x)
  %call1 = tail call double @log(double noundef %call) #6
  ret double %call1
}

declare double @log(double noundef) local_unnamed_addr #2

define dso_local double @calculate_loss(double noundef %m, double noundef %b, double* nocapture noundef readonly %data, i32 noundef %n) local_unnamed_addr #3 {
entry:
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %3, %for.body ]
  ret double %loss.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %3, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv2 = sitofp i32 %0 to double
  %1 = tail call double @llvm.fmuladd.f64(double %conv2, double %m, double %b)
  %call = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double noundef %1, double noundef 1.000000e+00) #6
  %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
  %2 = load double, double* %arrayidx3, align 8
  %sub = fsub double %call, %2
  %3 = tail call double @llvm.fmuladd.f64(double %sub, double %sub, double %loss.021)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @__enzyme_sample(double (double, double)* noundef, double (double, double, double)* noundef, i8* noundef, double noundef, double noundef) local_unnamed_addr #4

declare double @llvm.fmuladd.f64(double, double, double) #5

define dso_local double @loss(double* nocapture noundef readonly %data, i32 noundef %n) #3 {
entry:
  %call = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double noundef 0.000000e+00, double noundef 1.000000e+00) #6
  %call1 = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double noundef 0.000000e+00, double noundef 1.000000e+00) #6
  %call2 = tail call double @calculate_loss(double noundef %call, double noundef %call1, double* noundef %data, i32 noundef %n)
  ret double %call2
}

define dso_local i8* @condition(double* noundef %data, i32 noundef %n, i8* noundef %trace, i8** noundef %interface) local_unnamed_addr #3 {
entry:
  %0 = load i32, i32* @enzyme_condition, align 4
  %1 = load i32, i32* @enzyme_interface, align 4
  %call = tail call i8* @__enzyme_condition(double (double*, i32)* noundef nonnull @loss, double* noundef %data, i32 noundef %n, i32 noundef %0, i8* noundef %trace, i32 noundef %1, i8** noundef %interface) #6
  ret i8* %call
}

declare i8* @__enzyme_condition(double (double*, i32)* noundef, double* noundef, i32 noundef, i32 noundef, i8* noundef, i32 noundef, i8** noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree noinline nounwind willreturn writeonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { noinline nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #6 = { nounwind }


; CHECK: define internal fastcc i8* @condition_loss(double* nocapture noundef readonly %data, i32 noundef %n, i8** nocapture readonly %0, i8* %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = getelementptr inbounds i8*, i8** %0, i64 4
; CHECK-NEXT:   %3 = bitcast i8** %2 to i8* ()**
; CHECK-NEXT:   %4 = load i8* ()*, i8* ()** %3, align 8
; CHECK-NEXT:   %trace = call i8* %4() #6
; CHECK-NEXT:   %5 = getelementptr inbounds i8*, i8** %0, i64 7
; CHECK-NEXT:   %6 = bitcast i8** %5 to i1 (i8*, i8*)**
; CHECK-NEXT:   %7 = load i1 (i8*, i8*)*, i1 (i8*, i8*)** %6, align 8
; CHECK-NEXT:   %8 = call i1 %7(i8* %1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0)) #6
; CHECK-NEXT:   br i1 %8, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %entry
; CHECK-NEXT:   %9 = alloca double, align 8
; CHECK-NEXT:   %10 = bitcast double* %9 to i8*
; CHECK-NEXT:   %11 = getelementptr inbounds i8*, i8** %0, i64 1
; CHECK-NEXT:   %12 = bitcast i8** %11 to i64 (i8*, i8*, i8*, i64)**
; CHECK-NEXT:   %13 = load i64 (i8*, i8*, i8*, i64)*, i64 (i8*, i8*, i8*, i64)** %12, align 8
; CHECK-NEXT:   %14 = call i64 %13(i8* %1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* nonnull %10, i64 8) #6
; CHECK-NEXT:   %15 = load double, double* %9, align 8
; CHECK-NEXT:   br label %condition.call.without.trace

; CHECK: condition.call.without.trace:                     ; preds = %entry, %condition.call.with.trace
; CHECK-NEXT:   %16 = phi double [ %15, %condition.call.with.trace ], [ 4.000000e+00, %entry ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %16)
; CHECK-NEXT:   %17 = alloca double, align 8
; CHECK-NEXT:   store double %16, double* %17, align 8
; CHECK-NEXT:   %18 = bitcast double* %17 to i8**
; CHECK-NEXT:   %19 = load i8*, i8** %18, align 8
; CHECK-NEXT:   %20 = getelementptr inbounds i8*, i8** %0, i64 3
; CHECK-NEXT:   %21 = bitcast i8** %20 to void (i8*, i8*, double, i8*, i64)**
; CHECK-NEXT:   %22 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %21, align 8
; CHECK-NEXT:   call void %22(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %19, i64 8) #6
; CHECK-NEXT:   %23 = load i1 (i8*, i8*)*, i1 (i8*, i8*)** %6, align 8
; CHECK-NEXT:   %24 = call i1 %23(i8* %1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0)) #6
; CHECK-NEXT:   br i1 %24, label %condition.call1.with.trace, label %condition.call1.without.trace

; CHECK: condition.call1.with.trace:                       ; preds = %condition.call.without.trace
; CHECK-NEXT:   %25 = alloca double, align 8
; CHECK-NEXT:   %26 = bitcast double* %25 to i8*
; CHECK-NEXT:   %27 = getelementptr inbounds i8*, i8** %0, i64 1
; CHECK-NEXT:   %28 = bitcast i8** %27 to i64 (i8*, i8*, i8*, i64)**
; CHECK-NEXT:   %29 = load i64 (i8*, i8*, i8*, i64)*, i64 (i8*, i8*, i8*, i64)** %28, align 8
; CHECK-NEXT:   %30 = call i64 %29(i8* %1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), i8* nonnull %26, i64 8) #6
; CHECK-NEXT:   %31 = load double, double* %25, align 8
; CHECK-NEXT:   br label %condition.call1.without.trace

; CHECK: condition.call1.without.trace:                    ; preds = %condition.call.without.trace, %condition.call1.with.trace
; CHECK-NEXT:   %32 = phi double [ %31, %condition.call1.with.trace ], [ 4.000000e+00, %condition.call.without.trace ]
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %32)
; CHECK-NEXT:   %33 = alloca double, align 8
; CHECK-NEXT:   store double %32, double* %33, align 8
; CHECK-NEXT:   %34 = bitcast double* %33 to i8**
; CHECK-NEXT:   %35 = load i8*, i8** %34, align 8
; CHECK-NEXT:   %36 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %21, align 8
; CHECK-NEXT:   call void %36(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %35, i64 8) #6
; CHECK-NEXT:   %37 = getelementptr inbounds i8*, i8** %0, i64 6
; CHECK-NEXT:   %38 = bitcast i8** %37 to i1 (i8*, i8*)**
; CHECK-NEXT:   %39 = load i1 (i8*, i8*)*, i1 (i8*, i8*)** %38, align 8
; CHECK-NEXT:   %40 = call i1 %39(i8* %1, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i64 0, i64 0)) #6
; CHECK-NEXT:   br i1 %40, label %condition.call2.with.trace, label %condition.call2.without.trace

; CHECK: condition.call2.with.trace:                       ; preds = %condition.call1.without.trace
; CHECK-NEXT:   %41 = bitcast i8** %0 to i8* (i8*, i8*)**
; CHECK-NEXT:   %42 = load i8* (i8*, i8*)*, i8* (i8*, i8*)** %41, align 8
; CHECK-NEXT:   %43 = call i8* %42(i8* %1, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i64 0, i64 0)) #6
; CHECK-NEXT:   br label %condition.call2.without.trace

; CHECK: condition.call2.without.trace:                    ; preds = %condition.call1.without.trace, %condition.call2.with.trace
; CHECK-NEXT:   %.sink = phi i8* [ %43, %condition.call2.with.trace ], [ null, %condition.call1.without.trace ]
; CHECK-NEXT:   %call26 = call fastcc { double, i8* } @condition_calculate_loss(double %16, double %32, double* %data, i32 %n, i8** %0, i8* %.sink)
; CHECK-NEXT:   %44 = extractvalue { double, i8* } %call26, 1
; CHECK-NEXT:   %45 = getelementptr inbounds i8*, i8** %0, i64 2
; CHECK-NEXT:   %46 = bitcast i8** %45 to void (i8*, i8*, i8*)**
; CHECK-NEXT:   %47 = load void (i8*, i8*, i8*)*, void (i8*, i8*, i8*)** %46, align 8
; CHECK-NEXT:   call void %47(i8* %trace, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i64 0, i64 0), i8* %44) #6
; CHECK-NEXT:   ret i8* %trace
; CHECK-NEXT: }


; CHECK: define internal fastcc { double, i8* } @condition_calculate_loss(double noundef %m, double noundef %b, double* nocapture noundef readonly %data, i32 noundef %n, i8** nocapture readonly %0, i8* %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = getelementptr inbounds i8*, i8** %0, i64 4
; CHECK-NEXT:   %3 = bitcast i8** %2 to i8* ()**
; CHECK-NEXT:   %4 = load i8* ()*, i8* ()** %3, align 8
; CHECK-NEXT:   %trace = call i8* %4() #6
; CHECK-NEXT:   %cmp19 = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   %wide.trip.count = zext i32 %n to i64
; CHECK-NEXT:   %5 = getelementptr inbounds i8*, i8** %0, i64 7
; CHECK-NEXT:   %6 = bitcast i8** %5 to i1 (i8*, i8*)**
; CHECK-NEXT:   %7 = getelementptr inbounds i8*, i8** %0, i64 1
; CHECK-NEXT:   %8 = bitcast i8** %7 to i64 (i8*, i8*, i8*, i64)**
; CHECK-NEXT:   %9 = getelementptr inbounds i8*, i8** %0, i64 3
; CHECK-NEXT:   %10 = bitcast i8** %9 to void (i8*, i8*, double, i8*, i64)**
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %condition.call.without.trace, %entry
; CHECK-NEXT:   %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %26, %condition.call.without.trace ]
; CHECK-NEXT:   %mrv = insertvalue { double, i8* } undef, double %loss.0.lcssa, 0
; CHECK-NEXT:   %mrv1 = insertvalue { double, i8* } %mrv, i8* %trace, 1
; CHECK-NEXT:   ret { double, i8* } %mrv1

; CHECK: for.body:                                         ; preds = %condition.call.without.trace, %for.body.preheader
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %condition.call.without.trace ]
; CHECK-NEXT:   %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %26, %condition.call.without.trace ]
; CHECK-NEXT:   %11 = trunc i64 %indvars.iv to i32
; CHECK-NEXT:   %conv2 = sitofp i32 %11 to double
; CHECK-NEXT:   %12 = tail call double @llvm.fmuladd.f64(double %conv2, double %m, double %b)
; CHECK-NEXT:   %13 = load i1 (i8*, i8*)*, i1 (i8*, i8*)** %6, align 8
; CHECK-NEXT:   %14 = call i1 %13(i8* %1, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0)) #6
; CHECK-NEXT:   br i1 %14, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %for.body
; CHECK-NEXT:   %15 = alloca double, align 8
; CHECK-NEXT:   %16 = bitcast double* %15 to i8*
; CHECK-NEXT:   %17 = load i64 (i8*, i8*, i8*, i64)*, i64 (i8*, i8*, i8*, i64)** %8, align 8
; CHECK-NEXT:   %18 = call i64 %17(i8* %1, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* nonnull %16, i64 8) #6
; CHECK-NEXT:   %19 = load double, double* %15, align 8
; CHECK-NEXT:   br label %condition.call.without.trace

; CHECK: condition.call.without.trace:                     ; preds = %for.body, %condition.call.with.trace
; CHECK-NEXT:   %20 = phi double [ %19, %condition.call.with.trace ], [ 4.000000e+00, %for.body ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double %12, double 1.000000e+00, double %20)
; CHECK-NEXT:   %21 = alloca double, align 8
; CHECK-NEXT:   store double %20, double* %21, align 8
; CHECK-NEXT:   %22 = bitcast double* %21 to i8**
; CHECK-NEXT:   %23 = load i8*, i8** %22, align 8
; CHECK-NEXT:   %24 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %10, align 8
; CHECK-NEXT:   call void %24(i8* %trace, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call, i8* %23, i64 8) #6
; CHECK-NEXT:   %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
; CHECK-NEXT:   %25 = load double, double* %arrayidx3, align 8
; CHECK-NEXT:   %sub = fsub double %20, %25
; CHECK-NEXT:   %26 = tail call double @llvm.fmuladd.f64(double %sub, double %sub, double %loss.021)
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }