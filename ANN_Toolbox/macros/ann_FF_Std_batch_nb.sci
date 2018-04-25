function W = ann_FF_Std_batch_nb(x, t, N, W, lp, T, af, ex, err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// Updates weight matrix of an ANN,
// based on standard backpropagation algorithm (batch version).
// this function is designed for networks without bias

// see ANN_FF (help)

// "af", "ex" and "err_deriv_y" are optional arguments

[lsh, rsh] = argn(0);

// define "af", "ex" and "err_deriv_y" if necessary
if rsh < 7, af = ['ann_log_activ','ann_d_log_activ'], end;
if rsh < 8, ex = " ", end;
if rsh < 9, err_deriv_y = 'ann_d_sum_of_sqr', end;

// repeat T times
for time = 1 : T
  // find gradient
  grad_E = ann_FF_grad_BP_nb(x, t, N, W, lp(2), af, err_deriv_y);

  // update weights
  W = W - lp(1) * grad_E;

  // execute "ex"
  execstr(ex);
end;

endfunction

