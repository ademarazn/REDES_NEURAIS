function VH = ann_FF_VHess(x, t, N, W, V, dW, af, err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculates the result of multiplication between a vector and Hessian
// trough a finite differences procedure

[lsh,rsh] = argn(0);

// define default parameters if necessary
if rsh < 7, af = ['ann_log_activ', 'ann_d_log_activ'], end;
if rsh < 8, err_deriv_y = 'ann_d_sum_of_sqr', end;

// calculate gradient to the +
grad_p = ann_FF_grad_BP(x, t, N, W + dW * V, 0, af, err_deriv_y);
// ... and to the -
grad_n = ann_FF_grad_BP(x, t, N, W - dW * V, 0, af, err_deriv_y);

// result, difference is 2 * dW
VH = (grad_p - grad_n) / (2 * dW);

endfunction

