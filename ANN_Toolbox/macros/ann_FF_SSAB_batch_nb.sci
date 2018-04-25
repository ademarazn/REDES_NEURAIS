function [W,Delta_W_old,Delta_W_oldold,mu]=ann_FF_SSAB_batch_nb(x,t,N,W,lp,Delta_W_old,Delta_W_oldold,T,mu,af,ex,err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// Updates weight matrix of an ANN,
// based on backpropagation with SuperSAB algorithm (batch version).
// this function is to be used on networks without bias

// see ANN_FF (help)

// "mu", "af", "ex" and "err_deriv_y" are optional arguments

[lsh, rsh] = argn(0);

// size of W hypermatrix, required in several places
size_W = size(W)';

// define default parameters if necessary
if rsh < 9, mu = lp(1) * hypermat(size_W,ones(prod(size_W),1)), end;
if rsh < 10, af = ['ann_log_activ','ann_d_log_activ'], end;
if rsh < 11, ex = " ", end;
if rsh < 12, err_deriv_y = 'ann_d_sum_of_sqr', end;

// repeat T times
for time = 1 : T
  // error gradient
  grad_E = ann_FF_grad_BP_nb(x,t,N,W,lp(2),af,err_deriv_y);
    
  // sign hypermatrix
  M = sign(sign(Delta_W_old .* Delta_W_oldold) ...
           + hypermat(size_W,ones(prod(size_W),1)));
    
  // mu hypermatrix update (former lp(1))
  mu = ( (lp(4) - lp(5)) * M ...
         + lp(5) * hypermat(size_W,ones(prod(size_W),1)) ) .* mu;

  // update weights
  // (the new Delta_W_old ! ;) will become old after weight update,
  // i.e on next loop or next call to this function)
  // same for Delta_W_oldold
  Delta_W_oldold = Delta_W_old;
  Delta_W_old = ...
      - mu .* grad_E ...
      - (lp(3) * Delta_W_old) .* (hypermat(size_W,ones(prod(size_W),1)) - M);
  W = W + Delta_W_old;

  // execute "ex"
  execstr(ex);
end;

endfunction

