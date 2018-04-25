function [W,Delta_W_old] = ann_FF_Mom_batch_nb(x,t,N,W,lp,T,Delta_W_old,af,ex,err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// Updates weight matrix of an ANN,
// based on backpropagation with momentum algorithm.
// this function is to be used on networks without bias

// see ANN_FF (help)

// "Delta_W_old", "af", "ex" and "err_deriv_y" are optional arguments

[lsh, rsh] = argn(0);

// define "Delta_w_old", "af", "ex" and "err_deriv_y" if necessary
if rsh < 7, Delta_W_old = hypermat(size(W)'), end;
if rsh < 8, af = ['ann_log_activ','ann_d_log_activ'], end;
if rsh < 9, ex = " ", end;
if rsh < 10, err_deriv_y = 'ann_d_sum_of_sqr', end;

// no. of layers
L = size(N, 'c');
// ... and patterns
P = size(x,'c');

// initialize "z" to avoid resizing
z = zeros(max(N), L);

size_W = size(W)';

// repeat T times
for time = 1 : T
  // grad_E_mod is a hypermatrix with the same layout as W
  // because of flat spot elimination the modified grad_E is calculated here
  // reinitialize at each loop
  grad_E_mod = hypermat(size_W);
  
  // go trough all patterns
  for p = 1 : P
    // find all neuronal outputs (activation) for current input pattern
    // first "z" column is exactly "x(:,p)"
    z(1:N(1),1) = x(:,p);
    for l = 2 : L
      // first calculate total input (as column vector) ...
      z(1:N(l),l) = W(1:N(l), 1:N(l-1), l-1) * z(1:N(l-1), l-1);
      // ... then activation
      execstr('z(1:N(l),l) = ' + af(1) + '(z(1:N(l),l))');
    end;

    // now for layer "L" (last), requiring special treatment on "err_dz"
    
    // "err_dz" for output layer, don't propagate smaller than lp(2)
    execstr('err_dz = clean(' + err_deriv_y + '(z(1:N(L),L),t(:,p)), lp(2))');
    
    // "deriv_af" for output layer, also add flat spot elimination
    execstr('deriv_af = ' + af(2) + '(z(1:N(L),L))' + ...
                        ' + lp(4) * ones(N(L),1)');
  
    // "err_dz_deriv_af" product is used twice
    err_dz_deriv_af = err_dz .* deriv_af;
    grad_E_mod(1:N(L), 1:N(L-1), L-1) = ...
        grad_E_mod(1:N(L), 1:N(L-1), L-1) + ...
        err_dz_deriv_af * z(1:N(L-1), L-1)';

    // backpropagate
    for l = L-1 : -1 : 2
      // new "err_dz" based on previous one
      // transpose two vectors instead of W
      err_dz = (err_dz_deriv_af' * W(1:N(l+1), 1:N(l), l))';
      
      // new "deriv_af", also add flat spot elimination
      execstr('deriv_af = ' + af(2) + '(z(1:N(l),l))' + ...
                          ' + lp(4) * ones(N(l),1)');

      // same as for layer L, "err_dz_deriv_af" also used on next loop above
      err_dz_deriv_af = err_dz .* deriv_af;
      grad_E_mod(1:N(l), 1:N(l-1), l-1) = ...
          grad_E_mod(1:N(l), 1:N(l-1), l-1) + ...
          err_dz_deriv_af * z(1:N(l-1),l-1)';
    end;
  end;

  // update weights
  // (the new Delta_W_old ! ;) will become old after weight update,
  // i.e on next loop or next call to this function)
  Delta_W_old = - lp(1) * grad_E_mod + lp(3) * Delta_W_old;
  W = W + Delta_W_old;

  // execute "ex"
  execstr(ex);
end;

endfunction

