function grad_E = ann_FF_grad_BP_nb(x, t, N, W, c, af, err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// Calculate the error gradient considering all patterns
// trough a backpropagation procedure
// this function is designed for networks without bias

// see ANN_FF (help)

[lsh,rsh] = argn(0);

// define default parameters if necessary
if rsh < 5, c = 0, end;
if rsh < 6, af = ['ann_log_activ','ann_d_log_activ'], end;
if rsh < 7, err_deriv_y = 'ann_d_sum_of_sqr', end;

// no. of layers
L = size(N, 'c');
// ... and patterns
P = size(x,'c');

// initialize "z" to avoid resizing
z = zeros(max(N), L);

// initialize grad_E, W is a hypermatrix, grad_E have same layout
grad_E = hypermat(size(W)');

// calculate grad_E

// go trough all patterns
for p = 1 : P
  // find all neuronal outputs (activation) for current input pattern
  // first "z" column is exactly "x(:,p)"
  z(1:N(1),1) = x(:,p);
  for l = 2 : L
    // first calculate total input (as column vector) ...
    z(1:N(l),l) = W(1:N(l), 1:N(l-1),l-1) * z(1:N(l-1), l-1);
    // ... then activation
    execstr('z(1:N(l),l) = ' + af(1) + '(z(1:N(l),l))');
  end;

  // now for layer "L" (last), requiring special treatment on "err_dz"

  // "err_dz" for output layer, don't propagate smaller than c
  execstr('err_dz = clean(' + err_deriv_y + '(z(1:N(L),L),t(:,p)), c)');

  // "deriv_af" for output layer
  execstr('deriv_af = ' + af(2) + '(z(1:N(L),L))');

  // "err_dz_deriv_af" product is used twice
  err_dz_deriv_af = err_dz .* deriv_af;

  // adding contribution of pattern p
  // using the transposed of z vector here
  grad_E(1:N(L), 1:N(L-1), L-1) = ...
      grad_E(1:N(L), 1:N(L-1), L-1) + ...
      err_dz_deriv_af * z(1:N(L-1), L-1)';

  // backpropagate
  for l = L-1 : -1 : 2
    // new "err_dz" based on previous one
    // transpose two vectors instead of W
    err_dz = (err_dz_deriv_af' * W(1:N(l+1), 1:N(l), l))';

    // new "deriv_af"
    execstr('deriv_af = ' + af(2) + '(z(1:N(l),l))');

    // same as for layer "L", "err_dz_deriv_af" also used on next loop above
    err_dz_deriv_af = err_dz .* deriv_af;
    grad_E(1:N(l), 1:N(l-1), l-1) = ...
        grad_E(1:N(l), 1:N(l-1), l-1) + ...
        err_dz_deriv_af * z(1:N(l-1), l-1)';
  end;
end;

endfunction

