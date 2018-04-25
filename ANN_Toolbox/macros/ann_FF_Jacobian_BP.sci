function J = ann_FF_Jacobian_BP(x,N,W,af)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculate the Jacobian following a backpropagation procedure

[lsh,rsh] = argn(0);

// optional parameters
if rsh < 4, af = ["ann_log_activ", "ann_d_log_activ"], end;

// no. of layers
L = size(N,'c');
// ... and patterns
P = size(x,'c');

// create the hypermatrix to hold (grad_{a(\ell)}} z^\T)^\T
grad_a_z = hypermat([N(L), max(N(2:L)), L-1]);

// the matrix containing the activities
d_f = zeros(max(N(2:L)), L-1);

// initialize J
J = hypermat([N(L),N(1),P]);

// for all patterns
for p = 1 : P
  // forward propagation
  // initial activation
  z = x(:,p);
  for l = 1 : L-1
    // find next activation, use extended z, i.e. bias
    execstr('z = ' + af(1) + '(W(1:N(l+1), 1:N(l)+1, l) * [1;z]);');
    // and store its derivative
    execstr('d_f(1:N(l+1),l) = ' + af(2) + '(z)');
  end;
  
  // backpropagation
  // initial values
  grad_a_z(:, 1:N(L), L-1) = diag(d_f(1:N(L),L-1));
  for l = L-2 : -1 : 1
    grad_a_z(:, 1:N(l+1), l) = ...
        (grad_a_z(:, 1:N(l+2), l+1) * ...
         W(1:N(l+2), 2:N(l+1)+1, l+1)) .* ...
        (ones(N(L),1) * d_f(1:N(l+1),l)')
  end;

  J(:,:,p) = grad_a_z(:, 1:N(2),1) * W(1:N(2), 2:N(1)+1, 1);
end;

endfunction

