function W = ann_FF_ConjugGrad(x, t, N, W, T, dW, ex, af, err_deriv_y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// trains the network for T epochs using the Conjugate gradient algorithm

// see ANN_FF (help)

[lsh,rsh] = argn(0);

// deal with default parameters
if rsh < 7, ex = [" "], end;
if rsh < 8, af = ['ann_log_activ','ann_d_log_activ'], end;
if rsh < 9, err_deriv_y = 'ann_d_sum_of_sqr', end;

// no. of layers
L = size(N,'c');

//-------------------------------------------------------------------
// first step for conjugate gradients is performed outside the loop,
// for proper initialization

// calculate first grad_E and initialize directly to grad_E_old
grad_E = ann_FF_grad_BP(x, t, N, W, 0, af, err_deriv_y);

// initialize direction to grad_E
D = - grad_E;

// iterate to T-1 (for T we need fewer calculations)
for time = 1 : T-1
  // calculate D^\T \circ Hessian
  D_circ_H = ann_FF_VHess(x, t, N, W, D, dW, af, err_deriv_y);

  // calculate D^\T \circ grad_E and D^\T \circ Hessian \circ D ...
  D_circ_grad_E = 0;
  D_circ_H_circ_D = 0;
  for l = 1 : L-1
    // using "old" grad_E
    D_circ_grad_E = D_circ_grad_E + sum(D(:,:,l) .* grad_E(:,:,l));
    D_circ_H_circ_D = D_circ_H_circ_D + sum(D_circ_H(:,:,l) .* D(:,:,l));
  end;
  // ... and alpha
  alpha = - D_circ_grad_E / D_circ_H_circ_D;

  // new weights
  W = W + alpha * D;

  // execute ex if necessary
  execstr(ex);

  // new gradient
  grad_E = ann_FF_grad_BP(x, t, N, W, 0, af, err_deriv_y);

  // calculate D^\T \circ grad_E (new) ...
  D_circ_grad_E = 0;
  for l = 1 : L-1
    D_circ_grad_E = D_circ_grad_E + sum(D(:,:,l) .* grad_E(:,:,l));
  end;
  // ... and beta
  beta = - D_circ_grad_E / D_circ_H_circ_D;

  // new direction
  D = - grad_E + beta * D;
end;

// for T only
// calculate D^\T \circ Hessian
D_circ_H = ann_FF_VHess(x, t, N, W, D, dW, af, err_deriv_y);

// calculate D^\T \circ grad_E and D^\T \circ Hessian \circ D ...
D_circ_grad_E = 0;
D_circ_H_circ_D = 0;
for l = 1 : L-1
  // using "old" grad_E
  D_circ_grad_E = D_circ_grad_E + sum(D(:,:,l) .* grad_E(:,:,l));
  D_circ_H_circ_D = D_circ_H_circ_D + sum(D_circ_H(:,:,l) .* D(:,:,l));
end;
// ... and alpha
alpha = - D_circ_grad_E / D_circ_H_circ_D;

// final weights
W = W + alpha * D;

// and final ex
execstr(ex);

endfunction

