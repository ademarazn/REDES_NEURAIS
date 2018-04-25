function grad_E = ann_FF_grad_nb(x,t,N,W,dW,af,ef)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// Calculates the error gradient following a finite difference procedure,
// i.e. perturbing each weight in turn;
// used for --- testing --- purposes only as is much slower than BP algorithm.
// this function is designed for networks without bias

// The gradient is calculated only for all patterns in "x" and "t"

// see ANN_FF (help)

[lsh,rsh] = argn(0);

// define optional parameters if necessary
if rsh < 6, af = 'ann_log_activ', end;
if rsh < 7, ef = 'ann_sum_of_sqr', end;

// create the return matrix
grad_E = hypermat(size(W)');

// rl - run between layers, parameter for ann_FF_run function
rl = [2,size(N,'c')];

// for each pattern
for p = 1 : size(x,'c')
  // for each layer
  for l = 2 : size(N,'c')
    // for each neuron in layer
    for n = 1 : N(l)
      // for each connection to previous layer
      for i = 1 : N(l-1)
        // hold the old value of W
        temp = W(n,i,l-1);
        // change W value
        W(n,i,l-1) = temp - dW;
        // run the net
        y = ann_FF_run_nb(x(:,p),N,W,rl,af);
        // calculate new error, to the "left"
        execstr('err_n = ' + ef + '(y,t(:,p))');
        // change W value
        W(n,i,l-1) = temp + dW;
        // run the net
        y = ann_FF_run_nb(x(:,p),N,W,rl,af);
        // calculate new error, to the "right"
        execstr('err_p = ' + ef + '(y,t(:,p))');
        // "2" because \Delta w = 2 * dW
        grad_E(n,i,l-1) = ...
            grad_E(n,i,l-1) + (err_p - err_n) / (2 * dW);
        // restore W
        W(n,i,l-1) = temp;
      end;
    end;
  end;
end;

endfunction

