function H = ann_FF_Hess(x, t, N, W, dW, dW2, af, ef)
// This file is part of:
//    ANN Toolbox for Scilab
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public Licence version 2

// Calculates the Hessian
// using a finite difference procedure by perturbing two weights
// used for --- testing --- purposes only as is very slow

// The Hessian is calculated considering the whole set of patterns

// see ANN_FF (help)

[lsh,rsh] = argn(0);

// define default parameters if necessary
if rsh < 7, af = 'ann_log_activ', end;
if rsh < 8, ef = 'ann_sum_of_sqr', end;

// no. of layers
L = size(N,'c');
// rl run between layers, parameter for ann_FF_run function
rl = [2, size(N,'c')];

// create the return hypermatrix,
// layout is of type W .*. W (NOT W .*. W')
H = hypermat([size(W)', size(W)']);

// first weights are W(k1,i1,l1-1), second ones are W(k2,i2,l2-1)
// for each layer
// WARNING: THIS WILL NOT CALCULATE CORECTLY THE "DIAGONAL" ELEMENTS
for l1 = 2 : L, for l2 = 2 : L
  // for each neuron in layer
  for k1 = 1 : N(l1), for k2 = 1 : N(l2)
    // for each connection from previous layer
    for i1 = 1 : N(l1-1)+1, for i2 = 1 : N(l2-1)+1
      // hold original weight values
      temp1 = W(k1,i1,l1-1);
      temp2 = W(k2,i2,l2-1);
      // first change: ++
      W(k1,i1,l1-1) = temp1 + dW;
      W(k2,i2,l2-1) = temp2 + dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err1 = ' + ef + '(y,t)');
      // second change -+
      W(k1,i1,l1-1) = temp1 - dW;
      W(k2,i2,l2-1) = temp2 + dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err2 = ' + ef + '(y,t)');
      // third change +-
      W(k1,i1,l1-1) = temp1 + dW;
      W(k2,i2,l2-1) = temp2 - dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err3 = ' + ef + '(y,t)');
      // fourth change --
      W(k1,i1,l1-1) = temp1 - dW;
      W(k2,i2,l2-1) = temp2 - dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err4 = ' + ef + '(y,t)');
      // restore weights
      W(k1,i1,l1-1) = temp1;
      W(k2,i2,l2-1) = temp2;
      // calculate hessian term
      // "4" factor because (\Delta W)^2 = (2 dW)^2
      H(k1,i1,l1-1,k2,i2,l2-1) = ...
          (err1 - err2 - err3 + err4) / (4 * dW^2);
    end, end;
  end, end;
end, end;

// NOW THE DIAGONAL ELEMENTS
// (avoid "if"-s above, it's too slow)
for l = 2 : L
  // for each neuron in layer
  for k = 1 : N(l)
    // for each connection from previous layer
    for i = 1 : N(l-1)+1
      // hold original weight values
      temp = W(k,i,l-1);
      // first change: +
      W(k,i,l-1) = temp + (1 + dW2) * dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err1 = ' + ef + '(y,t)');
      // second change +
      W(k,i,l-1) = temp + (1 - dW2) * dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err2 = ' + ef + '(y,t)');
      err_p = (err1 - err2) / (2 * dW2 * dW);
      // first change: -
      W(k,i,l-1) = temp - (1 - dW2) * dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err1 = ' + ef + '(y,t)');
      // second change -
      W(k,i,l-1) = temp - (1 + dW2) * dW;
      y = ann_FF_run(x,N,W,rl,af);
      execstr('err2 = ' + ef + '(y,t)');
      err_n = (err1 - err2) / (2 * dW2 * dW);
      // restore weight
      W(k,i,l-1) = temp;
      // calculate hessian term
      H(k,i,l-1,k,i,l-1) = ...
          (err_p - err_n) / (2 * dW);
    end;
  end;
end;

endfunction

