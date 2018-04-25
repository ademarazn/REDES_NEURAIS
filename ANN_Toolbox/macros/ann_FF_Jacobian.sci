function J = ann_FF_Jacobian(x,N,W,dx,af)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculates the Jacobian using a finite differences procedure

[lsh,rsh] = argn(0);

// optional parameters
if rsh < 5, af = "ann_log_activ", end;

// required for ann_FF_run
l = [2,size(N,'c')];
// no. of patterns
P = size(x,'c');

// initialize J
J = hypermat([N(size(N,'c')), N(1), P]);

// for each pattern
for p = 1 : P
  // for each input
  for i = 1 : N(1)
    temp = x(i,p);
    x(i,p) = temp + dx;
    y_p = ann_FF_run(x(:,p), N, W, l, af);
    x(i,p) = temp - dx;
    y_n = ann_FF_run(x(:,p), N, W, l, af);
    J(:,i,p) = (y_p - y_n) / (2 * dx);
  end;
end;

endfunction

