function y = ann_FF_run(x, N, W, l, af)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// runs the network, with biases,
// with input pattern(s) "x" injected al layer "l(1)"
// returning the activation at layer "l(2)"
// (defaults to whole network)

// see ANN_FF (help)

// "l" and "af" are optional

[lsh, rsh] = argn(0);

// "l" defaults to whole network
if rsh < 4, l = [2,size(N,'c')], end;

// "af" defaults to logistic activation function
if rsh < 5, af = 'ann_log_activ', end;

// no. of present patterns
P = size(x,'c');

// initialize "y"
y = zeros(N(l(2)), P);

// go trough all patterns
for p = 1 : P
  // first "input" layer uses "x(:,p)" and calculate total input ...
  // (an "1" is added to the input vector to represent bias)
  z = W(1:N(l(1)), 1:N(l(1)-1)+1, l(1)-1) * [1; x(:,p)];
  // ... then activation
  execstr("z = " + af + "(z)");

  // propagate, same as above but use "z"
  for ll = l(1)+1 : l(2)
    // ... use old "z" to find total input
    z = W(1:N(ll), 1:N(ll-1)+1, ll-1) * [1; z];
    // ... then compute activation
    execstr("z = " + af + "(z)");
  end;

  // collect data
  y(:,p) = z;
end;

endfunction

