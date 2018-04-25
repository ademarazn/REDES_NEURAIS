function W = ann_FF_init_nb(N, r)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// generate the weight matrix for an feedforward ANN defined by N
// this function is designed for networks without bias

// see ANN_FF (help)

// "r" is optional argument

[lsh, rsh] = argn(0);

// define "r" if necessary
if rsh < 2, r = [-1,1], end;

// don't create weight entries for input neurons (from layer 0),
// i.e. no. of matrices W(:,:,*) is size(N,'c')-1
W = hypermat([max(N), max(N), size(N,'c') - 1]);

// initialize (only the required values)
// with random numbers between "r(1)" and "r(2)"

for l = 2 : size(N,'c')
  W(1:N(l), 1:N(l-1), l-1) = ...
      (r(2) - r(1)) * rand(N(l), N(l-1)) + r(1) * ones(N(l), N(l-1));
end;

endfunction

