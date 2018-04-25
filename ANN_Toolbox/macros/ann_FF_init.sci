function W = ann_FF_init(N, r, rb)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// generate the weight matrix for an feedforward ANN defined by N

// see ANN_FF (help)

// "r" and "rb" are optional arguments

[lsh, rsh] = argn(0);

// define "r" if necessary
if rsh < 2, r = [-1,1], end;

// "+1" -- to alow room for biases (first column in each W)
// don't create weight entries for input neurons (from input layer 1)
// i.e. no. of matrices W(:,:,*) is size(N,'c')-1
W = hypermat([max(N), max(N)+1, size(N,'c') - 1]);

// initialize weights with random numbers between "r(1)" and "r(2)"
// (only the required values, first column, i.e. bias, later)
for l = 2 : size(N,'c')
  W(1:N(l), 2:N(l-1)+1, l-1) = ...
      (r(2) - r(1)) * rand(N(l), N(l-1)) + r(1) * ones(N(l), N(l-1));
end;

// biases, if required, otherwise leave them 0
if rsh > 2 ...
    then for l = 2 : size(N,'c')
           W(1:N(l), 1, l-1) = ...
               (rb(2) - rb(1)) * rand(N(l), 1) + rb(1) * ones(N(l), 1);
         end;
end;

endfunction

