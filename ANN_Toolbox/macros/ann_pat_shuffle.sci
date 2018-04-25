function [x,t] = ann_pat_shuffle(x,t)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// shuffles the patterns from "x" and the corresponding "t"

// see ANN_GEN (help)

// no. of patterns
P = size(x,'c');

my_rand = ceil(P * rand(P,1));

for p = 1 : P
  // shuffle x
  temp = x(:,my_rand(p));
  x(:,my_rand(p)) = x(:,p);
  x(:,p) = temp;
  // shuffle t same way (keep x(:,p) <-> t(:,p) correspondence)
  temp = t(:,my_rand(p));
  t(:,my_rand(p)) = t(:,p);
  t(:,p) = temp;
end;

endfunction

