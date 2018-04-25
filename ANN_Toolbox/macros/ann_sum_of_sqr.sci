function E = ann_sum_of_sqr(y,t)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculates sum-of-squares error between "y" and "t" patterns

// see ANN_GEN (help)

E = sum((y-t) .^ 2) / 2;

endfunction

