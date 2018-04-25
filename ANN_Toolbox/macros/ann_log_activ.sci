function y = ann_log_activ(x)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculates logistic activation function for each component of "x"

// see ANN_GEN (help)

y = 1 ./ (1+%e^(-x));

endfunction

