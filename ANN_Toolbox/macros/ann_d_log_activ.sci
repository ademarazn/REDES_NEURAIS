function z = ann_d_log_activ(y)
// This file is part of:
// ANN Toolbox for Scilab 5.x
// Copyright (C) Ryurick M. Hristev
// updated by Allan CORNET INRIA, May 2008
// released under GNU Public licence version 2

// calculates the derivative of logistic activation function,
//   given the actual value of the function

// see ANN_GEN (help)

z = y .* (1 - y);

endfunction