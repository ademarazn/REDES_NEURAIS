ANN Toolbox ver. 0.4.2.5 for Scilab 5.4
=======================================

This represents a toolbox for artificial neural networks,
based on my developments described in "Matrix ANN" book,
under development, if interested send me an email at
r.hristev@phys.canterbury.ac.nz
allan.cornet@scilab.org

Current feature:s
 - Only layered feedforward networks are supported *directly* at the moment
   (for others use the "hooks" provided)
 - Unlimited number of layers
 - Unlimited number of neurons per each layer separately
 - User defined activation function (defaults to logistic)
 - User defined error function (defaults to SSE)
 - Algorithms implemented so far:
    * standard (vanilla) with or without bias, on-line or batch
    * momentum with or without bias, on-line or batch
    * SuperSAB with or without bias, on-line or batch
    * Conjugate gradients
    * Jacobian computation
    * Computation of result of multiplication between "vector" and Hessian
 - Some helper functions provided

For full descriptions start with the toplevel "ANN" man page.

