# lambda-autodiff

Reverse-mode automatic differentiation in Lisp (Clojure).

Progress -> goals:

* Scalar values -> vectorized, non-broadcasted values
* Explict namespace for supported functions -> macros for overriding predefined primitive functions
* -> Integration with neural networks API in https://github.com/cloudkj/lambda-ml
