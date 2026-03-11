# lambda-autodiff

Reverse-mode automatic differentiation in Clojure. Basic support for common
vectorized operations on values of n-dimensional arrays. Also includes
lightweight extensions to enable use cases for specific neural network
architectures.

## Usage

Values can be scalars or n-dimensional arrays, represented as nodes in a
computational graph.

For example, we can declare an expression $Q = 3a^3 - b^2$ and evaluate it
with parameter values:

```clojure
=> (use 'lambda-autodiff.core)
=> (def a (make-node [2 3]))
=> (def b (make-node [6 4]))
=> (def Q (sub (mul (make-node 3) (pow a 3)) (mul b b)))
=> (.value Q)
[-12.0 65.0]
```

We can then compute the gradients of the expression with respect to the
parameters ($\frac{\partial Q}{\partial a} = 9a^2$ and
$\frac{\partial Q}{\partial b} = -2b$), and look up the values:

```clojure
=> (-> (differentiate Q) (get a))
[36.0 81.0]
=> (-> (differentiate Q) (get b))
[-12.0 -8.0]
```

## Examples

* [Multi-layer perceptron](https://cloudkj.github.io/lambda-autodiff/doc/examples/mlp/)
* Character-level recurrent neural network (RNN)
* [Convolutional neural network (CNN) for MNIST digit recognition](https://cloudkj.github.io/lambda-autodiff/doc/examples/cnn/)
* [GPT](https://cloudkj.github.io/lambda-autodiff/doc/examples/gpt/)
