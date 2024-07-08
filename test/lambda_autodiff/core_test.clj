(ns lambda-autodiff.core-test
  (:require [clojure.test :refer :all]
            [lambda-autodiff.core :refer :all]))

(deftest test1
  (let [x (make-node -4)
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (= -20 (.value y)))
    (is (= 46 (get grads x)))))

(deftest test2
  (let [a (make-node -4.0)
        b (make-node 2.0)
        c (add a b)
        d (add (mul a b) (pow b 3.0))
        c (add (add c c) (make-node 1.0))
        c (add (add c (make-node 1.0)) (add c (neg a)))
        d (add (add d (mul d (make-node 2.0))) (relu (add b a)))
        d (add (add d (mul (make-node 3.0) d)) (relu (sub b a)))
        e (sub c d)
        f (pow e 2.0)
        g (div f (make-node 2.0))
        g (add g (div (make-node 10.0) f))
        grads (differentiate g)]
    (is (< (abs (- 24.7041 (.value g))) 1e-4))
    (is (< (abs (- 138.8338 (get grads a))) 1e-4))
    (is (< (abs (- 645.5773 (get grads b))) 1e-4))))
