(ns lambda-autodiff.core-test
  (:require [clojure.test :refer :all]
            [lambda-autodiff.core :refer :all]))

(deftest test1
  "Source: https://github.com/karpathy/micrograd/blob/master/test/test_engine.py"
  (let [x (make-node -4)
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (= -20 (.value y)))
    (is (= 46 (get grads x)))))

(deftest test2
  "Source: https://github.com/karpathy/micrograd/blob/master/test/test_engine.py"
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
    (is (< (abs (- 645.5773 (get grads b))) 1e-4))
    (is (< (abs (- -6.9417 (get grads c))) 1e-4))
    (is (< (abs (- 6.9417 (get grads d))) 1e-4))
    (is (< (abs (- -6.9417 (get grads e))) 1e-4))
    (is (< (abs (- 0.4958 (get grads f))) 1e-4))))

(deftest test3
  "Source: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf"
  (let [x (make-node 0.12345)
        w (make-node 3.14)
        b (make-node 1.68)
        t1 (mul w x)
        z (add t1 b)
        t3 (neg z)
        t4 (exp t3)
        t5 (add (make-node 1) t4)
        y (div (make-node 1) t5)
        t6 (sub y t5)
        t7 (pow t6 2)
        L (div t7 (make-node 2))
        grads (differentiate L)]
    (is (< (abs (- 0.0285 (.value L))) 1e-4))
    (is (< (abs (- -0.0067 (get grads w))) 1e-4))
    (is (< (abs (- -0.0540 (get grads b))) 1e-4))))
