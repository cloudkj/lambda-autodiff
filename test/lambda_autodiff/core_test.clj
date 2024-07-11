(ns lambda-autodiff.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [lambda-autodiff.core :refer :all]))

(deftest test-add-broadcast
  (let [a (make-node [[[-4.0 3.0] [-2.0 5.5]] [[-4.0 3.0] [-2.0 5.5]]])
        b (make-node 2.0)
        c (add a b)
        grads (differentiate c)]
    (is (= [[[-2.0 5.0] [0.0 7.5]] [[-2.0 5.0] [0.0 7.5]]] (.value c)))
    (is (= [[[1 1] [1 1]] [[1 1] [1 1]]] (get grads a)))
    (is (= 8 (get grads b)))))

(deftest test-mul-broadcast
  (let [a (make-node [[[-2] [1] [3] [0]]])
        b (make-node [
            [[[1] [2] [3] [4]]]
            [[[1] [2] [3] [4]]]
        ])
        c (mul a b)
        grads (differentiate c)]
    (is (= [[[[-2] [2] [9] [0]]] [[[-2] [2] [9] [0]]]] (.value c)))
    (is (= [[[2] [4] [6] [8]]] (get grads a)))
    (is (= [[[[-2] [1] [3] [0]]] [[[-2] [1] [3] [0]]]] (get grads b)))))

;; Source: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
(deftest test0
  (let [x (make-node [0.52 1.12 0.77])
        y (make-node 1)
        w (make-node [-0.1 0 0.1])
        b (make-node 0.123)
        logit (add (sum (mul x w)) b)
        preds (mul (make-node 0.5) (add (tanh (div logit (make-node 2))) (make-node 1)))
        probs (add (mul preds y) (mul (sub (make-node 1) preds) (sub (make-node 1) y)))
        loss (neg (sum (log probs)))
        grads (differentiate loss)]
    (is (every? #(< % 1e-4) (m/abs (m/sub [0.6219] (.value loss)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-0.2408 -0.5186 -0.3566] (get grads w)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-0.4631] (get grads b)))))))

;; Source: https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
(deftest test1
  (let [x (make-node -4)
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (= -20 (.value y)))
    (is (= 46 (get grads x)))))

(deftest test1-vector
  (let [x (make-node [-4 3 -2])
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (= [-20 297 8] (.value y)))
    (is (= [46 202 -6] (get grads x)))))

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
    (is (< (abs (- 645.5773 (get grads b))) 1e-4))
    (is (< (abs (- -6.9417 (get grads c))) 1e-4))
    (is (< (abs (- 6.9417 (get grads d))) 1e-4))
    (is (< (abs (- -6.9417 (get grads e))) 1e-4))
    (is (< (abs (- 0.4958 (get grads f))) 1e-4))))

(deftest test2-vector
  (let [a (make-node [-4.0 3.0 -2.0])
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
    (is (every? #(< % 1e-4) (m/abs (m/sub [24.7041 14112.0004 1104.5045] (.value g)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [138.8338 4199.9999 939.9961] (get grads a)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [36384.5540] (get grads b)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-6.9417 -168.0 -46.9998] (get grads c)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [6.9417 168.0 46.9998] (get grads d)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-6.9417 -168.0 -46.9998] (get grads e)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [0.4958 0.5 0.5] (get grads f)))))))

;; Source: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
(deftest test3
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
        L (sum (div t7 (make-node 2)))
        grads (differentiate L)]
    (is (< (abs (- 0.0285 (.value L))) 1e-4))
    (is (< (abs (- -0.0067 (get grads w))) 1e-4))
    (is (< (abs (- -0.0540 (get grads b))) 1e-4))))

(deftest test3-vector
  (let [x (make-node [0.1 0.2 -0.1 0.0 -0.2])
        w (make-node [3.14 4.1 5.1 6.1 7.1])
        b (make-node 1.68)
        t1 (mul w x)
        z (add t1 b)
        t3 (neg z)
        t4 (exp t3)
        t5 (add (make-node 1) t4)
        y (div (make-node 1) t5)
        t6 (sub y t5)
        t7 (pow t6 2)
        L (sum (div t7 (make-node 2)))
        grads (differentiate L)]
    (is (every? #(< % 1e-4) (m/abs (m/sub [0.9817] (.value L)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-0.0062 -0.0048 0.0269 0.0 0.2454] (get grads w)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-1.6909] (get grads b)))))))

;; Source: https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/
(deftest test4
  (let [x (make-node 2.71)
        y (loop [i 0
                 y x]
            (if (= i 100)
              y
              (recur (inc i) (sin (add x y)))))
        grads (differentiate y)]
    (is (< (abs (- 0.2465 (.value y))) 1e-4))
    (is (< (abs (- -0.2050 (get grads x))) 1e-4))))

(deftest test4-vector
  (let [x (make-node [2.71 7.12 1.27])
        y (loop [i 0
                 y x]
            (if (= i 100)
              y
              (recur (inc i) (sin (add x y)))))
        grads (differentiate y)]
    (is (every? #(< % 1e-4) (m/abs (m/sub [0.2465 0.9718 0.8519] (.value y)))))
    (is (every? #(< % 1e-4) (m/abs (m/sub [-0.2050 -0.1907 -0.3437] (get grads x)))))))
