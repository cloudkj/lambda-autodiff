(ns lambda-autodiff.core-test
  (:require [clojure.test :refer :all]
            [lambda-autodiff.core :refer :all]
            [lambda-autodiff.array :as ma]))

(deftest test-add-broadcast
  (let [a (make-node [[[-4.0 3.0] [-2.0 5.5]] [[-4.0 3.0] [-2.0 5.5]]])
        b (make-node 2.0)
        c (add a b)
        grads (differentiate c)]
    (is (ma/equals [[[-2.0 5.0] [0.0 7.5]] [[-2.0 5.0] [0.0 7.5]]] (.value c)))
    (is (ma/equals [[[1 1] [1 1]] [[1 1] [1 1]]] (get grads a)))
    (is (ma/equals 8 (get grads b)))))

(deftest test-mul-broadcast
  (let [a (make-node [[[-2] [1] [3] [0]]])
        b (make-node [[[[1] [2] [3] [4]]]
                      [[[1] [2] [3] [4]]]])
        c (mul a b)
        grads (differentiate c)]
    (is (ma/equals [[[[-2] [2] [9] [0]]] [[[-2] [2] [9] [0]]]] (.value c)))
    (is (ma/equals [[[2] [4] [6] [8]]] (get grads a)))
    (is (ma/equals [[[[-2] [1] [3] [0]]] [[[-2] [1] [3] [0]]]] (get grads b)))))

(deftest test-div
  (let [a (make-node [[1 -1 2] [2 6 0]])
        b (make-node [[4 10 -2] [5 4 8]])
        c (div a b)
        grads (differentiate c)]
    (is (ma/equals [[0.25 -0.1 -1.0] [0.4 1.5 0.0]] (.value c)))
    (is (ma/equals [[0.25 0.1 -0.5] [0.2 0.25 0.125]] (get grads a)))
    (is (ma/equals [[-0.0625 0.01 -0.5] [-0.08 -0.375 0.0]] (get grads b)))))

(deftest test-mmul
  (let [a1 (make-node [1 2 3])
        b1 (make-node [4 5 6])
        c1 (mmul a1 b1)
        a2 (make-node [[1 2 3] [-1 0 1]])
        b2 (make-node [[4 1] [1 5] [6 6]])
        c2 (mmul a2 b2)]
    (is (ma/equals 32.0 (.value c1)))
    (is (ma/equals [4 5 6] (-> (differentiate c1) (get a1))))
    (is (ma/equals [1 2 3] (-> (differentiate c1) (get b1))))
    (is (ma/equals [[24.0 29.0] [2.0 5.0]] (.value c2)))
    (is (ma/equals [[5 6 12] [5 6 12]] (-> (differentiate c2) (get a2))))
    (is (ma/equals [[0.0 0.0] [2.0 2.0] [4.0 4.0]] (-> (differentiate c2) (get b2))))))

(deftest test-sum
  (let [a (make-node [[1 2]])
        b (make-node [[2 -1 15] [-1 0 1]])
        c1 (add a (sum a))
        c2 (add a (sum b))
        c3 (mul a (sum b))
        c4 (mul (sum b) b)]
    (is (ma/equals [[4 5]] (.value c1)))
    (is (ma/equals [[3 3]] (-> (differentiate c1) (get a))))
    (is (ma/equals [[17 18]] (.value c2)))
    (is (ma/equals [[1 1]] (-> (differentiate c2) (get a))))
    (is (ma/equals [[2 2 2] [2 2 2]] (-> (differentiate c2) (get b))))
    (is (ma/equals [[16 32]] (.value c3)))
    (is (ma/equals [[16 16]] (-> (differentiate c3) (get a))))
    (is (ma/equals [[3 3 3] [3 3 3]] (-> (differentiate c3) (get b))))
    (is (ma/equals [[32 -16 240] [-16 0 16]] (.value c4)))
    (is (ma/equals [[32 32 32] [32 32 32]] (-> (differentiate c4) (get b))))))

(deftest test-neg
  (let [a (make-node [[1.0 -2.0 9.0] [-3.0 4.0 0.0]])
        b (neg a)
        grads (differentiate b)]
    (is (ma/equals [[-1.0 2.0 -9.0] [3.0 -4.0 0.0]] (.value b)))
    (is (ma/equals [[-1.0 -1.0 -1.0] [-1.0 -1.0 -1.0]] (get grads a)))))

(deftest test-sub
  (let [a (make-node [[1 -1 2] [2 6 0]])
        b (make-node [[4 10 -2] [5 4 8]])
        c (sub a b)
        grads (differentiate c)]
    (is (ma/equals [[-3 -11 4] [-3 2 -8]] (.value c)))
    (is (ma/equals [[1 1 1] [1 1 1]] (get grads a)))
    (is (ma/equals [[-1 -1 -1] [-1 -1 -1]] (get grads b)))))

(deftest test-join
  (let [a (make-node [1 2])
        b (make-node [3 4])
        c (sum (pow (join a b) 2))
        grads (differentiate c)]
    (is (ma/equals [2.0 4.0] (get grads a)))
    (is (ma/equals [6.0 8.0] (get grads b)))))

(deftest test-join2
  (let [a (make-node [[1 2 3] [10 20 30]])
        b (make-node [[4 5 6]])
        c (make-node [[7 8 9] [77 88 99] [-1 -2 -3]])
        d (join (join a b) c)
        e (sum (pow d 2))
        grads (differentiate e)]
    (is (ma/equals [[2 4 6] [20 40 60]] (get grads a)))
    (is (ma/equals [[8 10 12]] (get grads b)))
    (is (ma/equals [[14 16 18] [154 176 198] [-2 -4 -6]] (get grads c)))))

(deftest test-join3
  (let [a (make-node [[11 22]])
        b (make-node [[30 40]])
        c (make-node [[5 6]])
        d0 (join (join a b 0) c 0)
        d0 (sum (mul d0 d0))
        d1 (join (join a b 1) c 1)
        d1 (sum (mul d1 d1))
        grads0 (differentiate d0)
        grads1 (differentiate d1)]
    (is (ma/equals [[22 44]] (get grads0 a)))
    (is (ma/equals [[22 44]] (get grads1 a)))
    (is (ma/equals [[60 80]] (get grads0 b)))
    (is (ma/equals [[60 80]] (get grads1 b)))
    (is (ma/equals [[10 12]] (get grads0 c)))
    (is (ma/equals [[10 12]] (get grads1 c)))))

(deftest test-select
  (let [a (make-node [1 2 3 4 5])
        b (select a (range 1 4))
        c (sum (mul b b))
        grads (differentiate c)]
    (is (ma/equals [0 4 6 8 0] (get grads a)))))

(deftest test-grad-shape
  (let [x (make-node [[1 2 3 4]])
        w (make-node [[1 2] [3 4] [5 6] [7 8]])
        b (make-node [-1 1])
        c (add (mmul x w) b)
        grads (differentiate c)]
    (is (ma/equals [4 2] (ma/shape (get grads w))))
    (is (ma/equals [2] (ma/shape (get grads b))))))

;; Adapted from: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
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
    (is (ma/equals 0.6219 (.value loss) 1e-4))
    (is (ma/equals [-0.2408 -0.5186 -0.3566] (get grads w) 1e-4))
    (is (ma/equals -0.4631 (get grads b) 1e-4))))

;; Adapted from: https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
(deftest test1
  (let [x (make-node -4)
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (ma/equals -20 (.value y)))
    (is (ma/equals 46 (get grads x)))))

(deftest test1-vector
  (let [x (make-node [-4 3 -2])
        z (add (add (mul (make-node 2) x) (make-node 2)) x)
        q (add (relu z) (mul z x))
        h (relu (mul z z))
        y (add (add h q) (mul q x))
        grads (differentiate y)]
    (is (ma/equals [-20 297 8] (.value y)))
    (is (ma/equals [46 202 -6] (get grads x)))))

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
    (is (ma/equals 24.7041 (.value g) 1e-4))
    (is (ma/equals 138.8338 (get grads a) 1e-4))
    (is (ma/equals 645.5773 (get grads b) 1e-4))
    (is (ma/equals -6.9417 (get grads c) 1e-4))
    (is (ma/equals 6.9417 (get grads d) 1e-4))
    (is (ma/equals -6.9417 (get grads e) 1e-4))
    (is (ma/equals 0.4958 (get grads f) 1e-4))))

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
    (is (ma/equals [24.7041 14112.0004 1104.5045] (.value g) 1e-4))
    (is (ma/equals [138.8338 4199.9999 939.9961] (get grads a) 1e-4))
    (is (ma/equals 36384.5540 (get grads b) 1e-4))
    (is (ma/equals [-6.9417 -168.0 -46.9998] (get grads c) 1e-4))
    (is (ma/equals [6.9417 168.0 46.9998] (get grads d) 1e-4))
    (is (ma/equals [-6.9417 -168.0 -46.9998] (get grads e) 1e-4))
    (is (ma/equals [0.4958 0.5 0.5] (get grads f) 1e-4))))

;; Adapted from: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
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
    (is (ma/equals 0.0285 (.value L) 1e-4))
    (is (ma/equals -0.0067 (get grads w) 1e-4))
    (is (ma/equals -0.0540 (get grads b) 1e-4))))

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
    (is (ma/equals 0.9817 (.value L) 1e-4))
    (is (ma/equals [-0.0062 -0.0048 0.0269 0.0 0.2454] (get grads w) 1e-4))
    (is (ma/equals -1.6909 (get grads b) 1e-4))))

;; Adapted from: https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/
(deftest test4
  (let [x (make-node 2.71)
        y (loop [i 0
                 y x]
            (if (= i 100)
              y
              (recur (inc i) (sin (add x y)))))
        grads (differentiate y)]
    (is (ma/equals 0.2465 (.value y) 1e-4))
    (is (ma/equals -0.2050 (get grads x) 1e-4))))

(deftest test4-vector
  (let [x (make-node [2.71 7.12 1.27])
        y (loop [i 0
                 y x]
            (if (= i 100)
              y
              (recur (inc i) (sin (add x y)))))
        grads (differentiate y)]
    (is (ma/equals [0.2465 0.9718 0.8519] (.value y) 1e-4))
    (is (ma/equals [-0.2050 -0.1907 -0.3437] (get grads x) 1e-4))))