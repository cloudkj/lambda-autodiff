(ns lambda-autodiff.util-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [lambda-autodiff.util :refer :all]))

(deftest test-asum
  (let [a [[[0 1] [1 0] [2 1] [1 3]]
           [[1 0] [5 1] [7 0] [-4 1]]
           [[-1 0] [0 3] [-1 1] [0 0]]]]
    (is (= [[0 1] [6 4] [8 2] [-3 4]] (asum a [0])))
    (is (= [[4 5] [9 2] [-2 4]] (asum a [1])))
    (is (= [[1 1 3 4] [1 6 7 -3] [-1 3 0 0]] (asum a [2])))
    (is (= [11 11] (asum a [0 1])))
    (is (= [11 11] (asum a [1 0])))
    (is (= [1 10 10 1] (asum a [0 2])))
    (is (= [1 10 10 1] (asum a [2 0])))
    (is (= [9 11 2] (asum a [2 1])))
    (is (= [9 11 2] (asum a [1 2])))
    (is (= 22 (asum a [2 0 1])))
    (is (= 22 (asum a [1 2 0])))))

(deftest test-broadcast-axes
  (is (= [1 3] (broadcast-axes [8 1 6 1] [7 1 5])))
  (is (= [0 2] (broadcast-axes [7 1 5] [8 1 6 1])))
  (is (= [0 1] (broadcast-axes [3] [256 256 3])))
  (is (empty? (broadcast-axes [5 4] [1])))
  (is (= [0 2] (broadcast-axes [3 1] [15 3 5])))
  (is (empty? (broadcast-axes [3] [4])))
  (is (empty? (broadcast-axes [2 1] [8 4 3]))))

(deftest test-dimension-max
  (let [a [[[1 1 2 4]
            [5 6 7 8]
            [3 2 1 0]
            [1 2 3 4]]
           [[4 4 2 1]
            [8 7 6 5]
            [0 1 2 3]
            [3 2 0 9]]
           [[1 1 2 4]
            [5 6 7 8]
            [3 2 1 0]
            [1 2 3 4]]]
        b [[1 2 0]
           [2 4 -1]
           [3 8 0]
           [0 3 2]]]
    (is (= [[4 4 2 4] [8 7 7 8] [3 2 2 3] [3 2 3 9]] (dimension-max a 0)))
    (is (= [[5 6 7 8] [8 7 6 9] [5 6 7 8]] (dimension-max a 1)))
    (is (= [[4 8 3 4] [4 8 3 9] [4 8 3 4]] (dimension-max a 2)))
    (is (= [3 8 2] (dimension-max b 0)))
    (is (= [2 4 8 3] (dimension-max b 1)))))
