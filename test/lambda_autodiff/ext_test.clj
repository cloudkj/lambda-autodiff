(ns lambda-autodiff.ext-test
  (:require [clojure.test :refer :all]
            [lambda-autodiff.core :refer :all]
            [lambda-autodiff.array :as ma]
            [lambda-autodiff.ext :refer :all]))

(deftest test-im2col
  (let [a (make-node [[[[0 0 0 0 0 0 0]
                        [0 2 1 0 2 2 0]
                        [0 1 2 1 1 2 0]
                        [0 2 2 0 2 2 0]
                        [0 1 2 2 1 1 0]
                        [0 0 0 1 1 1 0]
                        [0 0 0 0 0 0 0]]
                       [[0 0 0 0 0 0 0]
                        [0 0 2 1 1 2 0]
                        [0 0 1 2 0 1 0]
                        [0 0 1 2 0 1 0]
                        [0 1 0 1 0 2 0]
                        [0 2 2 2 2 0 0]
                        [0 0 0 0 0 0 0]]
                       [[0 0 0 0 0 0 0]
                        [0 0 1 0 1 0 0]
                        [0 1 1 1 1 2 0]
                        [0 1 2 1 2 0 0]
                        [0 0 0 2 2 2 0]
                        [0 2 2 2 0 2 0]
                        [0 0 0 0 0 0 0]]]])
        b (im2col a 3 3 2)
        grads (differentiate b)]
    (is (ma/equals [[[0 0 0 0 2 1 0 1 2 0 0 0 0 0 2 0 0 1 0 0 0 0 0 1 0 1 1]
                     [0 0 0 1 0 2 2 1 1 0 0 0 2 1 1 1 2 0 0 0 0 1 0 1 1 1 1]
                     [0 0 0 2 2 0 1 2 0 0 0 0 1 2 0 0 1 0 0 0 0 1 0 0 1 2 0]
                     [0 1 2 0 2 2 0 1 2 0 0 1 0 0 1 0 1 0 0 1 1 0 1 2 0 0 0]
                     [2 1 1 2 0 2 2 2 1 1 2 0 1 2 0 0 1 0 1 1 1 2 1 2 0 2 2]
                     [1 2 0 2 2 0 1 1 0 0 1 0 0 1 0 0 2 0 1 2 0 2 0 0 2 2 0]
                     [0 1 2 0 0 0 0 0 0 0 1 0 0 2 2 0 0 0 0 0 0 0 2 2 0 0 0]
                     [2 2 1 0 1 1 0 0 0 0 1 0 2 2 2 0 0 0 0 2 2 2 2 0 0 0 0]
                     [1 1 0 1 1 0 0 0 0 0 2 0 2 0 0 0 0 0 2 2 0 0 2 0 0 0 0]]] (.value b)))
    (is (= [1 3 7 7] (ma/shape (get grads a))))
    (is (ma/equals [[1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]] (-> (get grads a) (ma/select 0 0 :all :all))))
    (is (ma/equals [[1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]] (-> (get grads a) (ma/select 0 1 :all :all))))
    (is (ma/equals [[1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [2 2 4 2 4 2 2]
                    [1 1 2 1 2 1 1]
                    [1 1 2 1 2 1 1]] (-> (get grads a) (ma/select 0 2 :all :all))))))

(deftest test-maxpool2d
  (let [a (make-node [[[[1 1 2 4]
                        [5 6 7 8]
                        [3 2 1 0]
                        [1 2 3 4]]
                       [[4 4 2 1]
                        [8 7 6 5]
                        [0 1 4 3]
                        [3 2 0 9]]
                       [[1 1 2 4]
                        [5 6 7 8]
                        [3 2 1 0]
                        [1 2 3 4]]]])
        b (maxpool2d a 2 2 2)
        c (maxpool2d a 2 2 1)
        gradsb (differentiate b)
        gradsc (differentiate c)]
    (is (ma/equals [[[[6 8]
                     [3 4]]
                    [[8 6]
                     [3 9]]
                    [[6 8]
                     [3 4]]]] (.value b)))
    (is (= [1 3 4 4] (ma/shape (get gradsb a))))
    (is (ma/equals [[0 0 0 0]
                    [0 1 0 1]
                    [1 0 0 0]
                    [0 0 0 1]] (-> (get gradsb a) (ma/select 0 0 :all :all))))
    (is (ma/equals [[0 0 0 0]
                    [1 0 1 0]
                    [0 0 0 0]
                    [1 0 0 1]] (-> (get gradsb a) (ma/select 0 1 :all :all))))
    (is (ma/equals [[0 0 0 0]
                    [0 1 0 1]
                    [1 0 0 0]
                    [0 0 0 1]] (-> (get gradsb a) (ma/select 0 2 :all :all))))
    (is (ma/equals [[[[6 7 8]
                     [6 7 8]
                     [3 3 4]]
                    [[8 7 6]
                     [8 7 6]
                     [3 4 9]]
                    [[6 7 8]
                     [6 7 8]
                     [3 3 4]]]] (.value c)))
    (is (= [1 3 4 4] (ma/shape (get gradsc a))))
    (is (ma/equals [[0 0 0 0]
                    [0 2 2 2]
                    [1 0 0 0]
                    [0 0 1 1]] (-> (get gradsc a) (ma/select 0 0 :all :all))))
    (is (ma/equals [[0 0 0 0]
                    [2 2 2 0]
                    [0 0 1 0]
                    [1 0 0 1]] (-> (get gradsc a) (ma/select 0 1 :all :all))))
    (is (ma/equals [[0 0 0 0]
                    [0 2 2 2]
                    [1 0 0 0]
                    [0 0 1 1]] (-> (get gradsc a) (ma/select 0 2 :all :all))))))
