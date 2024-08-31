(ns lambda-autodiff.array
  (:refer-clojure :exclude [count flatten max min set])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [clojure.math.combinatorics :as combo])
  (:import [ai.djl.ndarray NDManager]
           [ai.djl.ndarray.types Shape]))

(def ^:dynamic *implementation* :core-matrix)

(def nd-manager (NDManager/newBaseManager))

(defn set-implementation!
  [impl]
  (alter-var-root (var *implementation*) (fn [_] impl)))

(defn array
  [data]
  (case *implementation*
    :core-matrix (m/array data)
    :djl (let [;; Optimistically compute shape of a collection in a depth-first manner
               shape (loop [coll data shape (list)]
                       (if (not (coll? coll))
                         (reverse shape)
                         (recur (first coll)
                                (cons (clojure.core/count coll) shape))))
               flat (clojure.core/flatten data)]
           (assert (= (clojure.core/count flat) (apply * shape)) "Mismatch between input data and shape")
           ;; TODO: figure out how to convert `flat` into properly typed array
           ;; (e.g. `long-array`, `float-array`, etc.)based on element type
           (.create nd-manager (into-array Double/TYPE flat) (Shape. (long-array shape))))))

;; Adapter functions - shim layer over underlying implementations

(defn add
  [a b]
  (case *implementation*
    :core-matrix (m/add a b)
    :djl (.add a b)))

(defn broadcast
  [a shape]
  (case *implementation*
    :core-matrix (m/broadcast a shape)
    :djl (.broadcast a (long-array shape))))

(defn cos
  [a]
  (case *implementation*
    :core-matrix (m/cos a)
    :djl (.cos a)))

(defn count
  [a]
  (case *implementation*
    :core-matrix (m/ecount a)
    :djl (.size a)))

(defn dimensionality
  [a]
  (case *implementation*
    :core-matrix (m/dimensionality a)
    :djl (.dimension (.getShape a))))

(defn div
  [a b]
  (case *implementation*
    :core-matrix (m/div a b)
    :djl (.div a b)))

(defn emap
  [f a]
  (case *implementation*
    :core-matrix (m/emap f a)
    ;; TODO: figure out how to create correctly typed primitive array based on type of output from `f`
    :djl (.create nd-manager (into-array Double/TYPE (map f (.toArray a))) (.getShape a))))

(defn exp
  [a]
  (case *implementation*
    :core-matrix (m/exp a)
    :djl (.exp a)))

(defn flatten
  [a]
  (case *implementation*
    :core-matrix (m/to-vector a)
    :djl (.flatten a)))

(defn log
  [a]
  (case *implementation*
    :core-matrix (m/log a)
    :djl (.log a)))

(defn max
  [a]
  (case *implementation*
    :core-matrix (m/emax a)
    ;; TODO: use correct getter (e.g. getFloat, getInt, ...) depending on numeric type of `a` elements
    :djl (.getDouble (.max a) (long-array 0))))

(defn min
  [a]
  (case *implementation*
   :core-matrix (m/emin a)
    ;; TODO: use correct getter (e.g. getFloat, getInt, ...) depending on numeric type of `a` elements
   :djl (.getDouble (.min a) (long-array 0))))

(defn mul
  [a b]
  (case *implementation*
    :core-matrix (m/mul a b)
    :djl (.mul a b)))

(defn negate
  [a]
  (case *implementation*
    :core-matrix (m/negate a)
    :djl (.neg a)))

(defn ones
  [shape]
  (case *implementation*
    :core-matrix (m/fill (m/new-array shape) 1)
    :djl (.ones nd-manager (Shape. (long-array shape)))))

(defn pow
  [a exponent]
  (case *implementation*
    :core-matrix (m/pow a exponent)
    :djl (.pow a exponent)))

(defn reshape
  [a shape]
  (case *implementation*
    :core-matrix (m/reshape a shape)
    :djl (.reshape a (Shape. (long-array shape)))))

(defn sample-normal
  [shape]
  (case *implementation*
    :core-matrix (mr/sample-normal shape)
    :djl (.randomNormal nd-manager (Shape. (long-array shape)))))

(defn select
  [a & indexes]
  (apply m/select (cons a indexes)))

(defn set
  [a & args]
  (apply m/set-selection (cons a args)))

(defn set-indexes
  [a indexes values]
  (m/set-indices a indexes values))

(defn shape
  [a]
  (m/shape a))

(defn sin
  [a]
  (m/sin a))

(defn slice
  [a index]
  (let [indexes (cons index (repeat (dec (dimensionality a)) :all))]
    (apply select (cons a indexes))))

(defn square
  [a]
  (m/square a))

(defn sub
  [a b]
  (m/sub a b))

(defn sum
  [a]
  (m/esum a))

(defn transpose
  [a]
  (m/transpose a))

(defn zeros
  [shape]
  (m/zero-array shape))

;; Testing and equality utility functions

(defn equals
  ([a b]
   (equals a b 0.0))
  ([a b epsilon]
   (m/equals a b epsilon)))

;; Utility functions

(defn asum
  "Calculates sum of numerical array elements over the given axes"
  [a axes]
  (reduce (fn [a axis]
            ;; Sum along a specific axis
            (let [size (nth (m/shape a) axis)]
            (reduce m/add (for [i (range size)] (m/slice a axis i)))))
          a
          ;; Sum along all axes from right to left so axis indices remain valid
          (reverse (sort axes))))

(defn batch-mmul
  "Extended matrix multiplication to allow batched matrix multiply if both arguments are N>2 dimensional"
  [a b]
  (cond (not (= (m/dimensionality a) (m/dimensionality b))) (m/mmul a b)
        (<= (m/dimensionality a) 2) (m/mmul a b)
        ;; For now, enforce strict requirement that batch dimensions match between two operands, and do not broadcast
        (not (= (drop-last 2 (m/shape a)) (drop-last 2 (m/shape b)))) (m/mmul a b)
        :else 
        ;; Every dimension except the last two are treated as batch dimensions
        (let [batch-shape (drop-last 2 (m/shape a))
              result-shape (-> (vec batch-shape)
                               (conj (last (butlast (m/shape a))))
                               (conj (last (m/shape b))))]
          (loop [selections (apply combo/cartesian-product (map range batch-shape))
                 result (m/zero-array result-shape)]
            (if (empty? selections)
              result
              (let [selection (conj (vec (first selections)) :all :all)
                    suba (apply m/select (cons a selection))
                    subb (apply m/select (cons b selection))]
                (recur (rest selections)
                       (->> (m/mmul suba subb)
                            (conj selection)
                            (cons result)
                            (apply m/set-selection)))))))))

(defn broadcast-axes
  "Returns the axes in a given shape which will be expanded during broadcasting with another shape"
  [s t]
  (let [d (- (clojure.core/count s) (clojure.core/count t))
        s (if (< d 0) (vec (concat (repeat (- d) 1) s)) s)
        t (if (> d 0) (vec (concat (repeat d 1) t)) t)]
    (if (not (every? true? (map #(or (= %1 %2) (= %1 1) (= %2 1)) s t)))
        [] ;; Not broadcastable
        (->> (map vector s t)
             (keep-indexed #(if (> (second %2) (first %2)) %1))))))

(defn clamp
  "Clamp all elements within an array to be within the range `[lo, hi]`"
  [a lo hi]
  (m/emap (fn [e] (cond (< e lo) lo
                        (> e hi) hi
                        :else e))
          a))

(defn flatten-argmax
  "Returns the index of the maximum value within an array after flattening"
  [a]
  (->> (m/to-vector a)
       (map-indexed vector)
       (apply max-key second)
       (first)))

(defn dimension-max
  "Returns the maximum value (or index of the value) within an array along a dimension"
  ([a dim]
   (dimension-max a dim false))
  ([a dim argmax?]
   (let [shape (m/shape a)
        ;; Helper to increment seq of indexes for all dimensions while staying in bounds of input shape
         inc-ranges (fn [ranges]
                      (loop [dim (- (clojure.core/count ranges) 1)
                             ranges ranges]
                        (cond (< dim 0) nil ;; No more dimensions to increment
                              (nil? (nth ranges dim)) (recur (dec dim) ranges) ;; Skip dimension
                             ;; Increment index for dimension when still within bounds
                              (< (nth ranges dim) (dec (nth shape dim))) (assoc ranges dim (inc (nth ranges dim)))
                             ;; Reset index and try to increment index for next dimension
                              :else (recur (dec dim) (assoc ranges dim 0)))))]
     (loop [indexes []
            values []
            ;; Ranges to take subarrays: start at index 0 for every dimension except the one over which we're taking max
            ranges (assoc (vec (repeat (m/dimensionality a) 0)) dim nil)]
       (if (nil? ranges)
         ;; Output is the same shape as input with given dimension dropped
         (-> (keep-indexed #(when (not (= %1 dim)) %2) shape)
             (m/zero-array)
             (m/set-indices indexes values))
         ;; Add length=1 to every index range in order to take subarray
         (let [slice (m/submatrix a (map (fn [e] (if (nil? e) e [e 1])) ranges))]
           (recur (conj indexes (filter #(not (nil? %)) ranges))
                  (conj values (if argmax? (flatten-argmax slice) (m/emax slice)))
                  (inc-ranges ranges))))))))

(defn one-hot
  "Returns an n-sized array with element at index i set to one"
  [n i]
  (m/mset (m/zero-array [n]) i 1.0))

(defn swap-last-dims
  "Swaps the last two dimensions of an array"
  [a]
  (let [dims (m/dimensionality a)]
    (if (<= dims 2)
      (m/transpose a)
      (let [ordering (-> (vec (range (- dims 2)))
                         (conj (- dims 1))
                         (conj (- dims 2)))]
        (m/transpose a ordering)))))