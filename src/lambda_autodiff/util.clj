(ns lambda-autodiff.util
  (:require [clojure.core.matrix :as m]
            [clojure.math.combinatorics :as combo]))

(defn choose
  "Given a probability distribution, sample from the distribution and return the selected index"
  [probs]
  (let [cdf (reduce (fn [cdf p] (conj cdf (+ (last cdf) p))) [0] probs) ;; Convert probabilities into CDF
        ranges (partition 2 1 cdf)] ;; Convert into pairs of [lower, upper] prbability ranges
    ;; Generate a random number within [0, 1] then search for and return the index of the corresponding number
    (java.util.Collections/binarySearch ranges
                                        (rand)
                                        (fn [r e] (cond (< e (first r)) 1
                                                        (> e (second r)) -1
                                                        :else 0)))))

;; Matrix utils

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
  (let [d (- (count s) (count t))
        s (if (< d 0) (vec (concat (repeat (- d) 1) s)) s)
        t (if (> d 0) (vec (concat (repeat d 1) t)) t)]
    (if (not (every? true? (map #(or (= %1 %2) (= %1 1) (= %2 1)) s t)))
        [] ;; Not broadcastable
        (->> (map vector s t)
             (keep-indexed #(if (> (second %2) (first %2)) %1))))))

;; TODO: can be replaced with core.matrix/clamp
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
                      (loop [dim (- (count ranges) 1)
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
  "Returns an n x 1 array with element at index i set to one"
  [n i]
  (m/mset (m/zero-array [n 1]) i 0 1.0))

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

;; Graph utils

(defn make-graphviz-dot
  "Generates the computational graph starting at a node in Graphviz DOT format"
  [root]
  (loop [dot (str "digraph G {\n")
         stack (list root)]
    (if (empty? stack)
      (str dot "}")
      (let [node (peek stack)
            dot' (str dot (format "%s [label=\"%s\"];\n" (hash node) (str node)))]
        (recur (->> (.children node)
                    (map (fn [[child weight]] (format "%s -> %s [label=\"%s\"];\n" (hash node) (hash child) weight)))
                    (reduce str dot'))
               (->> (.children node)
                    (map (fn [[child _]] child))
                    (reduce conj (pop stack))))))))

(defn count-graph
  "Counts number of nodes in graph starting at a given node"
  [root]
  (loop [stack (list root)
         result 0]
    (if (empty? stack)
      result
      (let [node (peek stack)]
        (recur (->> (.children node) (map first) (reduce conj (pop stack)))
               (+ result 1))))))
