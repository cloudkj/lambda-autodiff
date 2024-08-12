(ns lambda-autodiff.core
  (:require [clojure.core.matrix :as m]
            [lambda-autodiff.util :as util]))

;; Custom type to represent a node in the computational graph. Each node instance is a unique
;; entity, independent of underlying values. We use a custom type as workaround for Clojure
;; default value-based equality and hashing behavior.
(deftype Node [value label children]
  Object
  (toString [node]
    (if (nil? (.label node))
      (str (.value node))
      (format "%s [%s]" (.label node) (str (.value node))))))

(defn make-node
  "Creates a node in the computational graph"
  ([value]
   (make-node value nil))
  ([value label]
   (make-node value label []))
  ([value label children]
   (Node. value label children)))

;; Operations

(defn add
  [a b]
  (make-node (m/add (.value a) (.value b)) "+" [[a identity] [b identity]]))

(defn mul
  [a b]
  (make-node (m/mul (.value a) (.value b))
             "*"
             [[a (fn [upstream] (m/mul upstream (.value b)))]
              [b (fn [upstream] (m/mul upstream (.value a)))]]))

(defn mmul
  [a b]
  (make-node (util/batch-mmul (.value a) (.value b))
             "mmul"
             [[a (fn [upstream] (util/batch-mmul upstream (util/swap-last-dims (.value b))))]
              [b (fn [upstream] (util/batch-mmul (util/swap-last-dims (.value a)) upstream))]]))

(defn reshape
  [a shape]
  (make-node (m/reshape (.value a) shape)
             "reshape"
             [[a (fn [upstream] (m/reshape upstream (m/shape (.value a))))]]))

(defn transpose
  [a]
  (make-node (m/transpose (.value a)) "T" [[a m/transpose]]))

(defn sum
  [a]
  (make-node (m/esum (.value a))
             "sum"
             [[a (fn [upstream] (m/broadcast (m/esum upstream) (m/shape (.value a))))]]))

(defn pow
  [a exponent]
  ;; TODO: add restriction that `exponent` is scalar only and not another node
  (make-node (m/pow (.value a) exponent)
             "pow"
             [[a (fn [upstream] (m/mul upstream (m/mul exponent (m/pow (.value a) (- exponent 1)))))]]))

(defn exp
  [a]
  (let [out (m/exp (.value a))]
    (make-node out "exp" [[a (fn [upstream] (m/mul upstream out))]])))

(defn log
  [a]
  (make-node (m/log (.value a)) "log" [[a (fn [upstream] (m/mul upstream (m/div 1 (.value a))))]]))

(defn sin
  [a]
  (make-node (m/sin (.value a)) "sin" [[a (fn [upstream] (m/mul upstream (m/cos (.value a))))]]))

(defn tanh
  [a]
  (let [out (m/div (m/sub (m/exp (m/mul 2 (.value a))) 1)
                   (m/add (m/exp (m/mul 2 (.value a))) 1))]
    (make-node out "tanh" [[a (fn [upstream] (m/mul upstream (m/sub 1 (m/square out))))]])))

(defn relu
  [a]
  (make-node (m/emap #(if (> % 0) % 0) (.value a))
             "relu"
             [[a (fn [upstream] (m/mul upstream (m/emap #(if (> % 0) 1 0) (.value a))))]]))

(defn div
  [a b]
  (make-node (m/div (.value a) (.value b))
             "/"
             [[a (fn [upstream] (m/div upstream (.value b)))]
              [b (fn [upstream] (m/div (m/mul -1 upstream (.value a)) (m/square (.value b))))]]))

(defn neg
  [a]
  (make-node (m/negate (.value a)) "-" [[a m/negate]]))

(defn sub
  [a b]
  (make-node (m/sub (.value a) (.value b)) "-" [[a identity] [b m/negate]]))

;; Differentiation

(defn differentiate
  "Returns a map of nodes to partial derivative values"
  [root]
  (loop [gradients {}
         ;; Reshape base derivative to match root output shape
         stack (list [root (m/fill (m/new-array (m/shape (.value root))) 1)])]
    (if (empty? stack)
      ;; Post-process gradients at the end of accumulation to account for different shapes
      (->> gradients
           (map (fn [[c g]]
                  [c
                   ;; If child shape is smaller than gradient shape, sum gradient along axes that would be expanded during broadcasting
                   (cond (< (m/ecount (.value c)) (m/ecount g))
                         (->> (util/broadcast-axes (m/shape (.value c)) (m/shape g))
                              (util/asum g))
                         ;; If child shape is larger than gradient shape, broadcast gradient to match child shape
                         (> (m/ecount (.value c)) (m/ecount g))
                         (m/broadcast g (m/shape (.value c)))
                         :else g)]))
           ;; Reshape gradient to match child shape if dimensionality unequal
           (map (fn [[c g]] [c (if (not (= (m/dimensionality (.value c)) (m/dimensionality g))) (m/reshape g (m/shape (.value c))) g)]))
           (into {}))
      (let [[node upstream] (peek stack)
            ;; Apply chain rule along path from root to child with update function
            children (map (fn [[child update]] [child (update upstream)]) (.children node))]
        (recur (reduce (fn [gs [child downstream]]
                         ;; Accumulate gradients for each child to apply multivariate chain rule
                         (assoc gs child (m/add (get gs child 0) downstream)))
                       gradients
                       children)
               ;; Push all child nodes onto stack and pass derivatives, from root to child thus far, downstream
               (reduce conj (pop stack) children))))))
