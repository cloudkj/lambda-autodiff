(ns lambda-autodiff.core
  (:require [clojure.math :as math]
            [clojure.core.matrix :as m]
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
  (make-node (m/add (.value a) (.value b))
             "+"
             [[a 1] [b 1]]))

(defn mul
  [a b]
  (make-node (m/mul (.value a) (.value b))
             "*"
             [[a (.value b)] [b (.value a)]]))

(defn sum
  [a]
  (make-node (m/esum (.value a))
             "sum"
             [[a (m/fill (m/new-array (m/shape (.value a))) 1)]]))

(defn pow
  [a exponent]
  ;; TODO: add restriction that `exponent` is scalar only and not another node
  (make-node (m/pow (.value a) exponent)
             "pow"
             [[a (m/mul exponent (m/pow (.value a) (- exponent 1)))]]))

(defn exp
  [a]
  (let [out (m/exp (.value a))]
    (make-node out "exp" [[a out]])))

(defn log
  [a]
  (make-node (m/log (.value a)) "log" [[a (m/div 1 (.value a))]]))

(defn sin
  [a]
  (make-node (m/sin (.value a)) "sin" [[a (m/cos (.value a))]]))

(defn tanh
  [a]
  (let [out (m/div (m/sub (m/exp (m/mul 2 (.value a))) 1)
                   (m/add (m/exp (m/mul 2 (.value a))) 1))]
    (make-node out "tanh" [[a (m/sub 1 (m/square out))]])))

(defn relu
  [a]
  (make-node (m/emap #(if (> % 0) % 0) (.value a))
             "relu"
             [[a (m/emap #(if (> % 0) 1 0) (.value a))]]))

(defn div
  [a b]
  (mul a (pow b -1)))

(defn neg
  [a]
  (mul a (make-node -1)))

(defn sub
  [a b]
  (add a (neg b)))

;; API

(defn differentiate
  "Returns a map of nodes to partial derivative values"
  [root]
  (loop [gradients {}
         ;; Reshape base derivative to match root output shape
         stack (list [root (m/fill (m/new-array (m/shape (.value root))) 1)])]
    (if (empty? stack)
      gradients
      (let [[node upstream] (peek stack)
             children (map (fn [[child local]]
                             (let [child-shape (m/shape (.value child))
                                   ;; Update with `upstream*local` to apply chain rule along path from root to child
                                   update (m/mul upstream local)
                                   axes (util/broadcast-axes child-shape (m/shape (.value node)))]
                               ;; Account for broadcasting by determining axes to sum updated partial derivative
                               ;; based on child shape with relation to shape of upstream derivative
                               [child (-> update (util/asum axes) (m/reshape child-shape))]))
                           (.children node))]
        (recur (reduce (fn [gs [child downstream]]
                         ;; Accumulate gradients for each child to apply multivariate chain rule
                         (assoc gs child (m/add (get gs child 0) downstream)))
                       gradients
                       children)
               ;; Push all child nodes onto stack and pass derivatives, from root to child thus far, downstream
               (reduce conj (pop stack) children))))))
