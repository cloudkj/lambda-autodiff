(ns lambda-autodiff.core
  (:require [lambda-autodiff.array :as ma]))

(defprotocol LeafNode
  (leaf? [this] false))

;; Custom type to represent a node in the computational graph. Each node instance is a unique
;; entity, independent of underlying values. We use a custom type as workaround for Clojure
;; default value-based equality and hashing behavior.
(deftype Node [value label children]
  LeafNode
  (leaf? [this] (empty? children))
  Object
  (toString [node]
    (let [v (str (.value node))
          v (if (< (count v) 10) v (str (subs v 0 10) "..."))]
      (if (nil? (.label node))
        v
        (format "%s {%s}" (.label node) v)))))

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
  (make-node (ma/add (.value a) (.value b)) "+" [[a identity] [b identity]]))

(defn mul
  [a b]
  (make-node (ma/mul (.value a) (.value b))
             "*"
             [[a (fn [upstream] (ma/mul upstream (.value b)))]
              [b (fn [upstream] (ma/mul upstream (.value a)))]]))

(defn mmul
  [a b]
  (make-node (ma/batch-mmul (.value a) (.value b))
             "mmul"
             [[a (fn [upstream] (ma/batch-mmul upstream (ma/swap-last-dims (.value b))))]
              [b (fn [upstream] (ma/batch-mmul (ma/swap-last-dims (.value a)) upstream))]]))

(defn reshape
  [a shape]
  (make-node (ma/reshape (.value a) shape)
             "reshape"
             [[a (fn [upstream] (ma/reshape upstream (ma/shape (.value a))))]]))

(defn transpose
  [a]
  (make-node (ma/transpose (.value a)) "T" [[a ma/transpose]]))

(defn sum
  [a]
  (make-node (ma/sum (.value a))
             "sum"
             [[a (fn [upstream] (ma/broadcast (ma/sum upstream) (ma/shape (.value a))))]]))

(defn pow
  [a exponent]
  (assert (number? exponent))
  (make-node (ma/pow (.value a) exponent)
             "pow"
             [[a (fn [upstream] (ma/mul upstream (ma/mul exponent (ma/pow (.value a) (- exponent 1)))))]]))

(defn exp
  [a]
  (let [out (ma/exp (.value a))]
    (make-node out "exp" [[a (fn [upstream] (ma/mul upstream out))]])))

(defn log
  [a]
  (make-node (ma/log (.value a)) "log" [[a (fn [upstream] (ma/mul upstream (ma/div 1 (.value a))))]]))

(defn sin
  [a]
  (make-node (ma/sin (.value a)) "sin" [[a (fn [upstream] (ma/mul upstream (ma/cos (.value a))))]]))

(defn tanh
  [a]
  (let [out (ma/div (ma/sub (ma/exp (ma/mul 2 (.value a))) 1)
                    (ma/add (ma/exp (ma/mul 2 (.value a))) 1))]
    (make-node out "tanh" [[a (fn [upstream] (ma/mul upstream (ma/sub 1 (ma/square out))))]])))

(defn relu
  [a]
  (make-node (ma/emap #(if (> % 0) % 0) (.value a))
             "relu"
             [[a (fn [upstream] (ma/mul upstream (ma/emap #(if (> % 0) 1 0) (.value a))))]]))

(defn div
  [a b]
  (make-node (ma/div (.value a) (.value b))
             "/"
             [[a (fn [upstream] (ma/div upstream (.value b)))]
              [b (fn [upstream] (ma/div (ma/mul (ma/negate upstream) (.value a)) (ma/square (.value b))))]]))

(defn neg
  [a]
  (make-node (ma/negate (.value a)) "-" [[a ma/negate]]))

(defn sub
  [a b]
  (make-node (ma/sub (.value a) (.value b)) "-" [[a identity] [b ma/negate]]))

(defn join
  ([a b]
   (join a b 0))
  ([a b dim]
   (let [a-dims (ma/dimensionality (.value a))
         b-dims (ma/dimensionality (.value b))
         a-dim-count (ma/dimension-count (.value a) dim)
         b-dim-count (ma/dimension-count (.value b) dim)]
     (make-node (ma/join-along (.value a) (.value b) dim)
                "join"
                [[a (fn [upstream]
                        (apply ma/select (->> (range 0 a-dim-count)
                                              (assoc (vec (repeat a-dims :all)) dim)
                                              (cons upstream))))]
                 [b (fn [upstream]
                       (apply ma/select (->> (range a-dim-count (+ a-dim-count b-dim-count))
                                              (assoc (vec (repeat b-dims :all)) dim)
                                              (cons upstream))))]]))))

(defn select
  [a & indexes]
  (make-node (apply ma/select (cons (.value a) indexes))
              "select"
              [[a (fn [upstream] (->> upstream
                                      (conj (vec indexes))
                                      (cons (ma/zeros (ma/shape (.value a))))
                                      (apply ma/set)))]]))

;; Differentiation

(defn topological-sort
  [root]
  (loop [stack (list root)
         started #{}
         finished #{}
         sorted (list)]
    (if (empty? stack)
      sorted
      (let [node (peek stack)]
        (cond (not (contains? started node))
              (recur (reduce conj stack (map first (.children node))) ;; Push children onto stack
                     (conj started node)                              ;; Mark as started
                     finished
                     sorted)
              (not (contains? finished node))
              (recur (pop stack)          ;; Pop stack
                     started
                     (conj finished node) ;; Mark as finished
                     (cons node sorted))  ;; Store result
              :else
              (recur (pop stack) started finished sorted))))))

(defn shape-gradients
  "Post-process gradients at the end of accumulation to account for different shapes"
  [gradients]
  (->> gradients
       (map (fn [[c g]]
              [c
               ;; If child shape is smaller than gradient shape, sum gradient along axes that would be expanded during broadcasting
               (cond (< (ma/count (.value c)) (ma/count g))
                     (->> (ma/broadcast-axes (ma/shape (.value c)) (ma/shape g))
                          (ma/asum g))
                     ;; If child shape is larger than gradient shape, broadcast gradient to match child shape
                     (> (ma/count (.value c)) (ma/count g))
                     (ma/broadcast g (ma/shape (.value c)))
                     :else g)]))
       ;; Reshape gradient to match child shape if dimensionality unequal
       (map (fn [[c g]] [c (if (not (= (ma/dimensionality (.value c)) (ma/dimensionality g))) (ma/reshape g (ma/shape (.value c))) g)]))
       (into {})))

(defn differentiate
  "Returns a map of nodes to partial derivative values"
  [root]
  (loop [gradients {root (ma/ones (ma/shape (.value root)))}
         sorted (topological-sort root)]
    (if (empty? sorted)
        (shape-gradients gradients)
        (let [node (first sorted)
              upstream (get gradients node)
              ;; Apply chain rule by passing local derivative to children to compute downstream derivatives
              children (map (fn [[child update]] [child (update upstream)]) (.children node))]
          (recur ;; Accumulate gradients for each child to apply multivariate chain rule
                 (reduce (fn [gs [child downstream]] (assoc gs child (ma/add (get gs child 0) downstream)))
                         gradients
                         children)
                 (rest sorted))))))
