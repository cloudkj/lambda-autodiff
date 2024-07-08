(ns lambda-autodiff.core
  (:require [clojure.math :as math]))

;; Custom type to represent a node in the computational graph. Each node instance is a unique entity, independent of
;; underlying values. We use a custom type as workaround for Clojure default value-based equality/hashing behavior.
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
                    (map (fn [[child weight]] child))
                    (reduce conj (pop stack))))))))

;; Operations

(defn add
  [a b]
  (make-node (+ (.value a) (.value b))
             "+"
             [[a 1] [b 1]]))

(defn mul
  [a b]
  (make-node (* (.value a) (.value b))
             "*"
             [[a (.value b)] [b (.value a)]]))

(defn pow
  [a b]
  ;; TODO: add restriction that `b` is scalar-only and not another node
  (make-node (math/pow (.value a) b)
              "pow"
              [[a (* b (math/pow (.value a) (- b 1)))]]))

(defn exp
  [a]
  (let [out (math/exp (.value a))]
    (make-node out "exp" [[a out]])))

(defn tanh
  [a]
  (let [out (/ (- (math/exp (* 2 (.value a))) 1)
               (+ (math/exp (* 2 (.value a))) 1))]
    (make-node out "tanh" [[a (- 1 (* out out))]])))

(defn relu
  [a]
  (make-node (if (> (.value a) 0) (.value a) 0)
             "relu"
             [[a (if (> (.value a) 0) 1 0)]]))

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
         stack (list [root 1])]
    (if (empty? stack)
      gradients
      (let [[node partial] (peek stack)]
        (recur (->> (.children node)
                    (map (fn [[child local]]
                      [child
                       ;; Accumulating gradients for each child is application of multivariate chain rule
                       (+ (get gradients child 0)
                          ;; Partial * local is application of chain rule along the path from root node
                          (* partial local))]))
                    (into gradients))
               ;; Push all child nodes onto stack and update product of local derivatives from root to child thus far
               (->> (.children node)
                    (map (fn [[child local]] [child (* partial local)]))
                    (reduce conj (pop stack))))))))
