(ns lambda-autodiff.core
  (:require [clojure.math :as math]))

(defn make-node
  "Creates a node in the computational graph"
  ([value]
   (make-node value nil))
  ([value label]
   (make-node value label {}))
  ([value label children]
   {:value value
    :label label
    :children children}))

(defn add
  [a b]
  (make-node (+ (:value a) (:value b))
             "+"
             {a 1 b 1}))

(defn mul
  [a b]
  (make-node (* (:value a) (:value b))
             "*"
             {a (:value b) b (:value a)}))

(defn tanh
  [a]
  (let [out (/ (- (math/exp (* 2 (:value a))) 1)
               (+ (math/exp (* 2 (:value a))) 1))]
    (make-node out "tanh" {a (- 1 (* out out))})))

(defn differentiate
  "Returns a map of nodes to partial derivative values"
  [root]
  (loop [gradients {}
         stack (list [root 1])]
    (if (empty? stack)
      gradients
      (let [[node product] (peek stack)
            [gradients' stack'] (reduce (fn [[gs st] [child weight]]
                                          ;;(println (:label node) "---" weight "--->" (:label child))
                                          [(assoc gs child (+ (get gs child 0) (* product weight)))
                                           (conj st [child (* product weight)])])
                                        [gradients (pop stack)]
                                        (seq (:children node)))]
        (recur gradients' stack')))))

;; Temp; for testing
(defn -main
  [& args]
  (let [a (make-node 4 "a")
        b (make-node 3 "b")
        e (make-node 2 "e")
        c (add a b)
        d (mul a c)
        f (mul b e)
        g (add d f)
        grads (differentiate g)]
    (println (:value g))
    (doseq [[node grad] grads]
      (println (:label node) "\t" grad))))
