(ns lambda-autodiff.util
  (:require [lambda-autodiff.core :as core]))

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

;; Graph utils

(defn make-graphviz-dot
  "Generates the computational graph starting at a node in Graphviz DOT format"
  [root]
  (loop [stack (list root)
         nodes #{}
         edges {}]
    (if (empty? stack)
      (let [dot (str "digraph G {\n")
            dot (->> nodes
                     (map (fn [node]
                             (let [label (if (nil? (.label node)) (str node) (.label node))]
                               (format "%s [label=\"%s\"];\n" (hash node) label))))
                     (reduce str dot))
            dot (->> edges
                     (mapcat (fn [[node children-indexed]] (map #(list node %) (vals children-indexed))))
                     (map (fn [[node child]] (format "%s -> %s;\n" (hash node) (hash child))))
                     (reduce str dot))]
        (str dot "}"))
      (let [node (peek stack)]
        (recur (->> (if (core/leaf? node) (list) (.children node))
                    (map first)
                    (reduce conj (pop stack)))
               (conj nodes node)
               (->> (if (core/leaf? node) (list) (.children node))
                    (map-indexed (fn [i [child _]] [i child]))
                    (reduce (fn [e [i child]] (assoc-in e [node i] child)) edges)))))))

(defn count-graph
  "Counts number of nodes and edges in the graph starting at a given node"
  [root]
  (loop [stack (list root)
         nodes #{}
         edges 0]
    (if (empty? stack)
      [(count nodes) edges]
      (let [node (peek stack)]
          (recur (->> (.children node) (map first) (reduce conj (pop stack)))
                 (conj nodes node)
                 (+ edges (count (.children node))))))))
