(ns lambda-autodiff.util)

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
