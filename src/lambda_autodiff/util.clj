(ns lambda-autodiff.util
  (:require [clojure.core.matrix :as m]))

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

(defn broadcast-axes
  "Returns the axes in a given shape which will be expanded during broadcasting with another shape"
  [s t]
  (let [d (- (count s) (count t))
        s (if (< d 0) (vec (concat (repeat (- d) 1) s)) s)
        t (if (> d 0) (vec (concat (repeat d 1) t)) t)]
    (if (not (every? true? (map #(or (= %1 %2) (= %1 1) (= %2 1)) s t)))
        [] ;; Not broadcastable
        (keep-indexed #(if (= %2 1) %1) s))))

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
