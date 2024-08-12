(ns lambda-autodiff.examples.cnn
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [lambda-autodiff.core :refer :all]
            [lambda-autodiff.ext :refer :all]
            [lambda-autodiff.util :as util]
            [nextjournal.clerk :as clerk])
  (:import [java.awt Color]
           [java.awt.image BufferedImage]))

;; # MNIST CNN demo

;; Sample data: https://github.com/matthewdowney/clojure-neural-networks-from-scratch/raw/main/resources/mnist/validation_data.edn.gz

(def sample-url "https://github.com/matthewdowney/clojure-neural-networks-from-scratch/raw/main/resources/mnist/validation_data.edn.gz")

(defn load-data
  "Return a vector of data from the given `path`, which points to a gzipped file where each line is EDN data."
  [path]
  (with-open [rdr (clojure.java.io/reader
                   (java.util.zip.GZIPInputStream.
                    (clojure.java.io/input-stream path)))]
    (->> (line-seq rdr)
         (pmap read-string)
         (into []))))

(defn pixels->image
  [pixels]
  (let [w (count (first pixels))
        h (count pixels)]
    (reduce (fn [image [y row]]
              (reduce (fn [image [x pixel]]
                        (let [[r g b] (repeat 3 (int (* pixel 255)))]
                          (doto image
                            (.setRGB x y (.getRGB (Color. r g b))))))
                      image
                      (map-indexed vector row)))
            (BufferedImage. w h BufferedImage/TYPE_INT_ARGB)
            (map-indexed vector pixels))))

(def data (load-data sample-url))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Data size: " (count data)))

^{:nextjournal.clerk/visibility {:result :hide}}
(def channels 1)
^{:nextjournal.clerk/visibility {:result :hide}}
(def image-size 28)

;; Examples

(->> (take 10 data)
     (map first)
     (map #(m/reshape % [image-size image-size]))
     (map pixels->image)
     (clerk/row))

;; ### Hyperparameters

(m/set-current-implementation :vectorz)

^{:nextjournal.clerk/visibility {:result :hide}}
(def learning-rate 0.01)
^{:nextjournal.clerk/visibility {:result :hide}}
(def batch-size 1)

;; ### Model parameters
;;
;; ##### Conv layer

^{:nextjournal.clerk/visibility {:result :hide}}
(def filter-size 3)
^{:nextjournal.clerk/visibility {:result :hide}}
(def stride 1)
^{:nextjournal.clerk/visibility {:result :hide}}
(def num-filters 32)

^{:nextjournal.clerk/visibility {:result :hide}}
(def w1 (make-node (m/mul (mr/sample-normal [num-filters channels filter-size filter-size]) 0.01) "w1"))
^{:nextjournal.clerk/visibility {:result :hide}}
(def b1 (make-node (m/mul (mr/sample-normal [num-filters]) 0.01) "b1"))

;; ##### Pool layer

^{:nextjournal.clerk/visibility {:result :hide}}
(def pool-filter-size 2)
^{:nextjournal.clerk/visibility {:result :hide}}
(def pool-stride 2)

;; ##### Linear layers

;; TODO: compute flattened layer size from conv/pool parameters
^{:nextjournal.clerk/visibility {:result :hide}}
(def w2 (make-node (m/mul (mr/sample-normal [5408 100]) 0.01) "w2"))
^{:nextjournal.clerk/visibility {:result :hide}}
(def b2 (make-node (m/mul (mr/sample-normal [100]) 0.01) "b2"))

^{:nextjournal.clerk/visibility {:result :hide}}
(def w3 (make-node (mr/sample-normal [100 10])))
^{:nextjournal.clerk/visibility {:result :hide}}
(def b3 (make-node (mr/sample-normal [10])))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Total number of parameters: "
               (->> (map #(.value %) [w1 b1 w2 b2 w3 b3])
                    (map m/ecount)
                    (reduce +))))

;; ### Optimization

(def results
  (loop [data (take 1 data)
         ;;data (repeat 200 (first data))
         ;; Parameters
         w1 w1 b1 b1 w2 w2 b2 b2 w3 w3 b3 b3
         progress []]
    (if (not (empty? data))
      (let [xs (map first (take batch-size data))
            ys (map second (take batch-size data))
            ytargets (make-node (map (fn [y] (util/one-hot 10 y)) ys))
            batch-size (min batch-size (count xs)) ;; Clamp batch-size based on actual number of examples 
            ;; Convolution
            input (make-node (m/reshape xs [batch-size channels image-size image-size]))
            unfold (im2col input filter-size filter-size stride)
            weights (transpose (reshape w1 [num-filters (* channels filter-size filter-size)]))
            m (relu (add (mmul unfold weights) b1))
            out-size (+ (quot (- image-size filter-size) stride) 1)
            conv-out (reshape (transpose m) [batch-size num-filters out-size out-size])
            ;; Pooling
            pool-out (maxpool2d conv-out pool-filter-size pool-filter-size pool-stride)
            ;; Linear
            flattened (reshape pool-out [batch-size 5408])
            output1 (relu (add (mmul flattened w2) b2))
            output2 (add (mmul output1 w3) b3)
            probs (div (exp output2) (sum (exp output2)))
            loss (neg (sum (mul (log probs) ytargets)))
            ;; Backwards pass
            grads (differentiate loss)
            ;; Accuracy
            accurate (->> (.value probs)
                          (map (fn [p] (first (apply max-key second (map-indexed vector p)))))
                          (map #(if (= %1 %2) 1 0) ys)
                          (reduce +))]
        (println "loss:" (.value loss) "avg loss:" (/ (.value loss) batch-size)
                 "accuracy:" (float (/ (-> (last progress) (get :accurate 0) (+ accurate))
                                       (-> (last progress) (get :count 0) (+ batch-size)))))
        ;;(m/pm (get grads w1))
        (doseq [param [w1 b1 w2 b2 w3 b3]]
          (println "dparam: min" (m/emin (get grads param)) "max:" (m/emax (get grads param))))

        (recur (drop batch-size data)
               (make-node (m/sub (.value w1) (m/mul learning-rate (get grads w1))))
               (make-node (m/sub (.value b1) (m/mul learning-rate (get grads b1))))
               (make-node (m/sub (.value w2) (m/mul learning-rate (get grads w2))))
               (make-node (m/sub (.value b2) (m/mul learning-rate (get grads b2))))
               (make-node (m/sub (.value w3) (m/mul learning-rate (get grads w3))))
               (make-node (m/sub (.value b3) (m/mul learning-rate (get grads b3))))
               (conj progress {:loss (/ (.value loss) batch-size)
                               :accurate (-> (last progress) (get :accurate 0) (+ accurate))
                               :count (-> (last progress) (get :count 0) (+ batch-size))
                               :grads (->> (map vector ["w1" "b1" "w2" "b2" "w3" "b3"] [w1 b1 w2 b2 w3 b3])
                                           (reduce (fn [g [name param]] (-> (assoc-in g [name :min] (m/emin (get grads param)))
                                                                            (assoc-in [name :max] (m/emax (get grads param)))))
                                                   {}))})))
      {:progress progress
       :params [w1 b1 w2 b2 w3 b3]})))

;; ### Visualization

(clerk/vl {:data {:values (map (fn [p] (assoc p :accuracy (float (/ (:accurate p) (:count p))))) (:progress results))}
           :width 600 :height 400
           :encoding {:x {:field "count" :type "quantitative"}}
           :layer [{:mark "line" :encoding {:color {:value "#1f77b4"} :y {:field "loss" :type "quantitative"}}}
                   {:mark "line" :encoding {:color {:value "#ff7f0e"} :y {:field "accuracy" :type "quantitative"}}}]
           :resolve {:scale {:y "independent"}}})

;; Gradient stats:

(clerk/plotly {:data (let [x (map #(:count %) (:progress results))]
                       (for [n ["w1" "b1" "w2" "b2" "w3" "b3"] agg [:min :max]]
                         {:name (str (name agg) "(d" n ")")
                          :x x
                          :y (map (fn [p] (get-in p [:grads n agg])) results)}))})
