(ns lambda-autodiff.examples.char-rnn
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [clojure.java.io :as io]
            [clojure.math :as math]
            [lambda-autodiff.core :refer :all]
            [lambda-autodiff.util :as util]
            [taoensso.nippy :as nippy]
            [nextjournal.clerk :as clerk]))

;; # Character-level RNN demo
;;
;; Adapted from https://gist.github.com/karpathy/d4dee566867f8291f086
;;
;; ### Dataset
;;
;; Sources:
;; * Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
;; * Linux source code: https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/linux.txt

(def data (slurp "https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/linux.txt"))
(def vocab (seq (set data)))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Data size: " (count data) ", vocab size: " (count vocab)))

;; Forward and inverted mapping

(def ix-to-char (into {} (map-indexed (fn [i c] [i c]) vocab)))
(def char-to-ix (into {} (map-indexed (fn [i c] [c i]) vocab)))

^{:nextjournal.clerk/visibility {:result :hide}}
(defn sample
  "Helper function for generating snippet of text by sampling from model using a seed character index"
  [Wxh Whh Why bh by hprev n seed]
  (loop [index seed
         t 0
         hidden hprev
         indexes []]
    (if (>= t n)
      indexes
      (let [x (make-node (m/reshape (util/one-hot (count vocab) index) [(count vocab) 1]))
            h (tanh (add (mmul Wxh x) (add (mmul Whh hidden) bh)))
            y (add (mmul Why h) by)
            p (div (exp y) (sum (exp y)))
            index (util/choose (m/as-vector (.value p)))]
        (recur index (inc t) h (conj indexes index))))))

;; ### Hyperparameters

(m/set-current-implementation :vectorz)

^{:nextjournal.clerk/visibility {:result :hide}}
(def seq-length 32)
^{:nextjournal.clerk/visibility {:result :hide}}
(def hidden-size 128)
^{:nextjournal.clerk/visibility {:result :hide}}
(def learning-rate 0.0001)
^{:nextjournal.clerk/visibility {:result :hide}}
(def init-loss (* (- (math/log (/ 1 (count vocab)))) seq-length))

;; ### Model parameters

(def load-model-params? true)
(def save-model-params? false)

;; Load model weights:

(def model-name "char_rnn_linux")
(def model-weights
  (let [suffix "offset367680_iters5000_20240815_230852"]
    (loop [names ["Wxh" "Whh" "Why" "bh" "by" "hprev"]
           weights {}]
      (if (or (not load-model-params?) (empty? names))
        weights
        (let [name (first names)
              filename (str "data/" model-name "_" name "_" suffix ".dat")
              deserialized (with-open [in (io/input-stream (io/file filename))]
                             (->> (.readAllBytes in)
                                  (nippy/thaw)
                                  (nippy/read-quarantined-serializable-object-unsafe!)))]
          (println "Loaded:" filename "shape:" (m/shape deserialized))
          (recur (rest names) (assoc weights name deserialized)))))))

^{:nextjournal.clerk/visibility {:result :hide}}
(def Wxh (make-node (get model-weights "Wxh" (m/mul (mr/sample-normal [hidden-size (count vocab)]) 0.01))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def Whh (make-node (get model-weights "Whh" (m/mul (mr/sample-normal [hidden-size hidden-size]) 0.01))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def Why (make-node (get model-weights "Why" (m/mul (mr/sample-normal [(count vocab) hidden-size]) 0.01))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def bh (make-node (get model-weights "bh" (m/zero-array [hidden-size 1]))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def by (make-node (get model-weights "by" (m/zero-array [(count vocab) 1]))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def hprev (make-node (get model-weights "hprev" (m/zero-array [hidden-size 1]))))

;; Parameter distribution:

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/row (clerk/plotly {:data [{:x (seq (m/to-vector (.value Wxh)))
                                  :name "Wxh"
                                  :type "histogram"}
                                 {:x (seq (m/to-vector (.value Whh)))
                                  :name "Whh"
                                  :type "histogram"}
                                 {:x (seq (m/to-vector (.value Why)))
                                  :name "Why"
                                  :type "histogram"}]})
           (clerk/plotly {:data [{:x (seq (m/to-vector (.value bh)))
                                  :name "bh"
                                  :type "histogram"}
                                 {:x (seq (m/to-vector (.value by)))
                                  :name "by"
                                  :type "histogram"}
                                 {:x (seq (m/to-vector (.value hprev)))
                                  :name "hprev"
                                  :type "histogram"}]}))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Total number of parameters: "
               (->> (map #(.value %) [Wxh Whh Why bh by])
                    (map m/ecount)
                    (reduce +))))

;; ### Optimization

^{:nextjournal.clerk/visibility {:result :hide}}
(defn forward
  [inputs targets Wxh Whh Why bh by hprev]
  (loop [t 0
         hidden hprev
         loss (make-node 0)]
    (if (>= t (count inputs))
      [loss hidden]
      (let [x (make-node (m/reshape (util/one-hot (count vocab) (nth inputs t)) [(count vocab) 1]))
            ytarget (make-node (m/reshape (util/one-hot (count vocab) (nth targets t)) [(count vocab) 1]))
            h (tanh (add (mmul Wxh x) (add (mmul Whh hidden) bh)))
            y (add (mmul Why h) by)
            p (div (exp y) (sum (exp y)))
            l (neg (log (sum (mul p ytarget))))]
        (recur (+ t 1) h (add loss l))))))

(def iters 10)
(def offset 207744)

(def results
  (loop [n 0
         p offset
         start (java.lang.System/currentTimeMillis)
         ;; Parameters
         Wxh Wxh Whh Whh Why Why bh bh by by
         ;; Initial hidden state
         hprev hprev
         progress []]
    (if (< n iters)
      (let [;; Forward pass
            inputs (map char-to-ix (subs data p (+ p seq-length)))
            targets (map char-to-ix (subs data (+ p 1) (+ p seq-length 1)))
            [loss hidden] (forward inputs targets Wxh Whh Why bh by hprev)
            ;; Loss smoothing
            smooth-loss (-> (last progress) (get :smooth-loss init-loss))
            smooth-loss (+ (* smooth-loss 0.999) (* (.value loss) 0.001))
            ;; Backward pass
            grads (differentiate loss)
            [dWxh dWhh dWhy dbh dby] (map #(util/clamp (get grads %) -5 5) [Wxh Whh Why bh by])
            ;; Timing
            end (java.lang.System/currentTimeMillis)
            elapsed (- end start)]
        (if (= (mod n 100) 0)
          (println "n:" n "p:" p "loss:" smooth-loss "actual loss:" (.value loss) "graph size:" (util/count-graph loss) "elapsed:" elapsed))
        ;;;;;;
        (recur (inc n)
               (+ p seq-length)
               end
               (make-node (m/sub (.value Wxh) (m/mul learning-rate dWxh)))
               (make-node (m/sub (.value Whh) (m/mul learning-rate dWhh)))
               (make-node (m/sub (.value Why) (m/mul learning-rate dWhy)))
               (make-node (m/sub (.value bh) (m/mul learning-rate dbh)))
               (make-node (m/sub (.value by) (m/mul learning-rate dby)))
               (make-node (.value hidden))
               (conj progress {:iter n
                               :offset p
                               :elapsed-ms elapsed
                               :smooth-loss smooth-loss
                               :loss (.value loss)
                               :sample (if (= (mod n 200) 0)
                                         (->> (nth inputs 0)
                                              (sample Wxh Whh Why bh by hidden 128)
                                              (map ix-to-char)
                                              (clojure.string/join)))})))
      {:params [Wxh Whh Why bh by hprev]
       :progress (conj (pop progress)
                       (assoc (last progress) :sample
                              ;; Generate larger sample for final iteration
                              (->> (get char-to-ix (get data p))
                                   (sample Wxh Whh Why bh by hprev 256)
                                   (map ix-to-char)
                                   (clojure.string/join))))})))

(clerk/vl {:data {:values (:progress results)}
           :width 600 :height 400
           :encoding {:x {:field "iter" :type "quantitative"}}
           :layer [{:mark "line" :encoding {:color {:value "#1f77b4"} :y {:field "smooth-loss" :type "quantitative"}}}
                   {:mark "line" :encoding {:color {:value "#418637"} :y {:field "elapsed-ms" :type "quantitative"}}}]
           :resolve {:scale {:y "independent"}}})

;; Sampled RNN output:

(clerk/table {::clerk/page-size nil}
             {:head ["iter" "sampled text"]
              :rows (->> (butlast (:progress results))
                        ;; (map-indexed (fn [i p] (conj p {:iter i})))
                         (filter (fn [p] (not (nil? (:sample p)))))
                         (map (fn [p] [(:iter p) (:sample p)])))})

;; Final output:

(clerk/md (str "```\n" (:sample (last (:progress results))) "\n```"))

;; Save model weights:

(when save-model-params?
  (let [offset (->> (:progress results) (last) (:offset))
        iters (count (:progress results))
        timestamp (.format (java.text.SimpleDateFormat. "yyyyMMdd_HHMMss") (new java.util.Date))]
    (doseq [[name params] (map vector ["Wxh" "Whh" "Why" "bh" "by" "hprev"] (:params results))]
      (let [filename (str "data/" model-name "_" name "_offset" offset "_iters" iters "_" timestamp ".dat")]
        (println "Saving:" filename "shape:" (m/shape (.value params)))
        (with-open [out (io/output-stream filename)]
          (.write out (nippy/freeze (.value params))))))))
