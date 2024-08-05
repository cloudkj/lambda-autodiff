(ns lambda-autodiff.examples.char-rnn
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [clojure.math :as math]
            [lambda-autodiff.core :refer :all]
            [lambda-autodiff.util :as util]
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

(def data (slurp "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"))
(def vocab (seq (set data)))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Data size: " (count data) ", vocab size: " (count vocab)))

;; Forward and inverted mapping

(def ix-to-char (into {} (map-indexed (fn [i c] [i c]) vocab)))
(def char-to-ix (into {} (map-indexed (fn [i c] [c i]) vocab)))

^{:nextjournal.clerk/visibility {:result :hide}}
(defn forward
  [inputs targets Wxh Whh Why bh by hprev]
  (loop [t 0
         hidden hprev
         loss (make-node 0)]
    (if (>= t (count inputs))
      [loss hidden]
      (let [x (make-node (util/one-hot (count vocab) (nth inputs t)))
            ytarget (make-node (util/one-hot (count vocab) (nth targets t)))
            h (tanh (add (mmul Wxh x) (add (mmul Whh hidden) bh)))
            y (add (mmul Why h) by)
            p (div (exp y) (sum (exp y)))
            l (neg (log (sum (mul p ytarget))))]
        (recur (+ t 1) h (add loss l))))))

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
      (let [x (make-node (util/one-hot (count vocab) index))
            h (tanh (add (mmul Wxh x) (add (mmul Whh hidden) bh)))
            y (add (mmul Why h) by)
            p (div (exp y) (sum (exp y)))
            index (util/choose (m/as-vector (.value p)))]
        (recur index (inc t) h (conj indexes index))))))

;; ### Hyperparameters

^{:nextjournal.clerk/visibility {:result :hide}}
(def seq-length 25)
^{:nextjournal.clerk/visibility {:result :hide}}
(def hidden-size 100)
^{:nextjournal.clerk/visibility {:result :hide}}
(def learning-rate 0.005)
^{:nextjournal.clerk/visibility {:result :hide}}
(def init-loss (* (- (math/log (/ 1 (count vocab)))) seq-length))

;; ### Model parameters

^{:nextjournal.clerk/visibility {:result :hide}}
(def Wxh (make-node (m/mul (mr/sample-normal [hidden-size (count vocab)]) 0.01)))
^{:nextjournal.clerk/visibility {:result :hide}}
(def Whh (make-node (m/mul (mr/sample-normal [hidden-size hidden-size]) 0.01)))
^{:nextjournal.clerk/visibility {:result :hide}}
(def Why (make-node (m/mul (mr/sample-normal [(count vocab) hidden-size]) 0.01)))
^{:nextjournal.clerk/visibility {:result :hide}}
(def bh (make-node (m/zero-array [hidden-size 1])))
^{:nextjournal.clerk/visibility {:result :hide}}
(def by (make-node (m/zero-array [(count vocab) 1])))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/md (str "Total number of parameters: "
               (->> (map #(.value %) [Wxh Whh Why bh by])
                    (map m/ecount)
                    (reduce +))))

;; ### Optimization

(def results
  (loop [n 0
         p 0
         ;; Parameters
         Wxh Wxh Whh Whh Why Why bh bh by by
         ;; Initial hidden state
         hprev (make-node (m/zero-array [hidden-size 1]))
         progress []]
    (if (< n 1000)
      (let [;; Forward pass
            inputs (map char-to-ix (subs data p (+ p seq-length)))
            targets (map char-to-ix (subs data (+ p 1) (+ p seq-length 1)))
            [loss hidden] (forward inputs targets Wxh Whh Why bh by hprev)
            ;; Loss smoothing
            smooth-loss (-> (last progress) (get :smooth-loss init-loss))
            smooth-loss (+ (* smooth-loss 0.999) (* (.value loss) 0.001))
            ;; Backward pass
            grads (differentiate loss)
            [dWxh dWhh dWhy dbh dby] (map #(util/clamp (get grads %) -5 5) [Wxh Whh Why bh by])]
        ;; TODO: remove
        (if (= (mod n 100) 0)
          (println "n:" n "p:" p "loss:" smooth-loss "actual loss:" (.value loss) "graph size:" (util/count-graph loss)))
        ;;;;;;
        (recur (inc n)
               (+ p seq-length)
               (make-node (m/sub (.value Wxh) (m/mul learning-rate dWxh)))
               (make-node (m/sub (.value Whh) (m/mul learning-rate dWhh)))
               (make-node (m/sub (.value Why) (m/mul learning-rate dWhy)))
               (make-node (m/sub (.value bh) (m/mul learning-rate dbh)))
               (make-node (m/sub (.value by) (m/mul learning-rate dby)))
               (make-node (.value hidden))
               (conj progress {:smooth-loss smooth-loss
                               :loss (.value loss)
                               :sample (if (= (mod n 200) 0)
                                         (->> (nth inputs 0)
                                              (sample Wxh Whh Why bh by hidden 128)
                                              (map ix-to-char)
                                              (clojure.string/join)))})))
      ;; Generate larger sample for final iteration
      (conj (pop progress)
            (assoc (last progress) :sample
                   (->> (get char-to-ix (get data p))
                        (sample Wxh Whh Why bh by hprev 256)
                        (map ix-to-char)
                        (clojure.string/join)))))))

(clerk/vl {:data {:values (map-indexed (fn [i p] (conj p {:iter i})) results)}
           :width 600 :height 400
           :encoding {:x {:field "iter" :type "quantitative"}}
           :layer [{:mark "line"
                    :encoding {:color {:value "#1f77b4"}
                               :y {:field "smooth-loss" :type "quantitative"}}}]
           :resolve {:scale {:y "independent"}}})

;; ;; Sampled RNN output

(clerk/table {:head ["iter" "sampled text"]
              :rows (->> (butlast results)
                         (map-indexed (fn [i p] (conj p {:iter i})))
                         (filter (fn [p] (not (nil? (:sample p)))))
                         (map (fn [p] [(:iter p) (:sample p)])))})

;; ;; Final output:

(clerk/md (str "```\n" (:sample (last results) "\n```")))
