(ns lambda-autodiff.examples.char-rnn
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as mr]
            [lambda-autodiff.core :refer :all]
            [nextjournal.clerk :as clerk]))

(def data (slurp "input.txt"))

(def vocab (seq (set data)))

(println "data size:" (count data) "vocab size:" (count vocab))

(def ix-to-char (into {} (map-indexed (fn [i c] [i c]) vocab)))

(def char-to-ix (into {} (map-indexed (fn [i c] [c i]) vocab)))



;; TODO: add to utils
(defn one-hot
  [n i]
  (m/mset (m/zero-array [n 1]) i 0 1.0))

(defn forward
  [inputs targets Wxh Whh Why bh by hprev]
  (loop [t 0
         hidden hprev
         loss (make-node 0)]
    (if (>= t (count inputs))
      [loss hidden]
      (let [;; one-hot encode the character
            x (make-node (one-hot (count vocab) (nth inputs t)))
            ytarget (make-node (one-hot (count vocab) (nth targets t)))
            h (tanh (add (inner Wxh x) (add (inner Whh hidden) bh)))
            y (add (inner Why h) by)
            p (div (exp y) (sum (exp y)))
            l (neg (log (sum (mul p ytarget))))]
        (recur (+ t 1) h (add loss l))))))

(defn clip
  "Clip the values within an array to be in a certain range"
  [m lo hi]
  (m/emap (fn [e] (cond (< e lo) lo
                        (> e hi) hi
                        :else e))
          m))

(defn choice
  "Given vector of probability distribution, sample from distribution and return index"
  [p]
  (let [cdf (reduce (fn [cdf p] (conj cdf (+ (last cdf) p))) [0] p) ;; Convert probabilities into CDF
        ranges (partition 2 1 cdf)] ;; Convert into pairs of [lower, upper] prbability ranges
    (java.util.Collections/binarySearch ranges
                                        (rand)
                                        (fn [r e] (cond (< e (first r)) 1
                                                        (> e (second r)) -1
                                                        :else 0)))))

(defn sample
  [seed-ix n Wxh Whh Why bh by hprev]
  (loop [ix seed-ix
         t 0
         hidden hprev
         ixes []]
    (if (>= t n)
      ixes
      (let [;; one-hot encode the character
            x (make-node (one-hot (count vocab) ix))
            h (tanh (add (inner Wxh x) (add (inner Whh hidden) bh)))
            y (add (inner Why h) by)
            p (div (exp y) (sum (exp y)))
            ix (choice (m/as-vector (.value p)))]
        (recur ix (inc t) h (conj ixes ix))))))

(def seq-length 25)
(def hidden-size 100)
(def learning-rate 0.005)

(loop [n 0
       p 0
       Wxh (make-node (m/mul (mr/sample-normal [hidden-size (count vocab)]) 0.01) "Wxh")
       Whh (make-node (m/mul (mr/sample-normal [hidden-size hidden-size]) 0.01) "Whh")
       Why (make-node (m/mul (mr/sample-normal [(count vocab) hidden-size]) 0.01) "Why")
       bh (make-node (m/zero-array [hidden-size 1]) "bh")
       by (make-node (m/zero-array [(count vocab) 1]) "by")
       hprev (make-node (m/zero-array [hidden-size 1]))
       smooth-loss (* (- (clojure.math/log (/ 1 (count vocab)))) seq-length)
       losses []]
  (if (>= n 10000)
    (println "----\n" (->> (sample (char-to-ix (get data p)) 200 Wxh Whh Why bh by hprev)
                           (map ix-to-char)
                           (clojure.string/join)) " \n----")
    (let [inputs (map char-to-ix (subs data p (+ p seq-length)))
          targets (map char-to-ix (subs data (+ p 1) (+ p seq-length 1)))
          [loss hidden] (forward inputs targets Wxh Whh Why bh by hprev)
          smooth-loss (+ (* smooth-loss 0.999) (* (.value loss) 0.001))
          grads (differentiate loss)
          [dWxh dWhh dWhy dbh dby] (map #(clip (get grads %) -5 5) [Wxh Whh Why bh by])]

    ;;  (m/pm (.value Wxh))
    ;;  (println (m/shape (.value Wxh)) (m/esum (m/emap #(if (or (>= % 1) (<= % -1)) 1 0) (.value Wxh))))
    ;;   (m/pm (.value Whh))
    ;;   (println (m/shape (.value Whh)) (m/esum (m/emap #(if (or (>= % 5) (<= % -5)) 1 0) (.value Whh)))) 

    ;;   (if (= n 0)
    ;;     (do
    ;;       (println "Wxh range:" (m/emin (.value Wxh)) (m/emax (.value Wxh)))
    ;;       (println "Whh range:" (m/emin (.value Whh)) (m/emax (.value Whh)))
    ;;       (println "Why range:" (m/emin (.value Why)) (m/emax (.value Why)))
    ;;       ))

      (if (= (mod n 500) 0)
        (do
          (println "n:" n "p:" p "loss:" smooth-loss "actual loss:" (.value loss) "graph size:" (count-graph loss))))
        ;;   (m/pm dby)
          ;; Check size of gradients 
        ;;   (println "dWxh min:" (m/emin dWxh) "max:" (m/emax dWxh))
        ;;   (println "dWhh min:" (m/emin dWhh) "max:" (m/emax dWhh))
        ;;   (println "dWhy min:" (m/emin dWhy) "max:" (m/emax dWhy))
        ;;   (println "dbh min:" (m/emin dbh) "max:" (m/emax dbh))
        ;;   (println "dby min:" (m/emin dby) "max:" (m/emax dby))
      
      (if (=  (mod n 1000) 0)
        (do
          ;; Sample some text
          ;;(println (sample (nth inputs 0) 200 Wxh Whh Why bh by hidden))
          (println "----\n" (->> (sample (nth inputs 0) 200 Wxh Whh Why bh by hidden)
                                 (map ix-to-char)
                                 (clojure.string/join)) " \n----")))
          
      (recur (inc n)
             (+ p seq-length)
             ;;Wxh Whh Why bh by
             (make-node (m/sub (.value Wxh) (m/mul learning-rate dWxh)))
             (make-node (m/sub (.value Whh) (m/mul learning-rate dWhh)))
             (make-node (m/sub (.value Why) (m/mul learning-rate dWhy)))
             (make-node (m/sub (.value bh) (m/mul learning-rate dbh)))
             (make-node (m/sub (.value by) (m/mul learning-rate dby)))
             (make-node (.value hidden))
             smooth-loss
             (conj losses (.value loss))))))
