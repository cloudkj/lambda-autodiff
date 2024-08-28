(ns lambda-autodiff.ext
  (:require [lambda-autodiff.core :refer :all]
            [lambda-autodiff.array :as ma]))

(defn- im2col-helper
  [x filter-height filter-width stride]
  (let [[batch-size channels height width] (ma/shape x) ;; Assume input shape is (batch x channels x height x width)
        out-height (+ (quot (- height filter-height) stride) 1)
        out-width (+ (quot (- width filter-width) stride) 1)
        out (ma/zeros [batch-size (* out-height out-width) (* channels filter-height filter-width)])]
    (loop [b 0
           i 0
           j 0
           out out]
      (cond (>= b batch-size) out
            (>= i out-height) (recur (inc b) 0 0 out)
            (>= j out-width) (recur b (inc i) 0 out)
            :else (let [window (ma/select x (range b (+ b 1)) :all (range (* i stride) (+ (* i stride) filter-height)) (range (* j stride) (+ (* j stride) filter-width)))
                        col (ma/reshape window [(* channels filter-height filter-width)])]
                    ;; TODO: next - define replacement for `set-selection
                    (recur b i (inc j) (ma/set out b (+ (* i out-width) j) :all col)))))))

(defn im2col
  [a filter-height filter-width stride]
  (let [[batch-size channels height width] (ma/shape (.value a))]
    (make-node (im2col-helper (.value a) filter-height filter-width stride)
               "im2col"
               [[a (fn [upstream]
                     (let [out-height (+ (quot (- height filter-height) stride) 1)
                           out-width (+ (quot (- width filter-width) stride) 1)]
                       (loop [i 0
                              downstream (ma/zeros [batch-size channels height width])]
                         (if (>= i (* out-height out-width))
                           downstream
                           (let [r (* (quot i out-height) stride)
                                 c (* (mod i out-width) stride)
                                 ;; Indexes corresponding to the patch over which to sum gradients
                                 h-indexes (range r (+ r filter-height))
                                 w-indexes (range c (+ c filter-width))
                                 ;; Submatrix from upstream derivatives to be added to downstream output
                                 window (ma/reshape (ma/select upstream :all i :all) [batch-size channels filter-height filter-width])
                                 sum (ma/add (ma/select downstream :all :all h-indexes w-indexes) window)]
                             (recur (inc i)
                                    (ma/set downstream :all :all h-indexes w-indexes sum)))))))]])))

(defn maxpool2d
  [a filter-height filter-width stride]
  (let [[batch-size channels height width] (ma/shape (.value a))
        out-height (+ (quot (- height filter-height) stride) 1)
        out-width (+ (quot (- width filter-width) stride) 1)
        ;; Artifically combine channel and batch dimensions to allow for taking max along last dimension
        unfolded (-> (ma/reshape (.value a) [(* batch-size channels) 1 height width])
                     (im2col-helper filter-height filter-width stride))] ;; Shape: (channels x #patches x patchsize)
    (make-node (-> (ma/dimension-max unfolded 2) (ma/reshape [batch-size channels out-height out-width]))
               "maxpool2d"
               [[a (fn [upstream]
                     ;; TODO: to improve performance, get max and argmax values in single call to `dimension-max`
                     (let [argmax (-> (ma/dimension-max unfolded 2 true)
                                      (ma/reshape [batch-size channels (* out-height out-width)])) ;; Shape: (batch size x channels x #patches)
                           upstream (ma/reshape upstream (ma/shape argmax))]
                       (loop [b 0
                              c 0
                              downstream (ma/zeros [batch-size channels height width])]
                         (cond (>= b batch-size) downstream
                               (>= c channels) (recur (inc b) 0 downstream)
                               :else
                               ;; Compute indexes corresponding to downstream array from argmax
                               (let [indexes (->> (ma/slice (ma/slice argmax b) c)
                                                  (map-indexed (fn [pi mi]
                                                                 [b c ;; Set first dimensions in index to current batch, channel
                                                                  (+ (quot mi filter-width) (* (quot pi out-width) stride))
                                                                  (+ (mod mi filter-width) (* (mod pi out-width) stride))])))
                                     ;; Sum upstream derivatives and route through indexes
                                     sums (->> (map vector indexes (ma/slice (ma/slice upstream b) c))
                                               (reduce (fn [s [index value]] (assoc s index (+ (get s index 0) value))) {}))]
                                 (recur b (inc c) (ma/set-indexes downstream (keys sums) (vals sums))))))))]])))
