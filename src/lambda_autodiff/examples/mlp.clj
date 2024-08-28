(ns lambda-autodiff.examples.mlp
  (:require [lambda-autodiff.core :refer :all]
            [lambda-autodiff.array :as ma]
            [nextjournal.clerk :as clerk]))

;; # MLP demo
;;
;; Adapted from https://github.com/karpathy/micrograd/blob/master/demo.ipynb
;;
;; ### Dataset

^{:nextjournal.clerk/visibility {:code :fold :result :hide}}
(def xs (make-node [[4.83751667e-01,  8.56774435e-01],
                    [-2.03904990e-01,  9.44174884e-01],
                    [-2.33699745e-01,  1.01830362e+00],
                    [1.38609631e+00, -5.32661759e-01],
                    [-1.14293576e+00, -4.61366058e-03],
                    [1.39662548e+00, -5.13164918e-01],
                    [2.93616968e-01,  1.09090689e+00],
                    [8.03167941e-01,  3.57168481e-01],
                    [-8.96254130e-01,  2.16812377e-04],
                    [8.70549437e-02, -9.30215693e-02],
                    [1.59306700e-01,  9.48193629e-01],
                    [9.73960289e-02,  1.79549506e-01],
                    [8.65172067e-01,  6.12766141e-01],
                    [-5.19497285e-01,  9.90586302e-01],
                    [1.44224214e+00, -5.27019733e-01],
                    [1.35053497e+00, -4.21139680e-01],
                    [-9.99352179e-01,  3.80436602e-01],
                    [1.86202631e+00,  4.68558925e-01],
                    [2.90887958e-01,  9.10398040e-01],
                    [6.53449090e-02, -5.79306007e-02],
                    [1.05734711e+00, -4.84291450e-02],
                    [5.51736115e-01, -4.49842754e-01],
                    [1.93063482e+00,  1.15405760e-01],
                    [2.73172889e-01, -2.28567765e-01],
                    [-9.64813064e-01,  1.45143558e-01],
                    [7.62070181e-01, -5.65163344e-01],
                    [-4.77310823e-01,  8.84639542e-01],
                    [9.10355234e-01, -4.68689304e-01],
                    [1.93875827e+00,  7.26697243e-02],
                    [1.73208804e+00, -1.93262700e-01],
                    [-3.26353887e-01,  6.90087596e-01],
                    [-2.21099110e-01,  1.15661950e+00],
                    [-8.95137402e-02,  9.11384932e-01],
                    [1.88273279e+00, -2.44446686e-01],
                    [1.86921112e+00, -2.73691230e-02],
                    [2.12077142e+00,  6.07686806e-01],
                    [1.63955929e+00, -1.47158741e-01],
                    [-6.77189711e-01,  6.26830690e-01],
                    [9.15137390e-01,  1.89654238e-01],
                    [-1.04849021e+00,  3.27590036e-01],
                    [-7.30883611e-01,  6.72981206e-01],
                    [1.69465941e+00, -1.68391103e-01],
                    [-9.44831010e-01,  4.95955548e-01],
                    [6.27544747e-01, -3.61080305e-01],
                    [3.02541995e-01, -4.52431066e-01],
                    [7.33487997e-01,  8.27952082e-01],
                    [-1.69478086e-01,  9.85844482e-01],
                    [8.05450265e-01, -4.71267205e-01],
                    [1.59746856e-01,  8.98055865e-01],
                    [8.39178428e-01, -5.34849294e-01],
                    [-8.88585847e-01,  2.86240689e-01],
                    [8.61828457e-01, -4.17327964e-01],
                    [8.06784387e-01,  7.01194342e-01],
                    [1.05094301e-01,  4.84206761e-01],
                    [1.68552875e-01, -9.23421067e-02],
                    [1.73327595e+00,  8.41985374e-02],
                    [1.12587112e+00, -5.38346966e-01],
                    [1.17909214e+00, -3.94321707e-01],
                    [8.61666502e-01,  7.75023966e-01],
                    [2.22884625e-01,  7.02723831e-02],
                    [1.94314808e+00,  2.74459012e-01],
                    [1.76361141e+00, -1.37607259e-01],
                    [-7.43690606e-01,  8.77144424e-01],
                    [5.62149940e-01,  8.10590176e-01],
                    [-9.02526312e-01,  1.04572866e-01],
                    [1.10617301e+00, -5.33226969e-01],
                    [2.20608947e-02,  2.24650145e-01],
                    [9.13122669e-01,  3.79985189e-01],
                    [3.28935321e-02,  1.14092842e+00],
                    [4.72539768e-02,  1.99645123e-01],
                    [8.88655401e-01,  4.76948075e-01],
                    [8.49385680e-01,  7.75531766e-01],
                    [2.43493811e-02, -1.39410197e-02],
                    [-1.04107088e+00,  2.92152081e-01],
                    [4.53886740e-01,  7.44897671e-01],
                    [5.45123013e-01, -4.45831985e-01],
                    [8.13262733e-01, -4.57414744e-01],
                    [2.51799670e-02,  7.43681431e-01],
                    [8.65667450e-01,  7.35433176e-02],
                    [4.45318729e-01, -2.82705554e-01],
                    [4.01135654e-02,  3.40379610e-01],
                    [3.35043140e-01,  9.34796108e-01],
                    [-7.79297659e-01,  7.90377827e-01],
                    [3.19439161e-01,  9.77267523e-01],
                    [1.72678529e+00, -2.55246765e-01],
                    [1.34879789e+00, -6.41216512e-01],
                    [6.64190630e-01,  8.11859195e-01],
                    [3.22395872e-02,  5.52261736e-02],
                    [1.89006081e+00,  6.89624840e-02],
                    [-8.45386950e-01,  5.74214921e-01],
                    [-5.42414365e-01,  9.50289611e-01],
                    [1.12187631e+00,  2.38887218e-01],
                    [1.97121222e+00,  4.31335860e-01],
                    [-6.68807592e-01,  6.06595826e-01],
                    [8.51996081e-01,  2.79775989e-01],
                    [1.29673316e+00, -4.80562723e-01],
                    [-6.99815191e-01,  7.21191325e-01],
                    [1.79858511e-01, -4.09686092e-01],
                    [1.01249371e+00,  6.37601450e-01],
                    [1.95035521e+00,  3.26622080e-01]]))

^{:nextjournal.clerk/visibility {:code :fold :result :hide}}
(def ys (make-node [-1, -1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1,
                    1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1,
                    1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,
                    1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,
                    -1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,  1,
                    1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1,  1]))

(clerk/vl {:data {:values (map (fn [x y] {:x1 (first x) :x2 (second x) :y y}) (.value xs) (.value ys))}
           :width 600 :height 400
           :encoding {:x {:field "x1" :type "quantitative"}
                      :y {:field "x2" :type "quantitative"}
                      :color {:field "y" :type "nominal"}}
           :mark "point"})

;; ### Model
;;
;; Define weights and biases:

^{:nextjournal.clerk/visibility {:result :hide}}
(def w1 (make-node (ma/sample-normal [16 2])))

^{:nextjournal.clerk/visibility {:result :hide}}
(def b1 (make-node (ma/sample-normal [16])))

^{:nextjournal.clerk/visibility {:result :hide}}
(def w2 (make-node (ma/sample-normal [16 16])))

^{:nextjournal.clerk/visibility {:result :hide}}
(def b2 (make-node (ma/sample-normal [16])))

^{:nextjournal.clerk/visibility {:result :hide}}
(def w3 (make-node (ma/sample-normal [1 16])))

^{:nextjournal.clerk/visibility {:result :hide}}
(def b3 (make-node (ma/sample-normal [1])))

;; Total number of parameters:

(->> (map #(.value %) [w1 b1 w2 b2 w3 b3])
     (map ma/count)
     (reduce +))

;; ### Optimization

(def iters 100)

(def results
  (loop [i 0
         progress []
         w1 w1 b1 b1 w2 w2 b2 b2 w3 w3 b3 b3]
    (if (< i iters)
      (let [learning-rate (- 1.0 (* 0.9 (/ i 100)))
            ;; Forward pass
            a1 (tanh (add (transpose (mmul w1 (transpose xs))) b1))
            a2 (tanh (add (transpose (mmul w2 (transpose a1))) b2))
            out (tanh (add (transpose (mmul w3 (transpose a2))) b3))
            ;; Loss function
            losses (relu (sub (make-node 1) (mul ys (transpose out))))
            dataloss (div (sum losses) (make-node (ma/count (.value losses))))
            regloss (->> [w1 b1 w2 b2 w3 b3]
                         (map #(sum (mul % %)))
                         (reduce add)
                         (mul (make-node 1e-4)))
            loss (add dataloss regloss)
            ;; Backward pass
            grads (differentiate loss)
            ;; Accuracy
            accuracy (map #(if (= (> %1 0) (> %2 0)) 1 0) (.value ys) (ma/flatten (.value out)))]
        (recur (inc i)
               (conj progress
                     {:loss (.value loss)
                      :learning-rate learning-rate
                      :accuracy (float (/ (reduce + accuracy) (count accuracy)))})
               (make-node (ma/sub (.value w1) (ma/mul learning-rate (get grads w1))))
               (make-node (ma/sub (.value b1) (ma/mul learning-rate (get grads b1))))
               (make-node (ma/sub (.value w2) (ma/mul learning-rate (get grads w2))))
               (make-node (ma/sub (.value b2) (ma/mul learning-rate (get grads b2))))
               (make-node (ma/sub (.value w3) (ma/mul learning-rate (get grads w3))))
               (make-node (ma/sub (.value b3) (ma/mul learning-rate (get grads b3))))))
      {:progress progress
       :params [w1 b1 w2 b2 w3 b3]})))

;; ### Visualization

(clerk/vl {:data {:values (map-indexed (fn [i p] (conj p {:iter i})) (:progress results))}
           :width 600 :height 400
           :encoding {:x {:field "iter" :type "quantitative"}}
           :layer [{:mark "line" :encoding {:color {:value "#1f77b4"} :y {:field "loss" :type "quantitative"}}}
                   {:mark "line" :encoding {:color {:value "#ff7f0e"} :y {:field "accuracy" :type "quantitative"}}}]
           :resolve {:scale {:y "independent"}}})

;; Plot decision boundary:

(let [step 0.05
      [xmin xmax] ((juxt #(apply min %) #(apply max %)) (map first (.value xs)))
      [ymin ymax] ((juxt #(apply min %) #(apply max %)) (map second (.value xs)))
      xrange (range (- xmin (* 2 step)) (+ xmax (* 2 step)) step)
      yrange (range (- ymin (* 2 step)) (+ ymax (* 2 step)) step)
      [w1 b1 w2 b2 w3 b3] (:params results)
      mesh-xs (make-node (for [x2 yrange x1 xrange] [x1 x2]))
      a1 (tanh (add (transpose (mmul w1 (transpose mesh-xs))) b1))
      a2 (tanh (add (transpose (mmul w2 (transpose a1))) b2))
      out (tanh (add (transpose (mmul w3 (transpose a2))) b3))
      pos (->> (map vector (.value xs) (.value ys))
               (filter (fn [[_ y]] (> y 0)))
               (map first))
      neg (->> (map vector (.value xs) (.value ys))
               (filter (fn [[_ y]] (<= y 0)))
               (map first))]
  (clerk/plotly
   {:data [{:z (partition (count xrange) (ma/flatten (.value out)))
            :x xrange
            :y yrange
            :type "heatmap"}
           {:x (map first pos)
            :y (map second pos)
            :mode "markers"
            :type "scatter"
            :name "y = 1"}
           {:x (map first neg)
            :y (map second neg)
            :mode "markers"
            :type "scatter"
            :name "y = -1"}]}))
