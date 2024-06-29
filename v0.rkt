#lang racket

;; values represented as functions
;; - forward pass by function invocation
;;
;; a = 3
;; x = 2 * a
;; z = x + y
;; dz/dx = dx/dx
;; dy

(define (output val)
  (car val))

(define (value x)
  (lambda ()
    (list x         ;; output
          (list 0)  ;; local derivatives
          (list)))) ;; children

(define (add v1 v2)
  (lambda ()
    (list (+ (output (v1)) (output (v2)))   ;; output
          (list 1 1)      ;; local derivatives
          (list v1 v2)))) ;; children
     
(define (mul v1 v2)
  (lambda ()
    (list (* (output (v1)) (output (v2)))
          (list (output (v2)) (output (v1)))
          (list v1 v2))))

;; test
;; (define a (value 4))
;; (define b (value 3))
;; (define c (add a b))
;; (define d (mul a c))

;; manual autodiff
(define a 4)
(define b 3)
(define c (+ a b))
(define d (* a c))

d
;; dd/da = dd/da + dd/dc * dc/da
