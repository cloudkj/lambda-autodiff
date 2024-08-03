(defproject lambda-autodiff "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [net.mikera/core.matrix "0.63.0"]
                 [net.mikera/vectorz-clj "0.48.0"]
                 [io.github.nextjournal/clerk "0.16.1016"]]
  :repl-options {:init-ns lambda-autodiff.core})
