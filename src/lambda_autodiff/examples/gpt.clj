(ns lambda-autodiff.examples.gpt
  (:require [lambda-autodiff.core :refer :all]
            [lambda-autodiff.array :as ma]
            [lambda-autodiff.util :as util]
            [clojure.core.matrix :as matrix]
            [clojure.math :as math]
            [clojure.set :as set]
            [nextjournal.clerk :as clerk]))

;; # GPT demo
;;
;; Implementation of a character-level GPT model, closely following https://karpathy.github.io/2026/02/12/microgpt/
;;
;; ### Dataset
;;
;; For the demo, we learn from a list of Pokemon names to generate names of new, non-existent Pokemon.

^{:nextjournal.clerk/visibility {:result :hide}}
(def docs (shuffle ["abomasnow" "abra" "absol" "accelgor" "aegislash" "aerodactyl" "aggron" "aipom" "alakazam" "alomomola" "altaria" "amaura" "ambipom" "amoonguss" "ampharos" "anorith" "arbok" "arcanine" "arceus" "archen" "archeops" "ariados" "armaldo" "aromatisse" "aron" "articuno" "audino" "aurorus" "avalugg" "axew" "azelf" "azumarill" "azurill" "bagon" "baltoy" "banette" "barbaracle" "barboach" "basculin" "bastiodon" "bayleef" "beartic" "beautifly" "beedrill" "beheeyem" "beldum" "bellossom" "bellsprout" "bergmite" "bibarel" "bidoof" "binacle" "bisharp" "blastoise" "blaziken" "blissey" "blitzle" "boldore" "bonsly" "bouffalant" "braixen" "braviary" "breloom" "bronzong" "bronzor" "budew" "buizel" "bulbasaur" "buneary" "bunnelby" "burmy" "butterfree" "cacnea" "cacturne" "camerupt" "carbink" "carnivine" "carracosta" "carvanha" "cascoon" "castform" "caterpie" "celebi" "chandelure" "chansey" "charizard" "charmander" "charmeleon" "chatot" "cherrim" "cherubi" "chesnaught" "chespin" "chikorita" "chimchar" "chimecho" "chinchou" "chingling" "cinccino" "clamperl" "clauncher" "clawitzer" "claydol" "clefable" "clefairy" "cleffa" "cloyster" "cobalion" "cofagrigus" "combee" "combusken" "conkeldurr" "corphish" "corsola" "cottonee" "cradily" "cranidos" "crawdaunt" "cresselia" "croagunk" "crobat" "croconaw" "crustle" "cryogonal" "cubchoo" "cubone" "cyndaquil" "darkrai" "darmanitan" "darumaka" "dedenne" "deerling" "deino" "delcatty" "delibird" "delphox" "deoxys" "dewgong" "dewott" "dialga" "diancie" "diggersby" "diglett" "ditto" "dodrio" "doduo" "donphan" "doublade" "dragalge" "dragonair" "dragonite" "drapion" "dratini" "drifblim" "drifloon" "drilbur" "drowzee" "druddigon" "ducklett" "dugtrio" "dunsparce" "duosion" "durant" "dusclops" "dusknoir" "duskull" "dustox" "dwebble" "eelektrik" "eelektross" "eevee" "ekans" "electabuzz" "electivire" "electrike" "electrode" "elekid" "elgyem" "emboar" "emolga" "empoleon" "entei" "escavalier" "espeon" "espurr" "excadrill" "exeggcute" "exeggutor" "exploud" "farfetch'd" "fearow" "feebas" "fennekin" "feraligatr" "ferroseed" "ferrothorn" "finneon" "flaaffy" "flabébé" "flareon" "fletchinder" "fletchling" "floatzel" "floette" "florges" "flygon" "foongus" "forretress" "fraxure" "frillish" "froakie" "frogadier" "froslass" "furfrou" "furret" "gabite" "gallade" "galvantula" "garbodor" "garchomp" "gardevoir" "gastly" "gastrodon" "genesect" "gengar" "geodude" "gible" "gigalith" "girafarig" "giratina" "glaceon" "glalie" "glameow" "gligar" "gliscor" "gloom" "gogoat" "golbat" "goldeen" "golduck" "golem" "golett" "golurk" "goodra" "goomy" "gorebyss" "gothita" "gothitelle" "gothorita" "gourgeist" "granbull" "graveler" "greninja" "grimer" "grotle" "groudon" "grovyle" "growlithe" "grumpig" "gulpin" "gurdurr" "gyarados" "happiny" "hariyama" "haunter" "hawlucha" "haxorus" "heatmor" "heatran" "heliolisk" "helioptile" "heracross" "herdier" "hippopotas" "hippowdon" "hitmonchan" "hitmonlee" "hitmontop" "honchkrow" "honedge" "ho-oh" "hoopa" "hoothoot" "hoppip" "horsea" "houndoom" "houndour" "huntail" "hydreigon" "hypno" "igglybuff" "illumise" "infernape" "inkay" "ivysaur" "jellicent" "jigglypuff" "jirachi" "jolteon" "joltik" "jumpluff" "jynx" "kabuto" "kabutops" "kadabra" "kakuna" "kangaskhan" "karrablast" "kecleon" "keldeo" "kingdra" "kingler" "kirlia" "klang" "klefki" "klink" "klinklang" "koffing" "krabby" "kricketot" "kricketune" "krokorok" "krookodile" "kyogre" "kyurem" "lairon" "lampent" "landorus" "lanturn" "lapras" "larvesta" "larvitar" "latias" "latios" "leafeon" "leavanny" "ledian" "ledyba" "lickilicky" "lickitung" "liepard" "lileep" "lilligant" "lillipup" "linoone" "litleo" "litwick" "lombre" "lopunny" "lotad" "loudred" "lucario" "ludicolo" "lugia" "lumineon" "lunatone" "luvdisc" "luxio" "luxray" "machamp" "machoke" "machop" "magby" "magcargo" "magikarp" "magmar" "magmortar" "magnemite" "magneton" "magnezone" "makuhita" "malamar" "mamoswine" "manaphy" "mandibuzz" "manectric" "mankey" "mantine" "mantyke" "maractus" "mareep" "marill" "marowak" "marshtomp" "masquerain" "mawile" "medicham" "meditite" "meganium" "meloetta" "meowstic" "meowth" "mesprit" "metagross" "metang" "metapod" "mew" "mewtwo" "mienfoo" "mienshao" "mightyena" "milotic" "miltank" "mime jr." "minccino" "minun" "misdreavus" "mismagius" "moltres" "monferno" "mothim" "mr. mime" "mudkip" "muk" "munchlax" "munna" "murkrow" "musharna" "natu" "nidoking" "nidoqueen" "nidoran" "nidoran♂" "nidorina" "nidorino" "nincada" "ninetales" "ninjask" "noctowl" "noibat" "noivern" "nosepass" "numel" "nuzleaf" "octillery" "oddish" "omanyte" "omastar" "onix" "oshawott" "pachirisu" "palkia" "palpitoad" "pancham" "pangoro" "panpour" "pansage" "pansear" "paras" "parasect" "patrat" "pawniard" "pelipper" "persian" "petilil" "phanpy" "phantump" "phione" "pichu" "pidgeot" "pidgeotto" "pidgey" "pidove" "pignite" "pikachu" "piloswine" "pineco" "pinsir" "piplup" "plusle" "politoed" "poliwag" "poliwhirl" "poliwrath" "ponyta" "poochyena" "porygon" "porygon2" "porygon-z" "primeape" "prinplup" "probopass" "psyduck" "pumpkaboo" "pupitar" "purrloin" "purugly" "pyroar" "quagsire" "quilava" "quilladin" "qwilfish" "raichu" "raikou" "ralts" "rampardos" "rapidash" "raticate" "rattata" "rayquaza" "regice" "regigigas" "regirock" "registeel" "relicanth" "remoraid" "reshiram" "reuniclus" "rhydon" "rhyhorn" "rhyperior" "riolu" "roggenrola" "roselia" "roserade" "rotom" "rufflet" "sableye" "salamence" "samurott" "sandile" "sandshrew" "sandslash" "sawk" "sawsbuck" "scatterbug" "sceptile" "scizor" "scolipede" "scrafty" "scraggy" "scyther" "seadra" "seaking" "sealeo" "seedot" "seel" "seismitoad" "sentret" "serperior" "servine" "seviper" "sewaddle" "sharpedo" "shaymin" "shedinja" "shelgon" "shellder" "shellos" "shelmet" "shieldon" "shiftry" "shinx" "shroomish" "shuckle" "shuppet" "sigilyph" "silcoon" "simipour" "simisage" "simisear" "skarmory" "skiddo" "skiploom" "skitty" "skorupi" "skrelp" "skuntank" "slaking" "slakoth" "sliggoo" "slowbro" "slowking" "slowpoke" "slugma" "slurpuff" "smeargle" "smoochum" "sneasel" "snivy" "snorlax" "snorunt" "snover" "snubbull" "solosis" "solrock" "spearow" "spewpa" "spheal" "spinarak" "spinda" "spiritomb" "spoink" "spritzee" "squirtle" "stantler" "staraptor" "staravia" "starly" "starmie" "staryu" "steelix" "stoutland" "stunfisk" "stunky" "sudowoodo" "suicune" "sunflora" "sunkern" "surskit" "swablu" "swadloon" "swalot" "swampert" "swanna" "swellow" "swinub" "swirlix" "swoobat" "sylveon" "taillow" "talonflame" "tangela" "tangrowth" "tauros" "teddiursa" "tentacool" "tentacruel" "tepig" "terrakion" "throh" "thundurus" "timburr" "tirtouga" "togekiss" "togepi" "togetic" "torchic" "torkoal" "tornadus" "torterra" "totodile" "toxicroak" "tranquill" "trapinch" "treecko" "trevenant" "tropius" "trubbish" "turtwig" "tympole" "tynamo" "typhlosion" "tyranitar" "tyrantrum" "tyrogue" "tyrunt" "umbreon" "unfezant" "unown" "ursaring" "uxie" "vanillish" "vanillite" "vanilluxe" "vaporeon" "venipede" "venomoth" "venonat" "venusaur" "vespiquen" "vibrava" "victini" "victreebel" "vigoroth" "vileplume" "virizion" "vivillon" "volbeat" "volcanion" "volcarona" "voltorb" "vullaby" "vulpix" "wailmer" "wailord" "walrein" "wartortle" "watchog" "weavile" "weedle" "weepinbell" "weezing" "whimsicott" "whirlipede" "whiscash" "whismur" "wigglytuff" "wingull" "wobbuffet" "woobat" "wooper" "wormadam" "wurmple" "wynaut" "xatu" "xerneas" "yamask" "yanma" "yanmega" "yveltal" "zangoose" "zapdos" "zebstrika" "zekrom" "zigzagoon" "zoroark" "zorua" "zubat" "zweilous" "zygarde"]))

;; ### Tokenizer

(def uchars (into {} (map-indexed (fn [i c] [(char c) i]) (set (clojure.string/join docs)))))
^{:nextjournal.clerk/visibility {:result :hide}}
(def BOS (count uchars))
(def vocab-size (inc (count uchars)))

^{:nextjournal.clerk/visibility {:result :hide}}
(def n-layer 1)
^{:nextjournal.clerk/visibility {:result :hide}}
(def n-embd 16)
^{:nextjournal.clerk/visibility {:result :hide}}
(def block-size 16)
^{:nextjournal.clerk/visibility {:result :hide}}
(def n-head 4)
(def head-dim (/ n-embd n-head))

;; ### Parameters

^{:nextjournal.clerk/visibility {:result :hide}}
(def state-dict
  (let [sample-normal (fn [shape] (ma/sample-normal shape 0.08))]
    (loop [state-dict {"wte" (make-node (sample-normal [vocab-size n-embd]) "wte")
                       "wpe" (make-node (sample-normal [block-size n-embd]) "wpe")
                       "lm_head" (make-node (sample-normal [vocab-size n-embd]) "lm-head")}
           i 0]
        (if (>= i n-layer)
            state-dict
            (recur (-> state-dict
                       (assoc (str "layer" i ".attn_wq") (make-node (sample-normal [n-embd n-embd]) (str "layer" i ".attn_wq")))
                       (assoc (str "layer" i ".attn_wk") (make-node (sample-normal [n-embd n-embd]) (str "layer" i ".attn_wk")))
                       (assoc (str "layer" i ".attn_wv") (make-node (sample-normal [n-embd n-embd]) (str "layer" i ".attn_wv")))
                       (assoc (str "layer" i ".attn_wo") (make-node (sample-normal [n-embd n-embd]) (str "layer" i ".attn_wo")))
                       (assoc (str "layer" i ".mlp_fc1") (make-node (sample-normal [(* 4 n-embd) n-embd]) (str "layer" i ".mlp_fc1")))
                       (assoc (str "layer" i ".mlp_fc2") (make-node (sample-normal [n-embd (* 4 n-embd)]) (str "layer" i ".mlp_fc2"))))
                   (inc 1))))))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/table (clerk/use-headers (cons ["params" "shape"] (map (fn [[k v]] [k (ma/shape (.value v))]) state-dict))))

;; ### Architecture

;; ##### Helpers

^{:nextjournal.clerk/visibility {:result :hide}}
(defn softmax
  [logits]
  (let [max-val (ma/max (.value logits))
        exps (exp (sub logits (make-node max-val)))
        total (sum exps)]
    (div exps total)))

^{:nextjournal.clerk/visibility {:result :hide}}
(defn rmsnorm
  [x]
  (let [ms (div (sum (mul x x)) (make-node (ma/count (.value x))))
        scale (pow (add ms (make-node 1e-5)) -0.5)]
    (mul x scale)))

;; ##### Model

^{:nextjournal.clerk/visibility {:result :hide}}
(defn gpt
  [token-id pos-id keys values state-dict]
  (let [{wte "wte" wpe "wpe" lm-head "lm_head"} state-dict
        tok-emb (select wte token-id :all)
        pos-emb (select wpe pos-id :all)
        x (add tok-emb pos-emb)
        x (reshape x (cons 1 (ma/shape (.value x)))) ;; essentially setting batch_size=1
        x (rmsnorm x)]
    (loop [x x
           li 0
           keys keys
           values values]
        (if (>= li n-layer)
            [(mmul x (transpose lm-head)) keys values]
            (let [;; Multi-Head Attention block
                  x-residual x
                  x (rmsnorm x)
                  q (mmul x (get state-dict (str "layer" li ".attn_wq")))
                  k (mmul x (get state-dict (str "layer" li ".attn_wk")))
                  v (mmul x (get state-dict (str "layer" li ".attn_wv")))
                  keys (if (nil? (nth keys li nil))
                           (conj keys k)
                           (assoc keys li (join (nth keys li) k)))
                  values (if (nil? (nth values li nil))
                             (conj values v)
                             (assoc values li (join (nth values li) v)))
                  head-outs (for [h (range n-head)]
                              (let [hs (* h head-dim)
                                    q-h (select q :all (range hs (+ hs head-dim)))
                                    k-h (select (nth keys li) :all (range hs (+ hs head-dim)))
                                    v-h (select (nth values li) :all (range hs (+ hs head-dim)))
                                    attn-logits (div (mmul q-h (transpose k-h)) (make-node (math/pow head-dim 0.5)))
                                    attn-weights (softmax attn-logits)]
                                (mmul attn-weights v-h)))
                  x-attn (reduce #(join %1 %2 1) head-outs)
                  x (mmul x-attn (get state-dict (str "layer" li ".attn_wo")))
                  x (add x x-residual)
                  ;; MLP block
                  x-residual x
                  x (rmsnorm x)
                  x (mmul (get state-dict (str "layer" li ".mlp_fc1")) (transpose x))
                  x (relu x)
                  x (mmul (get state-dict (str "layer" li ".mlp_fc2")) x)
                  x (add (transpose x) x-residual)]
                (recur x (inc li) keys values))))))

;; ### Training loop

^{:nextjournal.clerk/visibility {:result :hide}}
(def learning-rate 0.01)
^{:nextjournal.clerk/visibility {:result :hide}}
(def beta1 0.85)
^{:nextjournal.clerk/visibility {:result :hide}}
(def beta2 0.99)
^{:nextjournal.clerk/visibility {:result :hide}}
(def eps-adam 1e-8)

^{:nextjournal.clerk/visibility {:result :hide}}
(defn train-step
  [step num-steps state-dict m v]
  (let [doc (nth docs (mod step (count docs)))
        _ (println "doc:" doc)
        tokens (->> (map (fn [c] (get uchars c)) doc)
                    (cons BOS)
                    (reverse)
                    (cons BOS)
                    (reverse))
        n (min block-size (dec (count tokens)))
        loss (loop [pos-id 0
                    keys []
                    values []
                    losses []]
                  (if (>= pos-id n)
                      (mul (div (make-node 1) (make-node n)) (reduce add losses))
                      (let [token-id (nth tokens pos-id)
                            target-id (nth tokens (inc pos-id))
                            [logits keys values] (gpt token-id pos-id keys values state-dict)
                            probs (softmax logits)
                            loss-t (neg (log (select probs :all target-id)))]
                      (recur (inc pos-id) keys values (conj losses loss-t)))))
         grads (differentiate loss)
         lr-t (* learning-rate (- 1 (/ step num-steps)))]
     (loop [params (keys state-dict)
            updates {}
            m m
            v v]
       (if (empty? params)
         [updates (first (.value loss)) m v]
         (let [p (first params)
               mi (ma/add (ma/mul beta1 (m p)) (ma/mul (- 1 beta1) (grads (state-dict p))))
               vi (ma/add (ma/mul beta2 (v p)) (ma/mul (- 1 beta2) (ma/pow (grads (state-dict p)) 2)))
               m-hat (ma/div mi (- 1 (math/pow beta1 (inc step))))
               v-hat (ma/div vi (- 1 (math/pow beta2 (inc step))))]
           (recur (rest params)
                  (assoc updates p (-> (.value (state-dict p))
                                        (ma/sub (ma/div (ma/mul lr-t m-hat) (ma/add (ma/pow v-hat 0.5) eps-adam)))
                                        (make-node p)))
                  (assoc m p mi)
                  (assoc v p vi)))))))

;; ### Inference

^{:nextjournal.clerk/visibility {:result :hide}}
(defn infer
  [state-dict sample-size]
  (let [chars (set/map-invert uchars)]
    (for [_ (range sample-size)]
      (let [sample (loop [pos-id 0
                          token-id nil
                          keys []
                          values []
                          sample []]
                     (cond (>= pos-id block-size) sample
                           (= token-id BOS) (butlast sample)
                           :else (let [token-id (if (nil? token-id) BOS token-id)
                                       [logits keys values] (gpt token-id pos-id keys values state-dict)
                                       probs (softmax logits)
                                       token-id (util/choose (ma/flatten (.value probs)))]
                                    (recur (inc pos-id) token-id keys values (conj sample (chars token-id))))))]
        (apply str sample)))))
 
^{:nextjournal.clerk/visibility {:result :hide}}
(defn train
  [num-steps sample-step]
  (loop [step 0
         progress []
         state-dict state-dict
         m (update-vals state-dict #(ma/zeros (ma/shape (.value %))))
         v (update-vals state-dict #(ma/zeros (ma/shape (.value %))))]
    (if (>= step num-steps)
        {:params state-dict
         :progress progress}
        (do
          (let [samples (when (= (mod step sample-step) 0) (infer state-dict 5))
                [state-dict loss m v] (train-step step num-steps state-dict m v)]
            (when (not (nil? samples)) (dorun (map-indexed #(println "sample" %1 ": {" %2 "}") samples)))
            (println "step:" (inc step) "/" num-steps " | loss:" loss)
            (recur (inc step)
                   (conj progress (cond-> {:step step :loss loss} (not (nil? samples)) (assoc :samples samples)))
                   state-dict
                   m
                   v))))))

;; ### Results

(def results
  (let [num-steps 1000
        sample-step 50]
    (train num-steps sample-step)))

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/vl {:data {:values (:progress results)}
           :width 600 :height 400
           :encoding {:x {:field "step" :type "quantitative"}}
           :layer [{:mark "line" :encoding {:color {:value "#1f77b4"} :y {:field "loss" :type "quantitative"}}}]
           :resolve {:scale {:y "independent"}}})

;; ##### Sample output

^{:nextjournal.clerk/visibility {:code :hide}}
(clerk/table {::clerk/page-size nil}
             {:head ["step" "sample"]
              :rows (->> (:progress results)
                         (filter (fn [p] (not (nil? (:samples p)))))
                         (mapcat (fn [p] (map #(list (:step p) %) (:samples p)))))})

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(defn build-static-html
  "Run with `lein run -m lambda-autodiff.examples.gpt/build-static-html`"
  []
  (clerk/clear-cache!)
  (clerk/build! {:paths ["src/lambda_autodiff/examples/gpt.clj"]
                 :out-path "doc/examples/gpt"}))
