"""Experimento 3 combinaciones student - mide routing + generalización."""

import random, re, pickle, time
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cognitive_swarm.tools.student_router import TRAINING_DATA
from cognitive_swarm.tools.reasoning_primitives import detect_reasoning_pattern
from cognitive_swarm.evaluation.leveled_benchmark import LEVELED_PROBLEMS

# --- Generar 226 dataset balanceado (32 patrones x5 + 66 orig) como en v2 ---
import sys
sys.path.insert(0, ".")

all_templates = {
    "trick_no_smoke": ["Electric train going {n} km/h wind {d} which way smoke?", "Which way does smoke blow from electric train?"],
    "trick_survivors_not_buried": ["Plane crashes on border where do you bury survivors?", "Crash on border of two countries where are survivors buried?"],
    "all_months": ["How many months have 28 days?", "Which months have at least 28 days?"],
    "all_but": ["Farmer has {total} sheep all but {remain} die how many left?"],
    "dogs_see_cats": ["Each dog sees {n} cats how many cats?"],
    "same_weight": ["Which weighs more a pound of feathers or a pound of {mat}?"],
    "letter_riddle": ["What appears once in a minute twice in a moment and never in a thousand?"],
    "simple_probability": ["Bag has {a} blue and {b} red balls probability of red?"],
    "rate_machines": ["{n} machines make {n} widgets in {t} minutes how long for {m} machines to make {m} widgets?"],
    "rate_distance": ["Car uses {c} liters per 100 km how many liters for {d} km?"],
    "rate_workers": ["{n} gardeners take {t} hours to weed how long for {m} gardeners?"],
    "snail_well": ["Snail at bottom of {depth} foot well climbs {climb} feet each day slides back {slide} feet how many days to top?"],
    "simple_subtract": ["You have {have} apples and give away {give} how many left?"],
    "handshake": ["In a group of {n} people each person shakes hands with every other person exactly once how many handshakes occur?"],
    "triangular_sequence": ["What comes next in sequence {s}?"],
    "nth_odd": ["What is the {n}th odd positive integer?"],
    "painted_cube": ["Cube cut into {n} equal smaller cubes how many have exactly one face painted?"],
    "rope_cut": ["Rope {total} cm cut one piece is {k} times as long as the other how long is shorter?"],
    "bat_and_ball": ["Bat and ball cost ${d}.10 together bat costs ${d} more than ball how much is ball?"],
    "fathers_sons": ["Two fathers and two sons go fishing each catches one fish how many fish total?"],
    "doctors_brother": ["Doctor is Robert's brother but Robert is not doctor's brother how?"],
    "mislabeled_boxes": ["3 boxes labeled Apples Oranges Apples&Oranges every label wrong which box to pick?"],
    "min_sisters": ["Family has {n} children each child has at least one sister minimum children?"],
    "fraction_compare": ["Which is larger {a}/{b} or {c}/{d}?"],
    "recipe_scaling": ["{a} eggs for {b} people how many eggs for {c} people?"],
    "invalid_syllogism": ["All cats are animals some animals are dogs can we conclude some cats are dogs?"],
    "work_inverse": ["{n1} people can build a house in {t1} days how many people are needed to build it in {t2} days?"],
    "trains_meet": ["Two trains {d} km apart travel toward each other at {v1} km/h and {v2} km/h when meet?"],
    "jug_impossible": ["You have {a} liter jug and {b} liter jug what exact amount cannot be measured?"],
    "bridge_crossing": ["Four people need to cross a bridge at night flashlight {t1} {t2} {t3} {t4} minutes shortest time?"],
    "system_equations": ["{n1} jars of honey and {n2} jars of jam cost {c1} cents {n3} jars of honey and {n4} jars of jam cost {c2} cents how much does one jar of honey cost?"],
    "alice_bob": ["Alice and Bob have total of {total} apples Bob gives Alice {t} apples Now Alice has twice as many as Bob how many did Alice start with?"],
}

samples=[]
for pat, tmpls in all_templates.items():
    for _ in range(5):
        tmpl = random.choice(tmpls)
        kw={}
        for ph in re.findall(r"\{(\w+)\}", tmpl):
            if ph=="mat": kw[ph]=random.choice(["iron","bricks"])
            elif ph=="s": kw[ph]="2, 6, 12, 20"
            else: kw[ph]=str(random.randint(2,50))
        if pat=="fraction_compare":
            a,b,c,d = int(kw["a"]), int(kw["b"]), int(kw["c"]), int(kw["d"])
            if a*d==c*b: kw["c"]=str((c%8)+1)
        try: samples.append((tmpl.format(**kw), pat))
        except: pass

combined = TRAINING_DATA + samples
texts=[q for q,_ in combined]
labels=[l for _,l in combined]
print(f"Dataset: {len(combined)} samples, {len(set(labels))} classes")

# 20 novel problems with numbers never seen in training
novel = [
    ("There are 47 people at a conference each pair shakes hands once how many handshakes?", "handshake", 1081),
    ("12 people can build a shed in 15 days how many people are needed to build it in 3 days?", "work_inverse", 60),
    ("You have 19 apples and give away 7 how many left?", "simple_subtract", 12),
    ("Which is larger 7/8 or 5/6?", "fraction_compare", "7/8"),
    ("You have 7 liter jug and 4 liter jug what amount cannot be measured?", "jug_impossible", "Cannot determine / none"),
    ("Car uses 5 liters per 100 km how many liters for 60 km?", "rate_distance", "3"),
    ("Snail at bottom of 50 foot well climbs 7 feet each day slides back 3 feet how many days to top?", "snail_well", "12"),
    ("Family has 9 children each child has at least one sister minimum children?", "min_sisters", "6"),
    ("Rope 84 cm cut one piece is 3 times as long as the other how long is shorter?", "rope_cut", "21"),
    ("Cube cut into 125 equal smaller cubes how many have exactly one face painted?", "painted_cube", "54"),
    ("What is the 15th odd positive integer?", "nth_odd", "29"),
    ("What comes next in sequence 2, 6, 12, 20, 30?", "triangular_sequence", "42"),
    ("Two trains 500 km apart travel toward each other at 100 km/h and 50 km/h when meet?", "trains_meet", "3.3"),
    ("An electric train going east at 80 km/h wind west which way does smoke blow?", "trick_no_smoke", "no smoke"),
    ("Farmer has 31 sheep all but 12 die how many left?", "all_but", "12"),
    ("Which weighs more a pound of feathers or a pound of lead?", "same_weight", "same"),
    ("3 boxes labeled Apples Oranges Apples&Oranges every label wrong which box to pick?", "mislabeled_boxes", "mixed"),
    ("Bag has 9 blue and 3 red balls probability of red?", "simple_probability", "0.25"),
    ("Alice and Bob have total of 50 apples Bob gives Alice 5 Now Alice has twice as many as Bob how many did Alice start with?", "alice_bob", "25"),
    ("Four people need to cross a bridge at night flashlight 1 2 7 10 minutes shortest time?", "bridge_crossing", "17"),
]

def eval_model(name, pipeline):
    t0=time.time()
    scores=cross_val_score(pipeline, texts, labels, cv=3)
    pipeline.fit(texts, labels)
    train_time=time.time()-t0
    # routing on 120 reasoning
    ok=0; total=0
    for p in LEVELED_PROBLEMS:
        if p["category"]!="reasoning": continue
        total+=1
        pred=pipeline.predict([p["q"]])[0]
        regex=detect_reasoning_pattern(p["q"])
        rl=regex[0] if regex else None
        if pred==rl: ok+=1
    # novel
    nok=0
    for q, exp_pat, exp_ans in novel:
        pred=pipeline.predict([q])[0]
        if pred==exp_pat: nok+=1
    # size
    import pickle, io
    buf=io.BytesIO(); pickle.dump(pipeline, buf); size_kb=len(buf.getvalue())/1024
    # latency
    t0=time.time()
    for _ in range(100): pipeline.predict(["test question handshake people"])
    lat=(time.time()-t0)/100*1000
    return {
        "name": name,
        "cv": float(scores.mean()),
        "train_acc": 1.0, # will compute
        "routing": f"{ok}/{total}",
        "routing_pct": ok/total,
        "novel": f"{nok}/{len(novel)}",
        "novel_pct": nok/len(novel),
        "size_kb": size_kb,
        "lat_ms": lat,
    }

configs = [
    ("A: TF-IDF word 1-2 + LogReg (v2 baseline)", Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=800)), ("clf", LogisticRegression(max_iter=1000))]) ),
    ("B: TF-IDF char 3-5 + LogReg", Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=800)), ("clf", LogisticRegression(max_iter=1000))]) ),
    ("C: Count + Naive Bayes", Pipeline([("vec", CountVectorizer(lowercase=True, ngram_range=(1,2), max_features=800)), ("clf", MultinomialNB())]) ),
]

results=[]
for name, pipe in configs:
    r=eval_model(name, pipe)
    # train acc
    pipe.fit(texts, labels)
    preds=pipe.predict(texts)
    r["train_acc"]=sum(p==t for p,t in zip(preds, labels))/len(labels)
    results.append(r)
    print(f"\n{name}")
    print(f"  CV {r['cv']:.3f} train {r['train_acc']:.3f} routing {r['routing']} novel {r['novel']} size {r['size_kb']:.0f}KB lat {r['lat_ms']:.2f}ms")

# pick winner by combined score = 0.5*routing +0.5*novel (generalization matters more)
for r in results:
    r["score"]=0.4*r["routing_pct"]+0.6*r["novel_pct"]
winner=max(results, key=lambda x: x["score"])
print(f"\nWinner: {winner['name']} score {winner['score']:.3f}")

# save report
import json
Path("/tmp/student_exp.json").write_text(json.dumps(results, indent=2))
print("saved /tmp/student_exp.json")
# --- Dual-router note (v9) ---
# M2 dual-router (TF-IDF vs Qwen 0.5B LoRA) will reuse this harness with --dual:
# both routers trained on same 231 samples (33 patterns incl. stock_take_away, ≥4/clase)
# and evaluated on same suites (120 closed + 20 novel + 60 human + 100 breaking).
# See docs/DUAL_ROUTER_BENCHMARK.md for design; qwen_router.py + qwen_lora_finetune.py
# will expose same predict(question)->(pattern,conf) interface for apples-to-apples.
# Until Qwen weights are trained, this file reports TF-IDF baseline; Qwen column = proxy/not-trained.
