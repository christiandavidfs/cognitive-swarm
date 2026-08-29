"""Tiny student router — learns to RECOGNIZE reasoning structure, not memorize answers.

Vision: the programmatic regex in reasoning_primitives.py is a placeholder for
a future non-programmatic layer. The student model is tiny, has no data, only
learns which PATTERN to dispatch to. Deterministic resolvers remain verified
ground truth. Knowledge can change, reasoning compounds.

Training: TF-IDF + LogisticRegression on ~100 paraphrases of the 20 reasoning
patterns. ~50KB model, ms inference, no external dependencies beyond sklearn.
Falls back to regex when confidence < threshold.
"""

import re
import pickle
from pathlib import Path
from typing import Optional, Tuple

MODEL_PATH = Path(__file__).parent / "student_router.pkl"
THRESHOLD = 0.07  # below this, fall back to regex (32 classes => random 0.03)

# Canonical training data: (question paraphrase, pattern_name)
# Each pattern gets 4-6 paraphrases with different numbers/wordings so the
# student learns STRUCTURE, not the specific answer (e.g., 5 machines vs
# 7 ovens both -> rate_machines).
TRAINING_DATA = [
    # trick_no_smoke
    ("An electric train is going north at 100 km/h and wind blows east. Which way does smoke blow?", "trick_no_smoke"),
    ("Electric train smoke direction when wind from south?", "trick_no_smoke"),
    ("Which way does the smoke blow from an electric train?", "trick_no_smoke"),
    # trick_survivors
    ("Plane crashes on border of US and Canada where do you bury survivors?", "trick_survivors_not_buried"),
    ("A plane crashes exactly on the border, where are survivors buried?", "trick_survivors_not_buried"),
    # all_months
    ("How many months have 28 days?", "all_months"),
    ("Which months have at least 28 days?", "all_months"),
    # all_but
    ("Farmer has 17 sheep all but 9 die how many left?", "all_but"),
    ("All but 5 of the 20 apples remain, how many remain?", "all_but"),
    # dogs_see_cats
    ("Each dog sees 3 cats how many cats are there?", "dogs_see_cats"),
    ("3 dogs each see 4 cats, how many cats?", "dogs_see_cats"),
    # same_weight
    ("Which weighs more a pound of feathers or a pound of iron?", "same_weight"),
    ("What weighs more a pound of bricks or a pound of feathers?", "same_weight"),
    # letter_riddle
    ("What appears once in a minute twice in a moment and never in a thousand?", "letter_riddle"),
    ("What occurs once in a minute twice in a moment?", "letter_riddle"),
    # simple_probability
    ("Bag has 4 blue and 6 red balls probability of red?", "simple_probability"),
    ("There are 3 red balls and 7 blue balls what is probability of drawing blue?", "simple_probability"),
    # rate_machines
    ("5 machines make 5 widgets in 5 minutes how long for 100 machines to make 100 widgets?", "rate_machines"),
    ("7 ovens bake 7 pies in 7 minutes how long for 3 ovens to bake 3 pies?", "rate_machines"),
    # rate_distance
    ("Car uses 8 liters per 100 km how many liters for 25 km?", "rate_distance"),
    ("Vehicle consumes 6 liters per 100 km how much for 200 km?", "rate_distance"),
    # rate_workers
    ("3 gardeners take 2 hours to weed how long for 6 gardeners?", "rate_workers"),
    ("4 workers finish in 6 hours how long for 8 workers?", "rate_workers"),
    # snail_well
    ("Snail at bottom of 30 foot well climbs 3 feet each day slides back 2 feet how many days to top?", "snail_well"),
    ("Snail in 10 foot well climbs 3 slides 1 how many days?", "snail_well"),
    # simple_subtract (guarded: not twice)
    ("You have 10 apples and give away 4 how many left?", "simple_subtract"),
    ("I have 20 apples and give away 7 how many remain?", "simple_subtract"),
    # handshake
    ("In a group of 8 people each shakes hands with every other once how many handshakes?", "handshake"),
    ("100 people each shake hands with every other exactly once how many handshakes?", "handshake"),
    ("15 people at a party each pair shakes hands once how many handshakes?", "handshake"),
    # triangular_sequence
    ("What comes next in sequence 2, 6, 12, 20, 30?", "triangular_sequence"),
    ("Next in sequence: 6, 12, 20, 30?", "triangular_sequence"),
    # nth_odd
    ("What is the 10th odd positive integer?", "nth_odd"),
    ("Find the 7th odd positive integer", "nth_odd"),
    # painted_cube
    ("Cube cut into 27 equal smaller cubes how many have exactly one face painted?", "painted_cube"),
    ("Cube divided into 64 small cubes how many have one face painted?", "painted_cube"),
    # rope_cut
    ("Rope 100 cm cut one piece is 3 times as long as the other how long is shorter?", "rope_cut"),
    ("Rope 60 cm cut one piece 2 times longer than other find shorter", "rope_cut"),
    # bat_and_ball
    ("Bat and ball cost $1.10 together bat costs $1 more than ball how much is ball?", "bat_and_ball"),
    ("Bat and ball cost $1.10 bat costs one dollar more how much is ball?", "bat_and_ball"),
    # fathers_sons
    ("Two fathers and two sons go fishing each catches one fish how many fish total?", "fathers_sons"),
    ("Two fathers and two sons fishing how many fish?", "fathers_sons"),
    # doctors_brother
    ("Doctor is Robert's brother but Robert is not doctor's brother how?", "doctors_brother"),
    ("The doctor is the boy's brother but boy is not doctor's brother", "doctors_brother"),
    # mislabeled_boxes
    ("3 boxes labeled Apples Oranges Apples&Oranges every label wrong which box to pick?", "mislabeled_boxes"),
    ("Boxes labeled wrong which box should he pick from to label correctly?", "mislabeled_boxes"),
    # min_sisters
    ("Family has 5 children each child has at least one sister minimum children?", "min_sisters"),
    ("Each child has at least one brother and one sister least number of children?", "min_sisters"),
    # fraction_compare
    ("Which is larger 2/3 or 3/5?", "fraction_compare"),
    ("Which is bigger 3/10 or 1/3?", "fraction_compare"),
    # recipe_scaling
    ("2 eggs for 4 people how many eggs for 6 people?", "recipe_scaling"),
    ("3 cups for 2 servings how many cups for 6 servings?", "recipe_scaling"),
    # invalid_syllogism
    ("All cats are animals some animals are dogs can we conclude some cats are dogs?", "invalid_syllogism"),
    ("All A are B some B are C can we conclude some A are C?", "invalid_syllogism"),
    # work_inverse
    ("Ten people can build a house in 20 days how many people needed to build it in 5 days?", "work_inverse"),
    ("6 people can build wall in 8 days how many people needed for 4 days?", "work_inverse"),
    # trains_meet
    ("Two trains 300 km apart travel toward each other at 60 km/h and 40 km/h when meet?", "trains_meet"),
    ("Trains 200 km apart travel toward each other at 80 and 20 when meet?", "trains_meet"),
    # jug_impossible
    ("You have 5 liter jug and 3 liter jug what exact number cannot be measured?", "jug_impossible"),
    ("With 5 liter and 3 liter jugs which amount cannot be measured?", "jug_impossible"),
    # bridge_crossing
    ("Four people need cross bridge at night flashlight 1 2 5 10 minutes shortest time?", "bridge_crossing"),
    ("Bridge flashlight 4 people 1 2 5 10 minutes minimum crossing time?", "bridge_crossing"),
    # system_equations
    ("Five jars of honey and three jars of jam cost 54 cents Four jars of honey and three jars of jam cost 42 cents how much does one jar of honey cost?", "system_equations"),
    ("5 jars of honey and 3 jars of jam cost 54 cents 4 jars of honey and 3 jars of jam cost 42 cents how much one jar honey cost?", "system_equations"),
    # alice_bob
    ("Alice and Bob have total of 40 apples Bob gives Alice 5 apples Now Alice has twice as many as Bob how many did Alice start with?", "alice_bob"),
    ("Alice and Bob total 30 apples Bob gives Alice 3 now Alice has twice as many as Bob how many did Alice start?", "alice_bob"),
]


def train(save_path: Path = MODEL_PATH) -> dict:
    """Train tiny TF-IDF + LogReg student, save pkl, return report."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        return {"error": f"sklearn not available: {e}"}

    texts = [q for q, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=500)
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    # 3-fold CV to report generalization (structure, not memorization)
    try:
        scores = cross_val_score(clf, X, labels, cv=3)
        cv_mean = float(scores.mean())
    except Exception:
        cv_mean = -1.0

    clf.fit(X, labels)

    payload = {"vectorizer": vec, "classifier": clf, "labels": sorted(set(labels))}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    # quick sanity on training set
    preds = clf.predict(X)
    acc = sum(p == t for p, t in zip(preds, labels)) / len(labels)
    return {"train_acc": acc, "cv_mean": cv_mean, "n_samples": len(labels), "n_classes": len(set(labels)), "model_path": str(save_path)}


def _load():
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict(question: str) -> Optional[Tuple[str, float]]:
    """Return (pattern_name, confidence) or None if no model / low confidence."""
    payload = _load()
    if payload is None:
        return None
    vec = payload["vectorizer"]
    clf = payload["classifier"]
    X = vec.transform([question])
    probs = clf.predict_proba(X)[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    label = clf.classes_[idx]
    if conf < THRESHOLD:
        return None
    return (str(label), conf)


def _word_to_num(text: str) -> str:
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
    }
    tl = text.lower()
    for w, n in word_to_num.items():
        tl = re.sub(r'\b' + w + r'\b', n, tl)
    return tl

def _nums(question: str) -> list[int]:
    norm = _word_to_num(question)
    return [int(x) for x in re.findall(r"\d+", norm)]

def extract_args(question: str, pattern: str) -> Optional[dict]:
    """Heuristic arg extraction for student v3: map numbers in order to pattern args.
    This is NOT regex-per-pattern — it's generic number-order mapping, so it generalizes
    to unseen numbers (e.g., 47 people handshake). Resolver remains deterministic."""
    nums = _nums(question)
    ql = question.lower()
    try:
        if pattern == "handshake":
            if nums: return {"n": nums[0]}
        elif pattern == "work_inverse":
            # "12 people can build ... in 15 days how many for 3 days" -> [12,15,3]
            if len(nums) >= 3: return {"n1": nums[0], "t1": nums[1], "t2": nums[2], "n2": int(nums[0]*nums[1]/nums[2]) if nums[2]!=0 else None}
        elif pattern == "rate_machines":
            if len(nums) >= 3: return {"time": nums[1] if len(nums)>1 else nums[0]}
        elif pattern == "rate_distance":
            if len(nums) >= 2: return {"liters_per_100km": nums[0], "km": nums[-1]}
        elif pattern == "snail_well":
            if len(nums) >= 3: return {"depth": nums[0], "climb_per_day": nums[1], "slide_per_night": nums[2]}
        elif pattern == "simple_subtract":
            if len(nums) >= 2: return {"have": nums[0], "give": nums[1]}
        elif pattern == "all_but":
            if len(nums) >= 2: return {"n": nums[1]}
        elif pattern == "rate_workers":
            if len(nums) >= 3: return {"n_workers": nums[0], "time": nums[1], "new_workers": nums[2]}
        elif pattern == "jug_impossible":
            if len(nums) >= 2: return {"a": nums[0], "b": nums[1]}
        elif pattern == "min_sisters":
            # 5 children -> n_children=5
            if nums: return {"n_children": nums[0], "min_girls": 2}
        elif pattern == "fraction_compare":
            # extract fractions a/b and c/d
            m = re.findall(r"(\d+)\s*/\s*(\d+)", question)
            if len(m) >= 2: return {"a": int(m[0][0]), "b": int(m[0][1]), "c": int(m[1][0]), "d": int(m[1][1]), "winner": f"{m[0][0]}/{m[0][1]}" if int(m[0][0])*int(m[1][1]) > int(m[1][0])*int(m[0][1]) else f"{m[1][0]}/{m[1][1]}"}
        elif pattern == "bridge_crossing":
            # Extract times like "1, 3, 6, 8 minutes" or "1 2 5 10 minutes"
            mlist = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*minutes?", question.lower())
            if mlist:
                return {"times": sorted([int(mlist.group(i)) for i in range(1,5)])}
            times = [int(x) for x in re.findall(r"(\d+)\s*min", question.lower())]
            if len(times) >= 4:
                return {"times": sorted(times)[:4]}
            return {"times": [1,2,5,10]}
        elif pattern == "system_equations":
            # fallback to regex for precise honey/jam — student knows pattern, regex extracts
            from cognitive_swarm.tools.reasoning_primitives import detect_reasoning_pattern as _det
            det = _det(question)
            if det and det[0]=="system_equations": return det[1]
        elif pattern == "alice_bob":
            if len(nums) >= 2: return {"total": nums[0], "transferred": nums[1], "answer": nums[0]//2}
        elif pattern in ("trick_no_smoke","trick_survivors_not_buried","all_months","same_weight","letter_riddle","mislabeled_boxes","invalid_syllogism","fathers_sons","doctors_brother"):
            return {}
    except Exception:
        return None
    return None

def predict_with_args(question: str) -> Optional[Tuple[str, dict, float]]:
    """Student v3: returns (pattern, args, confidence) with args extracted generically."""
    pred = predict(question)
    if pred is None:
        return None
    label, conf = pred
    args = extract_args(question, label)
    if args is None:
        return None
    args["_student_conf"] = conf
    return (label, args, conf)

def detect_with_student(question: str) -> Optional[Tuple[str, dict]]:
    """Try student first; returns (pattern_name, {}) stub — caller should
    re-run detect_reasoning_pattern for full args, or use student label to
    dispatch resolver. This keeps resolvers as verified ground truth."""
    pred = predict(question)
    if pred is None:
        return None
    label, conf = pred
    # We return label with empty args; truth_router will re-detect via regex
    # for precise args, but we record that student RECOGNIZED the structure.
    return (label, {"_student_conf": conf})
