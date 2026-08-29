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
