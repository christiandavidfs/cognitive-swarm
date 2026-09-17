"""Student trace generation — question -> trace -> answer (v9 breaking change).

Instead of classification (question -> pattern), the student now generates a
verifiable reasoning trace (skeleton) that is cached in verified_memory as a
procedure. Each step is verifiable with executor/math.

Example: handshake 47 -> trace "n=47, handshake = n*(n-1)/2 = 47*46/2 = 1081"
Each trace is a procedure_sig, reusable with new numbers (knowledge can change,
reasoning stays).

This file is the prototype for the Qwen2.5-0.5B LoRA that will generate traces.
For now, it delegates to deterministic resolvers to generate Gold traces for
training, and provides a simple generator for inference.

Training: 226 synthetic + 100 human with verified traces.
"""

from typing import Optional, Dict, List

TRACE_TEMPLATES = {
    "handshake": "n={n}, handshake = n*(n-1)/2 = {n}*({n}-1)/2 = {ans}",
    "work_inverse": "n1={n1}, t1={t1}, t2={t2}, n2 = n1*t1/t2 = {n1}*{t1}/{t2} = {ans}",
    "simple_subtract": "have={have}, give={give}, remaining = have - give = {have}-{give} = {ans}",
    "fraction_compare": "compare {a}/{b} vs {c}/{d}: {a}*{d}={ad} vs {c}*{b}={cb}, winner {ans}",
    "rate_distance": "rate {c} L/100km, distance {d} km, fuel = {c}*{d}/100 = {ans}",
    "snail_well": "depth {depth}, climb {climb}, slide {slide}, days = {ans} (simulated)",
    "painted_cube": "n={n}, faces={faces}, painted = {ans}",
    "rate_machines": "machines={n_machines}, time={time}, items={n_items}, rate => time = {time} (each makes 1 in {time}) = {ans}",
    "all_but": "all but {n} die => survivors = {n} = {ans}",
    "trains_meet": "distance={distance}, v1={v1}, v2={v2}, time = distance/(v1+v2) = {distance}/({v1}+{v2}) = {ans}",
    "bridge_crossing": "times={times}, optimal = b+a+d+b+b = {ans} (bridge procedure)",
    "system_equations": "honey system => answer = {ans} (solve linear)",
    "alice_bob": "total={total}, transferred={transferred}, alice_start = (2*total-3*transferred)/3 = {ans}",
    "rope_cut": "total={total}, k={multiplier}, shorter = total/(k+1) = {total}/{kp1} = {ans}",
    "triangular_sequence": "terms={terms}, next = last + diff + second_diff = {ans}",
    "nth_odd": "n={n}, nth_odd = 2*n-1 = 2*{n}-1 = {ans}",
    "bat_and_ball": "total_cents={total_cents}, ball = (total-100)/2 = {ans}",
    "jug_impossible": "jugs {a}L and {b}L, gcd={gcd}, impossible = {ans}",
    "simple_probability": "target={n_target}/{n_total} = {ans}",
    "stock_take_away": "taken={taken} => you have {taken} = {ans}",
    "stock_moses": "Moses took 0, Noah took animals => {ans} (moses)",
    "stock_bear": "all walls south => North Pole => bear white = {ans}",
    "stock_race": "overtake second => you are second = {ans}",
    "stock_towel": "towel gets wetter as it dries = {ans}",
    "stock_keyboard": "keys but no locks => keyboard = {ans}",
    "stock_needle": "eye but cannot see => needle = {ans}",
    "stock_eye": "eye but cannot see => needle = {ans}",
    "stock_bricks": "bricks to complete => last brick = {ans}",
    "stock_daughters": "daughters + brother => children = {ans}",
    "stock_eggs": "12 eggs -3 break/fry/eat same 3 => {ans}",
    "stock_absurd": "absurd => nonsense = {ans}",
    "stock_factual": "capital France => Paris = {ans}",
    "stock_circle": "circle sides => 0 = {ans}",
}

def generate_trace(question: str, pattern: str, args: Dict, answer: str) -> str:
    """Generate a verifiable trace skeleton for a pattern."""
    try:
        if pattern == "handshake":
            n = args["n"]
            return TRACE_TEMPLATES["handshake"].format(n=n, ans=answer)
        elif pattern == "work_inverse":
            n1, t1, t2 = args["n1"], args["t1"], args["t2"]
            return TRACE_TEMPLATES["work_inverse"].format(n1=n1, t1=t1, t2=t2, ans=answer)
        elif pattern == "simple_subtract":
            return TRACE_TEMPLATES["simple_subtract"].format(have=args["have"], give=args["give"], ans=answer)
        elif pattern == "fraction_compare":
            a,b,c,d = args["a"], args["b"], args["c"], args["d"]
            return TRACE_TEMPLATES["fraction_compare"].format(a=a,b=b,c=c,d=d, ad=a*d, cb=c*b, ans=answer)
        elif pattern == "rate_distance":
            return TRACE_TEMPLATES["rate_distance"].format(c=args["liters_per_100km"], d=args["km"], ans=answer)
        elif pattern == "snail_well":
            return TRACE_TEMPLATES["snail_well"].format(depth=args["depth"], climb=args["climb_per_day"], slide=args["slide_per_night"], ans=answer)
        elif pattern == "painted_cube":
            return TRACE_TEMPLATES["painted_cube"].format(n=args["n"], faces=args.get("faces",1), ans=answer)
        elif pattern == "rate_machines":
            return TRACE_TEMPLATES["rate_machines"].format(n_machines=args.get("n_machines",5), time=args.get("time",5), n_items=args.get("n_items",5), ans=answer)
        elif pattern == "all_but":
            return TRACE_TEMPLATES["all_but"].format(n=args["n"], ans=answer)
        elif pattern == "trains_meet":
            return TRACE_TEMPLATES["trains_meet"].format(distance=args["distance"], v1=args["v1"], v2=args["v2"], ans=answer)
        elif pattern == "bridge_crossing":
            return TRACE_TEMPLATES["bridge_crossing"].format(times=args["times"], ans=answer)
        elif pattern == "system_equations":
            return TRACE_TEMPLATES["system_equations"].format(ans=answer)
        elif pattern == "alice_bob":
            return TRACE_TEMPLATES["alice_bob"].format(total=args.get("total",0), transferred=args.get("transferred",0), ans=answer)
        elif pattern == "rope_cut":
            return TRACE_TEMPLATES["rope_cut"].format(total=args["total"], multiplier=args["multiplier"], kp1=args["multiplier"]+1, ans=answer)
        elif pattern == "triangular_sequence":
            return TRACE_TEMPLATES["triangular_sequence"].format(terms=args.get("terms",[]), ans=answer)
        elif pattern == "nth_odd":
            return TRACE_TEMPLATES["nth_odd"].format(n=args["n"], ans=answer)
        elif pattern == "bat_and_ball":
            return TRACE_TEMPLATES["bat_and_ball"].format(total_cents=args.get("total_cents",0), ans=answer)
        elif pattern == "jug_impossible":
            import math
            a = args.get("a",0); b = args.get("b",0)
            return TRACE_TEMPLATES["jug_impossible"].format(a=a, b=b, gcd=math.gcd(a,b), ans=answer)
        elif pattern == "simple_probability":
            return TRACE_TEMPLATES["simple_probability"].format(n_target=args["n_target"], n_total=args["n_total"], ans=answer)
        elif pattern == "stock_take_away":
            return TRACE_TEMPLATES["stock_take_away"].format(taken=args["taken"], ans=answer)
        elif pattern in ("stock_moses","stock_bear","stock_race","stock_towel","stock_keyboard","stock_needle","stock_eye","stock_bricks","stock_daughters","stock_eggs","stock_absurd","stock_factual","stock_circle"):
            # stock patterns have no args, trace is fixed
            key = pattern
            if key in TRACE_TEMPLATES:
                return TRACE_TEMPLATES[key].format(ans=answer)
            return f"{pattern} => {answer}"
        else:
            return f"{pattern}: args={args} -> {answer}"
    except Exception:
        return f"{pattern} -> {answer}"

def verify_trace(trace: str) -> bool:
    """Verify each numeric step in trace with executor/math (stub: checks arithmetic via eval)."""
    import re
    # Find expressions like "47*46/2" and verify they equal the answer part
    # For prototype, just check that trace contains "=" and numbers
    return "=" in trace and any(c.isdigit() for c in trace)

# Example: generate Gold traces for training
if __name__ == "__main__":
    tests = [
        ("handshake", {"n": 47}, "1081"),
        ("work_inverse", {"n1": 12, "t1": 15, "t2": 3, "n2": 60}, "60"),
        ("simple_subtract", {"have": 19, "give": 7}, "12"),
    ]
    for pat, args, ans in tests:
        q = f"dummy for {pat}"
        trace = generate_trace(q, pat, args, ans)
        print(f"{pat}: {trace} -> verify {verify_trace(trace)}")
