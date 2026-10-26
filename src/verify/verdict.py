"""Combine layer scores into SAME / MINOR / DIFFERENT.

Each kind of input has one primary layer whose score drives the verdict. The
other layers are reported but do not move the verdict until labels show they
add signal. Thresholds come from config/thresholds.json, which the evaluation
writes. The MINOR band is [minor, diff); its width is a guess (see evaluate).
"""
import json

from . import config

PRIMARY = {"static": ("pixel", "score"), "gif": ("temporal", "score")}


def load_thresholds(path=config.THRESHOLDS):
    with open(path) as f:
        return json.load(f)


def decide(kind, layers, thresholds):
    layer, field = PRIMARY[kind]
    result = layers[layer]
    t = thresholds[kind]
    if not result.get("available") or result.get(field) is None:
        return {"verdict": None, "primary": f"{layer}.{field}", "reason": "primary layer unavailable"}
    score = result[field]
    if score >= t["diff"]:
        verdict = "DIFFERENT"
    elif score >= t["minor"]:
        verdict = "MINOR"
    else:
        verdict = "SAME"
    return {"verdict": verdict, "primary": f"{layer}.{field}", "score": score,
            "thresholds": {"diff": t["diff"], "minor": t["minor"]}}
