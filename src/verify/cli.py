"""verify <reference> <candidate>: per-layer scores and a verdict, as JSON.

Pass two PNGs (static) or two GIFs (animation). Mixing kinds is an error.
Writes JSON to stdout, or to --out.
"""
import argparse
import json
import sys
from pathlib import Path

from . import SCHEMA_VERSION, config
from .layers import layout, pixel, temporal
from .verdict import decide, load_thresholds


def kind_of(ref, cand):
    exts = {ref.suffix.lower(), cand.suffix.lower()}
    if exts == {".gif"}:
        return "gif"
    if exts <= {".png", ".jpg", ".jpeg", ".bmp"} and ".gif" not in exts:
        return "static"
    raise SystemExit(f"reference and candidate must both be GIFs or both be images, got {ref.suffix} and {cand.suffix}")


def verify(ref, cand, layout_ref=None, layout_cand=None, thresholds_path=config.THRESHOLDS):
    kind = kind_of(ref, cand)
    layers = {"pixel": pixel.compare(ref, cand)}
    if kind == "gif":
        layers["temporal"] = temporal.compare(ref, cand)
    layers["layout"] = layout.compare(layout_ref, layout_cand)
    decision = decide(kind, layers, load_thresholds(thresholds_path))
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": str(ref),
        "candidate": str(cand),
        "kind": kind,
        "verdict": decision["verdict"],
        "verdict_detail": decision,
        "layers": layers,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="verify", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="compare a reference capture with a candidate capture")
    v.add_argument("reference", type=Path)
    v.add_argument("candidate", type=Path)
    v.add_argument("--layout-ref", type=Path, help="layout tree of the reference (layer is a stub)")
    v.add_argument("--layout-cand", type=Path, help="layout tree of the candidate (layer is a stub)")
    v.add_argument("--thresholds", type=Path, default=config.THRESHOLDS)
    v.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = p.parse_args(argv)

    for path in (args.reference, args.candidate):
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
    result = verify(args.reference, args.candidate, args.layout_ref, args.layout_cand, args.thresholds)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0
