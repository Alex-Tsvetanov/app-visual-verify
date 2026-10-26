"""Layer 'layout': deterministic diff of layout trees exported by the port harness.

STUB. The input format is not defined yet: the C++ harness does not export a
layout tree. Until it does, this layer reports itself unavailable and the
verdict ignores it. When the format exists, implement `compare` here; the
contract is a score where higher means more different, plus details.
"""

REASON = "layout-tree export format not defined yet (C++ harness has no exporter)"


def compare(ref_tree_path=None, cand_tree_path=None):
    if not ref_tree_path or not cand_tree_path:
        return {"available": False, "score": None, "reason": "no layout trees supplied"}
    return {"available": False, "score": None, "reason": REASON}
