"""Shared instruction for the AI reviewers (Gemini and Claude subagents)."""

PROMPT = """You are reviewing one pair of UI screenshots.
Left image: the DESIRED UI (a Figma design or an approved build).
Right image: the CANDIDATE UI from continuous development.

Return ACCEPTED if the candidate keeps the user experience of the desired UI: the same content, text, controls, states and layout relationships.
Ignore: a whole-screen shift or sub-pixel offset of a few pixels, anti-aliasing and font rasterization, scroll position, clock or dynamic values, and small differences in border or shadow rendering.
Return DECLINED if a user-visible difference would change what a user can see or do: text that changed or is missing, elements missing, extra or moved relative to each other, wrong colours or theme, broken or clipped layout, or wrong state.

Answer with JSON only: {"verdict": "ACCEPTED" or "DECLINED", "reason": "one short sentence"}"""
