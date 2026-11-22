"""Online deterministic generators for Category 1 & Category 4 developmental corpora.

Implements Engineering Blueprint Section 4.2 & 4.3:
  - Category 1a: 2D Grid Physics World State Transition streams (object permanence,
                 containment, occlusion, action-outcome causality).
  - Category 1b: Dyck-k nested bracket & indented code structure streams (depth 1..14).
  - Category 4:  Online Fast-Mapping Pseudo-word Episode Synthesizer (5-step algorithm:
                 phonotactic pseudo-words, 50/50 word-first vs definition-first, 2-5
                 distractor sentences forcing K+1 -> K eviction, target high-weight mask
                 and optional morphological plural/past/method inflection!).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_ONSETS = ["bl", "sp", "v", "z", "dr", "kr", "fl", "gl", "tr", "pr", "sn", "w", "m", "n", "t", "k", "f", "g", "d", "b"]
_VOWELS = ["a", "e", "i", "o", "u"]
_CODAS = ["ck", "rg", "sh", "g", "p", "t", "b", "m", "n", "nd", "st", "lf", "rk", "z"]
_ZH_RARE_PHONETIC = list("狴犴鹀鸮麇猯貐鲦玃鸧鹙狯獬廌獝鹓鹔猨")
_BRACKET_PAIRS = [("{", "}"), ("(", ")"), ("[", "]")]

_CONCRETE_ENTITIES = [
    ("bird", "a tiny nocturnal desert bird that drinks cactus dew", "tiny desert creature with wings"),
    ("frog", "a bright emerald tree frog that hides under wet leaves", "emerald forest amphibian"),
    ("lizard", "a striped canyon lizard that basks on warm stones", "striped stone reptile"),
    ("tool", "a curved brass carving tool used by wooden boat makers", "curved brass instrument"),
    ("fruit", "a sour purple mountain berry that ripens after frost", "sour frost-ripened berry"),
    ("vessel", "a narrow cedar river boat steered with a single pole", "narrow pole-steered boat"),
]

_DISTRACTOR_POOL = [
    "Meanwhile the afternoon wind shifted across the eastern valley and cooled the dusty path.",
    "Two travelers paused near the old stone bridge to check their wooden lanterns before dusk.",
    "Heavy clouds drifted slowly over the ridge while distant bells echoed from the harbor tower.",
    "Inside the quiet workshop, clean jars of oil and neatly coiled ropes lined the cedar shelf.",
    "After the brief shower passed, bright puddles reflected the pale sky along the courtyard wall.",
    "The baker swept flour from the counter and stacked warm loaves beside the brick oven.",
]


@dataclass
class FastMappingEpisode:
    text: str
    pseudoword: str
    plural_form: str
    lang_id: int
    target_byte_start: int
    target_byte_end: int
    pseudoword_byte_spans: list[tuple[int, int]]


def generate_pseudoword(rng: random.Random, lang: str = "en") -> str:
    """Step 2 of Section 4.3: phonotactically valid OOV pseudo-word."""
    if lang == "zh":
        return rng.choice(_ZH_RARE_PHONETIC) + rng.choice(_ZH_RARE_PHONETIC)
    if lang in ("python", "rust"):
        stem1 = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
        stem2 = rng.choice(["buf", "idx", "tag", "slot", "node", "span", "ctx"])
        return f"{stem1}_{stem2}"
    # English CVC / CCVC (4..8 bytes)
    for _ in range(16):
        w = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
        if 4 <= len(w) <= 8:
            return w
    return "blicket"


def _pluralize_en(word: str) -> str:
    if word.endswith(("s", "x", "z", "sh", "ch")):
        return word + "es"
    return word + "s"


def generate_grid_physics_episode(rng: random.Random, num_steps: int = 12) -> str:
    """Category 1a: 2D Grid Physics state transitions (permanence, containment, occlusion)."""
    objects = ["red_ball", "blue_cube", "brass_key", "clay_cup"]
    containers = ["wooden_box", "iron_pot", "woven_basket"]
    pos = {obj: [rng.randint(0, 4), rng.randint(0, 4)] for obj in objects}
    inside: dict[str, str | None] = {obj: None for obj in objects}
    lines = ["[WORLD_INIT]"]
    for obj in objects:
        lines.append(f"state: {obj} at ({pos[obj][0]},{pos[obj][1]}) visible=true")

    for step in range(num_steps):
        obj = rng.choice(objects)
        act = rng.choice(["move", "put_in", "take_out", "occlude"])
        if act == "move" and inside[obj] is None:
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            pos[obj][0] = max(0, min(4, pos[obj][0] + dx))
            pos[obj][1] = max(0, min(4, pos[obj][1] + dy))
            lines.append(f"step {step}: move {obj} ({dx:+d},{dy:+d}) -> {obj} at ({pos[obj][0]},{pos[obj][1]})")
        elif act == "put_in" and inside[obj] is None:
            box = rng.choice(containers)
            inside[obj] = box
            lines.append(f"step {step}: place {obj} inside {box} -> {obj} hidden_in {box} exists=true")
        elif act == "take_out" and inside[obj] is not None:
            box = inside[obj]
            inside[obj] = None
            lines.append(f"step {step}: lift {obj} from {box} -> {obj} at ({pos[obj][0]},{pos[obj][1]}) visible=true")
        else:
            box = rng.choice(containers)
            lines.append(
                f"step {step}: screen {box} blocks view of {obj} -> permanence({obj})=true at ({pos[obj][0]},{pos[obj][1]})"
            )
    return "\n".join(lines) + "\n"


def generate_dyck_sequence(
    rng: random.Random,
    max_depth: int = 12,
    min_bytes: int = 256,
    rust_flavored: bool = False,
) -> tuple[str, list[tuple[int, str]]]:
    """Category 1b: Nested Dyck-k bracket / Rust block sequence with exact closing labels.

    Returns:
      seq: UTF-8 string whose brackets {}()[] are 100% balanced up to depth max_depth.
      closing_positions: list of (byte_offset, closing_char) for Gate 4 evaluation.
    """
    stack: list[str] = []
    chunks: list[str] = []
    closing_positions: list[tuple[int, str]] = []
    total_bytes = 0
    reached_max = False

    def _emit(s: str) -> None:
        nonlocal total_bytes
        chunks.append(s)
        total_bytes += len(s.encode("utf-8"))

    # Drive up to max_depth first so long-range spans are guaranteed
    while len(stack) < max_depth:
        op, cl = rng.choice(_BRACKET_PAIRS)
        if rust_flavored and op == "{":
            prefix = f"fn blk_{len(stack)}() "
            _emit(prefix)
        _emit(op)
        stack.append(cl)
        filler = f" let x{len(stack)} = {rng.randint(1, 99)}; " if rust_flavored else f" a{len(stack)} "
        _emit(filler)
    reached_max = True

    # Interleave pushes, filler, and pops until exceeding min_bytes
    while total_bytes < min_bytes:
        if len(stack) == 0 or (len(stack) < max_depth and rng.random() < 0.55):
            op, cl = rng.choice(_BRACKET_PAIRS)
            _emit(op)
            stack.append(cl)
            _emit(f" v{rng.randint(0, 999)} " if not rust_flavored else f" x += {rng.randint(1, 9)}; ")
        else:
            cl = stack.pop()
            closing_positions.append((total_bytes, cl))
            _emit(cl)
            _emit(" ")

    # Close all remaining open brackets in strict LIFO order
    while stack:
        _emit(" end " if not rust_flavored else " ok; ")
        cl = stack.pop()
        closing_positions.append((total_bytes, cl))
        _emit(cl)

    assert reached_max
    return "".join(chunks), closing_positions


def generate_fast_mapping_episode(
    rng: random.Random,
    lang: str = "en",
    word_first: bool | None = None,
) -> FastMappingEpisode:
    """Category 4: Online 5-step Fast-Mapping Episode (Engineering Doc Section 4.3)."""
    pw = generate_pseudoword(rng, lang=lang)
    plural = _pluralize_en(pw) if lang == "en" else pw
    _, definition, summary = rng.choice(_CONCRETE_ENTITIES)

    if word_first is None:
        word_first = rng.random() < 0.5

    if word_first:
        intro = f"A {pw} is {definition}."
    else:
        intro = f"People call {definition} a {pw}."

    num_distractors = rng.randint(2, 5)
    distractors = " ".join(rng.sample(_DISTRACTOR_POOL, k=num_distractors))
    target = f"Look at one {pw} on the rock; now there are two {plural}, each a {summary}."

    prefix = f"{intro} {distractors} "
    full_text = prefix + target

    prefix_bytes = prefix.encode("utf-8")
    full_bytes = full_text.encode("utf-8")
    target_start = len(prefix_bytes)
    target_end = len(full_bytes)

    # Locate all byte spans of the pseudoword stem for optional Stage D weak ACh hint
    pw_bytes = pw.encode("utf-8")
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        idx = full_bytes.find(pw_bytes, pos)
        if idx < 0:
            break
        spans.append((idx, idx + len(pw_bytes)))
        pos = idx + len(pw_bytes)

    lang_id = 1 if lang == "en" else (0 if lang == "zh" else 4)
    return FastMappingEpisode(
        text=full_text,
        pseudoword=pw,
        plural_form=plural,
        lang_id=lang_id,
        target_byte_start=target_start,
        target_byte_end=target_end,
        pseudoword_byte_spans=spans,
    )
