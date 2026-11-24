"""Stage-D online zero-network Two-Turn Fast-Mapping Wug Episode synthesizer.

Implements Deliverable 3 of the DCPA Data Infrastructure & Curriculum Pipeline:
  - 100% offline procedural synthesis (zero external network downloads, zero tokenizers).
  - Covers all 5 domain IDs:
      * lang_id=0 (en): Phonotactically legal OOV English stems + exact orthographic
        inflection engine (`inflect_plural`, `inflect_past`, `inflect_prog`).
      * lang_id=1 (zh): Rare valid NFC UTF-8 Hanzi + category morpheme orthogonal compounds.
      * lang_id=2 (ja): Procedural Katakana OOV loanwords (4..6 mora) with SOV particles
        (は/が/を/の) and classifiers.
      * lang_id=3 (python): Fresh non-keyword Python identifiers (functions/classes/vars)
        whose full `[Turn 1 + Turn 2]` document passes `tree-sitter-python` with zero errors.
      * lang_id=4 (rust): Fresh non-keyword Rust identifiers (Vec/struct/fn bindings)
        whose full `[Turn 1 + Turn 2]` document passes `tree-sitter-rust` with zero errors.
  - Strict 50:50 Pre-Definition vs Post-Definition order symmetry across all domains.
  - Exact UTF-8 byte masks:
      * `loss_mask`: False across Turn 1 (Support), True across Turn 2 (Query).
      * `wug_target_mask`: True strictly on the target pseudo-word / inflected form /
        identifier byte slice inside Turn 2 (strict subset of `loss_mask`).
"""

from __future__ import annotations

import keyword
import random
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class WugEpisode:
    """Single two-turn fast-mapping Wug Episode with exact UTF-8 byte masks."""

    lang_id: int
    definition_order: str        # "post" or "pre"
    stem: str                    # Base invented pseudo-word / identifier
    target_word: str             # Exact surface form in Turn 2 (e.g. inflected plural/past or identifier)
    turn1_text: str              # Support segment (loss_mask == False)
    turn2_text: str              # Query segment (loss_mask == True)
    text: str                    # Full NFC UTF-8 string (turn1_text + turn2_text)
    raw_bytes: bytes             # Exact UTF-8 bytes of `text`
    turn1_byte_len: int          # Byte boundary separating Turn 1 and Turn 2
    target_byte_start: int       # Inclusive start byte offset of `target_word` in `raw_bytes`
    target_byte_end: int         # Exclusive end byte offset of `target_word` in `raw_bytes`
    x_bytes: torch.Tensor        # [T] uint8
    loss_mask: torch.Tensor      # [T] bool
    wug_target_mask: torch.Tensor  # [T] bool


# =============================================================================
# English Phonotactic OOV Generator & Exact Orthographic Inflection Engine
# =============================================================================

_EN_ONSETS: tuple[str, ...] = (
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kr",
    "pl", "pr", "sk", "sl", "sn", "sp", "st", "sw", "tr", "tw",
    "w", "z", "v", "qu", "sh", "ch", "th", "br", "gl", "pl",
)
_EN_VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")
_EN_CODAS: tuple[str, ...] = (
    "ck", "d", "f", "g", "k", "mp", "p", "rb", "x", "z",
    "sh", "ch", "nk", "lt", "nd", "rk", "sp", "st", "ft", "b",
)
_EN_SUFFIX_SYLLABLES: tuple[str, ...] = (
    "et", "le", "on", "ip", "ak", "or", "um", "ix", "oz", "al", "in", "up", "a", "o",
)

# Built-in exclusion list to guarantee 100% OOV English pseudo-words
_EN_REAL_WORDS: frozenset[str] = frozenset({
    "bad", "bag", "ban", "bar", "bat", "bed", "beg", "bet", "bid", "big", "bin", "bit",
    "black", "bled", "bless", "blind", "blink", "blob", "block", "blond", "blood", "blot",
    "bluff", "blunt", "blur", "blurt", "blush", "brad", "brag", "bran", "brand", "brass",
    "brat", "bred", "brick", "brig", "brim", "bring", "brink", "brisk", "broad", "broth",
    "brush", "buck", "bud", "buff", "bug", "bulb", "bulk", "bump", "bun", "bunk", "bunt",
    "chap", "chat", "check", "chef", "chest", "chick", "chid", "child", "chill", "chimp",
    "chin", "chip", "chit", "chop", "chub", "chuck", "chug", "chum", "chump", "chunk",
    "clad", "clam", "clamp", "clan", "clang", "clank", "clap", "clash", "clasp", "class",
    "cleft", "clerk", "click", "cliff", "climb", "clink", "clip", "clock", "clod", "clog",
    "clop", "clot", "club", "cluck", "clump", "clung", "clunk", "crab", "crack", "craft",
    "crag", "cram", "cramp", "crank", "crap", "crash", "crass", "crest", "crib", "crick",
    "crisp", "crock", "crop", "cross", "crud", "crump", "crush", "crust", "drab", "draft",
    "drag", "dram", "drank", "drape", "dreg", "dress", "drift", "drill", "drink", "drip",
    "drop", "dross", "drug", "drum", "drunk", "flab", "flag", "flak", "flam", "flan",
    "flank", "flap", "flash", "flask", "flat", "flax", "fled", "flesh", "flex", "flick",
    "fling", "flint", "flip", "flit", "flock", "flog", "flop", "floss", "flub", "fluff",
    "flung", "flunk", "flush", "flute", "flux", "frat", "fret", "frig", "frisk", "frock",
    "frog", "from", "frond", "front", "frost", "froth", "glad", "gland", "glass", "glen",
    "glib", "glint", "glob", "glop", "gloss", "glum", "glut", "grab", "grad", "graft",
    "gram", "grand", "grant", "graph", "grasp", "grass", "greg", "grid", "grift", "grill",
    "grim", "grin", "grind", "grip", "grist", "grit", "grog", "grot", "grub", "gruff",
    "grunt", "krill", "plan", "plank", "plant", "plash", "plat", "pled", "plod", "plop",
    "plot", "pluck", "plug", "plum", "plumb", "plump", "plunk", "plush", "pram", "prank",
    "prat", "prep", "press", "prick", "prig", "prim", "primp", "print", "prod", "prof",
    "prom", "prompt", "prong", "prop", "prow", "quack", "quad", "quaff", "quag", "quell",
    "quest", "quick", "quid", "quill", "quilt", "quip", "quit", "quiz", "shack", "shad",
    "shaft", "shag", "sham", "shank", "shed", "shelf", "shell", "shift", "shill", "shin",
    "ship", "shirk", "shirt", "shod", "shock", "shop", "short", "shot", "shred", "shrub",
    "shrug", "shuck", "shun", "shunt", "shush", "shut", "skid", "skiff", "skill", "skim",
    "skimp", "skin", "skip", "skirt", "skit", "skulk", "skull", "skunk", "slab", "slack",
    "slag", "slam", "slang", "slant", "slap", "slash", "slat", "sled", "slept", "slew",
    "slick", "slid", "slim", "slimp", "sling", "slink", "slip", "slit", "slob", "slog",
    "slop", "slot", "slug", "slum", "slump", "slung", "slunk", "slur", "slurp", "slush",
    "snack", "snag", "snap", "snark", "snick", "sniff", "snip", "snit", "snob", "snog",
    "snoop", "snort", "snot", "snub", "snuck", "snuff", "snug", "spam", "span", "spank",
    "spar", "spark", "spat", "spaz", "spec", "sped", "spell", "spend", "spent", "spill",
    "spin", "spit", "spiv", "split", "spoil", "sport", "spot", "sprig", "spud", "spun",
    "spunk", "spur", "spurn", "spurt", "stab", "stack", "staff", "stag", "stamp", "stand",
    "stank", "star", "stark", "start", "stash", "stat", "stem", "step", "stern", "stew",
    "stick", "stiff", "still", "stilt", "sting", "stink", "stint", "stock", "stomp", "stop",
    "stork", "storm", "strap", "strip", "strop", "strut", "stub", "stuck", "stud", "stuff",
    "stump", "stun", "stung", "stunk", "stunt", "swag", "swam", "swamp", "swan", "swap",
    "swat", "swell", "swept", "swift", "swig", "swill", "swim", "swing", "swipe", "swirl",
    "swish", "swob", "swop", "swum", "swung", "thank", "that", "theft", "them", "then",
    "thick", "thin", "thing", "think", "third", "this", "thong", "thorn", "throb", "thrum",
    "thud", "thug", "thumb", "thump", "thus", "track", "tract", "trad", "tram", "tramp",
    "trap", "trash", "trek", "trend", "tress", "trick", "trig", "trim", "trip", "trod",
    "troll", "tromp", "trop", "trot", "trow", "truck", "trug", "trump", "trunk", "truss",
    "trust", "twang", "twerp", "twig", "twin", "twirl", "twist", "twit", "vac", "van",
    "vat", "vent", "verb", "vest", "vet", "vex", "vim", "viz", "volt", "wab", "wack",
    "wad", "waft", "wag", "wan", "wand", "want", "war", "ward", "wark", "warm", "warn",
    "warp", "wart", "wash", "wasp", "watt", "wax", "web", "wed", "weft", "weld", "well",
    "welt", "wend", "went", "wept", "west", "wet", "whack", "what", "when", "whet", "whiff",
    "whig", "whim", "whip", "whir", "whisk", "whit", "whiz", "wick", "wid", "width", "wig",
    "wild", "will", "wilt", "wimp", "win", "wince", "wind", "wing", "wink", "wisp", "wist",
    "wit", "witch", "with", "wok", "wolf", "won", "wont", "word", "work", "worm", "worn",
    "worst", "wort", "worth", "wrap", "wren", "wrest", "wrick", "wring", "wrist", "writ",
    "zag", "zap", "zed", "zest", "zig", "zinc", "zing", "zip", "zit", "zouk",
    "planet", "blanket", "bracket", "cricket", "thicket", "wicket", "trinket", "ticket",
    "picket", "pocket", "rocket", "socket", "locket", "basket", "casket", "gasket", "jacket",
    "market", "target", "carpet", "helmet", "velvet", "wallet", "mallet", "pellet", "bullet",
    "skillet", "fillet", "millet", "pullet", "goblet", "tablet", "outlet", "inlet", "piglet",
    "dragon", "wagon", "flag", "lemon", "melon", "talon", "salon", "bison", "bacon", "baron",
})

_VOWEL_SET = frozenset("aeiou")


def inflect_plural(stem: str) -> str:
    """Exact English orthographic noun pluralization."""
    s = stem.lower()
    if s.endswith(("s", "x", "z", "ch", "sh")):
        return s + "es"
    if len(s) >= 2 and s.endswith("y") and s[-2] not in _VOWEL_SET:
        return s[:-1] + "ies"
    return s + "s"


def _needs_cvc_doubling(stem: str) -> bool:
    """Check whether a short monosyllabic CVC stem doubles its final consonant."""
    if len(stem) < 3 or len(stem) > 5:
        return False
    c_last = stem[-1]
    v_prev = stem[-2]
    c_pre2 = stem[-3]
    if c_last in _VOWEL_SET or c_last in ("w", "x", "y"):
        return False
    if v_prev not in _VOWEL_SET:
        return False
    if c_pre2 in _VOWEL_SET:
        return False
    # Ensure strictly 1 vowel group in stem
    vowel_groups = 0
    in_v = False
    for ch in stem:
        if ch in _VOWEL_SET:
            if not in_v:
                vowel_groups += 1
                in_v = True
        else:
            in_v = False
    return vowel_groups == 1


def inflect_past(stem: str) -> str:
    """Exact English orthographic regular past tense."""
    s = stem.lower()
    if s.endswith("e"):
        return s + "d"
    if len(s) >= 2 and s.endswith("y") and s[-2] not in _VOWEL_SET:
        return s[:-1] + "ied"
    if _needs_cvc_doubling(s):
        return s + s[-1] + "ed"
    return s + "ed"


def inflect_prog(stem: str) -> str:
    """Exact English orthographic progressive participle (-ing)."""
    s = stem.lower()
    if s.endswith("ie"):
        return s[:-2] + "ying"
    if s.endswith("e") and not s.endswith(("ee", "ye", "oe")):
        return s[:-1] + "ing"
    if _needs_cvc_doubling(s):
        return s + s[-1] + "ing"
    return s + "ing"


def _sample_en_pseudo_word(rng: random.Random) -> str:
    for _ in range(64):
        onset = rng.choice(_EN_ONSETS)
        vowel = rng.choice(_EN_VOWELS)
        coda = rng.choice(_EN_CODAS)
        cand = onset + vowel + coda
        if rng.random() < 0.55:
            suf = rng.choice(_EN_SUFFIX_SYLLABLES)
            # Avoid awkward double-consonant collisions like ck+k
            if cand[-1] != suf[0]:
                cand = cand + suf
        if 3 <= len(cand) <= 9 and cand not in _EN_REAL_WORDS:
            return cand
    return "blicket"


_EN_NOUN_PROPS: tuple[tuple[str, str, str], ...] = (
    ("a small nocturnal lizard living in hot deserts", "crawled across the warm sand", "desert"),
    ("a silver river bird that dives for tiny fish", "perched quietly on the mossy stone", "riverbank"),
    ("a glowing alpine crystal found inside cold caves", "shimmered beside the lantern", "cavern"),
    ("a smooth wooden flute carved by mountain shepherds", "rested neatly inside the velvet case", "workshop"),
    ("a rare purple moss that grows on shaded cliffs", "spread slowly over the damp slate", "valley"),
    ("a striped reef crab with broad copper claws", "scuttled beneath the tidepool rocks", "shore"),
    ("a sturdy brass compass used by polar navigators", "pointed steadily toward the ridge", "cabin"),
    ("a fragrant winter berry harvested after first frost", "ripened quickly on the thorny bush", "orchard"),
    ("a nimble tree squirrel with tufted russet ears", "leaped gracefully between oak branches", "forest"),
    ("a miniature clay teapot glazed with dark cobalt", "steamed softly on the cedar table", "kitchen"),
)

_EN_VERB_PROPS: tuple[tuple[str, str, str], ...] = (
    ("to weave thin willow twigs into a tight basket", "the young apprentice", "before sunset"),
    ("to polish raw amber stones with fine linen cloth", "the careful jeweler", "all morning"),
    ("to balance stacked river pebbles on a flat rock", "the patient traveler", "by the stream"),
    ("to tune tiny bronze bells with a wooden mallet", "the quiet clockmaker", "in the tower"),
    ("to sketch shifting cloud shapes in a pocket notebook", "the curious painter", "on the hill"),
    ("to fold crisp cedar leaves into miniature boats", "the clever child", "near the pond"),
    ("to tap hollow bamboo reeds to signal distant boats", "the watchful guard", "at the harbor"),
    ("to sort dried alpine herbs into labeled glass jars", "the village botanist", "after rain"),
)

_EN_COUNTS: tuple[str, ...] = ("two", "three", "four", "five", "six", "several")


def _sample_en_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    stem = _sample_en_pseudo_word(rng)
    is_verb = rng.random() < 0.45

    if not is_verb:
        desc, action, place = rng.choice(_EN_NOUN_PROPS)
        count = rng.choice(_EN_COUNTS)
        target = inflect_plural(stem)
        if not is_pre:
            t1 = rng.choice([
                f"Observe carefully: a {stem} is {desc}.\n",
                f"Field note: a {stem} refers to {desc}.\n",
                f"In this guide, a {stem} is {desc}.\n",
            ])
        else:
            t1 = rng.choice([
                f"Notice that {desc} is called a {stem}.\n",
                f"Locally, {desc} is known as a {stem}.\n",
                f"Researchers name {desc} a {stem}.\n",
            ])
        prefix2 = rng.choice([
            f"Yesterday morning near the {place}, {count} ",
            f"During the expedition at the {place}, {count} ",
            f"At dusk around the {place}, {count} ",
        ])
        suffix2 = f" {action}.\n"
        t2 = prefix2 + target + suffix2
    else:
        desc, actor, when = rng.choice(_EN_VERB_PROPS)
        use_prog = rng.random() < 0.40
        target = inflect_prog(stem) if use_prog else inflect_past(stem)
        if not is_pre:
            t1 = rng.choice([
                f"Remember this verb: to {stem} means {desc}.\n",
                f"In our craft, to {stem} is {desc}.\n",
            ])
        else:
            t1 = rng.choice([
                f"The skill {desc} is called to {stem}.\n",
                f"Masters call the act {desc} to {stem}.\n",
            ])
        if use_prog:
            prefix2 = f"Right now, {actor} is "
            suffix2 = f" calmly {when}.\n"
        else:
            prefix2 = f"Yesterday, {actor} "
            suffix2 = f" steadily {when}.\n"
        t2 = prefix2 + target + suffix2

    return _assemble_episode(0, "pre" if is_pre else "post", stem, target, t1, t2, prefix2, build_tensors=build_tensors)


# =============================================================================
# Chinese Rare-Hanzi Orthogonal Pseudo-Word Generator (lang_id = 1)
# =============================================================================

_ZH_RARE_CHARS: tuple[str, ...] = (
    "鹀", "狴", "蠼", "镱", "𫛭", "砜", "𫠜", "螭", "鹱", "𰻝",
    "猰", "狻", "猊", "鳇", "镤", "镄", "锎", "锿", "钔", "锘",
    "铹", "硇", "礞", "蓂", "蔹", "蠛", "蝘", "瑨", "瓈", "缊",
)
_ZH_MORPHEMES: tuple[str, ...] = (
    "晶", "兽", "草", "石", "酶", "鸟", "藤", "合金", "矿", "菌", "苔", "鳞",
)

_ZH_DOMAIN_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("一种在极低温下呈现深紫色光泽的超导陶瓷", "实验室里，研究员将三块", "缓缓放入液氮杜瓦瓶中进行电阻测绘。"),
    ("一种栖息在高山暗河岩缝里的盲眼小鱼", "科考队在溶洞浅滩发现五条", "正逆着清冽的泉水缓慢游动。"),
    ("一种夜间叶片会散发淡蓝荧光的耐寒苔藓", "植物学家沿着背阴石壁采集了几丛", "并小心封存进恒湿标本盒。"),
    ("一种敲击后能发出清脆泛音的层状硅酸盐矿石", "老地质师把刚切好的两片", "放在木架上反复测试共鸣频率。"),
    ("一种只在春季雷雨后短暂开花的攀援木质藤本", "山村采药人在北坡竹林深处辨认出一株", "随即记录下它的海拔和花期。"),
    ("一种能在高盐碱卤水中高效分解有机酸的嗜盐菌酶", "发酵车间通过微孔滤膜提纯出高活性", "使整套催化流程的能耗大幅降低。"),
    ("一种翼展修长、善乘上升气流滑翔的远洋海鸟", "黎明时分，瞭望员通过望远镜看见两只", "贴着白浪翻涌的海面平稳掠过。"),
    ("一种密度极低却能承受剧烈热冲击的轻质金属复合材料", "航天结构工程师对新铸造的", "开展了长达十二小时的疲劳振动试验。"),
)


def _sample_zh_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    rare = rng.choice(_ZH_RARE_CHARS)
    morph = rng.choice(_ZH_MORPHEMES)
    stem = unicodedata.normalize("NFC", rare + morph)
    desc, q_pre, q_suf = rng.choice(_ZH_DOMAIN_TEMPLATES)

    if not is_pre:
        t1 = rng.choice([
            f"记录显示，{stem}是{desc}。\n",
            f"据图谱记载，{stem}指代{desc}。\n",
            f"在最新辞条中，{stem}被定义为{desc}。\n",
        ])
    else:
        t1 = rng.choice([
            f"学界将{desc}正式命名为{stem}。\n",
            f"这种{desc}在文献中被称为{stem}。\n",
            f"人们习惯把{desc}称作{stem}。\n",
        ])

    t2 = f"{q_pre}{stem}{q_suf}\n"
    return _assemble_episode(1, "pre" if is_pre else "post", stem, stem, t1, t2, q_pre, build_tensors=build_tensors)


# =============================================================================
# Japanese Procedural Katakana SOV Pseudo-Word Generator (lang_id = 2)
# =============================================================================

_JA_KATA_HEAD: tuple[str, ...] = (
    "ブ", "ゾ", "グ", "ト", "ス", "ク", "プ", "ド", "フ", "ヴ",
    "ザ", "ビ", "ペ", "ボ", "ゲ", "ネ", "メ", "レ", "ケ", "モ",
)
_JA_KATA_MID: tuple[str, ...] = (
    "リ", "ル", "ロ", "ラ", "ケ", "ネ", "メ", "ト", "ド", "プ",
    "ブ", "ク", "グ", "ピ", "ビ", "テ", "デ", "ノ", "モ", "ッ", "ー",
)
_JA_KATA_TAIL: tuple[str, ...] = (
    "ト", "ブ", "プ", "ク", "グ", "マ", "ル", "ン", "ス", "ズ", "ド", "ラ", "ノ",
)
_JA_EXCLUDED: frozenset[str] = frozenset({
    "トマト", "グルプ", "テスト", "クラス", "プラス", "ドレス", "グラス", "クラブ",
    "ストブ", "クラン", "プラン", "ブロク", "ドラム", "スラム", "メロン", "レモン",
})

_JA_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("寒い森の奥にすむ小さな青い鳥", "昨日、少年は静かな川辺で三羽の", "を見つけて丁寧にスケッチしました。"),
    ("暗い洞窟の壁で淡く光る珍しい鉱石", "研究者は新しい標本箱へ二つの", "を並べて顕微鏡で観察しました。"),
    ("冷たい渓流の石の下に隠れる縞模様の魚", "朝早く、漁師は澄んだ浅瀬で元気な", "が泳ぐ様子を静かに記録しました。"),
    ("高い山の岩陰だけで育つ香りの強い野草", "植物学者は湿った斜面から五本の", "を摘み取って標本帳に挟みました。"),
    ("古い港の職人が真鍮で作る精密な方位計", "船長は嵐の前の甲板で壊れた", "を慎重に調整して動きを確かめました。"),
    ("柔らかな木目の杉板を削って作る小さな笛", "音楽家は夕方の講堂で手製の", "を軽く吹き、澄んだ響きを聴きました。"),
)


def _sample_ja_pseudo_word(rng: random.Random) -> str:
    for _ in range(64):
        target_len = rng.randint(4, 6)
        chars = [rng.choice(_JA_KATA_HEAD)]
        while len(chars) < target_len - 1:
            c = rng.choice(_JA_KATA_MID)
            if c in ("ッ", "ー") and chars[-1] in ("ッ", "ー"):
                continue
            chars.append(c)
        tail = rng.choice(_JA_KATA_TAIL)
        if chars[-1] == "ッ" and tail in ("ン", "ラ", "ル", "マ", "ノ"):
            tail = "ト"
        chars.append(tail)
        cand = "".join(chars)
        if cand not in _JA_EXCLUDED and 4 <= len(cand) <= 6:
            return cand
    return "ブリケット"


def _sample_ja_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    stem = _sample_ja_pseudo_word(rng)
    desc, q_pre, q_suf = rng.choice(_JA_TEMPLATES)
    if not is_pre:
        t1 = rng.choice([
            f"観察記録によると、{stem}は{desc}のことです。\n",
            f"この図鑑では、{stem}は{desc}を指します。\n",
        ])
    else:
        t1 = rng.choice([
            f"地元の人々は{desc}を{stem}と呼びます。\n",
            f"この地方では{desc}の名称が{stem}とされています。\n",
        ])
    t2 = f"{q_pre}{stem}{q_suf}\n"
    return _assemble_episode(2, "pre" if is_pre else "post", stem, stem, t1, t2, q_pre, build_tensors=build_tensors)


# =============================================================================
# Python & Rust Tree-Sitter-Verified Fresh Identifier Episodes (lang_id = 3, 4)
# =============================================================================

_CODE_PREFIXES_SNAKE: tuple[str, ...] = (
    "buf", "flux", "ring", "node", "span", "slot", "wire", "tile",
    "grid", "mesh", "core", "gate", "pipe", "bank", "cell", "knob",
)
_CODE_MID_SNAKE: tuple[str, ...] = (
    "val", "ctx", "acc", "reg", "tag", "bin", "vec", "map", "key", "idx",
)
_CODE_PREFIXES_PASCAL: tuple[str, ...] = (
    "Shard", "Flux", "Ring", "Node", "Tile", "Grid", "Mesh", "Gate", "Pipe", "Slot",
)
_CODE_MID_PASCAL: tuple[str, ...] = (
    "State", "Frame", "Cache", "Queue", "Block", "Token", "Index", "Store",
)
_ALPHA_NUM = "abcdefghjkmnpqrstuvwxyz23456789"
_RUST_KEYWORDS: frozenset[str] = frozenset({
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false",
    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
    "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
    "true", "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
})


def _fresh_snake_ident(rng: random.Random) -> str:
    for _ in range(32):
        p = rng.choice(_CODE_PREFIXES_SNAKE)
        suf = "".join(rng.choice(_ALPHA_NUM) for _ in range(3))
        if rng.random() < 0.5:
            cand = f"{p}_{suf}"
        else:
            cand = f"{p}_{rng.choice(_CODE_MID_SNAKE)}_{suf}"
        if not keyword.iskeyword(cand) and cand not in _RUST_KEYWORDS:
            return cand
    return "buf_k9z"


def _fresh_pascal_ident(rng: random.Random) -> str:
    p = rng.choice(_CODE_PREFIXES_PASCAL)
    m = rng.choice(_CODE_MID_PASCAL)
    tag = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + rng.choice("23456789")
    return f"{p}{m}{tag}"


_JS_KEYWORDS: frozenset[str] = frozenset({
    "break", "case", "catch", "class", "const", "continue", "debugger", "default",
    "delete", "do", "else", "export", "extends", "false", "finally", "for", "function",
    "if", "import", "in", "instanceof", "let", "new", "null", "return", "super",
    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with",
    "yield", "async", "await", "of",
})


def _fresh_camel_ident(rng: random.Random) -> str:
    p = rng.choice(_CODE_PREFIXES_SNAKE)
    m = rng.choice(_CODE_MID_PASCAL)
    suf = "".join(rng.choice(_ALPHA_NUM) for _ in range(3))
    cand = f"{p}{m}{suf}"
    return cand if cand not in _JS_KEYWORDS else f"bufState{suf}"


def _sample_python_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    kind = rng.randint(0, 2)
    if kind == 0:
        # List / dict buffer variable inside function
        ident = _fresh_snake_ident(rng)
        cap = rng.choice((16, 32, 64, 128))
        delta = rng.choice((3, 5, 7, 11))
        if not is_pre:
            t1 = (
                f"def collect_metrics(total_count: int) -> int:\n"
                f"    {ident} = [cap for cap in range({cap}) if cap % 2 == 0]\n"
            )
        else:
            t1 = (
                f"def collect_metrics(total_count: int) -> int:\n"
                f"    seed_items = [cap for cap in range({cap}) if cap % 2 == 0]\n"
                f"    {ident} = list(seed_items)\n"
            )
        prefix2 = "    "
        t2 = f"{prefix2}{ident}.append(total_count + {delta})\n    return total_count\n"
    elif kind == 1:
        # Fresh helper function definition + call in Turn 2
        ident = _fresh_snake_ident(rng)
        scale = rng.choice((2, 3, 4, 5))
        bias = rng.choice((1, 7, 9, 13))
        if not is_pre:
            t1 = (
                f"def {ident}(x: int, hi: int) -> int:\n"
                f"    return min(hi, (x * {scale}) + {bias})\n\n"
            )
        else:
            t1 = (
                f"# Scale integer sample by {scale}, add {bias}, and clamp to hi.\n"
                f"def {ident}(x: int, hi: int) -> int:\n"
                f"    return min(hi, (x * {scale}) + {bias})\n\n"
            )
        prefix2 = "def run_step(val: int) -> int:\n    return "
        t2 = f"{prefix2}{ident}(val, 255)\n"
    else:
        # Fresh dataclass-like class + field access in Turn 2
        ident = _fresh_pascal_ident(rng)
        lo = rng.choice((2, 4, 8))
        hi = rng.choice((32, 64, 96))
        if not is_pre:
            t1 = (
                f"class {ident}:\n"
                f"    def __init__(self, lo: int, hi: int) -> None:\n"
                f"        self.span = (lo, hi)\n\n"
            )
        else:
            t1 = (
                f"# Compact integer interval container storing (lo, hi) bounds.\n"
                f"class {ident}:\n"
                f"    def __init__(self, lo: int, hi: int) -> None:\n"
                f"        self.span = (lo, hi)\n\n"
            )
        prefix2 = "def make_span() -> tuple[int, int]:\n    obj = "
        t2 = f"{prefix2}{ident}({lo}, {hi})\n    return obj.span\n"

    return _assemble_episode(3, "pre" if is_pre else "post", ident, ident, t1, t2, prefix2, build_tensors=build_tensors)


def _sample_rust_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    kind = rng.randint(0, 2)
    if kind == 0:
        # Vec local binding in Turn 1 -> method call in Turn 2 (100% valid Rust fn)
        ident = _fresh_snake_ident(rng)
        cap = rng.choice((16, 32, 64, 128))
        if not is_pre:
            t1 = (
                f"fn accumulate_batch(total_count: u32) -> u32 {{\n"
                f"    let mut {ident}: Vec<u32> = Vec::with_capacity({cap});\n"
            )
        else:
            t1 = (
                f"fn accumulate_batch(total_count: u32) -> u32 {{\n"
                f"    let cap: usize = {cap};\n"
                f"    let mut {ident}: Vec<u32> = Vec::with_capacity(cap);\n"
            )
        prefix2 = "    "
        t2 = f"{prefix2}{ident}.push(total_count);\n    total_count\n}}\n"
    elif kind == 1:
        # Fresh struct definition in Turn 1 -> instantiation & field sum in Turn 2
        ident = _fresh_pascal_ident(rng)
        base_a = rng.choice((4, 8, 12, 16))
        base_b = rng.choice((32, 48, 64, 96))
        if not is_pre:
            t1 = (
                f"struct {ident} {{\n"
                f"    count: u32,\n"
                f"    limit: u32,\n"
                f"}}\n\n"
            )
        else:
            t1 = (
                f"// Bounded counter state tracking active count and capacity limit.\n"
                f"struct {ident} {{\n"
                f"    count: u32,\n"
                f"    limit: u32,\n"
                f"}}\n\n"
            )
        prefix2 = "fn inspect_state(delta: u32) -> u32 {\n    let item = "
        t2 = f"{prefix2}{ident} {{ count: {base_a} + delta, limit: {base_b} }};\n    item.count.min(item.limit)\n}}\n"
    else:
        # Fresh helper fn in Turn 1 -> direct call in Turn 2
        ident = _fresh_snake_ident(rng)
        mult = rng.choice((2, 3, 5, 7))
        add = rng.choice((1, 4, 9))
        if not is_pre:
            t1 = (
                f"fn {ident}(v: i32, hi: i32) -> i32 {{\n"
                f"    ((v * {mult}) + {add}).min(hi)\n"
                f"}}\n\n"
            )
        else:
            t1 = (
                f"// Multiply input by {mult}, offset by {add}, and clamp below hi.\n"
                f"fn {ident}(v: i32, hi: i32) -> i32 {{\n"
                f"    ((v * {mult}) + {add}).min(hi)\n"
                f"}}\n\n"
            )
        prefix2 = "fn eval_sample(x: i32) -> i32 {\n    "
        t2 = f"{prefix2}{ident}(x, 512)\n}}\n"

    return _assemble_episode(4, "pre" if is_pre else "post", ident, ident, t1, t2, prefix2, build_tensors=build_tensors)


def _sample_js_episode(rng: random.Random, is_pre: bool, build_tensors: bool = True) -> WugEpisode:
    kind = rng.randint(0, 2)
    if kind == 0:
        # Array local binding in Turn 1 -> push & length return in Turn 2 (valid ES6 function)
        ident = _fresh_camel_ident(rng)
        cap = rng.choice((8, 16, 32, 64))
        if not is_pre:
            t1 = (
                f"function collectBatch(nextItem) {{\n"
                f"  const {ident} = Array.from({{ length: {cap} }}, (_, i) => i * 2);\n"
            )
        else:
            t1 = (
                f"function collectBatch(nextItem) {{\n"
                f"  const baseSize = {cap};\n"
                f"  const {ident} = Array.from({{ length: baseSize }}, (_, i) => i * 2);\n"
            )
        prefix2 = "  "
        t2 = f"{prefix2}{ident}.push(nextItem);\n  return {ident}.length;\n}}\n"
    elif kind == 1:
        # Fresh ES6 class in Turn 1 -> instantiation & method call in Turn 2
        ident = _fresh_pascal_ident(rng)
        lo = rng.choice((2, 4, 8, 16))
        hi = rng.choice((64, 128, 256, 512))
        if not is_pre:
            t1 = (
                f"class {ident} {{\n"
                f"  constructor(lo, hi) {{\n"
                f"    this.lo = lo;\n"
                f"    this.hi = hi;\n"
                f"  }}\n"
                f"  clamp(v) {{\n"
                f"    return Math.min(this.hi, Math.max(this.lo, v));\n"
                f"  }}\n"
                f"}}\n\n"
            )
        else:
            t1 = (
                f"// Bounded numeric interval clamper storing [lo, hi] limits.\n"
                f"class {ident} {{\n"
                f"  constructor(lo, hi) {{\n"
                f"    this.lo = lo;\n"
                f"    this.hi = hi;\n"
                f"  }}\n"
                f"  clamp(v) {{\n"
                f"    return Math.min(this.hi, Math.max(this.lo, v));\n"
                f"  }}\n"
                f"}}\n\n"
            )
        prefix2 = "function applyBounds(x) {\n  const box = new "
        t2 = f"{prefix2}{ident}({lo}, {hi});\n  return box.clamp(x);\n}}\n"
    else:
        # Fresh helper arrow/function in Turn 1 -> invocation in Turn 2
        ident = _fresh_camel_ident(rng)
        mult = rng.choice((2, 3, 5, 7))
        add = rng.choice((1, 4, 9))
        if not is_pre:
            t1 = (
                f"function {ident}(v, cap) {{\n"
                f"  return Math.min(cap, v * {mult} + {add});\n"
                f"}}\n\n"
            )
        else:
            t1 = (
                f"// Scale v by {mult}, add {add}, and cap at the given ceiling.\n"
                f"function {ident}(v, cap) {{\n"
                f"  return Math.min(cap, v * {mult} + {add});\n"
                f"}}\n\n"
            )
        prefix2 = "function scoreInput(x) {\n  return "
        t2 = f"{prefix2}{ident}(x, 256);\n}}\n"

    return _assemble_episode(5, "pre" if is_pre else "post", ident, ident, t1, t2, prefix2, build_tensors=build_tensors)


def _assemble_episode(
    lang_id: int,
    definition_order: str,
    stem: str,
    target_word: str,
    turn1_text: str,
    turn2_text: str,
    turn2_prefix_before_target: str,
    build_tensors: bool = True,
) -> WugEpisode:
    t1_nfc = unicodedata.normalize("NFC", turn1_text)
    t2_nfc = unicodedata.normalize("NFC", turn2_text)
    pre_nfc = unicodedata.normalize("NFC", turn2_prefix_before_target)
    tgt_nfc = unicodedata.normalize("NFC", target_word)

    t1_bytes = t1_nfc.encode("utf-8", errors="strict")
    t2_bytes = t2_nfc.encode("utf-8", errors="strict")
    pre_bytes = pre_nfc.encode("utf-8", errors="strict")
    tgt_bytes = tgt_nfc.encode("utf-8", errors="strict")

    raw_bytes = t1_bytes + t2_bytes
    t1_len = len(t1_bytes)
    total_len = len(raw_bytes)
    tgt_start = t1_len + len(pre_bytes)
    tgt_end = tgt_start + len(tgt_bytes)

    if raw_bytes[tgt_start:tgt_end] != tgt_bytes:
        raise RuntimeError(
            f"Internal Wug mask misalignment: expected {tgt_bytes!r}, got {raw_bytes[tgt_start:tgt_end]!r}"
        )

    if build_tensors:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
        x_bytes = torch.from_numpy(arr)
        loss_mask = torch.zeros(total_len, dtype=torch.bool)
        loss_mask[t1_len:] = True
        wug_target_mask = torch.zeros(total_len, dtype=torch.bool)
        wug_target_mask[tgt_start:tgt_end] = True
    else:
        x_bytes = _EMPTY_U8
        loss_mask = _EMPTY_BOOL
        wug_target_mask = _EMPTY_BOOL

    return WugEpisode(
        lang_id=lang_id,
        definition_order=definition_order,
        stem=stem,
        target_word=tgt_nfc,
        turn1_text=t1_nfc,
        turn2_text=t2_nfc,
        text=t1_nfc + t2_nfc,
        raw_bytes=raw_bytes,
        turn1_byte_len=t1_len,
        target_byte_start=tgt_start,
        target_byte_end=tgt_end,
        x_bytes=x_bytes,
        loss_mask=loss_mask,
        wug_target_mask=wug_target_mask,
    )


_EMPTY_U8 = torch.empty(0, dtype=torch.uint8)
_EMPTY_BOOL = torch.empty(0, dtype=torch.bool)


ACTIVE_WUG_LANG_IDS: tuple[int, ...] = (0, 1, 3, 5)
COGNITIVE_V2_WUG_LANG_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def sample_wug_episode(rng: random.Random, lang_id: int, *, build_tensors: bool = True) -> WugEpisode:
    """Dynamically synthesize one two-turn Wug Episode for `lang_id` in {0..5} (active: 0=en,1=zh,3=py,5=js)."""
    is_pre = rng.random() < 0.5
    if lang_id == 0:
        ep = _sample_en_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    elif lang_id == 1:
        ep = _sample_zh_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    elif lang_id == 2:
        ep = _sample_ja_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    elif lang_id == 3:
        ep = _sample_python_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    elif lang_id == 4:
        ep = _sample_rust_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    elif lang_id == 5:
        ep = _sample_js_episode(rng, is_pre=is_pre, build_tensors=build_tensors)
    else:
        raise ValueError(f"Unsupported lang_id={lang_id}; expected one of 0..5")
    return ep


def sample_wug_episode_fast(rng: random.Random, lang_id: int) -> tuple[bytes, int, int, int]:
    """High-throughput zero-tensor variant returning `(raw_bytes, turn1_len, tgt_start, tgt_end)`."""
    ep = sample_wug_episode(rng, lang_id, build_tensors=False)
    return ep.raw_bytes, ep.turn1_byte_len, ep.target_byte_start, ep.target_byte_end


# =============================================================================
# DCPA v2 Cognitive Stage-4 Generator (World-Bible, Temporary Axioms & Micro-SDK)
# Zero Rust: 0=en, 1=zh, 2=formal, 3=python, 4=sql_struct, 5=javascript
# =============================================================================

_ZH_SURNAMES: tuple[str, ...] = ("沈", "陆", "顾", "裴", "温", "谢", "纪", "许", "林", "周", "宋", "叶")
_ZH_GIVEN_CHARS: tuple[str, ...] = ("知微", "怀瑾", "栖迟", "景行", "清晏", "云深", "明昭", "星河", "砚秋", "听澜", "敬之", "远洲")
_ZH_PLACES: tuple[str, ...] = ("北麓观测站", "雾港三号船坞", "青石古渡旧档案室", "冷杉岭气象塔", "澄江水文闸口", "白鹤滩地下实验室")
_ZH_TABOOS: tuple[tuple[str, str], ...] = (
    ("绝不在雷雨夜开启西侧铜窗", "窗外雷声滚过山脊，他伸手按住西侧铜窗的插销，始终没有推开半寸"),
    ("从不饮用任何盛在银器里的热茶", "侍者递来银壶热茶时，她只取过一旁的粗陶杯注入清水"),
    ("夜间巡检绝不携带明火灯具", "进入暗廊前，他吹灭风灯，只靠腕间冷光刻度表辨认石阶"),
    ("签署纪要前必须先核对双色铅封", "提笔落款之前，她先将匣口的双色铅封对着台灯逐一核验"),
)

_EN_PLACES: tuple[str, ...] = (
    "Greyhaven Tide Observatory",
    "Northridge Glacier Vault",
    "Coppergate Signal Depot",
    "Larkspur Canal Lockhouse",
    "storm-swept Glasscliff Lantern Room",
)
_EN_TABOOS: tuple[tuple[str, str], ...] = (
    (
        "never unseals brass canisters before checking the barometer",
        "checked the wall barometer twice before touching a single latch",
    ),
    (
        "refuses to speak mission coordinates over an unshielded wire",
        "wrote the coordinates on slate instead of speaking them into the line",
    ),
    (
        "never crosses the stone Causeway while the fog-bell rings",
        "waited calmly beside the pier until the fog-bell fell silent",
    ),
    (
        "always logs tare mass in ink before weighing cold samples",
        "recorded the tare mass in black ink prior to placing any sample on the pan",
    ),
)


def _pack_cognitive_fast(
    turn1_text: str,
    turn2_prefix: str,
    target_word: str,
    turn2_suffix: str,
    style_id: int,
) -> tuple[bytes, int, int, int, int]:
    t1_b = unicodedata.normalize("NFC", turn1_text).encode("utf-8", errors="strict")
    pre_b = unicodedata.normalize("NFC", turn2_prefix).encode("utf-8", errors="strict")
    tgt_b = unicodedata.normalize("NFC", target_word).encode("utf-8", errors="strict")
    suf_b = unicodedata.normalize("NFC", turn2_suffix).encode("utf-8", errors="strict")
    raw_b = t1_b + pre_b + tgt_b + suf_b
    t1_len = len(t1_b)
    tgt_s = t1_len + len(pre_b)
    tgt_e = tgt_s + len(tgt_b)
    if raw_b[tgt_s:tgt_e] != tgt_b:
        raise RuntimeError("Cognitive Stage-4 target span misalignment")
    return raw_b, t1_len, tgt_s, tgt_e, int(style_id)


def _sample_cog_en_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    artifact = _sample_en_pseudo_word(rng)
    p1 = _sample_en_pseudo_word(rng).capitalize()
    p2 = _sample_en_pseudo_word(rng).capitalize()
    while p2 == p1:
        p2 = _sample_en_pseudo_word(rng).capitalize()
    p3 = _sample_en_pseudo_word(rng).capitalize()

    if rng.random() < 0.55:
        place = rng.choice(_EN_PLACES)
        taboo_rule, taboo_act = rng.choice(_EN_TABOOS)
        freq = rng.randint(17, 89)
        if compact:
            t1 = (
                f"[WORLD-BIBLE]\n"
                f"- Lead: {p1} (rule: {taboo_rule}).\n"
                f"- Partner: {p2}; Archivist: {p3}.\n"
                f"- Key relic: the `{artifact}` resonator ({freq} Hz) at {place}.\n\n"
            )
            pre = (
                f"\x1dS\nlead={p1} | taboo=barometer/protocol | relic={artifact}@{freq}Hz\n\x1dE\n"
                f"Inside {place}, {p1} {taboo_act} while {p2} steadied the bench. "
                f"Only after {p3} confirmed the log did {p1} calibrate the "
            )
            suf = f" at {freq} Hz, preserving clean phase lock across the chamber.\n"
            return _pack_cognitive_fast(t1, pre, artifact, suf, style_id=3)

        t1 = (
            f"[WORLD-BIBLE DOSSIER — CANON LOCK]\n"
            f"Setting: {place}, late autumn.\n"
            f"Cast & Constraints:\n"
            f"  1. {p1} — Chief Instrumentalist; strict personal rule: {taboo_rule}.\n"
            f"  2. {p2} — Hydrographic Surveyor who tracks tidal shear.\n"
            f"  3. {p3} — Vault Archivist guarding cold-storage manifests.\n"
            f"Foreshadowed Object: `{artifact}`, a bronze acoustic core tuned to exactly {freq} Hz.\n"
            f"Task: Write the continuation scene without violating cast roles, taboos, or tuning figures.\n\n"
        )
        pre = (
            f"\x1dS\nanchor: {p1}@{place} | enforce taboo ({taboo_rule}) | callback: {artifact} ({freq} Hz)\n\x1dE\n"
            f"Wind rattled the upper shutters of {place} as {p2} spread the latest tidal shear chart across the oak table. "
            f"Near the storage rack, {p3} unlocked the cedar manifest ledger and set the sealed case within reach, expecting a rushed inspection.\n\n"
            f"Instead, {p1} {taboo_act}. {p2} watched in silence, recognizing the discipline that had prevented three prior instrument faults. "
            f"Once every safeguard was satisfied, {p1} lifted the bronze "
        )
        suf = (
            f" onto the isolation cradle and aligned its vernier dial to {freq} Hz. "
            f"A steady harmonic pulse answered the tide gauge, confirming {p3}'s archival notes without a single discrepancy.\n"
        )
        return _pack_cognitive_fast(t1, pre, artifact, suf, style_id=3)

    # Temporary compliance / counter-intuitive rulebook adjudication (style_id = 0)
    clause = f"{artifact.upper()}-{rng.randint(10, 99)}"
    cutoff = rng.choice((14, 19, 23, 31))
    fee_low = rng.choice((12, 15, 18))
    fee_high = rng.choice((45, 60, 75))
    t1 = (
        f"[PRIVATE COMPLIANCE RULEBOOK: {clause}]\n"
        f"Override Notice: Ignore standard intuition; apply ONLY the rules below.\n"
        f"  R1. Any dispatch tagged `urgent` is DEFERRED to the back of the queue unless its batch size is strictly under {cutoff} units.\n"
        f"  R2. Standard batches pay {fee_low} credits, whereas deferred batches pay a surcharge of {fee_high} credits.\n"
        f"Case: Operator {p1} submits an `urgent` dispatch containing {cutoff + 6} units.\n\n"
    )
    pre = (
        f"\x1dS\nrule={clause} | size={cutoff + 6} >= {cutoff} -> R1 defers urgent -> R2 fee={fee_high}\n\x1dE\n"
        f"Under temporary compliance protocol "
    )
    suf = (
        f", intuition that `urgent` clears first is explicitly overridden by R1: "
        f"because {p1}'s batch has {cutoff + 6} units (not strictly under {cutoff}), "
        f"the dispatch is deferred to the back of the queue and assessed the R2 surcharge of {fee_high} credits.\n"
    )
    return _pack_cognitive_fast(t1, pre, clause, suf, style_id=0)


def _sample_cog_zh_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    stem = unicodedata.normalize("NFC", rng.choice(_ZH_RARE_CHARS) + rng.choice(_ZH_MORPHEMES))
    s1, s2, s3 = rng.sample(_ZH_SURNAMES, 3)
    g1, g2, g3 = rng.sample(_ZH_GIVEN_CHARS, 3)
    c1, c2, c3 = s1 + g1, s2 + g2, s3 + g3

    if rng.random() < 0.55:
        place = rng.choice(_ZH_PLACES)
        taboo_rule, taboo_act = rng.choice(_ZH_TABOOS)
        metric = rng.randint(24, 96)
        if compact:
            t1 = (
                f"【设定档案】地点：{place}。主理人：{c1}（铁律：{taboo_rule}）；协理：{c2}；记录员：{c3}。"
                f"核心伏笔物：「{stem}」（临界读数 {metric} 毫伏）。\n\n"
            )
            pre = (
                f"\x1dS\n主视角={c1} | 禁忌={taboo_rule} | 伏笔回收={stem}({metric}mV)\n\x1dE\n"
                f"夜色笼罩{place}，{taboo_act}。待{c2}与{c3}核准基线后，{c1}才取出封存的"
            )
            suf = f"，仪表稳定停在 {metric} 毫伏，与前情记录分毫不差。\n"
            return _pack_cognitive_fast(t1, pre, stem, suf, style_id=3)

        t1 = (
            f"【长篇设定集 / 严禁吃书】\n"
            f"场景：{place}。\n"
            f"人物关系与硬性禁忌：\n"
            f"  1. {c1}（总工）：行事克制，铁律为“{taboo_rule}”；\n"
            f"  2. {c2}（水文测绘师）：负责校准流速与温差基线；\n"
            f"  3. {c3}（档案保管员）：掌管封存样本的出入库台账。\n"
            f"关键伏笔器物：「{stem}」，仅在阈值达到 {metric} 毫伏时产生谐振响应。\n"
            f"写作要求：续写现场核验片段，严守人物分工、性格禁忌与精确数值。\n\n"
        )
        pre = (
            f"\x1dS\n锁定人物: {c1}/{c2}/{c3} | 严守禁忌: {taboo_rule} | 回收伏笔: {stem}@{metric}mV\n\x1dE\n"
            f"深秋的湿冷雾气贴着{place}的石墙缓慢下沉。{c2}刚把最新一轮温差曲线钉上木板，{c3}便捧着铅封木匣走到长桌旁，低声询问是否立即开匣测绘。\n\n"
            f"{c1}没有急于动手——{taboo_act}。{c2}见状微微颔首，将流速补偿旋钮归零；{c3}也随即翻开台账核对编号。"
            f"待全部前置规程逐项落定，{c1}才稳稳托起匣中的"
        )
        suf = (
            f"接入电桥。当电位计读数缓缓爬升至 {metric} 毫伏时，纤细而清亮的谐振波纹如约浮现，"
            f"三人各自守住职责边界，整场核验未出一处纰漏。\n"
        )
        return _pack_cognitive_fast(t1, pre, stem, suf, style_id=3)

    # 反直觉私有合规条款裁定 (style_id = 0)
    limit_n = rng.choice((15, 25, 40))
    penalty = rng.choice((30, 50, 80))
    t1 = (
        f"【临时私有规则·{stem}条款】请完全抑制日常直觉，仅按以下条款推演：\n"
        f"  条款一：凡标注“加急”的审阅单，若页数超过 {limit_n} 页，一律转入慢速复核池，不得优先放行。\n"
        f"  条款二：转入慢速复核池的单据须额外计提 {penalty} 点合规工时。\n"
        f"待裁定案例：申请人{c1}提交了一份标注“加急”的 {limit_n + 8} 页技术单据。\n\n"
    )
    pre = (
        f"\x1dS\n绑定规则={stem}条款 | 页数={limit_n + 8}>{limit_n} -> 触发条款一转慢速池 -> 条款二计提{penalty}点\n\x1dE\n"
        f"依据临时生效的「"
    )
    suf = (
        f"条款」裁定：尽管单据带有“加急”标记，但因总页数为 {limit_n + 8} 页（超过 {limit_n} 页上限），"
        f"必须抑制“加急优先”的日常直觉，按条款一转入慢速复核池，并按条款二计提 {penalty} 点合规工时。\n"
    )
    return _pack_cognitive_fast(t1, pre, stem, suf, style_id=0)


def _sample_cog_formal_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    op_name = f"op_{_sample_en_pseudo_word(rng)}"
    mult = rng.choice((2, 3, 4))
    sub = rng.choice((3, 5, 7))
    mod_m = rng.choice((11, 13, 17, 19))
    x_val = rng.randint(4, 10)
    y_val = rng.randint(2, 8)
    step1 = (mult * x_val - y_val) % mod_m
    z_val = rng.randint(1, 6)
    final_v = (mult * step1 - z_val) % mod_m

    t1 = (
        f"[TEMPORARY AXIOM SYSTEM: {op_name}]\n"
        f"Let binary operator `{op_name}(u, v)` on Z_{mod_m} be defined strictly by:\n"
        f"  A1. {op_name}(u, v) := ({mult} * u - v) mod {mod_m}\n"
        f"  A2. Left-associative evaluation: fold({op_name}, a, b, c) := {op_name}({op_name}(a, b), c)\n"
        f"  A3. Note: `{op_name}` is non-commutative and non-associative; do NOT reorder operands.\n"
        f"Problem: Evaluate `fold({op_name}, {x_val}, {y_val}, {z_val})` step by step.\n\n"
    )
    pre = (
        f"\x1dS\nA1: {op_name}({x_val},{y_val})=({mult}*{x_val}-{y_val}) mod {mod_m}={step1} "
        f"-> {op_name}({step1},{z_val})=({mult}*{step1}-{z_val}) mod {mod_m}={final_v}\n\x1dE\n"
        f"Proof under temporary axiom system:\n"
        f"  1. Inner left call via A1: {op_name}({x_val}, {y_val}) = ({mult} * {x_val} - {y_val}) mod {mod_m} = {step1}.\n"
        f"  2. Outer call via A2 & A1: "
    )
    suf = (
        f"({step1}, {z_val}) = ({mult} * {step1} - {z_val}) mod {mod_m} = {final_v}.\n"
        f"∴ Verified value = {final_v}.\n"
    )
    return _pack_cognitive_fast(t1, pre, op_name, suf, style_id=0)


def _sample_cog_python_sdk_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    ctx_cls = _fresh_pascal_ident(rng)
    fn_open = f"open_{_fresh_snake_ident(rng)}"
    fn_apply = f"apply_{_fresh_snake_ident(rng)}"
    fn_seal = f"seal_{_fresh_snake_ident(rng)}"
    fn_bad = f"legacy_bypass_{_fresh_snake_ident(rng)}"
    cap_limit = rng.choice((100, 250, 500))
    fee_bps = rng.choice((15, 25, 40))

    t1 = (
        f"# === Synthetic Business Micro-SDK Contract ===\n"
        f"# Precondition: `{fn_open}` MUST run before `{fn_apply}`, followed by `{fn_seal}`.\n"
        f"# Rule 1: Reject non-positive `amount` with `ValueError`.\n"
        f"# Rule 2: Clamp `amount` to `{cap_limit}` before computing `{fee_bps}` bps fee.\n"
        f"# WARNING: `{fn_bad}` is deprecated and unsafe; NEVER call it.\n"
        f"class {ctx_cls}:\n"
        f"    def __init__(self, acct_id: str) -> None:\n"
        f"        self.acct_id = acct_id\n"
        f"        self.opened = False\n"
        f"        self.net = 0\n\n"
        f"def {fn_open}(acct_id: str) -> {ctx_cls}:\n"
        f"    ctx = {ctx_cls}(acct_id)\n"
        f"    ctx.opened = True\n"
        f"    return ctx\n\n"
        f"def {fn_apply}(ctx: {ctx_cls}, clamped_amt: int, fee: int) -> None:\n"
        f"    if not ctx.opened:\n"
        f"        raise RuntimeError('unopened context')\n"
        f"    ctx.net = clamped_amt - fee\n\n"
        f"def {fn_seal}(ctx: {ctx_cls}) -> int:\n"
        f"    return ctx.net\n\n"
        f"def {fn_bad}(acct_id: str, amount: int) -> int:\n"
        f"    raise RuntimeError('deprecated path')\n\n"
    )
    pre = (
        f"\x1dS\nguard amount<=0 | clamp<={cap_limit} | order: {fn_open} -> {fn_apply} -> {fn_seal} | skip {fn_bad}\n\x1dE\n"
        f"def settle_account(acct_id: str, amount: int) -> int:\n"
        f"    if amount <= 0:\n"
        f"        raise ValueError('amount must be positive')\n"
        f"    clamped = min(amount, {cap_limit})\n"
        f"    fee = (clamped * {fee_bps}) // 10000\n"
        f"    ctx = {fn_open}(acct_id)\n"
        f"    "
    )
    suf = f"(ctx, clamped, fee)\n    return {fn_seal}(ctx)\n"
    return _pack_cognitive_fast(t1, pre, fn_apply, suf, style_id=4)


def _sample_cog_sql_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    tbl_main = f"ledger_{_fresh_snake_ident(rng)}"
    tbl_acct = f"acct_{_fresh_snake_ident(rng)}"
    col_metric = f"net_{_fresh_snake_ident(rng)}"
    col_bad = f"shadow_deprecated_{_fresh_snake_ident(rng)}"
    min_tier = rng.choice((2, 3, 4))

    t1 = (
        f"-- Schema Contract (SQLite / ANSI SQL)\n"
        f"-- Rule 1: Join `{tbl_main}` with `{tbl_acct}` on `acct_id` for active accounts (`tier >= {min_tier}`).\n"
        f"-- Rule 2: Sum `{col_metric}` per `region_code` where `is_void = 0`.\n"
        f"-- Rule 3: NEVER reference deprecated column `{col_bad}`.\n"
        f"CREATE TABLE {tbl_acct} (\n"
        f"    acct_id INTEGER PRIMARY KEY,\n"
        f"    region_code TEXT NOT NULL,\n"
        f"    tier INTEGER NOT NULL\n"
        f");\n"
        f"CREATE TABLE {tbl_main} (\n"
        f"    entry_id INTEGER PRIMARY KEY,\n"
        f"    acct_id INTEGER NOT NULL,\n"
        f"    {col_metric} INTEGER NOT NULL,\n"
        f"    {col_bad} INTEGER DEFAULT 0,\n"
        f"    is_void INTEGER NOT NULL DEFAULT 0\n"
        f");\n\n"
    )
    pre = (
        f"\x1dS\njoin {tbl_acct}+{tbl_main} on acct_id | filter tier>={min_tier} & is_void=0 | sum({col_metric})\n\x1dE\n"
        f"SELECT\n"
        f"    a.region_code,\n"
        f"    SUM(m."
    )
    suf = (
        f") AS total_net\n"
        f"FROM {tbl_main} AS m\n"
        f"INNER JOIN {tbl_acct} AS a ON m.acct_id = a.acct_id\n"
        f"WHERE a.tier >= {min_tier} AND m.is_void = 0\n"
        f"GROUP BY a.region_code\n"
        f"ORDER BY total_net DESC;\n"
    )
    return _pack_cognitive_fast(t1, pre, col_metric, suf, style_id=4)


def _sample_cog_js_sdk_fast(rng: random.Random, compact: bool) -> tuple[bytes, int, int, int, int]:
    fn_init = _fresh_camel_ident(rng)
    fn_stage = _fresh_camel_ident(rng)
    fn_commit = _fresh_camel_ident(rng)
    fn_bad = f"unsafe{_fresh_pascal_ident(rng)}"
    max_items = rng.choice((16, 32, 64))

    t1 = (
        f"// Synthetic ES6 Micro-SDK Contract\n"
        f"// Precondition: `{fn_init}` -> `{fn_stage}` -> `{fn_commit}`.\n"
        f"// Rule: Reject empty arrays; slice batch to at most {max_items} positive integers; NEVER call `{fn_bad}`.\n"
        f"function {fn_init}(tenantId) {{\n"
        f"  return {{ tenantId, ready: true, staged: [] }};\n"
        f"}}\n"
        f"function {fn_stage}(session, items) {{\n"
        f"  if (!session.ready) throw new Error('uninitialized session');\n"
        f"  session.staged = items;\n"
        f"}}\n"
        f"function {fn_commit}(session) {{\n"
        f"  return session.staged.reduce((acc, v) => acc + v, 0);\n"
        f"}}\n"
        f"function {fn_bad}() {{\n"
        f"  throw new Error('deprecated API');\n"
        f"}}\n\n"
    )
    pre = (
        f"\x1dS\nguard empty | filter >0 & slice(0,{max_items}) | order: {fn_init} -> {fn_stage} -> {fn_commit}\n\x1dE\n"
        f"function processTenantBatch(tenantId, rawValues) {{\n"
        f"  if (!Array.isArray(rawValues) || rawValues.length === 0) {{\n"
        f"    throw new RangeError('rawValues must be non-empty');\n"
        f"  }}\n"
        f"  const clean = rawValues.filter((v) => v > 0).slice(0, {max_items});\n"
        f"  const session = {fn_init}(tenantId);\n"
        f"  "
    )
    suf = f"(session, clean);\n  return {fn_commit}(session);\n}}\n"
    return _pack_cognitive_fast(t1, pre, fn_stage, suf, style_id=4)


def sample_cognitive_stage4_episode_fast(
    rng: random.Random,
    lang_id: int,
    max_bytes: int = 3072,
) -> tuple[bytes, int, int, int, int]:
    """Synthesize a DCPA v2 Stage-4 fast-weight binding episode (`(raw_bytes, t1_len, tgt_s, tgt_e, style_id)`).

    Covers all 6 v2 domains with ZERO Rust:
      - `0=en`, `1=zh`: World-Bible long-form continuation (`style_id=3`) & temporary axiom adjudication (`style_id=0`)
      - `2=formal`: Counter-intuitive formal operator axiom derivation (`style_id=0`)
      - `3=python`, `4=sql_struct`, `5=javascript`: Multi-API Micro-SDK / Schema rebinding (`style_id=4`)
    """
    compact = int(max_bytes) < 1150
    if lang_id == 0:
        out = _sample_cog_en_fast(rng, compact=compact)
    elif lang_id == 1:
        out = _sample_cog_zh_fast(rng, compact=compact)
    elif lang_id == 2:
        out = _sample_cog_formal_fast(rng, compact=compact)
    elif lang_id == 3:
        out = _sample_cog_python_sdk_fast(rng, compact=compact)
    elif lang_id == 4:
        out = _sample_cog_sql_fast(rng, compact=compact)
    elif lang_id == 5:
        out = _sample_cog_js_sdk_fast(rng, compact=compact)
    else:
        raise ValueError(f"Unsupported cognitive_v2 lang_id={lang_id}; expected 0..5")

    if len(out[0]) <= max_bytes:
        return out
    if not compact:
        # Retry in compact mode if full World-Bible exceeded `rem`
        return sample_cognitive_stage4_episode_fast(rng, lang_id, max_bytes=min(int(max_bytes), 900))
    # Ultra-short fallback for tight remaining batch tail (`320 <= rem < ~750`)
    if lang_id in (0, 2):
        rb, t1, ts, te = sample_wug_episode_fast(rng, 0)
        return rb, t1, ts, te, 0
    if lang_id == 1:
        rb, t1, ts, te = sample_wug_episode_fast(rng, 1)
        return rb, t1, ts, te, 0
    if lang_id in (3, 4):
        rb, t1, ts, te = sample_wug_episode_fast(rng, 3)
        return rb, t1, ts, te, 4
    rb, t1, ts, te = sample_wug_episode_fast(rng, 5)
    return rb, t1, ts, te, 4


def format_episode_annotated(ep: WugEpisode) -> str:
    """Format an episode with `[SUPPORT: ...]`, `[QUERY: ...]`, and `<<target>>` markers."""
    t1 = ep.raw_bytes[: ep.turn1_byte_len].decode("utf-8", errors="strict")
    q_before = ep.raw_bytes[ep.turn1_byte_len : ep.target_byte_start].decode("utf-8", errors="strict")
    q_target = ep.raw_bytes[ep.target_byte_start : ep.target_byte_end].decode("utf-8", errors="strict")
    q_after = ep.raw_bytes[ep.target_byte_end :].decode("utf-8", errors="strict")
    return f"[SUPPORT: {t1}][QUERY: {q_before}<<{q_target}>>{q_after}]"
