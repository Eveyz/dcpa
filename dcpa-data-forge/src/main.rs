//! dcpa-data-forge: Multi-threaded Rust developmental corpus builder for DCPA-Flat v3.3.
//!
//! Enforces DCPA-Flat v3.3 WS-1 Full-Stack Binary Protocol:
//!   - Strict raw UTF-8 NFC normalization (ZERO tokenizers / BPE).
//!   - Two-stage sanitization:
//!       (a) `normalize_raw_utf8_bytes`: NFC normalize, CRLF/CR -> LF (0x0A), keep TAB (0x09),
//!           strip all other ASCII < 0x20 and 0x7F.
//!       (b) `validate_framed_record`: allow the 8 protected protocol control bytes
//!           (`0x04, 0x11, 0x12, 0x13, 0x14, 0x1D, 0x1E, 0x1F`), reject any other ASCII < 0x20
//!           (except `\t`, `\n`) or `0x7F`, and verify `\x1dS\n` ... `\x1dE\n` scratchpad frames
//!           are strictly paired/closed with inner span <= 260 bytes.
//!   - Lexical string/comment & bracket-depth validator (`compute_code_max_depth`) for
//!     Python (`3`), SQL/JSON (`4`), and JavaScript (`5`) returning `max_depth` in `1..=16`.
//!   - Emits contiguous delimiter-free `shard.bin` + exact 16-byte little-endian `shard.idx`
//!     (`<QIBBBB`):
//!       start_offset: u64 | byte_length: u32 | lang_id: u8 | source_id: u8 | max_depth: u8 | style_id: u8

use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// Protected v3.3 protocol control bytes (all other ASCII < 0x20 except \t=0x09, \n=0x0A are disallowed).
pub const ALLOWED_PROTOCOL_CTRL_BYTES: [u8; 8] = [
    0x04, // EOT: assistant turn end (\x04\n)
    0x11, // DC1: tool call start (\x11C\n)
    0x12, // DC2: sandbox observation (\x12O\n)
    0x13, // DC3: suspend decoding for sandbox (\x13\n)
    0x14, // DC4: reserved external repo/doc paging frame
    0x1D, // GS:  scratchpad start/end (\x1dS\n ... \x1dE\n -> G2 Slot-0 pin)
    0x1E, // RS:  user turn start (\x1eU\n)
    0x1F, // US:  assistant turn start (\x1fA\n)
];

/// Maximum permitted inner byte span inside `\x1dS\n` ... `\x1dE\n` scratchpad frames.
pub const MAX_SCRATCHPAD_INNER_BYTES: usize = 260;

/// Exact 16-byte little-endian binary shard index record (`<QIBBBB`, DCPA-Flat v3.3 SSOT).
///
/// Canonical `lang_id` (`0..=5`):
///   `0=en, 1=zh, 2=formal, 3=python, 4=sql_struct, 5=javascript`
/// Canonical `style_id` (`0..=4`):
///   `0=rigorous, 1=concise_memo, 2=expository, 3=literary, 4=strict_eng`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardIndexRecord {
    pub start_offset: u64,
    pub byte_length: u32,
    pub lang_id: u8,
    pub source_id: u8,
    pub max_depth: u8,
    pub style_id: u8,
}

impl ShardIndexRecord {
    pub const SIZE: usize = 16;

    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.start_offset.to_le_bytes());
        out[8..12].copy_from_slice(&self.byte_length.to_le_bytes());
        out[12] = self.lang_id;
        out[13] = self.source_id;
        out[14] = self.max_depth;
        out[15] = self.style_id;
        out
    }

    pub fn from_le_bytes(b: &[u8; 16]) -> Self {
        Self {
            start_offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            byte_length: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            lang_id: b[12],
            source_id: b[13],
            max_depth: b[14],
            style_id: b[15],
        }
    }
}

#[inline]
pub fn is_allowed_protocol_ctrl(b: u8) -> bool {
    ALLOWED_PROTOCOL_CTRL_BYTES.contains(&b)
}

/// Stage 1 sanitization (pre-framing raw text):
/// NFC normalize, unify CRLF/CR to LF (0x0A), keep TAB (0x09), strip all other ASCII < 0x20 and 0x7F.
pub fn normalize_raw_utf8_bytes(input: &str) -> Vec<u8> {
    let nfc: String = input.nfc().collect();
    let bytes = nfc.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\r' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 1;
            }
            out.push(b'\n');
        } else if b == 0x09 || b == 0x0A || (b >= 0x20 && b != 0x7F) {
            out.push(b);
        }
        // Strip 0x00..=0x08, 0x0B..=0x1F, 0x7F
        i += 1;
    }
    out
}

/// Backwards-compatible alias for `normalize_raw_utf8_bytes`.
#[inline]
pub fn normalize_utf8_bytes(input: &str) -> Vec<u8> {
    normalize_raw_utf8_bytes(input)
}

/// Stage 2 sanitization (post-framing protocol record validation):
///   - Verifies valid UTF-8.
///   - Explicitly allows the 8 protected protocol control bytes (`0x04, 0x11, 0x12, 0x13, 0x14, 0x1D, 0x1E, 0x1F`)
///     plus TAB (`0x09`) and LF (`0x0A`).
///   - Rejects any other ASCII `< 0x20` or `0x7F`.
///   - Verifies every `\x1dS\n` ... `\x1dE\n` scratchpad frame is strictly paired/closed
///     with inner byte span `<= 260` bytes and does not cross turn boundaries (`0x04` / `0x1E`).
pub fn validate_framed_record(bytes: &[u8]) -> Result<(), &'static str> {
    if core::str::from_utf8(bytes).is_err() {
        return Err("invalid utf-8");
    }

    let n = bytes.len();
    let mut i = 0usize;
    let mut scratch_inner_start: Option<usize> = None;

    while i < n {
        let b = bytes[i];
        if (b < 0x20 && b != 0x09 && b != 0x0A && !is_allowed_protocol_ctrl(b)) || b == 0x7F {
            return Err("disallowed control byte");
        }

        if b == 0x1D {
            if i + 2 < n && bytes[i + 1] == b'S' && bytes[i + 2] == b'\n' {
                if scratch_inner_start.is_some() {
                    return Err("nested or unclosed scratchpad frame");
                }
                scratch_inner_start = Some(i + 3);
                i += 3;
                continue;
            } else if i + 2 < n && bytes[i + 1] == b'E' && bytes[i + 2] == b'\n' {
                if let Some(start) = scratch_inner_start.take() {
                    let inner_len = i - start;
                    if inner_len > MAX_SCRATCHPAD_INNER_BYTES {
                        return Err("scratchpad inner span exceeds 260 bytes");
                    }
                    i += 3;
                    continue;
                } else {
                    return Err("unmatched scratchpad end");
                }
            } else {
                return Err("malformed 0x1D scratchpad delimiter");
            }
        }

        if scratch_inner_start.is_some() && (b == 0x04 || b == 0x1E) {
            return Err("unclosed scratchpad frame before turn boundary");
        }

        i += 1;
    }

    if scratch_inner_start.is_some() {
        return Err("unclosed scratchpad frame");
    }

    Ok(())
}

/// Lexical state machine validating balanced `{}()[]` outside strings/comments for:
///   - Python (`lang_id == 3`): `#` line comments, `'...'`, `"..."`, `'''...'''`, `"""..."""`
///   - SQL/JSON (`lang_id == 4`): `--` line comments, `/* ... */` block comments, `'...'` (with `''` & `\'`), `"..."`
///   - JavaScript (`lang_id == 5`): `//` line comments, `/* ... */` block comments, `'...'`, `"..."`,
///     template literals `` `...` `` with nested `${...}` interpolation
///
/// Returns `Some(max_depth)` with `max_depth` in `1..=16`, or `None` if unbalanced, unclosed,
/// invalid UTF-8, unsupported `lang_id`, `max_depth == 0`, or `max_depth > 16`.
pub fn compute_code_max_depth(bytes: &[u8], lang_id: u8) -> Option<u8> {
    if !matches!(lang_id, 3 | 4 | 5) || bytes.is_empty() {
        return None;
    }
    if core::str::from_utf8(bytes).is_err() {
        return None;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LexMode {
        Code,
        LineComment,
        BlockComment,
        SingleQuote,
        DoubleQuote,
        TripleSingle,
        TripleDouble,
        TemplateLiteral,
    }

    let n = bytes.len();
    let mut i = 0usize;
    let mut mode = LexMode::Code;
    let mut stack: Vec<u8> = Vec::with_capacity(32);
    // Tracks bracket stack lengths at which JS `${` entered an interpolated expression inside a template literal.
    let mut template_expr_depths: Vec<usize> = Vec::new();
    let mut max_depth: usize = 0;

    while i < n {
        let c = bytes[i];
        let next = if i + 1 < n { bytes[i + 1] } else { 0 };
        let next2 = if i + 2 < n { bytes[i + 2] } else { 0 };

        match mode {
            LexMode::LineComment => {
                if c == b'\n' {
                    mode = LexMode::Code;
                }
                i += 1;
            }
            LexMode::BlockComment => {
                if c == b'*' && next == b'/' {
                    mode = LexMode::Code;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            LexMode::SingleQuote => {
                if lang_id == 4 {
                    // SQL single-quoted string supports '' doubled quote and \' escape
                    if c == b'\'' && next == b'\'' {
                        i += 2;
                        continue;
                    }
                    if c == b'\\' {
                        if i + 1 >= n {
                            return None;
                        }
                        i += 2;
                        continue;
                    }
                    if c == b'\'' {
                        mode = LexMode::Code;
                    }
                    i += 1;
                } else {
                    if c == b'\\' {
                        if i + 1 >= n {
                            return None;
                        }
                        i += 2;
                        continue;
                    }
                    if c == b'\n' {
                        return None;
                    }
                    if c == b'\'' {
                        mode = LexMode::Code;
                    }
                    i += 1;
                }
            }
            LexMode::DoubleQuote => {
                if c == b'\\' {
                    if i + 1 >= n {
                        return None;
                    }
                    i += 2;
                    continue;
                }
                if c == b'\n' && lang_id != 4 {
                    return None;
                }
                if c == b'"' {
                    mode = LexMode::Code;
                }
                i += 1;
            }
            LexMode::TripleSingle => {
                if c == b'\\' {
                    if i + 1 >= n {
                        return None;
                    }
                    i += 2;
                    continue;
                }
                if c == b'\'' && next == b'\'' && next2 == b'\'' {
                    mode = LexMode::Code;
                    i += 3;
                } else {
                    i += 1;
                }
            }
            LexMode::TripleDouble => {
                if c == b'\\' {
                    if i + 1 >= n {
                        return None;
                    }
                    i += 2;
                    continue;
                }
                if c == b'"' && next == b'"' && next2 == b'"' {
                    mode = LexMode::Code;
                    i += 3;
                } else {
                    i += 1;
                }
            }
            LexMode::TemplateLiteral => {
                if c == b'\\' {
                    if i + 1 >= n {
                        return None;
                    }
                    i += 2;
                    continue;
                }
                if c == b'$' && next == b'{' {
                    template_expr_depths.push(stack.len());
                    stack.push(b'}');
                    if stack.len() > max_depth {
                        max_depth = stack.len();
                        if max_depth > 16 {
                            return None;
                        }
                    }
                    mode = LexMode::Code;
                    i += 2;
                    continue;
                }
                if c == b'`' {
                    mode = LexMode::Code;
                }
                i += 1;
            }
            LexMode::Code => {
                // Comments per language
                if lang_id == 3 && c == b'#' {
                    mode = LexMode::LineComment;
                    i += 1;
                    continue;
                }
                if lang_id == 4 && c == b'-' && next == b'-' {
                    mode = LexMode::LineComment;
                    i += 2;
                    continue;
                }
                if lang_id == 5 && c == b'/' && next == b'/' {
                    mode = LexMode::LineComment;
                    i += 2;
                    continue;
                }
                if (lang_id == 4 || lang_id == 5) && c == b'/' && next == b'*' {
                    mode = LexMode::BlockComment;
                    i += 2;
                    continue;
                }

                // String & template literal openers
                if lang_id == 3 && c == b'"' && next == b'"' && next2 == b'"' {
                    mode = LexMode::TripleDouble;
                    i += 3;
                    continue;
                }
                if lang_id == 3 && c == b'\'' && next == b'\'' && next2 == b'\'' {
                    mode = LexMode::TripleSingle;
                    i += 3;
                    continue;
                }
                if c == b'"' {
                    mode = LexMode::DoubleQuote;
                    i += 1;
                    continue;
                }
                if c == b'\'' {
                    mode = LexMode::SingleQuote;
                    i += 1;
                    continue;
                }
                if lang_id == 5 && c == b'`' {
                    mode = LexMode::TemplateLiteral;
                    i += 1;
                    continue;
                }

                // Brackets
                match c {
                    b'{' => {
                        stack.push(b'}');
                        if stack.len() > max_depth {
                            max_depth = stack.len();
                            if max_depth > 16 {
                                return None;
                            }
                        }
                    }
                    b'(' => {
                        stack.push(b')');
                        if stack.len() > max_depth {
                            max_depth = stack.len();
                            if max_depth > 16 {
                                return None;
                            }
                        }
                    }
                    b'[' => {
                        stack.push(b']');
                        if stack.len() > max_depth {
                            max_depth = stack.len();
                            if max_depth > 16 {
                                return None;
                            }
                        }
                    }
                    b'}' => {
                        if stack.pop() != Some(b'}') {
                            return None;
                        }
                        if lang_id == 5 && template_expr_depths.last() == Some(&stack.len()) {
                            template_expr_depths.pop();
                            mode = LexMode::TemplateLiteral;
                        }
                    }
                    b')' | b']' => {
                        if stack.pop() != Some(c) {
                            return None;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
    }

    if !matches!(mode, LexMode::Code | LexMode::LineComment) {
        return None;
    }
    if !stack.is_empty() || !template_expr_depths.is_empty() {
        return None;
    }
    if max_depth < 1 || max_depth > 16 {
        return None;
    }
    Some(max_depth as u8)
}

/// Structural syntax & delimiter balance validator for Python (3), SQL/JSON (4), and JavaScript (5).
/// Checks byte length in [64, 8192] and strict `{}()[]` balance with `1 <= max_depth <= 16`.
pub fn validate_code_module(bytes: &[u8], lang_id: u8) -> bool {
    if bytes.len() < 64 || bytes.len() > 8192 {
        return false;
    }
    compute_code_max_depth(bytes, lang_id).is_some()
}

/// Deterministic XorShift64 PRNG for zero-dependency parallel synthetic generation.
struct Rng64(u64);

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

fn generate_grid_physics_doc(seed: u64) -> Vec<u8> {
    let mut rng = Rng64::new(seed);
    let objects = ["red_ball", "blue_cube", "brass_key", "clay_cup"];
    let containers = ["wooden_box", "iron_pot", "woven_basket"];
    let mut pos: [(i32, i32); 4] = [(0, 0); 4];
    let mut inside: [Option<usize>; 4] = [None; 4];

    let mut s = String::with_capacity(1024);
    s.push_str("[WORLD_INIT]\n");
    for i in 0..4 {
        pos[i] = (rng.gen_range(5) as i32, rng.gen_range(5) as i32);
        s.push_str(&format!(
            "state: {} at ({},{}) visible=true\n",
            objects[i], pos[i].0, pos[i].1
        ));
    }

    let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for step in 0..16 {
        let oi = rng.gen_range(4);
        let act = rng.gen_range(4);
        match act {
            0 if inside[oi].is_none() => {
                let (dx, dy) = dirs[rng.gen_range(4)];
                pos[oi].0 = (pos[oi].0 + dx).clamp(0, 4);
                pos[oi].1 = (pos[oi].1 + dy).clamp(0, 4);
                s.push_str(&format!(
                    "step {}: move {} ({:+},{:+}) -> {} at ({},{})\n",
                    step, objects[oi], dx, dy, objects[oi], pos[oi].0, pos[oi].1
                ));
            }
            1 if inside[oi].is_none() => {
                let ci = rng.gen_range(3);
                inside[oi] = Some(ci);
                s.push_str(&format!(
                    "step {}: place {} inside {} -> {} hidden_in {} exists=true\n",
                    step, objects[oi], containers[ci], objects[oi], containers[ci]
                ));
            }
            2 if inside[oi].is_some() => {
                let ci = inside[oi].take().unwrap();
                s.push_str(&format!(
                    "step {}: lift {} from {} -> {} at ({},{}) visible=true\n",
                    step, objects[oi], containers[ci], objects[oi], pos[oi].0, pos[oi].1
                ));
            }
            _ => {
                let ci = rng.gen_range(3);
                s.push_str(&format!(
                    "step {}: screen {} blocks view of {} -> permanence({})=true at ({},{})\n",
                    step, containers[ci], objects[oi], objects[oi], pos[oi].0, pos[oi].1
                ));
            }
        }
    }
    normalize_raw_utf8_bytes(&s)
}

fn generate_dyck_doc(seed: u64) -> (Vec<u8>, u8) {
    let mut rng = Rng64::new(seed);
    let pairs = [('{', '}'), ('(', ')'), ('[', ']')];
    let max_depth = 3 + rng.gen_range(12); // depth 3..=14
    let target_len = 300 + rng.gen_range(1200);
    let js_style = (seed & 1) == 1;

    let mut stack: Vec<char> = Vec::with_capacity(16);
    let mut out = String::with_capacity(target_len + 128);
    let mut peak_depth: usize = 0;

    while stack.len() < max_depth {
        let (op, cl) = pairs[rng.gen_range(3)];
        if js_style && op == '{' {
            out.push_str(&format!("function b_{}() ", stack.len()));
        }
        out.push(op);
        stack.push(cl);
        if stack.len() > peak_depth {
            peak_depth = stack.len();
        }
        if js_style {
            out.push_str(&format!(" const v{} = {}; ", stack.len(), rng.gen_range(99)));
        } else {
            out.push_str(&format!(" x{} ", stack.len()));
        }
    }

    while out.len() < target_len {
        if stack.is_empty() || (stack.len() < max_depth && rng.gen_range(100) < 55) {
            let (op, cl) = pairs[rng.gen_range(3)];
            out.push(op);
            stack.push(cl);
            if stack.len() > peak_depth {
                peak_depth = stack.len();
            }
            out.push_str(&format!(" n{} ", rng.gen_range(256)));
        } else {
            let cl = stack.pop().unwrap();
            out.push(cl);
            out.push(' ');
        }
    }

    while let Some(cl) = stack.pop() {
        out.push_str(if js_style { " return 0; " } else { " end " });
        out.push(cl);
    }
    out.push('\n');
    (normalize_raw_utf8_bytes(&out), peak_depth.min(16) as u8)
}

fn generate_fast_mapping_doc(seed: u64) -> Vec<u8> {
    let mut rng = Rng64::new(seed);
    let onsets = ["bl", "sp", "v", "z", "dr", "kr", "fl", "gl", "tr", "pr", "sn", "w"];
    let vowels = ["a", "e", "i", "o", "u"];
    let codas = ["ck", "rg", "sh", "g", "p", "t", "b", "m", "n", "nd", "st"];
    let entities = [
        ("a tiny nocturnal desert bird that drinks cactus dew", "desert bird"),
        ("a bright emerald tree frog hiding under wet leaves", "emerald frog"),
        ("a striped canyon lizard basking on warm red stones", "canyon lizard"),
        ("a curved brass carving tool used by boat makers", "brass tool"),
        ("a sour purple mountain berry ripening after frost", "mountain berry"),
    ];
    let distractors = [
        "Meanwhile the afternoon wind shifted across the eastern valley and cooled the dusty path.",
        "Two travelers paused near the old stone bridge to check their wooden lanterns before dusk.",
        "Heavy clouds drifted slowly over the ridge while distant bells echoed from the harbor tower.",
        "Inside the quiet workshop, clean jars of oil and neatly coiled ropes lined the cedar shelf.",
        "After the brief shower passed, bright puddles reflected the pale sky along the courtyard wall.",
    ];

    let pw = format!(
        "{}{}{}",
        onsets[rng.gen_range(onsets.len())],
        vowels[rng.gen_range(vowels.len())],
        codas[rng.gen_range(codas.len())]
    );
    let plural = if pw.ends_with("sh") || pw.ends_with('s') || pw.ends_with('z') {
        format!("{}es", pw)
    } else {
        format!("{}s", pw)
    };
    let (def, short) = entities[rng.gen_range(entities.len())];
    let word_first = (rng.next_u64() & 1) == 0;
    let intro = if word_first {
        format!("A {} is {}.", pw, def)
    } else {
        format!("People call {} a {}.", def, pw)
    };
    let n_dist = 2 + rng.gen_range(4);
    let mut dist = String::new();
    for k in 0..n_dist {
        if k > 0 {
            dist.push(' ');
        }
        dist.push_str(distractors[(rng.gen_range(distractors.len()) + k) % distractors.len()]);
    }
    let target = format!(
        "Look at one {} on the stone; now there are two {}, each a {}.\n",
        pw, plural, short
    );
    let full = format!("{} {} {}", intro, dist, target);
    normalize_raw_utf8_bytes(&full)
}

fn write_shard(prefix: &Path, docs: &[(Vec<u8>, u8, u8, u8, u8)]) -> std::io::Result<()> {
    if let Some(parent) = prefix.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let bin_path = prefix.with_extension("bin");
    let idx_path = prefix.with_extension("idx");

    let mut bin_w = BufWriter::new(File::create(&bin_path)?);
    let mut idx_w = BufWriter::new(File::create(&idx_path)?);

    let mut offset: u64 = 0;
    let mut written: usize = 0;
    for (bytes, lang_id, source_id, max_depth, style_id) in docs {
        if bytes.is_empty() {
            continue;
        }
        bin_w.write_all(bytes)?;
        let rec = ShardIndexRecord {
            start_offset: offset,
            byte_length: bytes.len() as u32,
            lang_id: *lang_id,
            source_id: *source_id,
            max_depth: *max_depth,
            style_id: *style_id,
        };
        idx_w.write_all(&rec.to_le_bytes())?;
        offset += bytes.len() as u64;
        written += 1;
    }
    bin_w.flush()?;
    idx_w.flush()?;
    println!(
        "Wrote shard {:?} ({} docs, {} total bytes) + {:?}",
        bin_path, written, offset, idx_path
    );
    Ok(())
}

pub fn parse_lang(s: &str) -> u8 {
    match s.to_ascii_lowercase().as_str() {
        "en" => 0,
        "zh" => 1,
        "formal" | "math" | "fol" | "lean" | "ja" => 2,
        "python" | "py" => 3,
        "sql_struct" | "sql" | "json" | "struct" | "rust" | "rs" => 4,
        "javascript" | "js" | "ts" | "typescript" => 5,
        _ => 0,
    }
}

pub fn parse_style(s: &str) -> u8 {
    match s.to_ascii_lowercase().as_str() {
        "rigorous" | "neutral" => 0,
        "concise_memo" | "memo" => 1,
        "expository" => 2,
        "literary" | "narrative" => 3,
        "strict_eng" | "strict_code_tool" | "code" | "tool" => 4,
        other => other.parse::<u8>().unwrap_or(0).min(4),
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn inspect_shard(prefix: &Path) -> std::io::Result<()> {
    let bin_path = prefix.with_extension("bin");
    let idx_path = prefix.with_extension("idx");
    let bin_data = fs::read(&bin_path)?;
    let idx_data = fs::read(&idx_path)?;
    assert_eq!(idx_data.len() % ShardIndexRecord::SIZE, 0, "Corrupt .idx length");
    let n = idx_data.len() / ShardIndexRecord::SIZE;
    let mut valid_utf8 = 0usize;
    let mut valid_framed = 0usize;
    let mut total_bytes = 0u64;
    for i in 0..n {
        let rec_bytes: &[u8; 16] = idx_data[i * 16..(i + 1) * 16].try_into().unwrap();
        let rec = ShardIndexRecord::from_le_bytes(rec_bytes);
        let start = rec.start_offset as usize;
        let end = start + rec.byte_length as usize;
        let slice = &bin_data[start..end];
        if core::str::from_utf8(slice).is_ok() {
            valid_utf8 += 1;
        }
        if validate_framed_record(slice).is_ok() {
            valid_framed += 1;
        }
        total_bytes += rec.byte_length as u64;
    }
    println!(
        "Inspected {:?}: {} records, {} bytes, {}/{} valid UTF-8, {}/{} valid framed",
        prefix, n, total_bytes, valid_utf8, n, valid_framed, n
    );
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  dcpa-data-forge forge <input_dir> <lang:en|zh|formal|python|sql_struct|javascript> <source_id> <out_prefix> [style_id]");
        eprintln!("  dcpa-data-forge synth-stage1 <num_physics> <num_dyck> <out_prefix>");
        eprintln!("  dcpa-data-forge synth-fastmap <num_episodes> <out_prefix>");
        eprintln!("  dcpa-data-forge inspect <shard_prefix>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "forge" => {
            let input_dir = PathBuf::from(&args[2]);
            let lang_id = parse_lang(&args[3]);
            let source_id: u8 = args[4].parse().expect("source_id must be u8");
            let out_prefix = PathBuf::from(&args[5]);
            let style_id: u8 = args
                .get(6)
                .map(|s| parse_style(s))
                .unwrap_or(if matches!(lang_id, 3 | 4 | 5) { 4 } else { 0 });

            let mut files = Vec::new();
            collect_files(&input_dir, &mut files)?;
            let is_code = matches!(lang_id, 3 | 4 | 5);

            let docs: Vec<(Vec<u8>, u8, u8, u8, u8)> = files
                .par_iter()
                .filter_map(|path| {
                    let mut raw = String::new();
                    File::open(path).ok()?.read_to_string(&mut raw).ok()?;
                    let norm = normalize_raw_utf8_bytes(&raw);
                    if norm.is_empty() {
                        return None;
                    }
                    if is_code {
                        if norm.len() < 64 || norm.len() > 8192 {
                            return None;
                        }
                        let depth = compute_code_max_depth(&norm, lang_id)?;
                        Some((norm, lang_id, source_id, depth, style_id))
                    } else {
                        Some((norm, lang_id, source_id, 0u8, style_id))
                    }
                })
                .collect();

            write_shard(&out_prefix, &docs)?;
        }
        "synth-stage1" => {
            let n_phys: usize = args[2].parse().expect("num_physics");
            let n_dyck: usize = args[3].parse().expect("num_dyck");
            let out_prefix = PathBuf::from(&args[4]);

            let mut docs: Vec<(Vec<u8>, u8, u8, u8, u8)> = (0..n_phys)
                .into_par_iter()
                .map(|i| (generate_grid_physics_doc(1000 + i as u64), 2u8, 1u8, 0u8, 0u8))
                .collect();
            let dyck_docs: Vec<(Vec<u8>, u8, u8, u8, u8)> = (0..n_dyck)
                .into_par_iter()
                .map(|i| {
                    let (b, d) = generate_dyck_doc(500_000 + i as u64);
                    (b, 5u8, 2u8, d, 4u8)
                })
                .collect();
            docs.extend(dyck_docs);
            write_shard(&out_prefix, &docs)?;
        }
        "synth-fastmap" => {
            let n_ep: usize = args[2].parse().expect("num_episodes");
            let out_prefix = PathBuf::from(&args[3]);
            let docs: Vec<(Vec<u8>, u8, u8, u8, u8)> = (0..n_ep)
                .into_par_iter()
                .map(|i| (generate_fast_mapping_doc(900_000 + i as u64), 0u8, 21u8, 0u8, 2u8))
                .collect();
            write_shard(&out_prefix, &docs)?;
        }
        "inspect" => {
            inspect_shard(Path::new(&args[2]))?;
        }
        other => {
            eprintln!("Unknown subcommand: {}", other);
            std::process::exit(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfc_and_control_byte_normalization() {
        // Stage 1: Decomposed "e\u{0301}" -> NFC "\u{00e9}", CRLF/CR -> LF, TAB kept, NUL/BEL/protocol/DEL stripped
        let raw = "a\x00b\x07\x1d\x1e\x7f\tc\r\nd\re\u{0301}";
        let norm = normalize_raw_utf8_bytes(raw);
        assert_eq!(core::str::from_utf8(&norm).unwrap(), "ab\tc\nd\n\u{00e9}");
        assert_eq!(normalize_utf8_bytes(raw), norm);
    }

    #[test]
    fn test_validate_framed_record_protocol_and_scratchpad() {
        // Valid framed dialogue exercising all 8 allowed protocol control bytes
        let mut valid = Vec::new();
        valid.extend_from_slice(b"\x14ctx\n\x1eU\nCompute 2+2\n\x1fA\n");
        valid.extend_from_slice(b"\x1dS\nplan: use calc\x1dE\n");
        valid.extend_from_slice(b"\x11C\ncalc(\"2+2\")\x13\n\x12O\n4\n\x1fA\nAnswer: 4\x04\n");
        assert!(validate_framed_record(&valid).is_ok());

        // Exact 260-byte inner scratchpad is allowed
        let mut exact_260 = b"\x1dS\n".to_vec();
        exact_260.extend(std::iter::repeat(b'a').take(260));
        exact_260.extend_from_slice(b"\x1dE\n");
        assert!(validate_framed_record(&exact_260).is_ok());

        // 261-byte inner scratchpad is rejected
        let mut over_260 = b"\x1dS\n".to_vec();
        over_260.extend(std::iter::repeat(b'a').take(261));
        over_260.extend_from_slice(b"\x1dE\n");
        assert!(validate_framed_record(&over_260).is_err());

        // Unclosed scratchpad rejected
        assert!(validate_framed_record(b"\x1dS\nopen forever").is_err());
        // Unmatched scratchpad end rejected
        assert!(validate_framed_record(b"oops\x1dE\n").is_err());
        // Nested scratchpad rejected
        assert!(validate_framed_record(b"\x1dS\na\x1dS\nb\x1dE\n\x1dE\n").is_err());
        // Scratchpad crossing EOT rejected
        assert!(validate_framed_record(b"\x1dS\nabc\x04\n\x1dE\n").is_err());
        // Disallowed control bytes (0x00, 0x07, 0x7F) rejected
        assert!(validate_framed_record(b"hello\x00world").is_err());
        assert!(validate_framed_record(b"hello\x07world").is_err());
        assert!(validate_framed_record(b"hello\x7fworld").is_err());
        // Invalid UTF-8 rejected
        assert!(validate_framed_record(&[0xFF, 0xFE]).is_err());
    }

    #[test]
    fn test_compute_code_max_depth_python_sql_js() {
        // Python (lang_id=3): triple-quoted string & comment containing unbalanced brackets ignored
        let py_ok = concat!(
            "def compute(xs: list[int]) -> dict[str, int]:\n",
            "    \"\"\"Doc with unbalanced ((( [ { ignored\"\"\"\n",
            "    # comment with ] } ) ignored\n",
            "    return {'sum': sum([v for v in xs if (v > 0)])}\n",
        );
        assert_eq!(compute_code_max_depth(py_ok.as_bytes(), 3), Some(4));
        assert!(validate_code_module(py_ok.as_bytes(), 3));

        let py_bad = "def broken(xs:\n    return [1, 2, 3]\n";
        assert_eq!(compute_code_max_depth(py_bad.as_bytes(), 3), None);

        // SQL / JSON (lang_id=4): SQL -- comment, /* */ comment, doubled '' quote escape
        let sql_ok = concat!(
            "-- ignore unmatched (\n",
            "/* block comment [ { */\n",
            "CREATE TABLE orders (id BIGINT PRIMARY KEY, note TEXT);\n",
            "SELECT COUNT(*) FROM orders WHERE note <> 'it''s (fine)';\n",
        );
        assert_eq!(compute_code_max_depth(sql_ok.as_bytes(), 4), Some(1));
        assert!(validate_code_module(sql_ok.as_bytes(), 4));

        let json_ok = r#"{"schema": {"fields": [{"name": "id", "type": "int"}], "ok": true}}"#;
        assert_eq!(compute_code_max_depth(json_ok.as_bytes(), 4), Some(4));
        assert!(validate_code_module(json_ok.as_bytes(), 4));

        let sql_bad = "SELECT (a + b FROM orders;\n";
        assert_eq!(compute_code_max_depth(sql_bad.as_bytes(), 4), None);

        // JavaScript (lang_id=5): // comment, /* */ comment, template literal with ${...}
        let js_ok = concat!(
            "// unmatched (( in comment\n",
            "export function render(items) {\n",
            "  /* block comment [ */\n",
            "  return items.map((x) => `val=${String([x.id])}`);\n",
            "}\n",
        );
        // Depth trace: { (1) -> .map( (2) -> `${ (3) -> String( (4) -> [ (5)
        assert_eq!(compute_code_max_depth(js_ok.as_bytes(), 5), Some(5));
        assert!(validate_code_module(js_ok.as_bytes(), 5));

        let js_bad = "function oops(x) { return [1, 2; }\n";
        assert_eq!(compute_code_max_depth(js_bad.as_bytes(), 5), None);

        // Depth boundary: 16 accepted, 17 rejected, 0 rejected
        let d16 = format!("{}{}{}", "(".repeat(16), "x", ")".repeat(16));
        assert_eq!(compute_code_max_depth(d16.as_bytes(), 3), Some(16));
        let d17 = format!("{}{}{}", "(".repeat(17), "x", ")".repeat(17));
        assert_eq!(compute_code_max_depth(d17.as_bytes(), 3), None);
        assert_eq!(compute_code_max_depth(b"x = 1\ny = 2\n", 3), None);
    }

    #[test]
    fn test_idx_record_roundtrip_qibbbb() {
        let rec = ShardIndexRecord {
            start_offset: 0x0102_0304_0506_0708,
            byte_length: 4096,
            lang_id: 5,
            source_id: 42,
            max_depth: 11,
            style_id: 4,
        };
        let bytes = rec.to_le_bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), rec.start_offset);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), rec.byte_length);
        assert_eq!(bytes[12], 5);
        assert_eq!(bytes[13], 42);
        assert_eq!(bytes[14], 11);
        assert_eq!(bytes[15], 4);

        let decoded = ShardIndexRecord::from_le_bytes(&bytes);
        assert_eq!(decoded, rec);
    }

    #[test]
    fn test_parse_lang_v3_ssot() {
        assert_eq!(parse_lang("en"), 0);
        assert_eq!(parse_lang("zh"), 1);
        assert_eq!(parse_lang("formal"), 2);
        assert_eq!(parse_lang("python"), 3);
        assert_eq!(parse_lang("sql_struct"), 4);
        assert_eq!(parse_lang("javascript"), 5);
    }
}
