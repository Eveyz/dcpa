//! dcpa-data-forge: Multi-threaded Rust developmental corpus builder for DCPA.
//!
//! Enforces Engineering Blueprint Section 4.1 - 4.3:
//!   - Strict raw UTF-8 NFC normalization (ZERO tokenizers / BPE).
//!   - Uniform single-byte LF (0x0A) line endings; preserves spaces (0x20) & TAB (0x09).
//!   - Strips invisible control bytes (0x00..=0x08, 0x0B..=0x1F, 0x7F).
//!   - Balanced AST/delimiter validation for Python & Rust code files (200..=8000 bytes).
//!   - Emits contiguous delimiter-free `shard.bin` + fixed 16-byte `shard.idx`:
//!       startOffset: u64 | byteLength: u32 | langDomainId: u16 | stageMask: u16
//!   - Built-in parallel generators for Category 1 (2D Grid Physics + Dyck-k) and
//!     Category 4 (Fast-Mapping Pseudo-word Episodes).

use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// 16-byte little-endian binary shard index record (Engineering Doc Section 4.1).
#[derive(Debug, Clone, Copy)]
pub struct ShardIndexRecord {
    pub start_offset: u64,
    pub byte_length: u32,
    pub lang_domain_id: u16, // 0=zh, 1=en, 2=ja, 3=python, 4=rust, 5=synthetic
    pub stage_mask: u16,     // bit0=StageA, bit1=StageB, bit2=StageC, bit3=StageD
}

impl ShardIndexRecord {
    pub const SIZE: usize = 16;

    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.start_offset.to_le_bytes());
        out[8..12].copy_from_slice(&self.byte_length.to_le_bytes());
        out[12..14].copy_from_slice(&self.lang_domain_id.to_le_bytes());
        out[14..16].copy_from_slice(&self.stage_mask.to_le_bytes());
        out
    }

    pub fn from_le_bytes(b: &[u8; 16]) -> Self {
        Self {
            start_offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            byte_length: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            lang_domain_id: u16::from_le_bytes(b[12..14].try_into().unwrap()),
            stage_mask: u16::from_le_bytes(b[14..16].try_into().unwrap()),
        }
    }
}

/// Normalize UTF-8 text to NFC, unify CRLF/CR to LF (0x0A), keep TAB (0x09), strip control bytes.
pub fn normalize_utf8_bytes(input: &str) -> Vec<u8> {
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

/// Structural syntax & delimiter balance validator for Python (3) and Rust (4) source files.
/// Checks byte length in [200, 8000] and strict {}()[] balance outside string literals & comments.
pub fn validate_code_module(bytes: &[u8], lang_domain_id: u16) -> bool {
    if bytes.len() < 200 || bytes.len() > 8000 {
        return false;
    }
    let Ok(text) = core::str::from_utf8(bytes) else {
        return false;
    };
    let mut stack: Vec<u8> = Vec::with_capacity(64);
    let b = text.as_bytes();
    let mut i = 0;
    let mut in_string: Option<u8> = None;
    let mut in_line_comment = false;

    while i < b.len() {
        let c = b[i];
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if let Some(q) = in_string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                in_string = None;
            }
            i += 1;
            continue;
        }

        // Line comment detection
        if lang_domain_id == 3 && c == b'#' {
            in_line_comment = true;
            i += 1;
            continue;
        }
        if lang_domain_id == 4 && c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if c == b'"' || (lang_domain_id == 3 && c == b'\'') {
            in_string = Some(c);
            i += 1;
            continue;
        }

        match c {
            b'{' => stack.push(b'}'),
            b'(' => stack.push(b')'),
            b'[' => stack.push(b']'),
            b'}' | b')' | b']' => {
                if stack.pop() != Some(c) {
                    return false;
                }
            }
            _ => {}
        }
        i += 1;
    }

    in_string.is_none() && stack.is_empty()
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
    normalize_utf8_bytes(&s)
}

fn generate_dyck_doc(seed: u64) -> Vec<u8> {
    let mut rng = Rng64::new(seed);
    let pairs = [('{' , '}'), ('(', ')'), ('[', ']')];
    let max_depth = 3 + rng.gen_range(12); // depth 3..=14
    let target_len = 300 + rng.gen_range(1200);
    let rust_style = (seed & 1) == 1;

    let mut stack: Vec<char> = Vec::with_capacity(16);
    let mut out = String::with_capacity(target_len + 128);

    while stack.len() < max_depth {
        let (op, cl) = pairs[rng.gen_range(3)];
        if rust_style && op == '{' {
            out.push_str(&format!("fn b_{}() ", stack.len()));
        }
        out.push(op);
        stack.push(cl);
        if rust_style {
            out.push_str(&format!(" let v{} = {}; ", stack.len(), rng.gen_range(99)));
        } else {
            out.push_str(&format!(" x{} ", stack.len()));
        }
    }

    while out.len() < target_len {
        if stack.is_empty() || (stack.len() < max_depth && rng.gen_range(100) < 55) {
            let (op, cl) = pairs[rng.gen_range(3)];
            out.push(op);
            stack.push(cl);
            out.push_str(&format!(" n{} ", rng.gen_range(256)));
        } else {
            let cl = stack.pop().unwrap();
            out.push(cl);
            out.push(' ');
        }
    }

    while let Some(cl) = stack.pop() {
        out.push_str(if rust_style { " ok; " } else { " end " });
        out.push(cl);
    }
    out.push('\n');
    normalize_utf8_bytes(&out)
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
    normalize_utf8_bytes(&full)
}

fn write_shard(prefix: &Path, docs: &[(Vec<u8>, u16, u16)]) -> std::io::Result<()> {
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
    for (bytes, lang_id, stage_mask) in docs {
        if bytes.is_empty() {
            continue;
        }
        bin_w.write_all(bytes)?;
        let rec = ShardIndexRecord {
            start_offset: offset,
            byte_length: bytes.len() as u32,
            lang_domain_id: *lang_id,
            stage_mask: *stage_mask,
        };
        idx_w.write_all(&rec.to_le_bytes())?;
        offset += bytes.len() as u64;
    }
    bin_w.flush()?;
    idx_w.flush()?;
    println!(
        "Wrote shard {:?} ({} docs, {} total bytes) + {:?}",
        bin_path,
        docs.len(),
        offset,
        idx_path
    );
    Ok(())
}

fn parse_lang(s: &str) -> u16 {
    match s {
        "zh" => 0,
        "en" => 1,
        "ja" => 2,
        "python" | "py" => 3,
        "rust" | "rs" => 4,
        _ => 5,
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
        total_bytes += rec.byte_length as u64;
    }
    println!(
        "Inspected {:?}: {} records, {} bytes, {}/{} valid UTF-8",
        prefix, n, total_bytes, valid_utf8, n
    );
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  dcpa-data-forge forge <input_dir> <lang:zh|en|ja|python|rust|synthetic> <stage_mask> <out_prefix>");
        eprintln!("  dcpa-data-forge synth-stage1 <num_physics> <num_dyck> <out_prefix>");
        eprintln!("  dcpa-data-forge synth-fastmap <num_episodes> <out_prefix>");
        eprintln!("  dcpa-data-forge inspect <shard_prefix>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "forge" => {
            let input_dir = PathBuf::from(&args[2]);
            let lang_id = parse_lang(&args[3]);
            let stage_mask: u16 = args[4].parse().expect("stage_mask must be u16");
            let out_prefix = PathBuf::from(&args[5]);

            let mut files = Vec::new();
            collect_files(&input_dir, &mut files)?;
            let is_code = lang_id == 3 || lang_id == 4;

            let docs: Vec<(Vec<u8>, u16, u16)> = files
                .par_iter()
                .filter_map(|path| {
                    let mut raw = String::new();
                    File::open(path).ok()?.read_to_string(&mut raw).ok()?;
                    let norm = normalize_utf8_bytes(&raw);
                    if norm.is_empty() {
                        return None;
                    }
                    if is_code && !validate_code_module(&norm, lang_id) {
                        return None;
                    }
                    Some((norm, lang_id, stage_mask))
                })
                .collect();

            write_shard(&out_prefix, &docs)?;
        }
        "synth-stage1" => {
            let n_phys: usize = args[2].parse().expect("num_physics");
            let n_dyck: usize = args[3].parse().expect("num_dyck");
            let out_prefix = PathBuf::from(&args[4]);

            let mut docs: Vec<(Vec<u8>, u16, u16)> = (0..n_phys)
                .into_par_iter()
                .map(|i| (generate_grid_physics_doc(1000 + i as u64), 5u16, 0b0011u16))
                .collect();
            let dyck_docs: Vec<(Vec<u8>, u16, u16)> = (0..n_dyck)
                .into_par_iter()
                .map(|i| (generate_dyck_doc(500_000 + i as u64), 5u16, 0b0111u16))
                .collect();
            docs.extend(dyck_docs);
            write_shard(&out_prefix, &docs)?;
        }
        "synth-fastmap" => {
            let n_ep: usize = args[2].parse().expect("num_episodes");
            let out_prefix = PathBuf::from(&args[3]);
            let docs: Vec<(Vec<u8>, u16, u16)> = (0..n_ep)
                .into_par_iter()
                .map(|i| (generate_fast_mapping_doc(900_000 + i as u64), 1u16, 0b1000u16))
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
        // Decomposed "e\u{0301}" -> NFC "\u{00e9}", CRLF -> LF, TAB kept, NUL/BEL stripped
        let raw = "a\x00b\x07\tc\r\nd\re\u{0301}";
        let norm = normalize_utf8_bytes(raw);
        assert_eq!(core::str::from_utf8(&norm).unwrap(), "ab\tc\nd\n\u{00e9}");
    }

    #[test]
    fn test_code_validator() {
        let good_rust = format!("fn test() {{\n{}\n}}\n", "    let x = (1 + 2);\n".repeat(12));
        assert!(validate_code_module(good_rust.as_bytes(), 4));
        let bad_rust = format!("fn broken() {{\n{}\n", "    let x = (1 + 2);\n".repeat(12));
        assert!(!validate_code_module(bad_rust.as_bytes(), 4));
    }

    #[test]
    fn test_idx_record_roundtrip() {
        let rec = ShardIndexRecord {
            start_offset: 123456789,
            byte_length: 4096,
            lang_domain_id: 4,
            stage_mask: 0b1111,
        };
        let bytes = rec.to_le_bytes();
        assert_eq!(bytes.len(), 16);
        let decoded = ShardIndexRecord::from_le_bytes(&bytes);
        assert_eq!(decoded.start_offset, rec.start_offset);
        assert_eq!(decoded.byte_length, rec.byte_length);
        assert_eq!(decoded.lang_domain_id, rec.lang_domain_id);
        assert_eq!(decoded.stage_mask, rec.stage_mask);
    }
}
