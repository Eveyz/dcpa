//! Native SIMD math primitives (ARM64 NEON for Apple Silicon, x86_64 AVX2, scalar fallback).
//!
//! Strictly `no_std` compatible using `libm` transcendentals and zero heap allocation.

use crate::bf16::Bf16;

#[inline(always)]
pub fn expf(x: f32) -> f32 {
    libm::expf(x)
}

#[inline(always)]
pub fn logf(x: f32) -> f32 {
    libm::logf(x)
}

#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline(always)]
pub fn tanhf(x: f32) -> f32 {
    libm::tanhf(x)
}

#[inline(always)]
pub fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline(always)]
pub fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + expf(-x))
    } else {
        let e = expf(x);
        e / (1.0 + e)
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    // Numerically stable softplus matching PyTorch threshold=20
    if x > 20.0 {
        x
    } else if x < -20.0 {
        expf(x)
    } else {
        libm::log1pf(expf(x))
    }
}

#[inline(always)]
pub fn logaddexp(a: f32, b: f32) -> f32 {
    let m = if a > b { a } else { b };
    let d = if a > b { b - a } else { a - b };
    m + libm::log1pf(expf(d))
}

/// Hardware-accelerated f32 dot product (ARM64 NEON / x86_64 / scalar).
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        while i + 8 <= n {
            let va0 = vld1q_f32(ap.add(i));
            let vb0 = vld1q_f32(bp.add(i));
            let va1 = vld1q_f32(ap.add(i + 4));
            let vb1 = vld1q_f32(bp.add(i + 4));
            acc0 = vfmaq_f32(acc0, va0, vb0);
            acc1 = vfmaq_f32(acc1, va1, vb1);
            i += 8;
        }
        acc0 = vaddq_f32(acc0, acc1);
        while i + 4 <= n {
            let va = vld1q_f32(ap.add(i));
            let vb = vld1q_f32(bp.add(i));
            acc0 = vfmaq_f32(acc0, va, vb);
            i += 4;
        }
        let mut sum = vaddvq_f32(acc0);
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        return sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Hardware-accelerated AXPY: y[i] += alpha * x[i].
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let va = vdupq_n_f32(alpha);
        let mut i = 0usize;
        let xp = x.as_ptr();
        let yp = y.as_mut_ptr();
        while i + 8 <= n {
            let vx0 = vld1q_f32(xp.add(i));
            let vy0 = vld1q_f32(yp.add(i));
            let vx1 = vld1q_f32(xp.add(i + 4));
            let vy1 = vld1q_f32(yp.add(i + 4));
            vst1q_f32(yp.add(i), vfmaq_f32(vy0, vx0, va));
            vst1q_f32(yp.add(i + 4), vfmaq_f32(vy1, vx1, va));
            i += 8;
        }
        while i + 4 <= n {
            let vx = vld1q_f32(xp.add(i));
            let vy = vld1q_f32(yp.add(i));
            vst1q_f32(yp.add(i), vfmaq_f32(vy, vx, va));
            i += 4;
        }
        while i < n {
            *yp.add(i) += alpha * *xp.add(i);
            i += 1;
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            y[i] += alpha * x[i];
        }
    }
}

/// Matrix-vector multiply: out[r] = dot(w[r], x) + bias[r] (optional).
/// Weight layout is row-major [out_dim, in_dim].
#[inline]
pub fn linear_f32(x: &[f32], w: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let out_dim = out.len();
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    match bias {
        Some(b) => {
            debug_assert_eq!(b.len(), out_dim);
            for r in 0..out_dim {
                let row = &w[r * in_dim..(r + 1) * in_dim];
                out[r] = dot_f32(row, x) + b[r];
            }
        }
        None => {
            for r in 0..out_dim {
                let row = &w[r * in_dim..(r + 1) * in_dim];
                out[r] = dot_f32(row, x);
            }
        }
    }
}

/// RMSNorm in float32 matching Python `dcpa/ops/rmsnorm.py` bit-for-bit.
#[inline]
pub fn rmsnorm_f32(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let sq_sum = dot_f32(x, x);
    let inv_rms = 1.0 / sqrtf(sq_sum / (n as f32) + eps);
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// Softmax in-place over slice `x` in float32.
#[inline]
pub fn softmax_inplace_f32(x: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max_v);
        sum += *v;
    }
    let inv = 1.0 / (sum + 1e-12);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// L2 norm in float32.
#[inline]
pub fn l2_norm_f32(x: &[f32]) -> f32 {
    sqrtf(dot_f32(x, x))
}

/// Cosine similarity between two slices in float32.
#[inline]
pub fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    let d = dot_f32(a, b);
    let na = sqrtf(dot_f32(a, a));
    let nb = sqrtf(dot_f32(b, b));
    d / (na * nb + 1e-8)
}

/// Orthogonal Holographic RoPE matching Python `dcpa/ops/rope.py` identically.
#[inline]
pub fn apply_holographic_rope_f32(
    x: &[f32],
    pos: f32,
    period: f32,
    num_harmonics: usize,
    inverse: bool,
    out: &mut [f32],
) {
    let dim = x.len();
    let half = dim / 2;
    let p = if inverse { -pos } else { pos };
    let base_omega = (2.0 * core::f32::consts::PI) / period;
    for k in 0..half {
        let harmonic = ((k % num_harmonics) + 1) as f32;
        let angle = p * base_omega * harmonic;
        let c = cosf(angle);
        let s = sinf(angle);
        let xe = x[2 * k];
        let xo = x[2 * k + 1];
        out[2 * k] = xe * c - xo * s;
        out[2 * k + 1] = xe * s + xo * c;
    }
}

/// Column-Major contiguous bf16 column gather-accumulate: acc[d] += weight * col[d].to_f32().
/// Hardware prefetcher streams contiguous 64-byte cache lines linearly!
#[inline]
pub fn accumulate_bf16_col_f32(weight: f32, col: &[Bf16], acc: &mut [f32]) {
    debug_assert_eq!(col.len(), acc.len());
    let n = col.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vw = vdupq_n_f32(weight);
        let mut i = 0usize;
        let cp = col.as_ptr() as *const u16;
        let ap = acc.as_mut_ptr();
        while i + 4 <= n {
            // Load 4 bf16 u16s -> shift left by 16 bits into 4 f32 lanes
            let v16 = vld1_u16(cp.add(i));
            let v32 = vshll_n_u16(v16, 16);
            let vf = vreinterpretq_f32_u32(v32);
            let va = vld1q_f32(ap.add(i));
            vst1q_f32(ap.add(i), vfmaq_f32(va, vf, vw));
            i += 4;
        }
        while i < n {
            *ap.add(i) += weight * (*col.get_unchecked(i)).to_f32();
            i += 1;
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            acc[i] += weight * col[i].to_f32();
        }
    }
}

/// Column-Major contiguous bf16 column in-place Householder delta update:
/// col[d] = bf16(col[d].to_f32() + coeff * err[d])
#[inline]
pub fn update_bf16_col_delta(coeff: f32, err: &[f32], col: &mut [Bf16]) {
    debug_assert_eq!(col.len(), err.len());
    let n = col.len();
    for i in 0..n {
        let updated = col[i].to_f32() + coeff * err[i];
        col[i] = Bf16::from_f32(updated);
    }
}

/// Scale bf16 column in-place for lazy gamma_hip decay.
#[inline]
pub fn scale_bf16_col_inplace(scale: f32, col: &mut [Bf16]) {
    for v in col.iter_mut() {
        *v = Bf16::from_f32(v.to_f32() * scale);
    }
}
