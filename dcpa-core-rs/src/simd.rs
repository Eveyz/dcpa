//! Native SIMD math primitives (ARM64 NEON for Apple Silicon, x86_64 AVX2+FMA, scalar fallback).
//!
//! Includes 4-row register-blocked matrix-vector micro-kernels (`linear_f32` & `linear_bf16_f32`)
//! that reuse each loaded input vector register across 4 matrix rows simultaneously.
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

/// Hardware-accelerated f32 dot product (ARM64 NEON / x86_64 AVX2 / scalar).
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        while i + 16 <= n {
            let va0 = _mm256_loadu_ps(ap.add(i));
            let vb0 = _mm256_loadu_ps(bp.add(i));
            let va1 = _mm256_loadu_ps(ap.add(i + 8));
            let vb1 = _mm256_loadu_ps(bp.add(i + 8));
            #[cfg(target_feature = "fma")]
            {
                acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
                acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            }
            #[cfg(not(target_feature = "fma"))]
            {
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
            }
            i += 16;
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        while i + 8 <= n {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            #[cfg(target_feature = "fma")]
            {
                acc0 = _mm256_fmadd_ps(va, vb, acc0);
            }
            #[cfg(not(target_feature = "fma"))]
            {
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
            }
            i += 8;
        }
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        return sum;
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// 4-row register-blocked f32 dot product: computes `(dot(r0, x), dot(r1, x), dot(r2, x), dot(r3, x))`
/// while loading `x` only ONCE from L1 cache.
#[inline]
fn dot4_f32(
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    r3: &[f32],
    x: &[f32],
) -> (f32, f32, f32, f32) {
    let n = x.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);

        let xp = x.as_ptr();
        let p0 = r0.as_ptr();
        let p1 = r1.as_ptr();
        let p2 = r2.as_ptr();
        let p3 = r3.as_ptr();
        let mut i = 0usize;

        while i + 8 <= n {
            let vx0 = vld1q_f32(xp.add(i));
            let vx1 = vld1q_f32(xp.add(i + 4));

            a0 = vfmaq_f32(a0, vld1q_f32(p0.add(i)), vx0);
            b0 = vfmaq_f32(b0, vld1q_f32(p0.add(i + 4)), vx1);

            a1 = vfmaq_f32(a1, vld1q_f32(p1.add(i)), vx0);
            b1 = vfmaq_f32(b1, vld1q_f32(p1.add(i + 4)), vx1);

            a2 = vfmaq_f32(a2, vld1q_f32(p2.add(i)), vx0);
            b2 = vfmaq_f32(b2, vld1q_f32(p2.add(i + 4)), vx1);

            a3 = vfmaq_f32(a3, vld1q_f32(p3.add(i)), vx0);
            b3 = vfmaq_f32(b3, vld1q_f32(p3.add(i + 4)), vx1);
            i += 8;
        }
        a0 = vaddq_f32(a0, b0);
        a1 = vaddq_f32(a1, b1);
        a2 = vaddq_f32(a2, b2);
        a3 = vaddq_f32(a3, b3);
        while i + 4 <= n {
            let vx = vld1q_f32(xp.add(i));
            a0 = vfmaq_f32(a0, vld1q_f32(p0.add(i)), vx);
            a1 = vfmaq_f32(a1, vld1q_f32(p1.add(i)), vx);
            a2 = vfmaq_f32(a2, vld1q_f32(p2.add(i)), vx);
            a3 = vfmaq_f32(a3, vld1q_f32(p3.add(i)), vx);
            i += 4;
        }
        let mut s0 = vaddvq_f32(a0);
        let mut s1 = vaddvq_f32(a1);
        let mut s2 = vaddvq_f32(a2);
        let mut s3 = vaddvq_f32(a3);
        while i < n {
            let xv = *xp.add(i);
            s0 += *p0.add(i) * xv;
            s1 += *p1.add(i) * xv;
            s2 += *p2.add(i) * xv;
            s3 += *p3.add(i) * xv;
            i += 1;
        }
        return (s0, s1, s2, s3);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        (dot_f32(r0, x), dot_f32(r1, x), dot_f32(r2, x), dot_f32(r3, x))
    }
}

/// Hardware-accelerated bf16-row dot f32-vector product (ARM64 NEON / x86_64 AVX2 / scalar).
#[inline]
pub fn dot_bf16_f32(row: &[Bf16], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), x.len());
    let n = x.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let rp = row.as_ptr() as *const u16;
        let xp = x.as_ptr();
        while i + 8 <= n {
            let u8_lanes = vld1q_u16(rp.add(i));
            let w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u8_lanes), 16));
            let w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u8_lanes), 16));
            acc0 = vfmaq_f32(acc0, w0, vld1q_f32(xp.add(i)));
            acc1 = vfmaq_f32(acc1, w1, vld1q_f32(xp.add(i + 4)));
            i += 8;
        }
        acc0 = vaddq_f32(acc0, acc1);
        while i + 4 <= n {
            let u4 = vld1_u16(rp.add(i));
            let w = vreinterpretq_f32_u32(vshll_n_u16(u4, 16));
            acc0 = vfmaq_f32(acc0, w, vld1q_f32(xp.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(acc0);
        while i < n {
            sum += (*row.get_unchecked(i)).to_f32() * *xp.add(i);
            i += 1;
        }
        return sum;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut acc0 = _mm256_setzero_ps();
        let mut i = 0usize;
        let rp = row.as_ptr() as *const __m128i;
        let xp = x.as_ptr();
        let mut k = 0usize;
        while i + 8 <= n {
            let u8_lanes = _mm_loadu_si128(rp.add(k));
            let w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(u8_lanes), 16);
            let wf = _mm256_castsi256_ps(w32);
            let vx = _mm256_loadu_ps(xp.add(i));
            #[cfg(target_feature = "fma")]
            {
                acc0 = _mm256_fmadd_ps(wf, vx, acc0);
            }
            #[cfg(not(target_feature = "fma"))]
            {
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(wf, vx));
            }
            i += 8;
            k += 1;
        }
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        while i < n {
            sum += (*row.get_unchecked(i)).to_f32() * *xp.add(i);
            i += 1;
        }
        return sum;
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += row[i].to_f32() * x[i];
        }
        sum
    }
}

/// 4-row register-blocked bf16 GEMV micro-kernel: halves weight memory bandwidth AND cuts x loads 4x!
#[inline]
fn dot4_bf16_f32(
    r0: &[Bf16],
    r1: &[Bf16],
    r2: &[Bf16],
    r3: &[Bf16],
    x: &[f32],
) -> (f32, f32, f32, f32) {
    let n = x.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);

        let xp = x.as_ptr();
        let p0 = r0.as_ptr() as *const u16;
        let p1 = r1.as_ptr() as *const u16;
        let p2 = r2.as_ptr() as *const u16;
        let p3 = r3.as_ptr() as *const u16;
        let mut i = 0usize;

        while i + 8 <= n {
            let vx0 = vld1q_f32(xp.add(i));
            let vx1 = vld1q_f32(xp.add(i + 4));

            let u0 = vld1q_u16(p0.add(i));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u0), 16)), vx0);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u0), 16)), vx1);

            let u1 = vld1q_u16(p1.add(i));
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u1), 16)), vx0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u1), 16)), vx1);

            let u2 = vld1q_u16(p2.add(i));
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u2), 16)), vx0);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u2), 16)), vx1);

            let u3 = vld1q_u16(p3.add(i));
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u3), 16)), vx0);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u3), 16)), vx1);
            i += 8;
        }
        a0 = vaddq_f32(a0, b0);
        a1 = vaddq_f32(a1, b1);
        a2 = vaddq_f32(a2, b2);
        a3 = vaddq_f32(a3, b3);
        while i + 4 <= n {
            let vx = vld1q_f32(xp.add(i));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p0.add(i)), 16)), vx);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p1.add(i)), 16)), vx);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p2.add(i)), 16)), vx);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p3.add(i)), 16)), vx);
            i += 4;
        }
        let mut s0 = vaddvq_f32(a0);
        let mut s1 = vaddvq_f32(a1);
        let mut s2 = vaddvq_f32(a2);
        let mut s3 = vaddvq_f32(a3);
        while i < n {
            let xv = *xp.add(i);
            s0 += (*r0.get_unchecked(i)).to_f32() * xv;
            s1 += (*r1.get_unchecked(i)).to_f32() * xv;
            s2 += (*r2.get_unchecked(i)).to_f32() * xv;
            s3 += (*r3.get_unchecked(i)).to_f32() * xv;
            i += 1;
        }
        return (s0, s1, s2, s3);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        (
            dot_bf16_f32(r0, x),
            dot_bf16_f32(r1, x),
            dot_bf16_f32(r2, x),
            dot_bf16_f32(r3, x),
        )
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm256_set1_ps(alpha);
        let mut i = 0usize;
        let xp = x.as_ptr();
        let yp = y.as_mut_ptr();
        while i + 8 <= n {
            let vx = _mm256_loadu_ps(xp.add(i));
            let vy = _mm256_loadu_ps(yp.add(i));
            #[cfg(target_feature = "fma")]
            let res = _mm256_fmadd_ps(vx, va, vy);
            #[cfg(not(target_feature = "fma"))]
            let res = _mm256_add_ps(vy, _mm256_mul_ps(vx, va));
            _mm256_storeu_ps(yp.add(i), res);
            i += 8;
        }
        while i < n {
            *yp.add(i) += alpha * *xp.add(i);
            i += 1;
        }
        return;
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        for i in 0..n {
            y[i] += alpha * x[i];
        }
    }
}

/// Matrix-vector multiply: out[r] = dot(w[r], x) + bias[r] (optional).
/// Uses 4-row register-blocked micro-kernel on row-major `[out_dim, in_dim]`.
#[inline]
pub fn linear_f32(x: &[f32], w: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    let out_dim = out.len();
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut r = 0usize;
    match bias {
        Some(b) => {
            debug_assert_eq!(b.len(), out_dim);
            while r + 4 <= out_dim {
                let (s0, s1, s2, s3) = dot4_f32(
                    &w[r * in_dim..(r + 1) * in_dim],
                    &w[(r + 1) * in_dim..(r + 2) * in_dim],
                    &w[(r + 2) * in_dim..(r + 3) * in_dim],
                    &w[(r + 3) * in_dim..(r + 4) * in_dim],
                    x,
                );
                out[r] = s0 + b[r];
                out[r + 1] = s1 + b[r + 1];
                out[r + 2] = s2 + b[r + 2];
                out[r + 3] = s3 + b[r + 3];
                r += 4;
            }
            while r < out_dim {
                out[r] = dot_f32(&w[r * in_dim..(r + 1) * in_dim], x) + b[r];
                r += 1;
            }
        }
        None => {
            while r + 4 <= out_dim {
                let (s0, s1, s2, s3) = dot4_f32(
                    &w[r * in_dim..(r + 1) * in_dim],
                    &w[(r + 1) * in_dim..(r + 2) * in_dim],
                    &w[(r + 2) * in_dim..(r + 3) * in_dim],
                    &w[(r + 3) * in_dim..(r + 4) * in_dim],
                    x,
                );
                out[r] = s0;
                out[r + 1] = s1;
                out[r + 2] = s2;
                out[r + 3] = s3;
                r += 4;
            }
            while r < out_dim {
                out[r] = dot_f32(&w[r * in_dim..(r + 1) * in_dim], x);
                r += 1;
            }
        }
    }
}

/// Matrix-vector multiply over compact `Bf16` row-major `[out_dim, in_dim]` weights into `f32` output.
#[inline]
pub fn linear_bf16_f32(x: &[f32], w: &[Bf16], bias: Option<&[f32]>, out: &mut [f32]) {
    let out_dim = out.len();
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut r = 0usize;
    match bias {
        Some(b) => {
            debug_assert_eq!(b.len(), out_dim);
            while r + 4 <= out_dim {
                let (s0, s1, s2, s3) = dot4_bf16_f32(
                    &w[r * in_dim..(r + 1) * in_dim],
                    &w[(r + 1) * in_dim..(r + 2) * in_dim],
                    &w[(r + 2) * in_dim..(r + 3) * in_dim],
                    &w[(r + 3) * in_dim..(r + 4) * in_dim],
                    x,
                );
                out[r] = s0 + b[r];
                out[r + 1] = s1 + b[r + 1];
                out[r + 2] = s2 + b[r + 2];
                out[r + 3] = s3 + b[r + 3];
                r += 4;
            }
            while r < out_dim {
                out[r] = dot_bf16_f32(&w[r * in_dim..(r + 1) * in_dim], x) + b[r];
                r += 1;
            }
        }
        None => {
            while r + 4 <= out_dim {
                let (s0, s1, s2, s3) = dot4_bf16_f32(
                    &w[r * in_dim..(r + 1) * in_dim],
                    &w[(r + 1) * in_dim..(r + 2) * in_dim],
                    &w[(r + 2) * in_dim..(r + 3) * in_dim],
                    &w[(r + 3) * in_dim..(r + 4) * in_dim],
                    x,
                );
                out[r] = s0;
                out[r + 1] = s1;
                out[r + 2] = s2;
                out[r + 3] = s3;
                r += 4;
            }
            while r < out_dim {
                out[r] = dot_bf16_f32(&w[r * in_dim..(r + 1) * in_dim], x);
                r += 1;
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

/// Unit-RMS normalize without affine scale (`x * rsqrt(mean(x^2) + eps)`).
#[inline]
pub fn unit_rmsnorm_f32(x: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let sq_sum = dot_f32(x, x);
    let inv_rms = 1.0 / sqrtf(sq_sum / (n as f32) + eps);
    for i in 0..n {
        out[i] = x[i] * inv_rms;
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
