//! IEEE-754 bfloat16 conversion & 64-byte cache-line aligned storage buffer.
//!
//! Enforces Engineering Blueprint Chapter 6 Law #1:
//! Every column of hippocampal fast weights M^{C<-L} and M^{L<-C} is stored
//! contiguously in Column-Major order with strict 64-byte cache-line alignment.

use alloc::alloc::{alloc_zeroed, dealloc, Layout};
use core::ptr::NonNull;
use core::slice;

/// Compact 16-bit bfloat16 scalar (no external dependencies, `no_std` compatible).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bf16(pub u16);

impl Bf16 {
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Round-to-nearest-even (RNE) matching PyTorch/hardware bf16 conversion
        if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
            // Preserve NaN
            return Self(((bits >> 16) | 0x0040) as u16);
        }
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        Self(((bits.wrapping_add(rounding_bias)) >> 16) as u16)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// 64-byte cache-line aligned heap allocation created ONCE at startup (never resized in hot loop).
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

impl<T: Copy> AlignedVec<T> {
    pub const ALIGN: usize = 64;

    pub fn zeroed(len: usize) -> Self {
        assert!(len > 0, "AlignedVec length must be > 0");
        let byte_size = len
            .checked_mul(core::mem::size_of::<T>())
            .expect("AlignedVec size overflow");
        let align = Self::ALIGN.max(core::mem::align_of::<T>());
        let layout = Layout::from_size_align(byte_size, align).expect("invalid layout");
        let raw = unsafe { alloc_zeroed(layout) as *mut T };
        let ptr = NonNull::new(raw).expect("AlignedVec allocation failed");
        debug_assert_eq!((ptr.as_ptr() as usize) % Self::ALIGN, 0);
        Self { ptr, len, layout }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}
