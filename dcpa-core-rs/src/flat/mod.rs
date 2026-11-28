//! DCPA-Flat v3.3 pure static `no_std`-compatible Rust inference engine & replay capture.

pub mod config;
pub mod engine;
pub mod state;
pub mod weights;

pub use config::DcpaFlatConfig;
pub use engine::{
    DcpaFlatEngine, FlatByteStepOut, FlatPmdStepOut, FlatPriorStepOut, FlatSegmentStepOut,
};
pub use state::{
    BrainSnapshotV3, CarrierGraphStateRs, FlatEligState, FlatPmdState, FlatRingState,
    FlatScratchPool, FlatSgscState, SleepCapturedFragment, SleepReplayCaptureRing,
};
pub use weights::{DcpaFlatWeights, PluginExpertPackRs};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_config_validation() {
        let cfg = DcpaFlatConfig::flat_micro();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.d_head(), 32);
        assert_eq!(cfg.copy_ring_dk(), 16);
        assert_eq!(cfg.ring_dim(), 64);
        assert!(cfg.is_window_layer(0));
        assert!(cfg.is_window_layer(3));
        assert!(cfg.is_window_layer(5));
        assert!(!cfg.is_window_layer(1));

        let mut bad = cfg.clone();
        bad.vocab_size = 128;
        assert!(bad.validate().is_err());

        let mut bad2 = cfg.clone();
        bad2.n_layers = 9;
        assert!(bad2.validate().is_err());

        let mut bad3 = cfg.clone();
        bad3.edge_decay = 1.05;
        assert!(bad3.validate().is_err());

        let cfg206 = DcpaFlatConfig::flat_206m();
        assert!(cfg206.validate().is_ok());
        assert_eq!(cfg206.copy_ring_dk(), 24);
    }

    #[test]
    fn test_brain_snapshot_v3_size_and_copy() {
        let cfg_206 = DcpaFlatConfig::flat_206m();
        let sz_full = BrainSnapshotV3::byte_size_estimate_bf16(&cfg_206);
        assert!(
            sz_full >= 110_000 && sz_full <= 125_000,
            "206M snapshot size unexpected: {} B",
            sz_full
        );

        let mut cfg_shared = cfg_206.clone();
        cfg_shared.share_stem_ring = true;
        let sz_shared = BrainSnapshotV3::byte_size_estimate_bf16(&cfg_shared);
        assert_eq!(sz_full - sz_shared, 128 * 128 * 2);

        let micro = DcpaFlatConfig::flat_micro();
        let mut s1 = BrainSnapshotV3::new(&micro);
        let mut s2 = BrainSnapshotV3::new(&micro);
        s1.sgsc.h[0] = 3.25;
        s1.g1.slots[5] = -0.42;
        s1.g2.slot0_pinned = true;
        s1.rings.ring_pos[3] = 17;
        s1.rings.ring_valid[3] = true;
        s1.prev_rot_stem[1] = 0.77;
        s1.prev_x3[2] = -0.88;
        s1.pmd.xi[0] = [0.1, 0.2, 0.3, 0.4];
        s1.tau_think = 1.234;

        s2.copy_from(&s1);
        assert_eq!(s2.sgsc.h[0], 3.25);
        assert_eq!(s2.g1.slots[5], -0.42);
        assert!(s2.g2.slot0_pinned);
        assert_eq!(s2.rings.ring_pos[3], 17);
        assert!(s2.rings.ring_valid[3]);
        assert_eq!(s2.prev_rot_stem[1], 0.77);
        assert_eq!(s2.prev_x3[2], -0.88);
        assert_eq!(s2.pmd.xi[0], [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(s2.tau_think, 1.234);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_sleep_replay_capture_ring_shard_triplet() {
        let mut ring = SleepReplayCaptureRing::new(4, 1024, 0.25);
        // Below beta threshold without explicit correction -> ignored
        assert!(!ring.maybe_record(b"low beta text", 0, 0, &[], 0.15, false));
        assert_eq!(ring.len(), 0);

        // Above beta threshold -> recorded with anchor span
        let frag1 = b"SELECT id FROM users WHERE name = 'zibber';\n";
        assert!(ring.maybe_record(frag1, 4, 4, &[(34, 42)], 0.62, false));
        // Explicit correction overrides low beta
        let frag2 = "修复(alpha): 完成!\n".as_bytes();
        assert!(ring.maybe_record(frag2, 1, 1, &[(0, 6)], 0.05, true));
        assert_eq!(ring.len(), 2);

        let tmp = std::env::temp_dir().join("dcpa_flat_sleep_ring_unit_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let n = ring.write_shard_triplet(&tmp, "sleep_replay_ring").expect("write shards");
        assert_eq!(n, 2);

        let bin_data = std::fs::read(tmp.join("sleep_replay_ring.bin")).unwrap();
        let idx_data = std::fs::read(tmp.join("sleep_replay_ring.idx")).unwrap();
        let mask_data = std::fs::read(tmp.join("sleep_replay_ring.loss_mask.bin")).unwrap();

        assert_eq!(bin_data.len(), frag1.len() + frag2.len());
        assert_eq!(mask_data.len(), bin_data.len());
        assert_eq!(idx_data.len(), 2 * 16);

        // Verify <QIBBBB record 0
        let off0 = u64::from_le_bytes(idx_data[0..8].try_into().unwrap());
        let len0 = u32::from_le_bytes(idx_data[8..12].try_into().unwrap());
        assert_eq!(off0, 0);
        assert_eq!(len0 as usize, frag1.len());
        assert_eq!(idx_data[12], 4); // lang_id
        assert_eq!(idx_data[13], 99); // source_id
        assert_eq!(idx_data[15], 4); // style_id

        // Verify <QIBBBB record 1
        let off1 = u64::from_le_bytes(idx_data[16..24].try_into().unwrap());
        let len1 = u32::from_le_bytes(idx_data[24..28].try_into().unwrap());
        assert_eq!(off1 as usize, frag1.len());
        assert_eq!(len1 as usize, frag2.len());
        assert_eq!(idx_data[28], 1); // lang_id
        assert_eq!(idx_data[30], 1); // max_depth for "(alpha)"
        assert_eq!(idx_data[31], 1); // style_id

        // Verify fixed-point x10 loss mask weights: 10 normal, 15 delimiter, 30 anchor
        assert_eq!(mask_data[0], 10); // 'S'
        assert_eq!(mask_data[frag1.len() - 1], 15); // '\n'
        assert_eq!(mask_data[34], 30); // anchor span start
        assert_eq!(mask_data[41], 30); // anchor span end-1
        assert_eq!(mask_data[frag1.len()], 30); // frag2 anchor start

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
