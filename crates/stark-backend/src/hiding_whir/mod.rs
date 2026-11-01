//! ProofCard's pinned Plonky3 0.8 hiding-WHIR candidate types.

pub mod bridge;
pub mod codec;
pub mod transcript;

use p3_baby_bear_08::{
    default_babybear_poseidon2_16, default_babybear_poseidon2_24, BabyBear, Poseidon2BabyBear,
};
use p3_challenger_08::DuplexChallenger;
use p3_dft_08::Radix2DFTSmallBatch;
use p3_field_08::{extension::BinomialExtensionField, Field, TwoAdicField};
use p3_merkle_tree_08::MerkleTreeMmcs;
use p3_symmetric_08::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir_08::{
    FoldingFactor, HidingWhirPcs, ProtocolParameters, SecurityAssumption, ZkParameters,
    ZkWhirConfig,
};
use rand_10::rngs::StdRng;

pub type HidingBase = BabyBear;
pub type HidingExtension = BinomialExtensionField<HidingBase, 8>;
pub type OuterBase = p3_baby_bear::BabyBear;
pub type OuterExtension = p3_field::extension::BinomialExtensionField<OuterBase, 4>;
pub type HidingPoseidon16 = Poseidon2BabyBear<16>;
pub type HidingPoseidon24 = Poseidon2BabyBear<24>;
pub type HidingMerkleHash = PaddingFreeSponge<HidingPoseidon24, 24, 16, 8>;
pub type HidingMerkleCompress = TruncatedPermutation<HidingPoseidon16, 2, 8, 16>;
pub type HidingPackedBase = <HidingBase as Field>::Packing;
pub type HidingMerkleMmcs = MerkleTreeMmcs<
    HidingPackedBase,
    HidingPackedBase,
    HidingMerkleHash,
    HidingMerkleCompress,
    2,
    8,
>;
pub type HidingCommitment = p3_symmetric_08::MerkleCap<HidingBase, [HidingBase; 8]>;
pub type HidingDft = Radix2DFTSmallBatch<HidingBase>;
pub type RawHidingChallenger = DuplexChallenger<HidingBase, HidingPoseidon16, 16, 8>;
pub type HidingChallenger = transcript::V9Transcript;
pub type HidingConfig = ZkWhirConfig<HidingExtension, HidingBase, HidingChallenger>;
pub type HidingPcs = HidingWhirPcs<
    HidingExtension,
    HidingBase,
    HidingDft,
    HidingMerkleMmcs,
    HidingChallenger,
    StdRng,
>;

/// Constructs the VK-bound first candidate profile. M1 checks it against the exact P3 build.
#[derive(Debug, thiserror::Error)]
pub enum CandidateConfigError {
    #[error("{num_variables} variables exceed the BabyBear domain limit of {max_variables}")]
    UnsupportedDimension {
        num_variables: usize,
        max_variables: usize,
    },
    #[error(transparent)]
    Whir(#[from] p3_whir_08::ZkConfigError),
}

pub fn candidate_config(num_variables: usize) -> Result<HidingConfig, CandidateConfigError> {
    let max_variables = HidingBase::TWO_ADICITY as usize - 1;
    if num_variables > max_variables {
        return Err(CandidateConfigError::UnsupportedDimension {
            num_variables,
            max_variables,
        });
    }
    HidingConfig::new(
        num_variables,
        ProtocolParameters {
            security_level: 128,
            pow_bits: 0,
            round_log_inv_rates: vec![4],
            folding_factor: FoldingFactor::ConstantFromSecondRound(8, 6),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 1,
        },
        ZkParameters {
            ell_zk: 3,
            mask_log_inv_rate: 3,
        },
    )
    .map_err(CandidateConfigError::from)
}

pub fn candidate_mmcs() -> HidingMerkleMmcs {
    HidingMerkleMmcs::new(
        HidingMerkleHash::new(default_babybear_poseidon2_24()),
        HidingMerkleCompress::new(default_babybear_poseidon2_16()),
        0,
    )
}

pub fn raw_candidate_challenger() -> RawHidingChallenger {
    RawHidingChallenger::new(default_babybear_poseidon2_16())
}

pub fn candidate_challenger() -> HidingChallenger {
    transcript::V9Transcript::new(raw_candidate_challenger())
}

pub fn candidate_dft() -> HidingDft {
    HidingDft::default()
}

pub fn candidate_pcs(config: HidingConfig, rng: StdRng) -> HidingPcs {
    HidingPcs::new(config, candidate_dft(), candidate_mmcs(), rng)
}

#[cfg(test)]
mod tests;
