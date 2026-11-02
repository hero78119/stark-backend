//! ProofCard's pinned Plonky3 0.8 hiding-WHIR candidate types.

pub mod bridge;
pub mod codec;
pub mod transcript;

use p3_baby_bear_08::{
    default_babybear_poseidon2_16, default_babybear_poseidon2_24, BabyBear, Poseidon2BabyBear,
};
use p3_challenger_08::DuplexChallenger;
use p3_commit_08::MultilinearPcs;
use p3_dft_08::Radix2DFTSmallBatch;
use p3_field_08::{
    extension::BinomialExtensionField, Field, PrimeCharacteristicRing, TwoAdicField,
};
use p3_merkle_tree_08::MerkleTreeMmcs;
use p3_symmetric_08::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir_08::{
    FoldingFactor, HidingWhirPcs, HidingWhirProverData, ProtocolParameters, SecurityAssumption,
    ZkParameters, ZkWhirConfig, ZkWhirProof,
};
use rand_10::{
    rngs::{StdRng, SysRng},
    SeedableRng,
};

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
pub type HidingProverData = HidingWhirProverData<HidingBase, HidingExtension, HidingMerkleMmcs>;
pub type HidingProof = ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs>;

#[derive(Debug, thiserror::Error)]
pub enum CandidatePcsError {
    #[error(transparent)]
    Bridge(#[from] bridge::BridgeError),
    #[error(transparent)]
    Prover(#[from] p3_whir_08::WhirConfigError),
    #[error(transparent)]
    Verifier(#[from] p3_whir_08::ZkVerifierError),
    #[error("packed polynomial has {actual} variables; PCS is configured for {expected}")]
    VariableCountMismatch { expected: usize, actual: usize },
    #[error("the PCS opening claim differs from the OpenVM packed claim")]
    ClaimMismatch,
    #[error("PCS commitment/proof count does not match the bounded profile")]
    CommitmentCountMismatch,
    #[error("failed to initialize PCS masking randomness: {0}")]
    Randomness(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateProfile {
    pub num_variables: usize,
    pub num_commitments: usize,
    pub split_selector: bool,
}

pub fn candidate_profile(
    actual_variables: usize,
) -> Result<CandidateProfile, CandidateConfigError> {
    let max_variables = HidingBase::TWO_ADICITY as usize - 1;
    if actual_variables > max_variables + 1 {
        return Err(CandidateConfigError::UnsupportedDimension {
            num_variables: actual_variables,
            max_variables,
        });
    }
    let (num_variables, num_commitments, split_selector) = match actual_variables {
        0..=20 => (20, 1, false),
        21..=22 => (22, 1, false),
        23..=26 => (26, 1, false),
        27 => (26, 2, true),
        _ => {
            return Err(CandidateConfigError::UnsupportedProfileDimension(
                actual_variables,
            ))
        }
    };
    Ok(CandidateProfile {
        num_variables,
        num_commitments,
        split_selector,
    })
}

/// Constructs the VK-bound first candidate profile. M1 checks it against the exact P3 build.
#[derive(Debug, thiserror::Error)]
pub enum CandidateConfigError {
    #[error("{num_variables} variables exceed the BabyBear domain limit of {max_variables}")]
    UnsupportedDimension {
        num_variables: usize,
        max_variables: usize,
    },
    #[error("{0} variables are not in the fixed ProofCard candidate profiles (20, 22, 26)")]
    UnsupportedProfileDimension(usize),
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
    let (folding_factor, round_log_inv_rates, security_level) = match num_variables {
        20 => (FoldingFactor::ConstantFromSecondRound(8, 6), vec![4], 128),
        22 => (FoldingFactor::ConstantFromSecondRound(8, 8), vec![4], 128),
        26 => (
            FoldingFactor::ConstantFromSecondRound(8, 6),
            vec![4, 4],
            132,
        ),
        _ => {
            return Err(CandidateConfigError::UnsupportedProfileDimension(
                num_variables,
            ))
        }
    };
    HidingConfig::new(
        num_variables,
        ProtocolParameters {
            security_level,
            pow_bits: 0,
            round_log_inv_rates,
            folding_factor,
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

pub fn candidate_validate_claims(pcs: &HidingPcs) -> Result<(), CandidatePcsError> {
    pcs.config.validate_initial_claims(1)?;
    Ok(())
}

pub fn candidate_pcs_from_os_rng(config: HidingConfig) -> Result<HidingPcs, String> {
    let mut sys_rng = SysRng;
    let rng = StdRng::try_from_rng(&mut sys_rng).map_err(|error| error.to_string())?;
    Ok(candidate_pcs(config, rng))
}

pub fn candidate_commit(
    pcs: &HidingPcs,
    columns: &[Vec<OuterBase>],
    profile: CandidateProfile,
    challenger: &mut HidingChallenger,
) -> Result<(Vec<HidingCommitment>, Vec<HidingProverData>), CandidatePcsError> {
    if pcs.num_vars() != profile.num_variables
        || profile.num_commitments != if profile.split_selector { 2 } else { 1 }
    {
        return Err(CandidatePcsError::CommitmentCountMismatch);
    }
    let column_height = columns
        .first()
        .filter(|column| !column.is_empty() && column.len().is_power_of_two())
        .ok_or(bridge::BridgeError::EmptyColumns)?
        .len();
    if columns.iter().any(|column| column.len() != column_height) {
        return Err(bridge::BridgeError::InvalidColumnShape.into());
    }
    let row_variables = column_height.trailing_zeros() as usize;
    let selector_variables = profile.num_variables.checked_sub(row_variables).ok_or(
        CandidatePcsError::VariableCountMismatch {
            expected: profile.num_variables,
            actual: row_variables,
        },
    )?;
    let target_width = 1usize
        .checked_shl(selector_variables as u32)
        .ok_or(bridge::BridgeError::WidthOverflow)?;

    let chunks: Vec<&[Vec<OuterBase>]> = if profile.split_selector {
        if profile.num_variables != 26 || columns.len() <= target_width {
            return Err(CandidatePcsError::CommitmentCountMismatch);
        }
        vec![&columns[..target_width], &columns[target_width..]]
    } else {
        if columns.len() > target_width {
            return Err(CandidatePcsError::VariableCountMismatch {
                expected: profile.num_variables,
                actual: row_variables + columns.len().next_power_of_two().trailing_zeros() as usize,
            });
        }
        vec![columns]
    };

    let mut commitments = Vec::with_capacity(chunks.len());
    let mut prover_data = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let witness = bridge::pack_columns_at_width(chunk, target_width)?;
        let (commitment, data) = MultilinearPcs::commit(pcs, witness, challenger)?;
        commitments.push(commitment);
        prover_data.push(data);
    }
    Ok((commitments, prover_data))
}

pub fn candidate_observe_commitment(
    pcs: &HidingPcs,
    commitment: &HidingCommitment,
    challenger: &mut HidingChallenger,
) {
    MultilinearPcs::observe_commitment(pcs, commitment, challenger);
}

pub fn candidate_open(
    pcs: &HidingPcs,
    prover_data: Vec<HidingProverData>,
    commitments: &[HidingCommitment],
    profile: CandidateProfile,
    mu: OuterExtension,
    width: usize,
    cube_point_little_endian: &[OuterExtension],
    expected_value: OuterExtension,
    challenger: &mut HidingChallenger,
) -> Result<Vec<HidingProof>, CandidatePcsError> {
    let (full_point, expected_packed_claim) =
        bridge::packed_opening(mu, width, cube_point_little_endian, expected_value)?;
    if prover_data.len() != profile.num_commitments
        || commitments.len() != profile.num_commitments
        || pcs.num_vars() != profile.num_variables
    {
        return Err(CandidatePcsError::CommitmentCountMismatch);
    }
    let points = profile_opening_points(&full_point, profile)?;
    let mut proofs = Vec::with_capacity(points.len());
    for ((data, commitment), point) in prover_data.into_iter().zip(commitments).zip(points) {
        candidate_observe_commitment(pcs, commitment, challenger);
        let proof = MultilinearPcs::open(pcs, data, vec![point], challenger)?;
        proofs.push(proof);
    }
    let combined_claim = combine_opening_claims(&full_point, profile, &proofs)?;
    if combined_claim != expected_packed_claim {
        return Err(CandidatePcsError::ClaimMismatch);
    }
    Ok(proofs)
}

fn profile_opening_points(
    full_point: &p3_multilinear_util_08::point::Point<HidingExtension>,
    profile: CandidateProfile,
) -> Result<Vec<p3_multilinear_util_08::point::Point<HidingExtension>>, CandidatePcsError> {
    if profile.split_selector {
        if full_point.num_variables() != profile.num_variables + 1 {
            return Err(CandidatePcsError::VariableCountMismatch {
                expected: profile.num_variables + 1,
                actual: full_point.num_variables(),
            });
        }
        Ok(vec![
            p3_multilinear_util_08::point::Point::new(
                full_point.as_slice()[1..].to_vec(),
            );
            2
        ])
    } else {
        let added = profile
            .num_variables
            .checked_sub(full_point.num_variables())
            .ok_or(CandidatePcsError::VariableCountMismatch {
                expected: profile.num_variables,
                actual: full_point.num_variables(),
            })?;
        Ok(vec![bridge::pad_opening_point(full_point, added)])
    }
}

fn combine_opening_claims(
    full_point: &p3_multilinear_util_08::point::Point<HidingExtension>,
    profile: CandidateProfile,
    proofs: &[HidingProof],
) -> Result<HidingExtension, CandidatePcsError> {
    if proofs.len() != profile.num_commitments || proofs.iter().any(|proof| proof.evals.len() != 1)
    {
        return Err(CandidatePcsError::CommitmentCountMismatch);
    }
    if profile.split_selector {
        let selector = *full_point
            .as_slice()
            .first()
            .ok_or(CandidatePcsError::CommitmentCountMismatch)?;
        Ok((HidingExtension::ONE - selector) * proofs[0].evals[0] + selector * proofs[1].evals[0])
    } else {
        Ok(proofs[0].evals[0])
    }
}

pub fn candidate_verify(
    pcs: &HidingPcs,
    commitments: &[HidingCommitment],
    proofs: &[HidingProof],
    profile: CandidateProfile,
    mu: OuterExtension,
    width: usize,
    cube_point_little_endian: &[OuterExtension],
    expected_value: OuterExtension,
    challenger: &mut HidingChallenger,
) -> Result<(), CandidatePcsError> {
    if commitments.len() != profile.num_commitments
        || proofs.len() != profile.num_commitments
        || pcs.num_vars() != profile.num_variables
    {
        return Err(CandidatePcsError::CommitmentCountMismatch);
    }
    let (full_point, expected_packed_claim) =
        bridge::packed_opening(mu, width, cube_point_little_endian, expected_value)?;
    let points = profile_opening_points(&full_point, profile)?;
    let combined_claim = combine_opening_claims(&full_point, profile, proofs)?;
    if combined_claim != expected_packed_claim {
        return Err(CandidatePcsError::ClaimMismatch);
    }
    for ((commitment, proof), point) in commitments.iter().zip(proofs).zip(points) {
        MultilinearPcs::verify(pcs, commitment, proof, challenger, vec![point])?;
    }
    Ok(())
}

pub fn candidate_observe_commitments(
    pcs: &HidingPcs,
    commitments: &[HidingCommitment],
    challenger: &mut HidingChallenger,
) {
    for commitment in commitments {
        candidate_observe_commitment(pcs, commitment, challenger);
    }
}

#[cfg(test)]
mod tests;
