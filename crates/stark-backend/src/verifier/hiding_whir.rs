use core::fmt::{Debug, Display};

use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use thiserror::Error;

use crate::{
    hiding_whir::{
        bridge::{self, BridgeError},
        candidate_config, candidate_observe_commitments, candidate_pcs_from_os_rng,
        candidate_profile, candidate_validate_claims, candidate_verify,
        transcript::V9Transcript,
        CandidateConfigError, CandidatePcsError, CandidateProfile, HidingPcs, OuterBase,
        OuterExtension,
    },
    keygen::types::MultiStarkVerifyingKey,
    proof::ProofV9,
    prover::stacked_pcs::StackedLayout,
    verifier::{proof_shape::verify_outer_proof_shape, verify_with_pcs, VerifierError},
    FiatShamirTranscript, StarkProtocolConfig,
};

#[derive(Debug, Error)]
pub enum HidingWhirVerifierError<EF: Debug + Display + PartialEq + Eq> {
    #[error(transparent)]
    Outer(#[from] VerifierError<EF>),
    #[error("unsupported V9 stacked PCS shape or commitment count")]
    UnsupportedCommitments,
    #[error("PCS has {actual} variables; verifier key requires {expected}")]
    VariableCountMismatch { expected: usize, actual: usize },
    #[error("HidingWhir configuration rejected its opening profile: {0}")]
    InvalidPcsConfig(#[from] CandidatePcsError),
    #[error("HidingWhir profile rejected: {0}")]
    CandidateConfig(#[from] CandidateConfigError),
    #[error("μ batching proof-of-work is invalid")]
    MuPowInvalid,
    #[error("selector denominator remained zero after 128 μ samples")]
    MuRetriesExhausted,
}

/// Verifies only the V9 candidate protocol; the prehash includes the full V9 profile.
pub fn verify_v9<SC>(
    config: &SC,
    mvk: &MultiStarkVerifyingKey<SC>,
    proof: &ProofV9<SC>,
    vk_pre_hash: SC::Digest,
    transcript: &mut V9Transcript,
) -> Result<(), HidingWhirVerifierError<SC::EF>>
where
    SC: StarkProtocolConfig<F = OuterBase, EF = OuterExtension, Digest = [OuterBase; 8]>,
    SC::EF: TwoAdicField,
{
    let outer = proof.outer_ref();
    if !outer.trace_vdata.iter().any(Option::is_some) {
        return Err(VerifierError::EmptyTraces.into());
    }
    let checked_layouts =
        verify_outer_proof_shape(&mvk.inner, &outer).map_err(VerifierError::from)?;
    let profile = profile_for_layout(mvk, proof, &checked_layouts)?;
    let pcs = candidate_pcs_from_os_rng(candidate_config(profile.num_variables)?)
        .map_err(CandidatePcsError::Randomness)?;
    candidate_validate_claims(&pcs)?;

    verify_with_pcs(
        config,
        mvk,
        vk_pre_hash,
        outer,
        transcript,
        |layouts| check_pcs_shape(mvk, proof, &pcs, profile, layouts),
        |transcript| candidate_observe_commitments(&pcs, &proof.common_main_commits, transcript),
        |transcript, layouts, _, u_cube| {
            let width = layouts
                .first()
                .map(StackedLayout::width)
                .ok_or(HidingWhirVerifierError::UnsupportedCommitments)?;
            let params = &mvk.inner.params;
            if !<V9Transcript as FiatShamirTranscript<SC>>::check_witness(
                transcript,
                params.whir.mu_pow_bits,
                proof.mu_pow_witness,
            ) {
                return Err(HidingWhirVerifierError::MuPowInvalid);
            }

            for _ in 0..128 {
                let mu: SC::EF = <V9Transcript as FiatShamirTranscript<SC>>::sample_ext(transcript);
                let expected_value = proof
                    .stacking_proof
                    .stacking_openings
                    .iter()
                    .flatten()
                    .zip(mu.powers())
                    .fold(SC::EF::ZERO, |sum, (&claim, power)| sum + claim * power);
                match bridge::packed_opening(mu, width, u_cube, expected_value) {
                    Ok(_) => {
                        return candidate_verify(
                            &pcs,
                            &proof.common_main_commits,
                            &proof.hiding_whir_proofs,
                            profile,
                            mu,
                            width,
                            u_cube,
                            expected_value,
                            transcript,
                        )
                        .map_err(HidingWhirVerifierError::from)
                    }
                    Err(BridgeError::ZeroSelectorDenominator) => continue,
                    Err(error) => return Err(CandidatePcsError::Bridge(error).into()),
                }
            }
            Err(HidingWhirVerifierError::MuRetriesExhausted)
        },
    )
}

fn profile_for_layout<SC>(
    mvk: &MultiStarkVerifyingKey<SC>,
    proof: &ProofV9<SC>,
    layouts: &[StackedLayout],
) -> Result<CandidateProfile, HidingWhirVerifierError<SC::EF>>
where
    SC: StarkProtocolConfig,
    SC::EF: Debug + Display + PartialEq + Eq,
{
    if layouts.len() != 1
        || mvk
            .inner
            .per_air
            .iter()
            .any(|air| air.preprocessed_data.is_some())
        || proof
            .trace_vdata
            .iter()
            .flatten()
            .any(|trace| !trace.cached_commitments.is_empty())
        || proof.stacking_proof.stacking_openings.len() != 1
    {
        return Err(HidingWhirVerifierError::UnsupportedCommitments);
    }
    let selector_variables = layouts[0]
        .width()
        .checked_next_power_of_two()
        .ok_or(HidingWhirVerifierError::UnsupportedCommitments)?
        .trailing_zeros() as usize;
    let actual_variables = mvk.inner.params.log_stacked_height() + selector_variables;
    let profile = candidate_profile(actual_variables)?;
    if proof.common_main_commits.len() != profile.num_commitments
        || proof.hiding_whir_proofs.len() != profile.num_commitments
    {
        return Err(HidingWhirVerifierError::UnsupportedCommitments);
    }
    Ok(profile)
}

fn check_pcs_shape<SC>(
    mvk: &MultiStarkVerifyingKey<SC>,
    proof: &ProofV9<SC>,
    pcs: &HidingPcs,
    profile: CandidateProfile,
    layouts: &[StackedLayout],
) -> Result<(), HidingWhirVerifierError<SC::EF>>
where
    SC: StarkProtocolConfig,
    SC::EF: Debug + Display + PartialEq + Eq,
{
    let derived = profile_for_layout(mvk, proof, layouts)?;
    if derived != profile {
        return Err(HidingWhirVerifierError::UnsupportedCommitments);
    }
    if pcs.config.num_variables() != profile.num_variables {
        return Err(HidingWhirVerifierError::VariableCountMismatch {
            expected: profile.num_variables,
            actual: pcs.config.num_variables(),
        });
    }
    Ok(())
}
