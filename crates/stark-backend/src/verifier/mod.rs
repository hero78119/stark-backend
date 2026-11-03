use core::cmp::Reverse;

use itertools::{izip, Itertools};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use thiserror::Error;

use crate::{
    keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
    poly_common::Squarable,
    proof::{OuterProofRef, Proof},
    verifier::{
        batch_constraints::{verify_zerocheck_and_logup, BatchConstraintError},
        proof_shape::{verify_outer_proof_shape, verify_whir_proof_shape, ProofShapeError},
        stacked_reduction::{verify_stacked_reduction, StackedReductionError},
        whir::{verify_whir, VerifyWhirError},
    },
    FiatShamirTranscript, StarkProtocolConfig,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum VerifierError<EF: core::fmt::Debug + core::fmt::Display + PartialEq + Eq> {
    #[error("Protocol and VerifyingKey has mismatch in SystemParams")]
    SystemParamsMismatch,

    #[error("Trace heights are too large")]
    TraceHeightsTooLarge,

    #[error("Preprocessed trace height does not match verifier trace data")]
    PreprocessedTraceHeightMismatch,

    /// A proof without any traces is always considered invalid.
    #[error("Proof has no traces")]
    EmptyTraces,

    #[error("Proof shape verification failed: {0}")]
    ProofShapeError(#[from] ProofShapeError),

    #[error("Batch constraint verification failed: {0}")]
    BatchConstraintError(#[from] BatchConstraintError<EF>),

    #[error("Stacked reduction verification failed: {0}")]
    StackedReductionError(#[from] StackedReductionError<EF>),

    #[error("Whir verification failed: {0}")]
    WhirError(#[from] VerifyWhirError),
}

pub mod batch_constraints;
pub mod evaluator;
pub mod fractional_sumcheck_gkr;
#[cfg(feature = "hiding-whir")]
pub mod hiding_whir;
pub mod proof_shape;
pub mod stacked_reduction;
#[cfg(test)]
mod transcript_extractor;
pub mod whir;

pub fn verify<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    config: &SC,
    mvk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
    transcript: &mut TS,
) -> Result<(), VerifierError<SC::EF>>
where
    SC::EF: p3_field::TwoAdicField,
{
    verify_with_pcs(
        config,
        mvk,
        mvk.pre_hash,
        OuterProofRef {
            trace_vdata: &proof.trace_vdata,
            public_values: &proof.public_values,
            gkr_proof: &proof.gkr_proof,
            batch_constraint_proof: &proof.batch_constraint_proof,
            stacking_proof: &proof.stacking_proof,
        },
        transcript,
        |layouts| {
            verify_whir_proof_shape(
                &mvk.inner,
                &proof.whir_proof,
                &proof.stacking_proof,
                layouts,
            )
            .map_err(VerifierError::from)
        },
        |transcript| transcript.observe_commit(proof.common_main_commit),
        |transcript, _layouts, trace_id_to_air_id, u_cube| {
            let mut commits = vec![proof.common_main_commit];
            for &air_id in trace_id_to_air_id {
                if let Some(preprocessed) = &mvk.inner.per_air[air_id].preprocessed_data {
                    commits.push(preprocessed.commit);
                }
                commits.extend(
                    &proof.trace_vdata[air_id]
                        .as_ref()
                        .unwrap()
                        .cached_commitments,
                );
            }
            verify_whir::<SC, TS>(
                transcript,
                config,
                &proof.whir_proof,
                &proof.stacking_proof.stacking_openings,
                &commits,
                u_cube,
            )
            .map_err(VerifierError::from)
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_with_pcs<SC, TS, E>(
    config: &SC,
    mvk: &MultiStarkVerifyingKey<SC>,
    vk_pre_hash: SC::Digest,
    proof: OuterProofRef<'_, SC>,
    transcript: &mut TS,
    check_pcs_shape: impl FnOnce(&[crate::prover::stacked_pcs::StackedLayout]) -> Result<(), E>,
    bind_main: impl FnOnce(&mut TS),
    verify_openings: impl FnOnce(
        &mut TS,
        &[crate::prover::stacked_pcs::StackedLayout],
        &[usize],
        &[SC::EF],
    ) -> Result<(), E>,
) -> Result<(), E>
where
    SC: StarkProtocolConfig,
    SC::EF: p3_field::TwoAdicField,
    TS: FiatShamirTranscript<SC>,
    E: From<VerifierError<SC::EF>>,
{
    if config.params() != &mvk.inner.params {
        return Err(VerifierError::SystemParamsMismatch.into());
    }
    let MultiStarkVerifyingKey0 {
        params,
        per_air,
        trace_height_constraints,
    } = &mvk.inner;
    let trace_vdata = proof.trace_vdata;
    let public_values = proof.public_values;
    let gkr_proof = proof.gkr_proof;
    let batch_constraint_proof = proof.batch_constraint_proof;
    let stacking_proof = proof.stacking_proof;
    let l_skip = params.l_skip;
    // Total number of AIRs is in vkey, committed to in vk_pre_hash
    let num_airs = per_air.len();
    // Number of present traces. This is implicitly hashed via the
    // transcript.observe(is_air_present) below.
    let num_traces = trace_vdata.iter().flatten().count();
    if num_traces == 0 {
        return Err(VerifierError::EmptyTraces.into());
    }
    // We verify the proof shape early to return error and prevent later panics
    let layouts =
        verify_outer_proof_shape::<SC>(&mvk.inner, &proof).map_err(VerifierError::from)?;
    check_pcs_shape(&layouts)?;

    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| {
        (
            trace_vdata[air_id].is_none(),
            trace_vdata[air_id]
                .as_ref()
                .map(|vdata| Reverse(vdata.log_height)),
            air_id,
        )
    });
    trace_id_to_air_id.truncate(num_traces);

    for constraint in trace_height_constraints {
        let sum = trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let log_height = trace_vdata[air_id].as_ref().unwrap().log_height;
                // Proof shape will check n <= n_stack is in bounds
                (1 << log_height.max(l_skip)) as u64 * constraint.coefficients[air_id] as u64
            })
            .sum::<u64>();
        if sum >= constraint.threshold as u64 {
            return Err(VerifierError::TraceHeightsTooLarge.into());
        }
    }

    let omega_skip = SC::F::two_adic_generator(l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();

    // Preamble
    transcript.observe_commit(vk_pre_hash);
    bind_main(transcript);

    for (trace_vdata, avk, pvs) in izip!(trace_vdata.iter(), per_air.iter(), public_values.iter()) {
        let is_air_present = trace_vdata.is_some();

        // Proof shape asserts that vk.is_required => is_air_present (see
        // ProofShapeVDataError::RequiredAirNoVData)
        if !avk.is_required {
            transcript.observe(SC::F::from_bool(is_air_present));
        }
        if let Some(trace_vdata) = trace_vdata {
            if let Some(pdata) = avk.preprocessed_data.as_ref() {
                if (pdata.hypercube_dim + l_skip as isize) as usize != trace_vdata.log_height {
                    return Err(VerifierError::PreprocessedTraceHeightMismatch.into());
                }
                transcript.observe_commit(pdata.commit);
            } else {
                transcript.observe(SC::F::from_usize(trace_vdata.log_height));
            }
            debug_assert_eq!(
                avk.params.width.cached_mains.len(),
                trace_vdata.cached_commitments.len()
            );
            for commit in &trace_vdata.cached_commitments {
                transcript.observe_commit(*commit);
            }
            debug_assert_eq!(avk.params.num_public_values, pvs.len());
        }
        for pv in pvs {
            transcript.observe(*pv);
        }
    }

    // n_per_trace.len() = num_traces
    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| trace_vdata[air_id].as_ref().unwrap().log_height as isize - l_skip as isize)
        .collect();
    let r = verify_zerocheck_and_logup::<SC, TS>(
        transcript,
        &mvk.inner,
        public_values,
        gkr_proof,
        batch_constraint_proof,
        &trace_id_to_air_id,
        &n_per_trace,
        &omega_skip_pows,
    )
    .map_err(VerifierError::from)?;

    let need_rot_per_trace = trace_id_to_air_id
        .iter()
        .map(|&air_id| per_air[air_id].params.need_rot)
        .collect_vec();
    let mut need_rot_per_commit = vec![need_rot_per_trace];
    for &air_id in &trace_id_to_air_id {
        let need_rot = per_air[air_id].params.need_rot;
        if per_air[air_id].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        let cached_len = trace_vdata[air_id]
            .as_ref()
            .unwrap()
            .cached_commitments
            .len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }

    let u_prism = verify_stacked_reduction::<SC, TS>(
        transcript,
        stacking_proof,
        &layouts,
        &need_rot_per_commit,
        l_skip,
        params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &r,
        &omega_skip_pows,
    )
    .map_err(VerifierError::from)?;

    let (&u0, u_rest) = u_prism.split_first().unwrap();
    let u_cube = u0
        .exp_powers_of_2()
        .take(l_skip)
        .chain(u_rest.iter().copied())
        .collect_vec();

    verify_openings(transcript, &layouts, &trace_id_to_air_id, &u_cube)
}
