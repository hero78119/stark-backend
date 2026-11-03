use itertools::Itertools;
use openvm_stark_backend::{
    hiding_whir::{
        bridge::{self, BridgeError},
        candidate_commit_witness, candidate_config, candidate_open_with_stage_observer,
        candidate_pcs_from_os_rng, candidate_profile, candidate_validate_claims,
        transcript::V9Transcript,
        CandidateConfigError, CandidatePcsError, CandidateProfile, HidingCommitment, HidingProof,
        OuterBase, OuterExtension,
    },
    poly_common::Squarable,
    proof::{BatchConstraintProof, GkrProof, ProofV9, StackingProof},
    prover::{
        stacked_pcs::{stacked_matrix, StackedLayout},
        stacked_reduction::prove_stacked_opening_reduction_from_state,
        ColMajorMatrix, DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
    },
    FiatShamirTranscript, StarkProtocolConfig,
};
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use thiserror::Error;

use crate::{
    backend::CpuBackend,
    device::{eval_to_coeff_cpu, CpuDevice},
    logup_zerocheck::prove_zerocheck_and_logup_with_stage_observer,
    stacked_reduction::{StackedMatrixView, StackedReductionCpuNew},
    two_adic::DftTwiddles,
};

#[derive(Debug, Error)]
pub enum HidingWhirProverError {
    #[error("HidingWhir supports only traces without preprocessed or cached commitments")]
    UnsupportedCommitments,
    #[error("HidingWhir requires at least one common-main trace")]
    EmptyTraces,
    #[error("PCS has {actual} variables; the stacked trace requires {expected}")]
    VariableCountMismatch { expected: usize, actual: usize },
    #[error("stacked PCS failed: {0}")]
    StackedPcs(#[from] openvm_stark_backend::prover::error::StackedPcsError),
    #[error("LogUp/GKR proving failed: {0}")]
    Logup(#[from] openvm_stark_backend::prover::error::LogupZerocheckError),
    #[error("HidingWhir PCS failed: {0}")]
    CandidatePcs(#[from] CandidatePcsError),
    #[error("HidingWhir profile rejected: {0}")]
    CandidateConfig(#[from] CandidateConfigError),
    #[error("selector denominator remained zero after 128 μ samples")]
    MuRetriesExhausted,
}

struct HidingCommitData<F> {
    matrix: ColMajorMatrix<F>,
    layout: StackedLayout,
    profile: CandidateProfile,
    prover_data: Vec<openvm_stark_backend::hiding_whir::HidingProverData>,
    commitments: Vec<HidingCommitment>,
}

fn stacked_shape<SC: StarkProtocolConfig>(
    params: &openvm_stark_backend::SystemParams,
    ctx: &ProvingContext<CpuBackend<SC>>,
) -> Result<(StackedLayout, usize), HidingWhirProverError> {
    if ctx.per_trace.is_empty() {
        return Err(HidingWhirProverError::EmptyTraces);
    }
    let mut dimensions = ctx
        .per_trace
        .iter()
        .map(|(air_id, trace)| {
            (
                *air_id,
                (
                    MatrixDimensions::width(&trace.common_main),
                    openvm_stark_backend::p3_util::log2_strict_usize(MatrixDimensions::height(
                        &trace.common_main,
                    )),
                ),
            )
        })
        .collect_vec();
    dimensions.sort_by_key(|(air_id, (_, log_height))| (std::cmp::Reverse(*log_height), *air_id));
    let layout = StackedLayout::new(
        params.l_skip,
        params.log_stacked_height(),
        dimensions.into_iter().map(|(_, shape)| shape).collect(),
    )?;
    let selector_variables = layout
        .width()
        .checked_next_power_of_two()
        .ok_or(HidingWhirProverError::VariableCountMismatch {
            expected: usize::MAX,
            actual: layout.width(),
        })?
        .trailing_zeros() as usize;
    Ok((layout, params.log_stacked_height() + selector_variables))
}

/// Proves the existing row-major SWIRL/LogUp/stacking protocol with the V9 hiding PCS.
/// The caller binds the V9 descriptor into `vk_pre_hash`; PCS/config are fixed from validated trace
/// shape.
pub fn prove_v9<SC>(
    device: &CpuDevice<SC>,
    mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
    ctx: ProvingContext<CpuBackend<SC>>,
    vk_pre_hash: SC::Digest,
    transcript: &mut V9Transcript,
) -> Result<ProofV9<SC>, HidingWhirProverError>
where
    SC: StarkProtocolConfig<F = OuterBase, EF = OuterExtension, Digest = [OuterBase; 8]>,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    CpuBackend<SC>: openvm_stark_backend::prover::ProverBackend<
        Val = SC::F,
        Challenge = SC::EF,
        Commitment = SC::Digest,
        Matrix = RowMajorMatrix<SC::F>,
    >,
{
    prove_v9_with_stage_observer(device, mpk, ctx, vk_pre_hash, transcript, |_| {})
}

/// V9 proving entry with coarse stage events for native/WASM profiling.
pub fn prove_v9_with_stage_observer<SC, O>(
    device: &CpuDevice<SC>,
    mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
    ctx: ProvingContext<CpuBackend<SC>>,
    vk_pre_hash: SC::Digest,
    transcript: &mut V9Transcript,
    on_stage: O,
) -> Result<ProofV9<SC>, HidingWhirProverError>
where
    SC: StarkProtocolConfig<F = OuterBase, EF = OuterExtension, Digest = [OuterBase; 8]>,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    O: Fn(&'static str),
    CpuBackend<SC>: openvm_stark_backend::prover::ProverBackend<
        Val = SC::F,
        Challenge = SC::EF,
        Commitment = SC::Digest,
        Matrix = RowMajorMatrix<SC::F>,
    >,
{
    on_stage("preflight_started");
    if mpk.per_air.iter().any(|pk| pk.preprocessed_data.is_some())
        || ctx
            .per_trace
            .iter()
            .any(|(_, trace)| !trace.cached_mains.is_empty())
    {
        return Err(HidingWhirProverError::UnsupportedCommitments);
    }

    let ctx = ctx.into_sorted();
    let (expected_layout, expected_variables) = stacked_shape(&mpk.params, &ctx)?;
    let profile = candidate_profile(expected_variables)?;
    let pcs = candidate_pcs_from_os_rng(candidate_config(profile.num_variables)?)
        .map_err(CandidatePcsError::Randomness)?;
    candidate_validate_claims(&pcs)?;
    on_stage("preflight_complete");
    let stage = &on_stage;

    let host = openvm_stark_backend::prover::prove_with_pcs::<
        SC,
        CpuBackend<SC>,
        V9Transcript,
        Vec<HidingCommitment>,
        HidingCommitData<SC::F>,
        (GkrProof<SC>, BatchConstraintProof<SC>),
        Vec<SC::EF>,
        (StackingProof<SC>, SC::F, Vec<HidingProof>),
        HidingWhirProverError,
    >(
        transcript,
        mpk,
        vk_pre_hash,
        ctx,
        |transcript: &mut V9Transcript, traces: &[&RowMajorMatrix<SC::F>]| {
            stage("column_conversion_started");
            let col_major_traces = traces
                .iter()
                .map(|trace| ColMajorMatrix::from_row_major(trace))
                .collect_vec();
            stage("column_conversion_complete");
            let col_major_refs = col_major_traces.iter().collect_vec();
            let (matrix, layout) =
                stacked_matrix(mpk.params.l_skip, mpk.params.n_stack, &col_major_refs)?;
            stage("stacked_matrix_complete");
            if layout.width() != expected_layout.width()
                || layout.height() != expected_layout.height()
            {
                return Err(HidingWhirProverError::VariableCountMismatch {
                    expected: expected_variables,
                    actual: params_variables(&layout),
                });
            }
            let row_variables = matrix.height().trailing_zeros() as usize;
            let selector_variables = profile.num_variables.checked_sub(row_variables).ok_or(
                HidingWhirProverError::VariableCountMismatch {
                    expected: profile.num_variables,
                    actual: row_variables,
                },
            )?;
            let target_width = 1usize.checked_shl(selector_variables as u32).ok_or(
                HidingWhirProverError::VariableCountMismatch {
                    expected: profile.num_variables,
                    actual: params_variables(&layout),
                },
            )?;
            let pieces = profile.num_commitments;
            let single_commitment = profile.selector_variables == 0;
            let width_is_single = matrix.width() <= target_width;
            let width_exceeds_pieces = matrix.width() > target_width.saturating_mul(pieces);
            if single_commitment != width_is_single || (!single_commitment && width_exceeds_pieces)
            {
                return Err(HidingWhirProverError::VariableCountMismatch {
                    expected: expected_variables,
                    actual: params_variables(&layout),
                });
            }
            let mut witnesses = Vec::with_capacity(pieces);
            let dft_twiddles = DftTwiddles::new(mpk.params.l_skip);
            for piece in 0..pieces {
                witnesses.push(
                    bridge::pack_eval_columns_at_width(
                        &matrix.values,
                        matrix.height(),
                        matrix.width(),
                        piece * target_width,
                        mpk.params.l_skip,
                        target_width,
                        |column| eval_to_coeff_cpu(column, &dft_twiddles),
                    )
                    .map_err(CandidatePcsError::Bridge)?,
                );
            }
            stage("column_coefficients_complete");
            stage("pcs_commit_started");
            let mut commitments = Vec::with_capacity(witnesses.len());
            let mut prover_data = Vec::with_capacity(witnesses.len());
            for witness in witnesses {
                let (commitment, data) =
                    candidate_commit_witness(&pcs, witness, profile, transcript)?;
                commitments.push(commitment);
                prover_data.push(data);
            }
            stage("pcs_commit_complete");
            Ok((
                commitments.clone(),
                HidingCommitData {
                    matrix,
                    layout,
                    profile,
                    prover_data,
                    commitments,
                },
            ))
        },
        |_, _| {},
        |transcript: &mut V9Transcript,
         mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
         ctx: &ProvingContext<CpuBackend<SC>>,
         _data: &HidingCommitData<SC::F>| {
            stage("gkr_logup_started");
            let (gkr, batch, r) =
                prove_zerocheck_and_logup_with_stage_observer(transcript, mpk, ctx, stage)?;
            stage("gkr_logup_complete");
            Ok(((gkr, batch), r))
        },
        |transcript: &mut V9Transcript,
         mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
         ctx: ProvingContext<CpuBackend<SC>>,
         data: HidingCommitData<SC::F>,
         r: Vec<SC::EF>| {
            stage("stacking_reduction_started");
            let HidingCommitData {
                matrix,
                layout,
                profile,
                prover_data,
                commitments,
            } = data;
            let need_rot = ctx
                .per_trace
                .iter()
                .map(|(air_id, _)| mpk.per_air[*air_id].vk.params.need_rot)
                .collect_vec();
            drop(ctx);
            let lambda: SC::EF = <V9Transcript as FiatShamirTranscript<SC>>::sample_ext(transcript);
            let state = StackedReductionCpuNew::from_views(
                device,
                vec![StackedMatrixView {
                    layout: &layout,
                    matrix: &matrix,
                }],
                vec![need_rot],
                &r,
                lambda,
            );
            let (stacking_proof, u_prisma) = prove_stacked_opening_reduction_from_state::<SC, _, _>(
                transcript,
                mpk.params.n_stack,
                state,
            );
            drop(matrix);
            stage("stacking_reduction_complete");
            let (&u0, u_rest) = u_prisma
                .split_first()
                .ok_or(HidingWhirProverError::EmptyTraces)?;
            let u_cube = u0
                .exp_powers_of_2()
                .take(mpk.params.l_skip)
                .chain(u_rest.iter().copied())
                .collect_vec();

            let mu_pow_witness = <V9Transcript as FiatShamirTranscript<SC>>::grind(
                transcript,
                mpk.params.whir.mu_pow_bits,
            );
            for _ in 0..128 {
                let mu: SC::EF = <V9Transcript as FiatShamirTranscript<SC>>::sample_ext(transcript);
                let expected_value = stacking_proof
                    .stacking_openings
                    .iter()
                    .flatten()
                    .zip(mu.powers())
                    .fold(SC::EF::ZERO, |sum, (&claim, power)| sum + claim * power);
                match bridge::packed_opening(mu, layout.width(), &u_cube, expected_value) {
                    Ok(_) => {
                        stage("pcs_open_started");
                        let hiding_proof = candidate_open_with_stage_observer(
                            &pcs,
                            prover_data,
                            &commitments,
                            profile,
                            mu,
                            layout.width(),
                            &u_cube,
                            expected_value,
                            transcript,
                            stage,
                        )?;
                        stage("pcs_open_complete");
                        return Ok((stacking_proof, mu_pow_witness, hiding_proof));
                    }
                    Err(BridgeError::ZeroSelectorDenominator) => continue,
                    Err(error) => return Err(CandidatePcsError::Bridge(error).into()),
                }
            }
            Err(HidingWhirProverError::MuRetriesExhausted)
        },
    )?;
    on_stage("proof_complete");

    let (gkr_proof, batch_constraint_proof) = host.constraints_proof;
    let (stacking_proof, mu_pow_witness, hiding_whir_proof) = host.opening_proof;
    Ok(ProofV9 {
        common_main_commits: host.common_main_commit,
        trace_vdata: host.trace_vdata,
        public_values: host.public_values,
        gkr_proof,
        batch_constraint_proof,
        stacking_proof,
        mu_pow_witness,
        hiding_whir_proofs: hiding_whir_proof,
    })
}

fn params_variables(layout: &StackedLayout) -> usize {
    openvm_stark_backend::p3_util::log2_strict_usize(layout.height())
        + layout
            .width()
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .trailing_zeros() as usize
}

#[cfg(test)]
mod tests;
