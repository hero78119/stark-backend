use itertools::{izip, Itertools};
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;
use tracing::{info, info_span, instrument};

#[cfg(feature = "metrics")]
use crate::prover::metrics::trace_metrics;
use crate::{
    proof::{BatchConstraintProof, GkrProof, Proof, StackingProof, TraceVData, WhirProof},
    FiatShamirTranscript, StarkProtocolConfig,
};

mod cpu_backend;
pub mod error;
mod hal;
mod logup_zerocheck;
mod matrix;
pub mod metrics;
pub mod poly;
pub mod stacked_pcs;
pub mod stacked_reduction;
pub mod sumcheck;
mod types;
pub mod whir;

pub use cpu_backend::*;
pub use error::*;
pub use hal::*;
pub use logup_zerocheck::*;
pub use matrix::*;
pub use types::*;

/// Trait for STARK/SNARK proving at the highest abstraction level.
pub trait Prover {
    type ProvingKeyView<'a>
    where
        Self: 'a;
    type ProvingContext<'a>
    where
        Self: 'a;
    type Proof;
    type Error;

    /// The prover should own the challenger, whose state mutates during proving.
    fn prove<'a>(
        &'a mut self,
        pk: Self::ProvingKeyView<'a>,
        ctx: Self::ProvingContext<'a>,
    ) -> Result<Self::Proof, Self::Error>;
}

/// Shared outer transcript and proof orchestration for backends with different PCS types.
pub fn prove_with_pcs<SC, PB, TS, C, D, CP, A, OP, E>(
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKey<PB>,
    vk_pre_hash: SC::Digest,
    unsorted_ctx: ProvingContext<PB>,
    commit: impl FnOnce(&mut TS, &[&PB::Matrix]) -> Result<(C, D), E>,
    bind_main: impl FnOnce(&mut TS, &C),
    prove_constraints: impl FnOnce(
        &mut TS,
        &DeviceMultiStarkProvingKey<PB>,
        &ProvingContext<PB>,
        &D,
    ) -> Result<(CP, A), E>,
    prove_openings: impl FnOnce(
        &mut TS,
        &DeviceMultiStarkProvingKey<PB>,
        ProvingContext<PB>,
        D,
        A,
    ) -> Result<OP, E>,
) -> Result<HostProof<SC, PB, CP, OP, C>, E>
where
    SC: StarkProtocolConfig,
    PB: ProverBackend<Val = SC::F, Challenge = SC::EF, Commitment = SC::Digest>,
    TS: FiatShamirTranscript<SC>,
{
    transcript.observe_commit(vk_pre_hash);

    let ctx = unsorted_ctx.into_sorted();
    // `ctx` should NOT be permuted anymore: the ordering by `trace_idx` is now fixed.

    let num_airs_present = ctx.per_trace.len();
    info!(num_airs_present);

    let _main_commit_span = info_span!("prover.main_trace_commit", phase = "prover").entered();
    let traces = ctx
        .common_main_traces()
        .map(|(_, trace)| trace)
        .collect_vec();
    let (common_main_commit, common_main_pcs_data) = commit(transcript, &traces)?;

    let mut trace_vdata: Vec<Option<TraceVData<SC>>> = vec![None; mpk.per_air.len()];
    let mut public_values: Vec<Vec<SC::F>> = vec![Vec::new(); mpk.per_air.len()];

    // Hypercube dimension per trace (present AIR).
    for (air_id, trace_ctx) in &ctx.per_trace {
        let trace_height = trace_ctx.common_main.height();
        let log_height = log2_strict_usize(trace_height);

        trace_vdata[*air_id] = Some(TraceVData::<SC> {
            log_height,
            cached_commitments: trace_ctx
                .cached_mains
                .iter()
                .map(|cd| cd.commitment.clone())
                .collect(),
        });
        public_values[*air_id] = trace_ctx.public_values.clone();
    }
    #[cfg(feature = "metrics")]
    trace_metrics::<SC, _>(mpk, &trace_vdata).emit();

    // Keep the common-main root at the original transcript position, after metadata collection.
    bind_main(transcript, &common_main_commit);
    drop(_main_commit_span);

    for (trace_vdata, pvs, pk) in izip!(&trace_vdata, &public_values, &mpk.per_air) {
        if !pk.vk.is_required {
            transcript.observe(SC::F::from_bool(trace_vdata.is_some()));
        }
        if let Some(trace_vdata) = trace_vdata {
            if let Some(cd) = &pk.preprocessed_data {
                transcript.observe_commit(cd.commitment.clone());
            } else {
                transcript.observe(SC::F::from_usize(trace_vdata.log_height));
            }
            for commit in &trace_vdata.cached_commitments {
                transcript.observe_commit(commit.clone());
            }
        }
        for pv in pvs {
            transcript.observe(*pv);
        }
    }

    let (constraints_proof, artifacts) =
        prove_constraints(transcript, mpk, &ctx, &common_main_pcs_data)?;
    let opening_proof = prove_openings(transcript, mpk, ctx, common_main_pcs_data, artifacts)?;

    Ok(HostProof {
        common_main_commit,
        trace_vdata,
        public_values,
        constraints_proof,
        opening_proof,
    })
}

pub struct Coordinator<SC: StarkProtocolConfig, PB: ProverBackend, PD, TS> {
    pub backend: PB,
    pub device: PD,
    pub(crate) transcript: TS,
    _sc: std::marker::PhantomData<SC>,
}

impl<SC: StarkProtocolConfig, PB: ProverBackend, PD, TS> Coordinator<SC, PB, PD, TS> {
    pub fn new(backend: PB, device: PD, transcript: TS) -> Self {
        Self {
            backend,
            device,
            transcript,
            _sc: std::marker::PhantomData,
        }
    }
}

impl<SC, PB, PD, TS> Prover for Coordinator<SC, PB, PD, TS>
where
    SC: StarkProtocolConfig,
    PB: ProverBackend<Val = SC::F, Challenge = SC::EF, Commitment = SC::Digest>,
    PD: ProverDevice<PB, TS>,
    PD::Artifacts: Into<PD::OpeningPoints>,
    PD::PartialProof: Into<(GkrProof<SC>, BatchConstraintProof<SC>)>,
    PD::OpeningProof: Into<(StackingProof<SC>, WhirProof<SC>)>,
    TS: FiatShamirTranscript<SC>,
{
    type Proof = Proof<SC>;
    type Error = <PD as ProverDevice<PB, TS>>::Error;
    type ProvingKeyView<'a>
        = &'a DeviceMultiStarkProvingKey<PB>
    where
        Self: 'a;

    type ProvingContext<'a>
        = ProvingContext<PB>
    where
        Self: 'a;

    /// Specialized prove for InteractiveAirs.
    /// Handles trace generation of the permutation traces.
    /// Assumes the main traces have been generated and committed already.
    ///
    /// The [DeviceMultiStarkProvingKey] should already be filtered to only include the relevant
    /// AIR's proving keys.
    #[instrument(
        name = "stark_prove_excluding_trace",
        level = "info",
        skip_all,
        fields(phase = "prover")
    )]
    fn prove<'a>(
        &'a mut self,
        mpk: &'a DeviceMultiStarkProvingKey<PB>,
        unsorted_ctx: ProvingContext<PB>,
    ) -> Result<Self::Proof, Self::Error> {
        let transcript = &mut self.transcript;
        let HostProof {
            common_main_commit,
            trace_vdata,
            public_values,
            constraints_proof,
            opening_proof,
        } = prove_with_pcs::<
            SC,
            PB,
            TS,
            PB::Commitment,
            PB::PcsData,
            PD::PartialProof,
            PD::Artifacts,
            PD::OpeningProof,
            Self::Error,
        >(
            transcript,
            mpk,
            mpk.vk_pre_hash.clone(),
            unsorted_ctx,
            |_, traces| self.device.commit(traces).map_err(Into::into),
            |transcript, commitment| transcript.observe_commit(commitment.clone()),
            |transcript, mpk, ctx, pcs_data| {
                self.device
                    .prove_rap_constraints(transcript, mpk, ctx, pcs_data)
                    .map_err(Into::into)
            },
            |transcript, mpk, ctx, pcs_data, artifacts| {
                self.device
                    .prove_openings(transcript, mpk, ctx, pcs_data, artifacts.into())
                    .map_err(Into::into)
            },
        )?;

        let (gkr_proof, batch_constraint_proof) = constraints_proof.into();
        let (stacking_proof, whir_proof) = opening_proof.into();

        Ok(Proof::<SC> {
            public_values,
            trace_vdata,
            common_main_commit,
            gkr_proof,
            batch_constraint_proof,
            stacking_proof,
            whir_proof,
        })
    }
}
