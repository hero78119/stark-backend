use std::{io::Cursor, sync::Arc};

use openvm_stark_backend::{
    codec::{Decode, EncodableConfig, Encode},
    hiding_whir::{
        bridge, candidate_challenger, raw_candidate_challenger, transcript::V9Transcript,
    },
    keygen::MultiStarkKeygenBuilder,
    prover::{
        stacked_reduction::{prove_stacked_opening_reduction_from_state, StackedReductionProver},
        DeviceDataTransporter, ProvingContext, TraceCommitter,
    },
    test_utils::{
        dummy_airs::fib_air::air::FibonacciAir, test_system_params_small, CachedFixture11,
        FibFixture, InteractionsFixture11, PreprocessedFibFixture, TestFixture,
    },
    verifier::hiding_whir::verify_v9,
    FiatShamirTranscript, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::prove_v9;
use crate::{
    device::eval_to_coeff_cpu,
    stacked_reduction::{StackedMatrixView, StackedReductionCpuNew},
    two_adic::DftTwiddles,
    CpuDevice,
};

type SC = BabyBearPoseidon2Config;
type F = <SC as StarkProtocolConfig>::F;
type EF = <SC as StarkProtocolConfig>::EF;

#[test]
fn hiding_whir_cpu_column_conversion_matches_generic() {
    use openvm_stark_backend::prover::poly::eval_to_coeff_rs_message;

    const POPULATED_COLUMNS: usize = 5;
    for (column_height, l_skip) in [(1, 0), (2, 1), (16, 4), (64, 4)] {
        let values = (0..column_height * POPULATED_COLUMNS)
            .map(|index| match index % 4 {
                0 => F::ZERO,
                1 => F::ONE,
                2 => F::from_u64(F::ORDER_U64 - 1),
                _ => F::from_u64((index as u64 * 0x9e37_79b9) % F::ORDER_U64),
            })
            .collect::<Vec<_>>();
        let twiddles = DftTwiddles::new(l_skip);

        // Five populated columns are converted in power-of-two pieces, including one
        // partially occupied piece and an out-of-range piece that must stay zero.
        for first_column in [0, 4, 8] {
            let generic = bridge::pack_eval_columns_at_width(
                &values,
                column_height,
                POPULATED_COLUMNS,
                first_column,
                l_skip,
                4,
                |column| eval_to_coeff_rs_message(l_skip, column),
            )
            .unwrap();
            let cpu = bridge::pack_eval_columns_at_width(
                &values,
                column_height,
                POPULATED_COLUMNS,
                first_column,
                l_skip,
                4,
                |column| eval_to_coeff_cpu(column, &twiddles),
            )
            .unwrap();
            assert_eq!(cpu.as_slice(), generic.as_slice());
        }

        assert!(matches!(
            bridge::pack_eval_columns_at_width(
                &values,
                column_height,
                POPULATED_COLUMNS,
                0,
                l_skip,
                1,
                |_| Vec::new(),
            ),
            Err(openvm_stark_backend::hiding_whir::bridge::BridgeError::InvalidConvertedColumnLength {
                expected,
                actual: 0,
            }) if expected == column_height
        ));
    }
}

fn setup(
    params: openvm_stark_backend::SystemParams,
) -> (
    SC,
    CpuDevice<SC>,
    openvm_stark_backend::keygen::types::MultiStarkProvingKey<SC>,
    openvm_stark_backend::keygen::types::MultiStarkVerifyingKey<SC>,
    ProvingContext<openvm_stark_backend::prover::CpuColMajorBackend<SC>>,
) {
    let config = SC::default_from_params(params);
    let mut keygen = MultiStarkKeygenBuilder::new(config.clone());
    keygen.add_air(Arc::new(FibonacciAir));
    let pk = keygen.generate_pk().unwrap();
    let vk = pk.get_vk();
    let fixture = FibFixture::new(0, 1, 2);
    let ctx = fixture.generate_proving_ctx();
    (config.clone(), CpuDevice::new(config), pk, vk, ctx)
}

#[test]
fn hiding_swirl_native_roundtrip() {
    hiding_swirl_native_roundtrip_for(test_system_params_small(2, 18, 4));
}

#[test]
fn hiding_swirl_dimension_22_native_roundtrip() {
    hiding_swirl_native_roundtrip_for(test_system_params_small(2, 20, 4));
}

fn hiding_swirl_native_roundtrip_for(params: openvm_stark_backend::SystemParams) {
    let (config, device, pk, vk, host_ctx) = setup(params);
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx = device.transport_proving_ctx_to_device(&host_ctx);
    let vk_pre_hash = [F::from_u64(0x5639); 8];
    let mut prover_transcript = V9Transcript::with_events(raw_candidate_challenger());
    let proof = prove_v9(&device, &d_pk, d_ctx, vk_pre_hash, &mut prover_transcript).unwrap();
    let mut encoded = Vec::new();
    proof.encode(&mut encoded).unwrap();
    hiding_swirl_bounded_decode(&encoded);
    let proof =
        openvm_stark_backend::proof::ProofV9::<SC>::decode(&mut Cursor::new(&encoded)).unwrap();

    let mut v8_prefix = Vec::new();
    3u32.encode(&mut v8_prefix).unwrap();
    assert!(
        openvm_stark_backend::proof::ProofV9::<SC>::decode(&mut Cursor::new(v8_prefix)).is_err()
    );

    let mut verifier_transcript = V9Transcript::with_events(raw_candidate_challenger());
    verify_v9(&config, &vk, &proof, vk_pre_hash, &mut verifier_transcript).unwrap();
    assert_transcript_events_match(&prover_transcript, &verifier_transcript);
    let prover_continuation =
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(
            &mut prover_transcript,
        );
    let verifier_continuation =
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(
            &mut verifier_transcript,
        );
    assert_eq!(prover_continuation, verifier_continuation);

    let mut wrong_descriptor_transcript = candidate_challenger();
    let wrong_descriptor = [F::from_u64(0x5638); 8];
    assert!(verify_v9(
        &config,
        &vk,
        &proof,
        wrong_descriptor,
        &mut wrong_descriptor_transcript,
    )
    .is_err());

    let mut mutated = proof;
    mutated.public_values[0][0] += F::ONE;
    let mut mutated_transcript = candidate_challenger();
    assert!(verify_v9(&config, &vk, &mutated, vk_pre_hash, &mut mutated_transcript,).is_err());

    let mut wrong_pcs_claim =
        openvm_stark_backend::proof::ProofV9::<SC>::decode(&mut Cursor::new(&encoded)).unwrap();
    wrong_pcs_claim.hiding_whir_proofs[0].evals.clear();
    assert!(verify_v9(
        &config,
        &vk,
        &wrong_pcs_claim,
        vk_pre_hash,
        &mut candidate_challenger(),
    )
    .is_err());

    let mut wrong_mu_pow =
        openvm_stark_backend::proof::ProofV9::<SC>::decode(&mut Cursor::new(&encoded)).unwrap();
    wrong_mu_pow.mu_pow_witness += F::ONE;
    assert!(verify_v9(
        &config,
        &vk,
        &wrong_mu_pow,
        vk_pre_hash,
        &mut candidate_challenger(),
    )
    .is_err());
}

fn hiding_swirl_bounded_decode(encoded: &[u8]) {
    type Proof = openvm_stark_backend::proof::ProofV9<SC>;
    assert!(Proof::decode_from_bytes(&encoded[..encoded.len() - 1]).is_err());
    let mut trailing = encoded.to_vec();
    trailing.push(0);
    assert!(Proof::decode_from_bytes(&trailing).is_err());

    let mut oversized_commitment = 5u32.to_le_bytes().to_vec();
    oversized_commitment.extend_from_slice(&1u32.to_le_bytes());
    oversized_commitment.extend_from_slice(&((64 * 1024 * 1024 + 1) as u32).to_le_bytes());
    assert!(Proof::decode_from_bytes(&oversized_commitment).is_err());

    let commitment_count = read_u32(encoded, 4) as usize;
    assert!((1..=2).contains(&commitment_count));
    let mut num_airs_offset = 8;
    for _ in 0..commitment_count {
        let commitment_len = read_u32(encoded, num_airs_offset) as usize;
        num_airs_offset += 4 + commitment_len;
    }
    let mut bad_commitment_count = encoded.to_vec();
    bad_commitment_count[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert!(Proof::decode_from_bytes(&bad_commitment_count).is_err());
    let num_airs = read_u32(encoded, num_airs_offset) as usize;
    let bitmap_offset = num_airs_offset + 4;
    let mut public_values_offset = bitmap_offset + num_airs.div_ceil(8);
    for air in 0..num_airs {
        if encoded[bitmap_offset + air / 8] & (1 << (air % 8)) != 0 {
            let cached_count = read_u32(encoded, public_values_offset + 4) as usize;
            public_values_offset += 8 + cached_count * 32;
        }
    }
    let mut wrong_public_value_count = encoded.to_vec();
    wrong_public_value_count[public_values_offset..public_values_offset + 4]
        .copy_from_slice(&0u32.to_le_bytes());
    assert!(Proof::decode_from_bytes(&wrong_public_value_count).is_err());

    let mut too_many_airs = encoded.to_vec();
    too_many_airs[num_airs_offset..num_airs_offset + 4].copy_from_slice(&65u32.to_le_bytes());
    assert!(Proof::decode_from_bytes(&too_many_airs).is_err());

    let public_value_count = read_u32(encoded, public_values_offset) as usize;
    public_values_offset += 4;
    let mut base_field = Vec::new();
    SC::encode_base_field(&F::ZERO, &mut base_field).unwrap();
    for _ in 0..public_value_count {
        let values_len = read_u32(encoded, public_values_offset) as usize;
        public_values_offset += 4 + values_len * base_field.len();
    }
    let mut extension_field = Vec::new();
    SC::encode_extension_field(&EF::ZERO, &mut extension_field).unwrap();
    let gkr_claim_count_offset = public_values_offset + base_field.len() + extension_field.len();
    let mut too_many_gkr_layers = encoded.to_vec();
    too_many_gkr_layers[gkr_claim_count_offset..gkr_claim_count_offset + 4]
        .copy_from_slice(&28u32.to_le_bytes());
    assert!(Proof::decode_from_bytes(&too_many_gkr_layers).is_err());
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn assert_transcript_events_match(prover: &V9Transcript, verifier: &V9Transcript) {
    let prover_events = prover.events();
    let verifier_events = verifier.events();
    let mismatch = prover_events
        .iter()
        .zip(&verifier_events)
        .position(|(prover, verifier)| prover != verifier);
    assert!(
        mismatch.is_none() && prover_events.len() == verifier_events.len(),
        "V9 transcript events differ: prover={}, verifier={}, first mismatch={:?}",
        prover_events.len(),
        verifier_events.len(),
        mismatch.map(|index| { (index, &prover_events[index], verifier_events.get(index),) })
    );
}

#[test]
fn hiding_swirl_interactions_and_cur_next_roundtrip() {
    let params = test_system_params_small(2, 18, 3);
    let config = SC::default_from_params(params);
    let fib = FibFixture::new(0, 1, 2);
    let interactions = InteractionsFixture11;
    let mut keygen = MultiStarkKeygenBuilder::new(config.clone());
    keygen.add_air(Arc::new(FibonacciAir));
    for air in interactions.airs() {
        keygen.add_air(air);
    }
    let pk = keygen.generate_pk().unwrap();
    assert!(pk.per_air.iter().any(|air| air.vk.params.need_rot));
    let vk = pk.get_vk();

    let mut per_trace = fib.generate_proving_ctx().per_trace;
    for (air_id, trace) in interactions.generate_proving_ctx().per_trace {
        per_trace.push((air_id + 1, trace));
    }
    let host_ctx = ProvingContext::new(per_trace);
    let device = CpuDevice::new(config.clone());
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx = device.transport_proving_ctx_to_device(&host_ctx);
    let vk_pre_hash = [F::from_u64(0x4939); 8];
    let mut prover_transcript = V9Transcript::with_events(raw_candidate_challenger());
    let proof = prove_v9(&device, &d_pk, d_ctx, vk_pre_hash, &mut prover_transcript).unwrap();
    let mut verifier_transcript = V9Transcript::with_events(raw_candidate_challenger());
    verify_v9(&config, &vk, &proof, vk_pre_hash, &mut verifier_transcript).unwrap();
    assert_transcript_events_match(&prover_transcript, &verifier_transcript);
    let prover_continuation =
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(
            &mut prover_transcript,
        );
    let verifier_continuation =
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(
            &mut verifier_transcript,
        );
    assert_eq!(prover_continuation, verifier_continuation);
}

#[test]
fn hiding_swirl_rejects_unsupported_profiles_before_transcript_use() {
    let params = test_system_params_small(2, 4, 3);
    let preprocessed = PreprocessedFibFixture {
        a: 0,
        b: 1,
        sels: vec![false, true, false, true],
    };
    assert_prover_rejects_before_transcript(
        preprocessed.airs(),
        preprocessed.generate_proving_ctx(),
        params,
    );

    let params = test_system_params_small(2, 4, 3);
    let config = SC::default_from_params(params.clone());
    let cached = CachedFixture11 { config };
    assert_prover_rejects_before_transcript(cached.airs(), cached.generate_proving_ctx(), params);
}

fn assert_prover_rejects_before_transcript(
    airs: Vec<openvm_stark_backend::AirRef<SC>>,
    host_ctx: ProvingContext<openvm_stark_backend::prover::CpuColMajorBackend<SC>>,
    params: openvm_stark_backend::SystemParams,
) {
    let config = SC::default_from_params(params);
    let mut keygen = MultiStarkKeygenBuilder::new(config.clone());
    for air in airs {
        keygen.add_air(air);
    }
    let pk = keygen.generate_pk().unwrap();
    let device = CpuDevice::new(config);
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx = device.transport_proving_ctx_to_device(&host_ctx);
    let mut transcript = candidate_challenger();
    assert!(matches!(
        prove_v9(&device, &d_pk, d_ctx, [F::ZERO; 8], &mut transcript),
        Err(crate::hiding_whir::HidingWhirProverError::UnsupportedCommitments)
    ));
    assert_eq!(
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(&mut transcript),
        <openvm_stark_backend::hiding_whir::transcript::V9Transcript as FiatShamirTranscript<SC>>::sample_ext(&mut candidate_challenger())
    );
}

#[test]
fn hiding_swirl_stacking_from_views_matches_legacy() {
    let (config, device, pk, _, host_ctx) = setup(test_system_params_small(2, 4, 3));
    let d_pk = device.transport_pk_to_device(&pk);
    let ctx = device
        .transport_proving_ctx_to_device(&host_ctx)
        .into_sorted();
    let traces = ctx
        .common_main_traces()
        .map(|(_, trace)| trace)
        .collect::<Vec<_>>();
    let (_, pcs_data) = device.commit(&traces).unwrap();
    let need_rot = d_pk.per_air[0].vk.params.need_rot;
    let r = vec![EF::ONE];

    let mut legacy_transcript = candidate_challenger();
    let lambda: EF = <openvm_stark_backend::hiding_whir::transcript::V9Transcript as openvm_stark_backend::FiatShamirTranscript<SC>>::sample_ext(&mut legacy_transcript);
    let legacy_state =
        StackedReductionCpuNew::new(&device, vec![&pcs_data], vec![vec![need_rot]], &r, lambda);

    let mut view_transcript = candidate_challenger();
    let view_lambda: EF = <openvm_stark_backend::hiding_whir::transcript::V9Transcript as openvm_stark_backend::FiatShamirTranscript<SC>>::sample_ext(&mut view_transcript);
    assert_eq!(lambda, view_lambda);
    let view_state = StackedReductionCpuNew::from_views(
        &device,
        vec![StackedMatrixView {
            layout: &pcs_data.layout,
            matrix: &pcs_data.matrix,
        }],
        vec![vec![need_rot]],
        &r,
        view_lambda,
    );

    let legacy = prove_stacked_opening_reduction_from_state::<SC, _, _>(
        &mut legacy_transcript,
        config.params().n_stack,
        legacy_state,
    );
    let from_views = prove_stacked_opening_reduction_from_state::<SC, _, _>(
        &mut view_transcript,
        config.params().n_stack,
        view_state,
    );
    assert_eq!(legacy, from_views);
}
