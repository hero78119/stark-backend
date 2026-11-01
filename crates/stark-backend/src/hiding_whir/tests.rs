use std::sync::OnceLock;

use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear as OuterBase, Poseidon2BabyBear};
use p3_challenger_08::{CanObserve, CanSample, CanSampleUniformBits};
use p3_commit_08::MultilinearPcs as _;
use p3_field::{
    BasedVectorSpace as OuterBvs, Field as OuterField, PrimeCharacteristicRing as OuterRing,
};
use p3_field_08::{
    BasedVectorSpace as HidingBvs, Field as HidingField, PrimeCharacteristicRing as HidingRing,
    PrimeField64 as HidingPrimeField64, TwoAdicField,
};
use p3_multilinear_util_08::{point::Point, poly::Poly};
use p3_symmetric_08::MerkleCap;
use rand_10::{rngs::StdRng, SeedableRng};

use crate::{
    config::{SystemParams, WhirProximityStrategy},
    hasher::Hasher,
    interaction::LogUpSecurityParameters,
    prover::poly::{eval_to_coeff_rs_message, evals_mobius_eq_hypercube, Mle},
    FiatShamirTranscript, StarkProtocolConfig,
};

use super::{
    bridge, candidate_challenger, candidate_config, candidate_pcs, raw_candidate_challenger,
    HidingBase, HidingCommitment, HidingExtension, HidingMerkleMmcs, HidingPcs, OuterExtension,
};

type OuterPerm = Poseidon2BabyBear<16>;
type OuterHash = p3_symmetric::PaddingFreeSponge<OuterPerm, 16, 8, 8>;
type OuterCompress = p3_symmetric::TruncatedPermutation<OuterPerm, 2, 8, 16>;
type TestHasher = Hasher<OuterBase, [OuterBase; 8], OuterHash, OuterCompress>;

#[derive(Clone, Debug)]
struct TestConfig {
    params: crate::SystemParams,
    hasher: TestHasher,
}

impl StarkProtocolConfig for TestConfig {
    type F = OuterBase;
    type EF = OuterExtension;
    type Digest = [OuterBase; 8];
    type Hasher = TestHasher;

    fn params(&self) -> &crate::SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

fn test_config() -> &'static TestConfig {
    static CONFIG: OnceLock<TestConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let permutation = default_babybear_poseidon2_16();
        TestConfig {
            params: SystemParams::new(
                2,
                4,
                4,
                16,
                2,
                0,
                0,
                WhirProximityStrategy::UniqueDecoding,
                32,
                LogUpSecurityParameters {
                    max_interaction_count: 1,
                    log_max_message_length: 1,
                    pow_bits: 0,
                },
                2,
                0,
                2,
            ),
            hasher: Hasher::new(
                p3_symmetric::PaddingFreeSponge::new(permutation.clone()),
                p3_symmetric::TruncatedPermutation::new(permutation),
            ),
        }
    })
}

fn outer_base(value: u64) -> OuterBase {
    OuterRing::from_u64(value)
}

fn outer_ext(coefficients: [u64; 4]) -> OuterExtension {
    <OuterExtension as OuterBvs<OuterBase>>::from_basis_coefficients_fn(|i| {
        outer_base(coefficients[i])
    })
}

fn hiding_coefficients(value: HidingExtension) -> [u64; 8] {
    <HidingExtension as HidingBvs<HidingBase>>::as_basis_coefficients_slice(&value)
        .iter()
        .map(|coefficient| HidingPrimeField64::as_canonical_u64(coefficient))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

#[test]
fn hiding_whir_field_embedding_preserves_field_ops() {
    let a = outer_ext([2, 3, 5, 7]);
    let b = outer_ext([11, 13, 17, 19]);
    let embed_a = bridge::extension_to_hiding(a);
    let embed_b = bridge::extension_to_hiding(b);

    assert_eq!(
        bridge::extension_to_hiding(OuterExtension::ZERO),
        HidingExtension::ZERO
    );
    assert_eq!(
        bridge::extension_to_hiding(OuterExtension::ONE),
        HidingExtension::ONE
    );
    assert_eq!(bridge::extension_to_hiding(a + b), embed_a + embed_b);
    assert_eq!(bridge::extension_to_hiding(a * b), embed_a * embed_b);
    assert_eq!(
        bridge::extension_to_hiding(a.try_inverse().unwrap()),
        embed_a.try_inverse().unwrap()
    );
}

#[test]
fn hiding_whir_p3_table_matches_rs_mle_and_mobius_oracles() {
    for l_skip in [0, 1, 4] {
        for boolean_variables in 0..=2 {
            let length = 1usize << (l_skip + boolean_variables);
            let source = (0..length)
                .map(|i| outer_base((i * 23 + i * i * 3 + 7) as u64))
                .collect::<Vec<_>>();
            let table = eval_to_coeff_rs_message(l_skip, &source);
            let p3 = Poly::new(
                table
                    .iter()
                    .copied()
                    .map(bridge::base_to_hiding)
                    .collect::<Vec<_>>(),
            );
            let mle = Mle::from_evaluations(&table);
            let dimension = l_skip + boolean_variables;
            let mut points = (0..(1usize << dimension))
                .map(|mask| {
                    (0..dimension)
                        .map(|coordinate| {
                            if mask & (1 << coordinate) == 0 {
                                OuterExtension::ZERO
                            } else {
                                OuterExtension::ONE
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            points.push(
                (0..dimension)
                    .map(|i| outer_ext([i as u64 + 2, 1, 4, 9]))
                    .collect(),
            );

            for point in points {
                let p3_point = point
                    .iter()
                    .rev()
                    .copied()
                    .map(bridge::extension_to_hiding)
                    .collect::<Vec<_>>();
                let p3_value = p3.eval_base(&Point::new(p3_point));
                let old_mle_value = mle.eval_at_point(&point);
                let mut old_coeff_evaluations = table.clone();
                Mle::coeffs_to_evals_inplace(&mut old_coeff_evaluations);
                let mobius = evals_mobius_eq_hypercube(&point);
                let old_mobius_value = old_coeff_evaluations
                    .iter()
                    .zip(mobius)
                    .fold(OuterExtension::ZERO, |sum, (&value, weight)| {
                        sum + OuterExtension::from(value) * weight
                    });

                assert_eq!(p3_value, bridge::extension_to_hiding(old_mle_value));
                assert_eq!(old_mle_value, old_mobius_value);
            }
        }
    }
    // Coefficients [2, 5] and evaluations [2, 5] are different polynomials.
    let evaluations = [outer_base(2), outer_base(5)];
    assert_eq!(
        Mle::from_evaluations(&evaluations).eval_at_point::<OuterBase, OuterBase>(&[outer_base(3)]),
        outer_base(11)
    );
    assert_eq!(
        Mle::from_coeffs(evaluations.to_vec())
            .eval_at_point::<OuterBase, OuterBase>(&[outer_base(3)]),
        outer_base(17)
    );
}

#[test]
fn hiding_whir_column_packing_matches_mu_batch_for_widths_and_padding() {
    let mu = outer_ext([7, 3, 2, 1]);
    for width in [1, 2, 3, 5] {
        let columns = (0..width)
            .map(|column| {
                let source = (0..8)
                    .map(|row| outer_base((column * 31 + row * row * 7 + 3) as u64))
                    .collect::<Vec<_>>();
                eval_to_coeff_rs_message(1, &source)
            })
            .collect::<Vec<_>>();
        let cube_point = vec![
            outer_ext([5, 2, 1, 3]),
            outer_ext([9, 4, 2, 6]),
            outer_ext([13, 5, 8, 2]),
        ];
        let column_claims = columns
            .iter()
            .map(|column| Mle::from_evaluations(column).eval_at_point(&cube_point))
            .collect::<Vec<OuterExtension>>();
        let expected = column_claims
            .iter()
            .enumerate()
            .fold(OuterExtension::ZERO, |sum, (index, &claim)| {
                sum + mu.exp_u64(index as u64) * claim
            });
        let packed = bridge::pack_columns(&columns).unwrap();
        let (point, claim) =
            bridge::packed_opening(mu, columns.len(), &cube_point, expected).unwrap();

        assert_eq!(
            hiding_coefficients(packed.eval_base(&point)),
            hiding_coefficients(claim),
            "width={width}"
        );
    }
    assert!(std::panic::catch_unwind(|| eval_to_coeff_rs_message(1, &[outer_base(1)])).is_err());
    assert!(std::panic::catch_unwind(|| eval_to_coeff_rs_message(4, &[outer_base(1); 8])).is_err());
    let one_column = [vec![outer_base(2), outer_base(5)]];
    let (point, claim) = bridge::packed_opening(
        OuterExtension::NEG_ONE,
        1,
        &[outer_ext([3, 2, 1, 4])],
        outer_ext([7, 1, 3, 2]),
    )
    .unwrap();
    assert_eq!(
        point.as_slice().len(),
        1,
        "width one has no selector coordinate"
    );
    assert_eq!(claim, bridge::extension_to_hiding(outer_ext([7, 1, 3, 2])));
    assert_eq!(
        bridge::pack_columns(&one_column).unwrap().eval_base(&point),
        HidingExtension::from(bridge::base_to_hiding(outer_base(2)))
            + (HidingExtension::from(bridge::base_to_hiding(outer_base(5)))
                - HidingExtension::from(bridge::base_to_hiding(outer_base(2))))
                * point[0]
    );
    assert_eq!(
        bridge::packed_opening(
            OuterExtension::NEG_ONE,
            2,
            &[OuterExtension::ZERO],
            OuterExtension::ZERO
        ),
        Err(bridge::BridgeError::ZeroSelectorDenominator)
    );
}

#[test]
fn hiding_whir_pinned_candidate_config_fits_and_reports() {
    assert!(candidate_config(16).is_err());
    assert!(candidate_config(24).is_err());
    assert!(candidate_config(usize::MAX).is_err());
    let accepted = (0..=HidingBase::TWO_ADICITY as usize - 1)
        .filter_map(|dimension| match candidate_config(dimension) {
            Ok(config) => Some((dimension, config)),
            Err(error) => {
                eprintln!("candidate dimension {dimension} rejected: {error}");
                None
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !accepted.is_empty(),
        "fixed 128-bit BabyBear candidate has no accepted dimensions"
    );

    for (dimension, config) in &accepted {
        config.validate_initial_claims(1).unwrap();
        assert!(*dimension + config.starting_log_inv_rate <= HidingBase::TWO_ADICITY as usize);
        assert!(config.sumcheck_mask().domain_size.is_power_of_two());
        assert!(
            config.sumcheck_mask().domain_size.trailing_zeros() as usize
                <= HidingExtension::TWO_ADICITY
        );
        for group in config.mask_groups() {
            assert!(
                group.shape.message_len + group.shape.randomness_len <= group.shape.domain_size
            );
            assert!(
                group.shape.domain_size.trailing_zeros() as usize <= HidingExtension::TWO_ADICITY
            );
        }
        eprintln!(
            "dimension={dimension}; base2_adicity={}; ext2_adicity={}; rounds={:?}; randomized_terminal={:?}; oracle_randomness={:?}; sumcheck_mask={:?}; switch_masks={:?}; security={:#?}",
            HidingBase::TWO_ADICITY,
            HidingExtension::TWO_ADICITY,
            config.folding_schedule(),
            config.randomized_terminal(),
            config.oracle_randomness(),
            config.sumcheck_mask(),
            config.switch_masks(),
            config.hiding_base_case_security_report(),
        );
    }
    eprintln!(
        "all accepted candidate dimensions: {:?}",
        accepted
            .iter()
            .map(|(dimension, _)| *dimension)
            .collect::<Vec<_>>()
    );
}

#[test]
fn hiding_whir_p3_challenger_matches_v9_outer_transcript_adapter() {
    let _ = test_config();
    let initial = raw_candidate_challenger();
    let mut adapted = super::transcript::V9Transcript::with_events(initial.clone());
    let mut reference = initial;
    let base = outer_base(27);

    FiatShamirTranscript::<TestConfig>::observe(&mut adapted, base);
    CanObserve::observe(&mut reference, bridge::base_to_hiding(base));
    assert_eq!(
        FiatShamirTranscript::<TestConfig>::sample(&mut adapted),
        bridge::base_to_outer(CanSample::sample(&mut reference))
    );

    let extension = outer_ext([31, 37, 41, 43]);
    FiatShamirTranscript::<TestConfig>::observe_ext(&mut adapted, extension);
    for &coefficient in
        <OuterExtension as OuterBvs<OuterBase>>::as_basis_coefficients_slice(&extension)
    {
        CanObserve::observe(&mut reference, bridge::base_to_hiding(coefficient));
    }
    let adapted_ext = FiatShamirTranscript::<TestConfig>::sample_ext(&mut adapted);
    let expected_ext = <OuterExtension as OuterBvs<OuterBase>>::from_basis_coefficients_fn(|_| {
        bridge::base_to_outer(CanSample::sample(&mut reference))
    });
    assert_eq!(adapted_ext, expected_ext);

    let cap = MerkleCap::<HidingBase, [HidingBase; 8]>::new(vec![[HidingBase::ONE; 8]]);
    CanObserve::observe(&mut adapted, cap.clone());
    CanObserve::observe(&mut reference, cap.clone());
    CanObserve::observe(&mut adapted, cap.clone());
    CanObserve::observe(&mut reference, cap);
    let p3_scalar = HidingBase::from_u64(47);
    CanObserve::observe(&mut adapted, p3_scalar);
    CanObserve::observe(&mut reference, p3_scalar);
    let p3_array = [HidingBase::from_u64(53), HidingBase::from_u64(59)];
    CanObserve::observe(&mut adapted, p3_array);
    CanObserve::observe(&mut reference, p3_array);

    let adapted_bits = FiatShamirTranscript::<TestConfig>::sample_bits(&mut adapted, 12);
    let reference_bits =
        HidingPrimeField64::as_canonical_u64(&CanSample::<HidingBase>::sample(&mut reference))
            & 0xfff;
    assert_eq!(adapted_bits, reference_bits);
    let events = adapted.events();
    assert!(events.contains(&super::transcript::TranscriptEvent::ObserveP3Base(47)));
    assert!(
        events.contains(&super::transcript::TranscriptEvent::ObserveP3Array(vec![
            53, 59
        ]))
    );
    let mut rejection_state = raw_candidate_challenger();
    rejection_state.output_buffer = vec![HidingBase::NEG_ONE];
    let mut rejection_adapter = super::transcript::V9Transcript::new(rejection_state.clone());
    assert!(
        CanSampleUniformBits::<HidingBase>::sample_uniform_bits::<false>(
            &mut rejection_adapter,
            15
        )
        .is_err()
    );
    assert!(
        CanSampleUniformBits::<HidingBase>::sample_uniform_bits::<false>(&mut rejection_state, 15)
            .is_err()
    );
    assert_eq!(
        CanSampleUniformBits::<HidingBase>::sample_uniform_bits::<true>(&mut rejection_adapter, 15)
            .unwrap(),
        CanSampleUniformBits::<HidingBase>::sample_uniform_bits::<true>(&mut rejection_state, 15)
            .unwrap()
    );
}

#[test]
fn hiding_whir_prover_verifier_transcript_events_match() {
    let dimension = (0..=HidingBase::TWO_ADICITY as usize - 1)
        .find(|&dimension| candidate_config(dimension).is_ok())
        .expect("the fixed candidate must accept at least one dimension");
    let config = candidate_config(dimension).unwrap();
    let pcs: HidingPcs = candidate_pcs(config, StdRng::seed_from_u64(0x4849_4449_4e47));
    let table = (0..(1usize << dimension))
        .map(|index| HidingBase::from_u64((index * 19 + 5) as u64))
        .collect::<Vec<_>>();
    let witness = Poly::new(table);
    let point = Point::new(vec![HidingExtension::ONE; dimension]);
    let expected_claim = witness.eval_base(&point);
    let mut prover_challenger =
        super::transcript::V9Transcript::with_events(raw_candidate_challenger());
    let (commitment, prover_data) =
        p3_commit_08::MultilinearPcs::commit(&pcs, witness, &mut prover_challenger).unwrap();
    let pow_witness = (0u64..4096)
        .map(outer_base)
        .find(|witness| {
            let mut probe = prover_challenger.clone_untraced();
            run_outer_transcript_prefix(&mut probe);
            FiatShamirTranscript::<TestConfig>::check_witness(&mut probe, 3, *witness)
        })
        .expect("3-bit outer PoW should have a witness in the first 4096 candidates");
    run_outer_transcript_segment(&mut prover_challenger, pow_witness);
    pcs.observe_commitment(&commitment, &mut prover_challenger);
    let proof = p3_commit_08::MultilinearPcs::open(
        &pcs,
        prover_data,
        vec![point.clone()],
        &mut prover_challenger,
    )
    .unwrap();

    let mut verifier_challenger =
        super::transcript::V9Transcript::with_events(raw_candidate_challenger());
    pcs.observe_commitment(&commitment, &mut verifier_challenger);
    run_outer_transcript_segment(&mut verifier_challenger, pow_witness);
    assert_eq!(
        proof.evals.as_slice(),
        &[expected_claim],
        "V9's independently computed packed claim must match the PCS statement"
    );
    assert_ne!(proof.evals[0], expected_claim + HidingExtension::ONE);
    p3_commit_08::MultilinearPcs::verify(
        &pcs,
        &commitment,
        &proof,
        &mut verifier_challenger,
        vec![point.clone()],
    )
    .unwrap();
    let prover_events = prover_challenger.events();
    let verifier_events = verifier_challenger.events();
    assert_eq!(
        prover_events, verifier_events,
        "initial root, outer transcript/PoW, opening root, claims, and P3 events must match"
    );
    let last_root = prover_events
        .iter()
        .rposition(|event| {
            matches!(
                event,
                super::transcript::TranscriptEvent::ObserveCommitment(_)
            )
        })
        .expect("the opening root is recorded");
    assert!(
        prover_events[last_root + 1..]
            .iter()
            .any(|event| matches!(event, super::transcript::TranscriptEvent::ObserveP3Base(_))),
        "the opening claims' domain separator and field absorbs must be recorded"
    );
    let mut prover_after = prover_challenger.clone_untraced();
    let mut verifier_after = verifier_challenger.clone_untraced();
    for _ in 0..16 {
        assert_eq!(
            CanSample::<HidingBase>::sample(&mut prover_after.challenger()),
            CanSample::<HidingBase>::sample(&mut verifier_after.challenger()),
            "matching absorb events must leave the same challenger state"
        );
    }
    let main_root = commitment
        .as_ref()
        .iter()
        .flat_map(|digest| digest.iter())
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    assert_eq!(
        prover_events
            .iter()
            .filter(|event| matches!(event, super::transcript::TranscriptEvent::ObserveCommitment(root) if root == &main_root))
            .count(),
        2,
        "the common-main root is bound at exactly the two prescribed positions"
    );

    let mut omitted_opening_root =
        super::transcript::V9Transcript::with_events(raw_candidate_challenger());
    pcs.observe_commitment(&commitment, &mut omitted_opening_root);
    run_outer_transcript_segment(&mut omitted_opening_root, pow_witness);
    let sampled_without_second_root =
        FiatShamirTranscript::<TestConfig>::sample(&mut omitted_opening_root);
    let mut expected_after_opening_root =
        super::transcript::V9Transcript::with_events(raw_candidate_challenger());
    pcs.observe_commitment(&commitment, &mut expected_after_opening_root);
    run_outer_transcript_segment(&mut expected_after_opening_root, pow_witness);
    pcs.observe_commitment(&commitment, &mut expected_after_opening_root);
    let sampled_with_second_root =
        FiatShamirTranscript::<TestConfig>::sample(&mut expected_after_opening_root);
    assert_ne!(sampled_without_second_root, sampled_with_second_root);

    let other_pcs: HidingPcs = candidate_pcs(
        candidate_config(dimension).unwrap(),
        StdRng::seed_from_u64(0x5a4b_5748_4952),
    );
    let other_witness = Poly::new(
        (0..(1usize << dimension))
            .map(|index| HidingBase::from_u64((index * 19 + 5) as u64))
            .collect(),
    );
    let (other_commitment, _) = p3_commit_08::MultilinearPcs::commit(
        &other_pcs,
        other_witness,
        &mut candidate_challenger(),
    )
    .unwrap();
    assert_ne!(
        commitment, other_commitment,
        "independent CSPRNG seeds must change the hiding commitment"
    );

    let mut storage = vec![0; 4 * 1024 * 1024];
    let wire = super::codec::encode_into(&proof, &mut storage).expect("bounded codec buffer fits");
    let decode_limits = super::codec::DecodeLimits::for_dimension(dimension, 64 * 1024 * 1024);
    let standalone_commitment_wire = postcard::to_allocvec(&commitment).unwrap();
    let _: HidingCommitment = super::codec::decode_bounded(
        &standalone_commitment_wire,
        super::codec::DecodeLimits::for_commitment(dimension, 64 * 1024 * 1024),
    )
    .unwrap();
    let proof_copy: p3_whir_08::ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs> =
        super::codec::decode_bounded(wire, decode_limits).unwrap();
    let mut commitment_storage = vec![0; 1024];
    let commitment_storage_limit = commitment_storage.len();
    let commitment_wire = super::codec::encode_into(&commitment, &mut commitment_storage).unwrap();
    let commitment_copy: HidingCommitment = super::codec::decode_bounded(
        commitment_wire,
        super::codec::DecodeLimits::for_commitment(dimension, commitment_storage_limit),
    )
    .unwrap();
    assert_eq!(commitment, commitment_copy);

    assert!(super::codec::decode_bounded::<
        p3_whir_08::ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs>,
    >(&wire[..wire.len() - 1], decode_limits)
    .is_err());
    let mut trailing = wire.to_vec();
    trailing.push(0);
    assert!(matches!(
        super::codec::decode_bounded::<
            p3_whir_08::ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs>,
        >(&trailing, decode_limits),
        Err(super::codec::CodecError::TrailingBytes)
    ));

    let mut final_verifier = candidate_challenger();
    pcs.observe_commitment(&commitment_copy, &mut final_verifier);
    run_outer_transcript_segment(&mut final_verifier, pow_witness);
    p3_commit_08::MultilinearPcs::verify(
        &pcs,
        &commitment_copy,
        &proof_copy,
        &mut final_verifier,
        vec![point.clone()],
    )
    .unwrap();

    assert_eq!(proof_copy.evals[0], expected_claim);
    let wrong_expected_claim = expected_claim + HidingExtension::ONE;
    assert_ne!(proof_copy.evals[0], wrong_expected_claim);
    let rejects_wrong_external_claim = |expected: HidingExtension| {
        if proof_copy.evals.first() != Some(&expected) {
            return false;
        }
        let mut challenger = candidate_challenger();
        pcs.observe_commitment(&commitment_copy, &mut challenger);
        run_outer_transcript_segment(&mut challenger, pow_witness);
        p3_commit_08::MultilinearPcs::verify(
            &pcs,
            &commitment_copy,
            &proof_copy,
            &mut challenger,
            vec![point.clone()],
        )
        .is_ok()
    };
    assert!(rejects_wrong_external_claim(expected_claim));
    assert!(!rejects_wrong_external_claim(wrong_expected_claim));

    let mut over_limit = wire.to_vec();
    let oversized_prefix = postcard::to_allocvec(&(decode_limits.max_sequence_len + 1)).unwrap();
    over_limit.splice(..1, oversized_prefix);
    assert!(matches!(
        super::codec::decode_bounded::<
            p3_whir_08::ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs>,
        >(&over_limit, decode_limits,),
        Err(super::codec::CodecError::AllocationLimit { .. })
    ));

    let first_sumcheck = proof
        .sumchecks
        .first()
        .expect("roundtrip fixture has sumchecks");
    let evals_wire = postcard::to_allocvec(&proof.evals).unwrap();
    let sumchecks_len = postcard::to_allocvec(&proof.sumchecks.len()).unwrap();
    let mu_tilde_wire = postcard::to_allocvec(&first_sumcheck.mu_tilde).unwrap();
    let nested_len_offset = evals_wire.len() + sumchecks_len.len() + mu_tilde_wire.len();
    let mut nested_over_limit = wire.to_vec();
    nested_over_limit.splice(
        nested_len_offset..nested_len_offset + 1,
        postcard::to_allocvec(&(decode_limits.max_sequence_len + 1)).unwrap(),
    );
    assert!(matches!(
        super::codec::decode_bounded::<
            p3_whir_08::ZkWhirProof<HidingBase, HidingExtension, HidingMerkleMmcs>,
        >(&nested_over_limit, decode_limits),
        Err(super::codec::CodecError::AllocationLimit { .. })
    ));

    let mut nested_vectors = vec![1];
    nested_vectors.extend(postcard::to_allocvec(&(decode_limits.max_sequence_len + 1)).unwrap());
    assert!(matches!(
        super::codec::decode_bounded::<Vec<Vec<u8>>>(
            &nested_vectors,
            super::codec::DecodeLimits::for_dimension(dimension, 64 * 1024 * 1024),
        ),
        Err(super::codec::CodecError::AllocationLimit { .. })
    ));

    let mut oversized_commitment = commitment_wire.to_vec();
    oversized_commitment.splice(
        ..1,
        postcard::to_allocvec(&(decode_limits.max_sequence_len + 1)).unwrap(),
    );
    assert!(matches!(
        super::codec::decode_bounded::<HidingCommitment>(
            &oversized_commitment,
            super::codec::DecodeLimits::for_commitment(dimension, 64 * 1024 * 1024),
        ),
        Err(super::codec::CodecError::AllocationLimit { .. })
    ));
    assert!(matches!(
        super::codec::decode_bounded::<Vec<u8>>(
            &[0],
            super::codec::DecodeLimits::for_commitment(dimension, 0)
        ),
        Err(super::codec::CodecError::TooLarge { .. })
    ));
    eprintln!("standalone WHIR proof bytes: {}", wire.len());
}

fn run_outer_transcript_segment(
    transcript: &mut super::transcript::V9Transcript,
    pow_witness: OuterBase,
) {
    run_outer_transcript_prefix(transcript);
    assert!(FiatShamirTranscript::<TestConfig>::check_witness(
        transcript,
        3,
        pow_witness
    ));
}

fn run_outer_transcript_prefix(transcript: &mut super::transcript::V9Transcript) {
    let digest = [outer_base(17); 8];
    FiatShamirTranscript::<TestConfig>::observe_commit(transcript, digest);
    FiatShamirTranscript::<TestConfig>::observe_ext(transcript, outer_ext([31, 37, 41, 43]));
    let _ = FiatShamirTranscript::<TestConfig>::sample_ext(transcript);
    let _ = FiatShamirTranscript::<TestConfig>::sample_bits(transcript, 7);
}
