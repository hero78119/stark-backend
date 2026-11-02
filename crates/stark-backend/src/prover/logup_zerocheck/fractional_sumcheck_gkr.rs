use std::ops::Add;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug, instrument};

use crate::{
    proof::GkrLayerClaims,
    prover::{
        error::LogupZerocheckError,
        poly::evals_eq_hypercube,
        sumcheck::{fold_mle_evals, sumcheck_round_poly_evals},
        ColMajorMatrix,
    },
    FiatShamirTranscript, StarkProtocolConfig,
};

/// Proof for fractional sumcheck protocol
pub struct FracSumcheckProof<SC: StarkProtocolConfig> {
    /// The fractional sum p_0 / q_0
    pub fractional_sum: (SC::EF, SC::EF),
    /// The claims for p_j(0, rho), p_j(1, rho), q_j(0, rho), and q_j(1, rho) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims<SC>>,
    /// Sumcheck polynomials for each layer, for each sumcheck round, given by their evaluations on
    /// {1, 2, 3}.
    pub sumcheck_polys: Vec<Vec<[SC::EF; 3]>>,
}

#[derive(Clone, Copy, Debug, Default, derive_new::new)]
#[repr(C)]
pub struct Frac<EF> {
    // PERF[jpw]: in the initial round, we can keep `p` in base field
    pub p: EF,
    pub q: EF,
}

impl<EF: Field> Add<Frac<EF>> for Frac<EF> {
    type Output = Frac<EF>;

    fn add(self, other: Frac<EF>) -> Self::Output {
        Frac {
            p: self.p * other.q + self.q * other.p,
            q: self.q * other.q,
        }
    }
}

/// Runs the fractional sumcheck protocol using GKR layered circuit.
///
/// # Arguments
/// * `transcript` - The Fiat-Shamir transcript
/// * `evals` - list of `(p, q)` pairs of fractions in projective coordinates representing
///   evaluations on the hypercube
/// * `assert_zero` - Whether to assert that the final sum is zero. If `true`, then the transcript
///   will not observe the numerator of the final sum.
///
/// # Returns
/// The fractional sumcheck proof and the final random evaluation vector.
#[instrument(level = "info", skip_all)]
pub fn fractional_sumcheck<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    transcript: &mut TS,
    evals: &[Frac<SC::EF>],
    assert_zero: bool,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    fractional_sumcheck_with_stage_observer(transcript, evals, assert_zero, |_| {})
}

pub fn fractional_sumcheck_with_stage_observer<
    SC: StarkProtocolConfig,
    TS: FiatShamirTranscript<SC>,
    O: Fn(&'static str),
>(
    transcript: &mut TS,
    evals: &[Frac<SC::EF>],
    assert_zero: bool,
    on_stage: O,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    if evals.is_empty() {
        return Ok((empty_frac_sumcheck_proof(), vec![]));
    }
    let total_rounds = log2_strict_usize(evals.len());
    on_stage("fractional_segment_tree_allocation_started");
    let mut tree_evals: Vec<Frac<SC::EF>> = vec![Frac::default(); 2 << total_rounds];
    on_stage("fractional_segment_tree_allocated");
    tree_evals[(1 << total_rounds)..].copy_from_slice(evals);
    on_stage("fractional_segment_tree_leaves_copied");

    fractional_sumcheck_from_segment_tree(
        transcript,
        tree_evals,
        total_rounds,
        assert_zero,
        on_stage,
    )
}

/// Runs fractional sumcheck using an owned input vector whose capacity includes the segment tree.
/// The input evaluations are moved to the tree leaves in place to avoid retaining a duplicate.
pub fn fractional_sumcheck_owned_with_stage_observer<
    SC: StarkProtocolConfig,
    TS: FiatShamirTranscript<SC>,
    O: Fn(&'static str),
>(
    transcript: &mut TS,
    mut evals: Vec<Frac<SC::EF>>,
    assert_zero: bool,
    on_stage: O,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    if evals.is_empty() {
        return Ok((empty_frac_sumcheck_proof(), vec![]));
    }
    let total_rounds = log2_strict_usize(evals.len());
    let input_len = evals.len();
    on_stage("fractional_segment_tree_allocation_started");
    evals.resize(2 << total_rounds, Frac::default());
    on_stage("fractional_segment_tree_allocated");
    evals.copy_within(0..input_len, input_len);
    on_stage("fractional_segment_tree_leaves_copied");

    fractional_sumcheck_from_segment_tree_in_place(
        transcript,
        evals,
        total_rounds,
        assert_zero,
        on_stage,
    )
}

fn empty_frac_sumcheck_proof<SC: StarkProtocolConfig>() -> FracSumcheckProof<SC> {
    FracSumcheckProof {
        fractional_sum: (SC::EF::ZERO, SC::EF::ONE),
        claims_per_layer: vec![],
        sumcheck_polys: vec![],
    }
}

fn fractional_sumcheck_from_segment_tree_in_place<
    SC: StarkProtocolConfig,
    TS: FiatShamirTranscript<SC>,
    O: Fn(&'static str),
>(
    transcript: &mut TS,
    mut tree_evals: Vec<Frac<SC::EF>>,
    total_rounds: usize,
    assert_zero: bool,
    on_stage: O,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);
    for node_idx in (1..(1 << total_rounds)).rev() {
        tree_evals[node_idx] = tree_evals[2 * node_idx] + tree_evals[2 * node_idx + 1];
    }
    on_stage("fractional_segment_tree_built");

    let frac_sum = tree_evals[1];
    if assert_zero && frac_sum.p != SC::EF::ZERO {
        return Err(LogupZerocheckError::NonZeroRootSum);
    }
    if !assert_zero {
        transcript.observe_ext(frac_sum.p);
    }
    transcript.observe_ext(frac_sum.q);

    let mut claims_per_layer = Vec::with_capacity(total_rounds);
    claims_per_layer.push(GkrLayerClaims::<SC> {
        p_xi_0: tree_evals[2].p,
        q_xi_0: tree_evals[2].q,
        p_xi_1: tree_evals[3].p,
        q_xi_1: tree_evals[3].q,
    });
    transcript.observe_ext(claims_per_layer[0].p_xi_0);
    transcript.observe_ext(claims_per_layer[0].q_xi_0);
    transcript.observe_ext(claims_per_layer[0].p_xi_1);
    transcript.observe_ext(claims_per_layer[0].q_xi_1);
    let mu_1 = transcript.sample_ext();
    let mut xi_prev = vec![mu_1];

    for round in 1..total_rounds {
        let eval_size = 1 << round;
        let lambda = transcript.sample_ext();
        let tree_layer_start = 2 * eval_size;
        for x in 0..eval_size {
            let left = tree_evals[tree_layer_start + 2 * x];
            let right = tree_evals[tree_layer_start + 2 * x + 1];
            set_frac_field(&mut tree_evals, 0, x, left.p);
            set_frac_field(&mut tree_evals, 0, eval_size + x, left.q);
            set_frac_field(&mut tree_evals, 0, 2 * eval_size + x, right.p);
            set_frac_field(&mut tree_evals, 0, 3 * eval_size + x, right.q);
        }
        on_stage("fractional_gkr_round_matrix_materialized");

        let mut eq_xis = evals_eq_hypercube(&xi_prev);
        let mut matrix_start = 0;
        let mut scratch_start = tree_layer_start;
        let mut active_height = eval_size;
        let mut round_polys_eval = Vec::with_capacity(round);
        let mut rho = Vec::with_capacity(round);

        while active_height > 1 {
            let s_evals = fractional_round_poly_evals(
                &tree_evals,
                matrix_start,
                active_height,
                &eq_xis,
                lambda,
            );
            for eval in s_evals {
                transcript.observe_ext(eval);
            }
            round_polys_eval.push(s_evals);

            let r_round = transcript.sample_ext();
            fold_frac_matrix_into(
                &mut tree_evals,
                matrix_start,
                scratch_start,
                active_height,
                r_round,
            );
            fold_eq_evals_in_place(&mut eq_xis, r_round);
            std::mem::swap(&mut matrix_start, &mut scratch_start);
            active_height >>= 1;
            rho.push(r_round);
        }

        on_stage("fractional_gkr_round_complete");
        claims_per_layer.push(GkrLayerClaims::<SC> {
            p_xi_0: frac_field(&tree_evals, matrix_start, 0),
            q_xi_0: frac_field(&tree_evals, matrix_start, 1),
            p_xi_1: frac_field(&tree_evals, matrix_start, 2),
            q_xi_1: frac_field(&tree_evals, matrix_start, 3),
        });
        let claims = &claims_per_layer[round];
        transcript.observe_ext(claims.p_xi_0);
        transcript.observe_ext(claims.q_xi_0);
        transcript.observe_ext(claims.p_xi_1);
        transcript.observe_ext(claims.q_xi_1);

        let mu = transcript.sample_ext();
        xi_prev = [vec![mu], rho].concat();
        sumcheck_polys.push(round_polys_eval);
    }

    Ok((
        FracSumcheckProof {
            fractional_sum: (frac_sum.p, frac_sum.q),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}

fn fractional_round_poly_evals<EF: Field>(
    matrix: &[Frac<EF>],
    matrix_start: usize,
    height: usize,
    eq_xis: &[EF],
    lambda: EF,
) -> [EF; 3] {
    (0..height / 2)
        .into_par_iter()
        .map(|y| {
            let eq_0 = eq_xis[2 * y];
            let eq_1 = eq_xis[2 * y + 1];
            std::array::from_fn(|x_idx| {
                let x = EF::from_usize(x_idx + 1);
                let eq = eq_0 + (eq_1 - eq_0) * x;
                let p_j0 = matrix_value(matrix, matrix_start, height, 0, 2 * y)
                    + (matrix_value(matrix, matrix_start, height, 0, 2 * y + 1)
                        - matrix_value(matrix, matrix_start, height, 0, 2 * y))
                        * x;
                let q_j0 = matrix_value(matrix, matrix_start, height, 1, 2 * y)
                    + (matrix_value(matrix, matrix_start, height, 1, 2 * y + 1)
                        - matrix_value(matrix, matrix_start, height, 1, 2 * y))
                        * x;
                let p_j1 = matrix_value(matrix, matrix_start, height, 2, 2 * y)
                    + (matrix_value(matrix, matrix_start, height, 2, 2 * y + 1)
                        - matrix_value(matrix, matrix_start, height, 2, 2 * y))
                        * x;
                let q_j1 = matrix_value(matrix, matrix_start, height, 3, 2 * y)
                    + (matrix_value(matrix, matrix_start, height, 3, 2 * y + 1)
                        - matrix_value(matrix, matrix_start, height, 3, 2 * y))
                        * x;
                let p_prev = p_j0 * q_j1 + p_j1 * q_j0;
                let q_prev = q_j0 * q_j1;
                eq * (p_prev + lambda * q_prev)
            })
        })
        .reduce(
            || [EF::ZERO; 3],
            |mut acc, values| {
                for (sum, value) in acc.iter_mut().zip(values) {
                    *sum += value;
                }
                acc
            },
        )
}

fn matrix_value<EF: Copy>(
    matrix: &[Frac<EF>],
    matrix_start: usize,
    height: usize,
    column: usize,
    row: usize,
) -> EF {
    frac_field(matrix, matrix_start, column * height + row)
}

fn fold_frac_matrix_into<EF: Field>(
    values: &mut [Frac<EF>],
    source_start: usize,
    destination_start: usize,
    height: usize,
    r: EF,
) {
    let output_len = height;
    let (source, destination) = if source_start < destination_start {
        let (lower, upper) = values.split_at_mut(destination_start);
        (
            &lower[source_start..source_start + 2 * height],
            &mut upper[..output_len],
        )
    } else {
        let (lower, upper) = values.split_at_mut(source_start);
        (
            &upper[..2 * height],
            &mut lower[destination_start..destination_start + output_len],
        )
    };
    let next_height = height / 2;
    destination
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, output)| {
            let flat_start = 2 * idx;
            let mut folded = [EF::ZERO; 2];
            for (part, value) in folded.iter_mut().enumerate() {
                let flat_idx = flat_start + part;
                let column = flat_idx / next_height;
                let row = flat_idx % next_height;
                let left = frac_field(source, 0, column * height + 2 * row);
                let right = frac_field(source, 0, column * height + 2 * row + 1);
                *value = left + (right - left) * r;
            }
            output.p = folded[0];
            output.q = folded[1];
        });
}

fn fold_eq_evals_in_place<EF: Field>(values: &mut Vec<EF>, r: EF) {
    let next_height = values.len() / 2;
    for idx in 0..next_height {
        let left = values[2 * idx];
        let right = values[2 * idx + 1];
        values[idx] = left + (right - left) * r;
    }
    values.truncate(next_height);
}

fn set_frac_field<EF>(values: &mut [Frac<EF>], start: usize, index: usize, value: EF) {
    let target = &mut values[start + index / 2];
    if index % 2 == 0 {
        target.p = value;
    } else {
        target.q = value;
    }
}

fn frac_field<EF: Copy>(values: &[Frac<EF>], start: usize, index: usize) -> EF {
    let source = &values[start + index / 2];
    if index % 2 == 0 {
        source.p
    } else {
        source.q
    }
}

fn fractional_sumcheck_from_segment_tree<
    SC: StarkProtocolConfig,
    TS: FiatShamirTranscript<SC>,
    O: Fn(&'static str),
>(
    transcript: &mut TS,
    mut tree_evals: Vec<Frac<SC::EF>>,
    total_rounds: usize,
    assert_zero: bool,
    on_stage: O,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    // sumcheck polys for layers j=2,...,total_rounds
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);

    for node_idx in (1..(1 << total_rounds)).rev() {
        tree_evals[node_idx] = tree_evals[2 * node_idx] + tree_evals[2 * node_idx + 1];
    }
    on_stage("fractional_segment_tree_built");
    let frac_sum = tree_evals[1];
    if assert_zero {
        if frac_sum.p != SC::EF::ZERO {
            return Err(LogupZerocheckError::NonZeroRootSum);
        }
    } else {
        transcript.observe_ext(frac_sum.p);
    }
    transcript.observe_ext(frac_sum.q);

    // Index i is for layer i+1
    let mut claims_per_layer: Vec<GkrLayerClaims<SC>> = Vec::with_capacity(total_rounds);

    // Process each GKR round
    // `j = round + 1` goes from `1, ..., total_rounds`
    //
    // Round `j = 1` is special since "sumcheck" is directly checked by verifier
    claims_per_layer.push(GkrLayerClaims::<SC> {
        p_xi_0: tree_evals[2].p,
        q_xi_0: tree_evals[2].q,
        p_xi_1: tree_evals[3].p,
        q_xi_1: tree_evals[3].q,
    });
    transcript.observe_ext(claims_per_layer[0].p_xi_0);
    transcript.observe_ext(claims_per_layer[0].q_xi_0);
    transcript.observe_ext(claims_per_layer[0].p_xi_1);
    transcript.observe_ext(claims_per_layer[0].q_xi_1);
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, mu = %mu_1);
    // ξ^{(j-1)}
    let mut xi_prev = vec![mu_1];

    // GKR rounds
    for round in 1..total_rounds {
        // Number of hypercube points
        let eval_size = 1 << round;
        // We apply batch sumcheck to the polynomials
        // \eq(ξ^{(j-1)}, Y) (\hat p_j(0, Y) \hat q_j(1, Y) + \hat p_j(1, Y) \hat q_j(0, Y))
        // \eq(ξ^{(j-1)}, Y) (\hat q_j(0, Y) \hat q_j(1, Y))
        // Note: these are polynomials of degree 3 in each Y_i coordinate.

        // Sample λ_j for batching
        let lambda = transcript.sample_ext();
        debug!(gkr_round = round, %lambda);

        // Columns are p_j0, q_j0, p_j1, q_j1
        // PERF: use a view instead of re-allocating memory
        let mut pq_j_evals = SC::EF::zero_vec(4 * eval_size);
        let segment = &tree_evals[2 * eval_size..4 * eval_size];
        for x in 0..eval_size {
            pq_j_evals[x] = segment[2 * x].p;
            pq_j_evals[eval_size + x] = segment[2 * x].q;
            pq_j_evals[2 * eval_size + x] = segment[2 * x + 1].p;
            pq_j_evals[3 * eval_size + x] = segment[2 * x + 1].q;
        }
        let mut pq_j_evals = ColMajorMatrix::new(pq_j_evals, 4);
        let mut eq_xis = ColMajorMatrix::new(evals_eq_hypercube(&xi_prev), 1);

        // Batch sumcheck where the round polynomials are evaluated at {1, 2, 3}
        let (round_polys_eval, rho) = {
            let n = round;
            let mut round_polys_eval = Vec::with_capacity(n);
            let mut r_vec = Vec::with_capacity(n);

            // Sumcheck rounds: apply fraction addition in projective coordinates to MLEs
            for sumcheck_round in 0..n {
                // Evaluate the univariate polynomial at {1, 2, 3}
                // :projective fraction addition is degree 2, and then another +1 for eq
                let [s_evals] = sumcheck_round_poly_evals(
                    n - sumcheck_round,
                    3,
                    &[eq_xis.as_view(), pq_j_evals.as_view()],
                    |_x, _y, row| {
                        let eq_xi = row[0][0];
                        let &[p_j0, q_j0, p_j1, q_j1] = row[1].as_slice() else {
                            unreachable!("pq_j_evals always has 4 columns")
                        };
                        let p_prev = p_j0 * q_j1 + p_j1 * q_j0;
                        let q_prev = q_j0 * q_j1;
                        // batch using lambda
                        [eq_xi * (p_prev + lambda * q_prev)]
                    },
                );
                let s_evals: [SC::EF; 3] = s_evals.try_into().unwrap();
                for &eval in &s_evals {
                    transcript.observe_ext(eval);
                }
                round_polys_eval.push(s_evals);

                let r_round = transcript.sample_ext();
                pq_j_evals = fold_mle_evals(pq_j_evals, r_round);
                eq_xis = fold_mle_evals(eq_xis, r_round);
                r_vec.push(r_round);
                debug!(gkr_round = round, %sumcheck_round, %r_round);
            }
            (round_polys_eval, r_vec)
        };
        claims_per_layer.push(GkrLayerClaims::<SC> {
            p_xi_0: pq_j_evals.column(0)[0],
            q_xi_0: pq_j_evals.column(1)[0],
            p_xi_1: pq_j_evals.column(2)[0],
            q_xi_1: pq_j_evals.column(3)[0],
        });
        transcript.observe_ext(claims_per_layer[round].p_xi_0);
        transcript.observe_ext(claims_per_layer[round].q_xi_0);
        transcript.observe_ext(claims_per_layer[round].p_xi_1);
        transcript.observe_ext(claims_per_layer[round].q_xi_1);

        // Sample μ_j for reduction to single evaluation point
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);

        // Update ξ^{(j)} = (μ_j, ρ^{(j-1)})
        xi_prev = [vec![mu], rho].concat();

        sumcheck_polys.push(round_polys_eval);
    }

    Ok((
        FracSumcheckProof {
            fractional_sum: (frac_sum.p, frac_sum.q),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}
