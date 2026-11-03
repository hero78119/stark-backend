use std::io::{Error, Read, Result, Write};

use derivative::Derivative;
use p3_field::PrimeCharacteristicRing;
#[cfg(feature = "hiding-whir")]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[cfg(feature = "hiding-whir")]
use crate::hiding_whir::{
    codec::{decode_bounded, DecodeLimits},
    HidingCommitment, HidingProof,
};
use crate::{
    codec::{vec_with_capped_capacity, DecodableConfig, Decode, EncodableConfig, Encode},
    StarkProtocolConfig,
};

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct Proof<SC: StarkProtocolConfig> {
    /// The commitment to the data in common_main.
    pub common_main_commit: SC::Digest,

    /// For each AIR in vkey order, the corresponding trace shape, or None if
    /// the trace is empty. In a valid proof, if `vk.per_air[i].is_required`,
    /// then `trace_vdata[i]` must be `Some(_)`.
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,

    /// For each AIR in vkey order, the public values. Public values should be empty if the AIR has
    /// an empty trace.
    pub public_values: Vec<Vec<SC::F>>,

    pub gkr_proof: GkrProof<SC>,
    pub batch_constraint_proof: BatchConstraintProof<SC>,
    pub stacking_proof: StackingProof<SC>,
    pub whir_proof: WhirProof<SC>,
}

/// V9 proof payload for the pinned hiding-WHIR PCS candidate.
#[cfg(feature = "hiding-whir")]
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofV9<SC: StarkProtocolConfig> {
    pub common_main_commits: Vec<HidingCommitment>,
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,
    pub public_values: Vec<Vec<SC::F>>,
    pub gkr_proof: GkrProof<SC>,
    pub batch_constraint_proof: BatchConstraintProof<SC>,
    pub stacking_proof: StackingProof<SC>,
    pub mu_pow_witness: SC::F,
    pub hiding_whir_proofs: Vec<HidingProof>,
}

#[cfg(feature = "hiding-whir")]
impl<SC: StarkProtocolConfig> ProofV9<SC> {
    pub fn outer_ref(&self) -> OuterProofRef<'_, SC> {
        OuterProofRef {
            trace_vdata: &self.trace_vdata,
            public_values: &self.public_values,
            gkr_proof: &self.gkr_proof,
            batch_constraint_proof: &self.batch_constraint_proof,
            stacking_proof: &self.stacking_proof,
        }
    }
}

/// The portions of a proof checked by the shared SWIRL/LogUp/stacking verifier.
pub struct OuterProofRef<'a, SC: StarkProtocolConfig> {
    pub trace_vdata: &'a [Option<TraceVData<SC>>],
    pub public_values: &'a [Vec<SC::F>],
    pub gkr_proof: &'a GkrProof<SC>,
    pub batch_constraint_proof: &'a BatchConstraintProof<SC>,
    pub stacking_proof: &'a StackingProof<SC>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Default(bound = "")
)]
#[serde(bound = "")]
pub struct TraceVData<SC: StarkProtocolConfig> {
    /// The base 2 logarithm of the trace height. This should be a nonnegative integer and is
    /// allowed to be `< l_skip`.
    ///
    /// If the corresponding AIR has a preprocessed trace, this must match the
    /// value in the vkey.
    pub log_height: usize,
    /// The cached commitments used.
    ///
    /// The length must match the value in the vkey.
    pub cached_commitments: Vec<SC::Digest>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct GkrProof<SC: StarkProtocolConfig> {
    /// The grinding proof-of-work witness before sampling LogUp alpha,beta
    pub logup_pow_witness: SC::F,
    /// The denominator of the root layer.
    ///
    /// Note that the numerator claim is always zero, so we don't include it in
    /// the proof. Despite that the numerator is zero, the representation of the
    /// denominator is important for the verification procedure and thus must be
    /// provided.
    pub q0_claim: SC::EF,
    /// The claims for p_j(xi, 0), p_j(xi, 1), q_j(xi, 0), and q_j(xi, 0) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims<SC>>,
    /// The sumcheck polynomials for each layer, for each sumcheck round, given by their
    /// evaluations on {1, 2, 3}.
    pub sumcheck_polys: Vec<Vec<[SC::EF; 3]>>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct GkrLayerClaims<SC: StarkProtocolConfig> {
    pub p_xi_0: SC::EF,
    pub p_xi_1: SC::EF,
    pub q_xi_0: SC::EF,
    pub q_xi_1: SC::EF,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct BatchConstraintProof<SC: StarkProtocolConfig> {
    /// The terms \textnormal{sum}_{\hat{p}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub numerator_term_per_air: Vec<SC::EF>,
    /// The terms \textnormal{sum}_{\hat{q}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub denominator_term_per_air: Vec<SC::EF>,

    /// Polynomial for initial round, given by `(vk.d + 1) * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<SC::EF>,
    /// For rounds `1, ..., n_max`; evaluations on `{1, ..., vk.d + 1}`.
    pub sumcheck_round_polys: Vec<Vec<SC::EF>>,

    /// Per AIR **in sorted AIR order**, per AIR part, per column index in that part, openings for
    /// the prismalinear column polynomial and (optionally) its rotational convolution. All column
    /// openings are stored in a flat way, so only column openings or them interleaved with
    /// rotations.
    /// For example, if the rotated claims are included for a trace part, then the corresponding
    /// list of openings will look like [col_1, rot_1, col_2, rot_2, ...], and should be treated
    /// as "the i-th column's plain and rotated claims are (col_i, rot_i)".
    /// Otherwise, it will look like [col_1, col_2, col_3, ...], and should be treated as "the
    /// i-th column's plain and rotated claims are (col_i, 0)".
    /// The trace parts are ordered: [CommonMain (part 0), Preprocessed (if any), Cached(0),
    /// Cached(1), ...]
    pub column_openings: Vec<Vec<Vec<SC::EF>>>,
}

pub fn column_openings_by_rot<'a, EF: PrimeCharacteristicRing + Copy + 'a>(
    openings: &'a [EF],
    need_rot: bool,
) -> Box<dyn Iterator<Item = (EF, EF)> + 'a> {
    if need_rot {
        Box::new(openings.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])))
    } else {
        Box::new(openings.iter().map(|&claim| (claim, EF::ZERO)))
    }
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct StackingProof<SC: StarkProtocolConfig> {
    /// Polynomial for round 0, given by `2 * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<SC::EF>,
    /// Rounds 1, ..., n_stack; evaluations at {1, 2}.
    pub sumcheck_round_polys: Vec<[SC::EF; 2]>,
    /// Per commit, per column.
    pub stacking_openings: Vec<Vec<SC::EF>>,
}

pub type MerkleProof<Digest> = Vec<Digest>;

/// WHIR polynomial opening proof for multiple polynomials of the same height, committed to in
/// multiple commitments.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct WhirProof<SC: StarkProtocolConfig> {
    /// Proof-of-work witness for μ batching challenge.
    pub mu_pow_witness: SC::F,
    /// Per sumcheck round; evaluations on {1, 2}. This list is "flattened" with respect to the
    /// WHIR rounds.
    pub whir_sumcheck_polys: Vec<[SC::EF; 2]>,
    /// The codeword commits after each fold, except the final round.
    pub codeword_commits: Vec<SC::Digest>,
    /// The out-of-domain values "y0" per round, except the final round.
    pub ood_values: Vec<SC::EF>,
    /// For each sumcheck round, the folding PoW witness. Length is `num_whir_sumcheck_rounds =
    /// num_whir_rounds * k_whir`.
    pub folding_pow_witnesses: Vec<SC::F>,
    /// For each WHIR round, the query phase PoW witness. Length is `num_whir_rounds`.
    pub query_phase_pow_witnesses: Vec<SC::F>,
    /// For the initial round: per committed matrix, per in-domain query.
    // num_commits x num_queries x (1 << k) x stacking_width[i]
    pub initial_round_opened_rows: Vec<Vec<Vec<Vec<SC::F>>>>,
    pub initial_round_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>>,
    /// Per non-initial round, per in-domain-query.
    pub codeword_opened_values: Vec<Vec<Vec<SC::EF>>>,
    pub codeword_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>>,
    /// Coefficients of the polynomial after the final round.
    pub final_poly: Vec<SC::EF>,
}

// ==================== Encode implementations ====================

impl<SC: EncodableConfig> Encode for TraceVData<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.log_height.encode(writer)?;
        SC::encode_digest_slice(&self.cached_commitments, writer)
    }
}

impl<SC: EncodableConfig> Encode for GkrLayerClaims<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_extension_field(&self.p_xi_0, writer)?;
        SC::encode_extension_field(&self.p_xi_1, writer)?;
        SC::encode_extension_field(&self.q_xi_0, writer)?;
        SC::encode_extension_field(&self.q_xi_1, writer)?;
        Ok(())
    }
}

/// Codec version should change only when proof system or proof format changes.
/// The codec version **must** change if the verifying key serialization/deserialization format
/// changes. It does _not_ correspond to the main openvm version (which may change more frequently).
pub(crate) const CODEC_VERSION: u32 = 3;

impl<SC: EncodableConfig> Encode for Proof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // We explicitly implement Encode for Proof to add CODEC_VERSION
        CODEC_VERSION.encode(writer)?;
        SC::encode_digest(&self.common_main_commit, writer)?;

        // We encode trace_vdata by encoding the number of AIRs, encoding a bitmap of
        // which AIRs are present, and then encoding each present TraceVData.
        let num_airs: usize = self.trace_vdata.len();
        num_airs.encode(writer)?;
        for chunk in self.trace_vdata.chunks(8) {
            let mut ret = 0u8;
            for (i, vdata) in chunk.iter().enumerate() {
                ret |= (vdata.is_some() as u8) << (i as u8);
            }
            ret.encode(writer)?;
        }
        for vdata in self.trace_vdata.iter().flatten() {
            vdata.encode(writer)?;
        }

        // public_values: Vec<Vec<SC::F>>
        self.public_values.len().encode(writer)?;
        for pv in &self.public_values {
            SC::encode_base_field_slice(pv, writer)?;
        }
        self.gkr_proof.encode(writer)?;
        self.batch_constraint_proof.encode(writer)?;
        self.stacking_proof.encode(writer)?;
        self.whir_proof.encode(writer)
    }
}

#[cfg(feature = "hiding-whir")]
const V9_CODEC_VERSION: u32 = 6;
#[cfg(feature = "hiding-whir")]
const V9_MAX_COMMITMENTS: usize = 4;

#[cfg(feature = "hiding-whir")]
const V9_MAX_WIRE_BYTES: usize = 64 * 1024 * 1024;
#[cfg(feature = "hiding-whir")]
const V9_MAX_OUTER_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;
#[cfg(feature = "hiding-whir")]
const V9_MAX_AIRS: usize = 64;
#[cfg(feature = "hiding-whir")]
const V9_MAX_DIMENSION: usize = 26;
#[cfg(feature = "hiding-whir")]
const V9_MAX_FIELD_VECTOR_LEN: usize = 1 << 16;
#[cfg(feature = "hiding-whir")]
const V9_MAX_SUMCHECK_COEFFICIENTS: usize = 1 << 20;

#[cfg(feature = "hiding-whir")]
impl<SC: EncodableConfig> Encode for ProofV9<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        let mut writer = V9LimitedWriter::new(writer);
        V9_CODEC_VERSION.encode(&mut writer)?;
        if self.common_main_commits.is_empty()
            || self.common_main_commits.len() > V9_MAX_COMMITMENTS
            || self.hiding_whir_proofs.len() != self.common_main_commits.len()
        {
            return Err(Error::other("invalid V9 PCS commitment/proof count"));
        }
        self.common_main_commits.len().encode(&mut writer)?;
        for commitment in &self.common_main_commits {
            encode_postcard_payload(commitment, &mut writer)?;
        }

        let num_airs = self.trace_vdata.len();
        num_airs.encode(&mut writer)?;
        for chunk in self.trace_vdata.chunks(8) {
            let mut bitmap = 0u8;
            for (index, vdata) in chunk.iter().enumerate() {
                bitmap |= (vdata.is_some() as u8) << index;
            }
            bitmap.encode(&mut writer)?;
        }
        for vdata in self.trace_vdata.iter().flatten() {
            vdata.encode(&mut writer)?;
        }

        self.public_values.len().encode(&mut writer)?;
        for values in &self.public_values {
            SC::encode_base_field_slice(values, &mut writer)?;
        }
        self.gkr_proof.encode(&mut writer)?;
        self.batch_constraint_proof.encode(&mut writer)?;
        self.stacking_proof.encode(&mut writer)?;
        SC::encode_base_field(&self.mu_pow_witness, &mut writer)?;
        for proof in &self.hiding_whir_proofs {
            encode_postcard_payload(proof, &mut writer)?;
        }
        Ok(())
    }
}

#[cfg(feature = "hiding-whir")]
struct V9LimitedWriter<'a, W> {
    inner: &'a mut W,
    written: usize,
}

#[cfg(feature = "hiding-whir")]
impl<'a, W> V9LimitedWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self { inner, written: 0 }
    }
}

#[cfg(feature = "hiding-whir")]
impl<W: Write> Write for V9LimitedWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() > V9_MAX_WIRE_BYTES.saturating_sub(self.written) {
            return Err(Error::other("V9 proof exceeds the 64 MiB wire limit"));
        }
        let written = self.inner.write(buf)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(feature = "hiding-whir")]
fn encode_postcard_payload<T: Serialize, W: Write>(value: &T, writer: &mut W) -> Result<()> {
    let bytes = postcard::to_allocvec(value).map_err(Error::other)?;
    if bytes.len() > V9_MAX_WIRE_BYTES {
        return Err(Error::other("V9 PCS payload exceeds the 64 MiB wire limit"));
    }
    bytes.len().encode(writer)?;
    writer.write_all(&bytes)
}

#[cfg(feature = "hiding-whir")]
fn decode_postcard_payload<T: DeserializeOwned, R: Read>(
    reader: &mut R,
    limits: DecodeLimits,
) -> Result<T> {
    let len = usize::decode(reader)?;
    if len > limits.max_wire_bytes {
        return Err(Error::other(format!(
            "HidingWhir payload is {len} bytes; limit is {}",
            limits.max_wire_bytes
        )));
    }
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes)?;
    decode_bounded(&bytes, limits).map_err(Error::other)
}

#[cfg(feature = "hiding-whir")]
struct V9LimitedReader<'a, R> {
    inner: &'a mut R,
    read: usize,
}

#[cfg(feature = "hiding-whir")]
impl<'a, R> V9LimitedReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self { inner, read: 0 }
    }
}

#[cfg(feature = "hiding-whir")]
impl<R: Read> Read for V9LimitedReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let remaining = V9_MAX_WIRE_BYTES.saturating_sub(self.read);
        if remaining == 0 {
            return Err(Error::other("V9 proof exceeds the 64 MiB wire limit"));
        }
        let max_read = buf.len().min(remaining);
        let read = self.inner.read(&mut buf[..max_read])?;
        self.read += read;
        Ok(read)
    }
}

#[cfg(feature = "hiding-whir")]
#[derive(Default)]
struct V9DecodeBudget {
    allocated_bytes: usize,
}

#[cfg(feature = "hiding-whir")]
impl V9DecodeBudget {
    fn vec<T>(&mut self, len: usize, max_len: usize, name: &str) -> Result<Vec<T>> {
        if len > max_len {
            return Err(Error::other(format!(
                "V9 {name} length {len} exceeds limit {max_len}"
            )));
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>().max(1))
            .ok_or_else(|| Error::other(format!("V9 {name} allocation size overflow")))?;
        self.allocated_bytes = self
            .allocated_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::other("V9 allocation budget overflow"))?;
        if self.allocated_bytes > V9_MAX_OUTER_ALLOCATION_BYTES {
            return Err(Error::other(
                "V9 outer proof exceeds the 64 MiB allocation limit",
            ));
        }
        Ok(Vec::with_capacity(len))
    }
}

#[cfg(feature = "hiding-whir")]
fn decode_v9_field_vec<R: Read, T>(
    reader: &mut R,
    budget: &mut V9DecodeBudget,
    max_len: usize,
    name: &str,
    mut decode_field: impl FnMut(&mut R) -> Result<T>,
) -> Result<Vec<T>> {
    let len = usize::decode(reader)?;
    let mut values = budget.vec(len, max_len, name)?;
    for _ in 0..len {
        values.push(decode_field(reader)?);
    }
    Ok(values)
}

#[cfg(feature = "hiding-whir")]
fn decode_v9_trace_metadata<R: Read>(
    reader: &mut R,
    num_airs: usize,
) -> Result<Vec<Option<usize>>> {
    if num_airs > V9_MAX_AIRS {
        return Err(Error::other(format!(
            "V9 AIR count {num_airs} exceeds limit {V9_MAX_AIRS}"
        )));
    }

    let bitmap_len = num_airs.div_ceil(8);
    let mut bitmap = [0u8; V9_MAX_AIRS.div_ceil(8)];
    for byte in bitmap.iter_mut().take(bitmap_len) {
        *byte = u8::decode(reader)?;
    }

    let mut trace_metadata = Vec::with_capacity(num_airs);
    for (byte_idx, &byte) in bitmap[..bitmap_len].iter().enumerate() {
        for bit in 0u8..8 {
            let trace_idx = byte_idx * 8 + bit as usize;
            if trace_idx >= num_airs {
                if byte >> bit != 0 {
                    return Err(Error::other("trace_vdata bitmap padding bits set"));
                }
                break;
            }
            if byte & (1u8 << bit) != 0 {
                let log_height = usize::decode(reader)?;
                if log_height > V9_MAX_DIMENSION {
                    return Err(Error::other("V9 trace height exceeds PCS dimension limit"));
                }
                let num_cached_commitments = usize::decode(reader)?;
                if num_cached_commitments != 0 {
                    return Err(Error::other("V9 does not support cached trace commitments"));
                }
                trace_metadata.push(Some(log_height));
            } else {
                trace_metadata.push(None);
            }
        }
    }

    Ok(trace_metadata)
}

#[cfg(feature = "hiding-whir")]
fn decode_v9_gkr<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    budget: &mut V9DecodeBudget,
) -> Result<GkrProof<SC>> {
    let logup_pow_witness = SC::decode_base_field(reader)?;
    let q0_claim = SC::decode_extension_field(reader)?;

    let num_claims = usize::decode(reader)?;
    let mut claims_per_layer = budget.vec(num_claims, V9_MAX_DIMENSION + 1, "GKR layer claims")?;
    for _ in 0..num_claims {
        claims_per_layer.push(GkrLayerClaims::decode(reader)?);
    }

    let num_rounds = num_claims.saturating_sub(1);
    let mut sumcheck_polys =
        budget.vec::<Vec<[SC::EF; 3]>>(num_rounds, V9_MAX_DIMENSION, "GKR sumcheck rounds")?;
    for round in 0..num_rounds {
        let len = round + 1;
        let mut round_poly = budget.vec(len, V9_MAX_DIMENSION, "GKR round coefficients")?;
        for _ in 0..len {
            round_poly.push([
                SC::decode_extension_field(reader)?,
                SC::decode_extension_field(reader)?,
                SC::decode_extension_field(reader)?,
            ]);
        }
        sumcheck_polys.push(round_poly);
    }

    Ok(GkrProof {
        logup_pow_witness,
        q0_claim,
        claims_per_layer,
        sumcheck_polys,
    })
}

#[cfg(feature = "hiding-whir")]
fn decode_v9_batch_constraints<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    budget: &mut V9DecodeBudget,
) -> Result<BatchConstraintProof<SC>> {
    let numerator_term_per_air = decode_v9_field_vec::<_, _>(
        reader,
        budget,
        V9_MAX_AIRS,
        "batch numerator terms",
        SC::decode_extension_field,
    )?;
    let num_present_airs = numerator_term_per_air.len();
    let mut denominator_term_per_air =
        budget.vec(num_present_airs, V9_MAX_AIRS, "batch denominator terms")?;
    for _ in 0..num_present_airs {
        denominator_term_per_air.push(SC::decode_extension_field(reader)?);
    }

    let univariate_round_coeffs = decode_v9_field_vec::<_, _>(
        reader,
        budget,
        V9_MAX_SUMCHECK_COEFFICIENTS,
        "batch univariate coefficients",
        SC::decode_extension_field,
    )?;

    let num_rounds = usize::decode(reader)?;
    let mut sumcheck_round_polys =
        budget.vec::<Vec<SC::EF>>(num_rounds, V9_MAX_DIMENSION, "batch sumcheck rounds")?;
    if num_rounds > 0 {
        let degree_plus_one = usize::decode(reader)?;
        if degree_plus_one > 64 {
            return Err(Error::other("V9 batch sumcheck degree exceeds limit 64"));
        }
        for _ in 0..num_rounds {
            let mut round = budget.vec(degree_plus_one, 64, "batch sumcheck coefficients")?;
            for _ in 0..degree_plus_one {
                round.push(SC::decode_extension_field(reader)?);
            }
            sumcheck_round_polys.push(round);
        }
    }

    let mut column_openings =
        budget.vec(num_present_airs, V9_MAX_AIRS, "batch AIR column openings")?;
    for _ in 0..num_present_airs {
        let num_parts = usize::decode(reader)?;
        let mut parts = budget.vec::<Vec<SC::EF>>(num_parts, 1, "V9 commitment parts")?;
        for _ in 0..num_parts {
            parts.push(decode_v9_field_vec::<_, _>(
                reader,
                budget,
                V9_MAX_FIELD_VECTOR_LEN,
                "column opening values",
                SC::decode_extension_field,
            )?);
        }
        column_openings.push(parts);
    }

    Ok(BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    })
}

#[cfg(feature = "hiding-whir")]
fn decode_v9_stacking<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    budget: &mut V9DecodeBudget,
) -> Result<StackingProof<SC>> {
    let univariate_round_coeffs = decode_v9_field_vec::<_, _>(
        reader,
        budget,
        V9_MAX_SUMCHECK_COEFFICIENTS,
        "stacking univariate coefficients",
        SC::decode_extension_field,
    )?;

    let num_rounds = usize::decode(reader)?;
    let mut sumcheck_round_polys =
        budget.vec(num_rounds, V9_MAX_DIMENSION, "stacking sumcheck rounds")?;
    for _ in 0..num_rounds {
        sumcheck_round_polys.push([
            SC::decode_extension_field(reader)?,
            SC::decode_extension_field(reader)?,
        ]);
    }

    let num_openings = usize::decode(reader)?;
    let mut stacking_openings = budget.vec(num_openings, 1, "stacked commitment openings")?;
    for _ in 0..num_openings {
        stacking_openings.push(decode_v9_field_vec::<_, _>(
            reader,
            budget,
            V9_MAX_FIELD_VECTOR_LEN,
            "stacked opening values",
            SC::decode_extension_field,
        )?);
    }

    Ok(StackingProof {
        univariate_round_coeffs,
        sumcheck_round_polys,
        stacking_openings,
    })
}

impl<SC: EncodableConfig> Encode for GkrProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_base_field(&self.logup_pow_witness, writer)?;
        SC::encode_extension_field(&self.q0_claim, writer)?;
        self.claims_per_layer.encode(writer)?;
        // We should know the length of sumcheck_polys and each nested vector based
        // on the length of claims_per_layer.
        for round in &self.sumcheck_polys {
            for arr in round {
                SC::encode_extension_field_iter(arr.iter(), writer)?;
            }
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for BatchConstraintProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Length of numerator_term_per_air is number of present AIRs
        SC::encode_extension_field_slice(&self.numerator_term_per_air, writer)?;
        SC::encode_extension_field_iter(self.denominator_term_per_air.iter(), writer)?;

        SC::encode_extension_field_slice(&self.univariate_round_coeffs, writer)?;

        // Each nested vector should be the same length
        let n_max = self.sumcheck_round_polys.len();
        n_max.encode(writer)?;
        if n_max > 0 {
            self.sumcheck_round_polys[0].len().encode(writer)?;
            for round_polys in &self.sumcheck_round_polys {
                SC::encode_extension_field_iter(round_polys.iter(), writer)?;
            }
        }

        // There is one outer vector per present AIR
        // column_openings: Vec<Vec<Vec<SC::EF>>>
        for part_col_openings in &self.column_openings {
            // part_col_openings: Vec<Vec<SC::EF>>
            part_col_openings.len().encode(writer)?;
            for col_opening in part_col_openings {
                SC::encode_extension_field_slice(col_opening, writer)?;
            }
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for StackingProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_extension_field_slice(&self.univariate_round_coeffs, writer)?;
        // sumcheck_round_polys: Vec<[SC::EF; 2]>
        self.sumcheck_round_polys.len().encode(writer)?;
        for arr in &self.sumcheck_round_polys {
            SC::encode_extension_field_iter(arr.iter(), writer)?;
        }
        // stacking_openings: Vec<Vec<SC::EF>>
        self.stacking_openings.len().encode(writer)?;
        for opening in &self.stacking_openings {
            SC::encode_extension_field_slice(opening, writer)?;
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for WhirProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_base_field(&self.mu_pow_witness, writer)?;
        // whir_sumcheck_polys: Vec<[SC::EF; 2]>
        self.whir_sumcheck_polys.len().encode(writer)?;
        for arr in &self.whir_sumcheck_polys {
            SC::encode_extension_field_iter(arr.iter(), writer)?;
        }
        let num_whir_sumcheck_rounds = self.whir_sumcheck_polys.len();

        // Each length can be derived from num_whir_rounds
        SC::encode_digest_slice(&self.codeword_commits, writer)?;
        SC::encode_extension_field_iter(self.ood_values.iter(), writer)?;
        let num_whir_rounds = self.codeword_commits.len() + 1;
        if !num_whir_sumcheck_rounds.is_multiple_of(num_whir_rounds) {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "num_whir_sumcheck_rounds must be a multiple of num_whir_rounds",
            ));
        }
        assert_eq!(num_whir_rounds, self.query_phase_pow_witnesses.len());
        SC::encode_base_field_iter(self.folding_pow_witnesses.iter(), writer)?;
        SC::encode_base_field_iter(self.query_phase_pow_witnesses.iter(), writer)?;

        let num_commits = self.initial_round_opened_rows.len();
        assert!(num_commits > 0);
        num_commits.encode(writer)?;
        let initial_num_whir_queries = self.initial_round_opened_rows[0].len();
        initial_num_whir_queries.encode(writer)?;

        if initial_num_whir_queries > 0 {
            let merkle_depth = self.initial_round_merkle_proofs[0][0].len();
            merkle_depth.encode(writer)?;

            // We avoid per-row Vec length prefixes by encoding each commit's stacked width,
            // which we can use to determine the shapes of the remaining WHIR proof fields.
            let widths: Vec<usize> = self
                .initial_round_opened_rows
                .iter()
                .map(|commit_rows| {
                    // If there are any queries/rows, infer width from the first row.
                    commit_rows
                        .first()
                        .and_then(|q| q.first())
                        .map(|row| row.len())
                        .unwrap_or(0)
                })
                .collect();

            // Encode widths (length is implicit via num_commits).
            for w in &widths {
                w.encode(writer)?;
            }

            // Encode all opened row values (no per-row length prefixes).
            for (commit_rows, &width) in self.initial_round_opened_rows.iter().zip(&widths) {
                debug_assert_eq!(commit_rows.len(), initial_num_whir_queries);
                for query_rows in commit_rows {
                    for row in query_rows {
                        debug_assert_eq!(row.len(), width);
                        SC::encode_base_field_iter(row.iter(), writer)?;
                    }
                }
            }

            for merkle_proofs in &self.initial_round_merkle_proofs {
                for proof in merkle_proofs {
                    SC::encode_digest_iter(proof.iter(), writer)?;
                }
            }
        }

        // Length of outer vector is num_whir_rounds
        for non_init_round in &self.codeword_opened_values {
            let num_queries = non_init_round.len();
            num_queries.encode(writer)?;
            // Length of nested vector is num_whir_queries, then k_whir_exp.
            for query_vals in non_init_round {
                SC::encode_extension_field_iter(query_vals.iter(), writer)?;
            }
        }

        // Length of outer vector is num_whir_rounds, then num_whir_queries. Each
        // inner vector length is one less than the one that precedes it.
        let mut first_merkle_depth = 0;
        if num_whir_rounds > 1 && initial_num_whir_queries > 0 {
            first_merkle_depth = self.codeword_merkle_proofs[0][0].len();
        }
        first_merkle_depth.encode(writer)?;
        for round_proofs in &self.codeword_merkle_proofs {
            for proof in round_proofs {
                SC::encode_digest_iter(proof.iter(), writer)?;
            }
        }

        SC::encode_extension_field_slice(&self.final_poly, writer)
    }
}

// ==================== Decode implementations ====================

impl<SC: DecodableConfig> Decode for TraceVData<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            log_height: usize::decode(reader)?,
            cached_commitments: SC::decode_digest_vec(reader)?,
        })
    }
}

impl<SC: DecodableConfig> Decode for GkrLayerClaims<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            p_xi_0: SC::decode_extension_field(reader)?,
            p_xi_1: SC::decode_extension_field(reader)?,
            q_xi_0: SC::decode_extension_field(reader)?,
            q_xi_1: SC::decode_extension_field(reader)?,
        })
    }
}

impl<SC: DecodableConfig> Decode for Proof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        // We explicitly implement Decode for Proof to check CODEC_VERSION
        let codec_version = u32::decode(reader)?;
        if codec_version != CODEC_VERSION {
            return Err(Error::other(format!(
                "CODEC_VERSION mismatch, expected: {}, actual: {}",
                CODEC_VERSION, codec_version
            )));
        }
        let common_main_commit = SC::decode_digest(reader)?;

        let num_airs = usize::decode(reader)?;
        let bitmap_len = num_airs.div_ceil(8);
        let mut bitmap: Vec<u8> = vec_with_capped_capacity(bitmap_len);
        for _ in 0..bitmap_len {
            bitmap.push(u8::decode(reader)?);
        }
        let mut trace_vdata = vec_with_capped_capacity(num_airs);
        for byte in bitmap {
            for i in 0u8..8 {
                if trace_vdata.len() >= num_airs {
                    // Padding bits must be zero for the encoding to be canonical.
                    if byte >> i != 0 {
                        return Err(Error::other("trace_vdata bitmap padding bits set"));
                    }
                    break;
                }
                if byte & (1u8 << i) != 0 {
                    trace_vdata.push(Some(TraceVData::decode(reader)?));
                } else {
                    trace_vdata.push(None);
                }
            }
        }

        // public_values: Vec<Vec<SC::F>>
        let num_pvs = usize::decode(reader)?;
        let mut public_values = vec_with_capped_capacity(num_pvs);
        for _ in 0..num_pvs {
            public_values.push(SC::decode_base_field_vec(reader)?);
        }

        Ok(Self {
            common_main_commit,
            trace_vdata,
            public_values,
            gkr_proof: GkrProof::decode(reader)?,
            batch_constraint_proof: BatchConstraintProof::decode(reader)?,
            stacking_proof: StackingProof::decode(reader)?,
            whir_proof: WhirProof::decode(reader)?,
        })
    }
}

#[cfg(feature = "hiding-whir")]
impl<SC: DecodableConfig> Decode for ProofV9<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mut reader = V9LimitedReader::new(reader);
        let codec_version = u32::decode(&mut reader)?;
        if codec_version != V9_CODEC_VERSION {
            return Err(Error::other(format!(
                "V9 CODEC_VERSION mismatch, expected: {}, actual: {}",
                V9_CODEC_VERSION, codec_version
            )));
        }

        let num_commitments = usize::decode(&mut reader)?;
        if !(1..=V9_MAX_COMMITMENTS).contains(&num_commitments) {
            return Err(Error::other("invalid V9 PCS commitment/proof count"));
        }
        let mut commitment_limits = DecodeLimits::for_commitment(V9_MAX_DIMENSION, 1024);
        commitment_limits.max_sequence_len = 64;
        commitment_limits.max_total_elements = 64;
        commitment_limits.max_total_allocation_bytes = 4096;
        let mut common_main_commits = Vec::with_capacity(num_commitments);
        for _ in 0..num_commitments {
            common_main_commits.push(decode_postcard_payload(&mut reader, commitment_limits)?);
        }

        let mut budget = V9DecodeBudget::default();
        let num_airs = usize::decode(&mut reader)?;
        let mut trace_vdata =
            budget.vec::<Option<TraceVData<SC>>>(num_airs, V9_MAX_AIRS, "AIR trace metadata")?;
        for log_height in decode_v9_trace_metadata(&mut reader, num_airs)? {
            trace_vdata.push(log_height.map(|log_height| TraceVData {
                log_height,
                cached_commitments: Vec::new(),
            }));
        }

        let num_pvs = usize::decode(&mut reader)?;
        if num_pvs != num_airs {
            return Err(Error::other(
                "V9 public value count does not match AIR trace count",
            ));
        }
        let mut public_values =
            budget.vec::<Vec<SC::F>>(num_pvs, V9_MAX_AIRS, "AIR public values")?;
        for _ in 0..num_pvs {
            public_values.push(decode_v9_field_vec::<_, _>(
                &mut reader,
                &mut budget,
                V9_MAX_FIELD_VECTOR_LEN,
                "AIR public input values",
                SC::decode_base_field,
            )?);
        }

        let gkr_proof = decode_v9_gkr::<SC, _>(&mut reader, &mut budget)?;
        let batch_constraint_proof =
            decode_v9_batch_constraints::<SC, _>(&mut reader, &mut budget)?;
        let stacking_proof = decode_v9_stacking::<SC, _>(&mut reader, &mut budget)?;
        let mu_pow_witness = SC::decode_base_field(&mut reader)?;
        let mut proof_limits = DecodeLimits::for_dimension(V9_MAX_DIMENSION, V9_MAX_WIRE_BYTES);
        proof_limits.max_total_allocation_bytes = V9_MAX_WIRE_BYTES;
        proof_limits.max_total_allocation_bytes = V9_MAX_WIRE_BYTES / num_commitments;
        let mut hiding_whir_proofs = Vec::with_capacity(num_commitments);
        for _ in 0..num_commitments {
            hiding_whir_proofs.push(decode_postcard_payload(&mut reader, proof_limits)?);
        }

        Ok(Self {
            common_main_commits,
            trace_vdata,
            public_values,
            gkr_proof,
            batch_constraint_proof,
            stacking_proof,
            mu_pow_witness,
            hiding_whir_proofs,
        })
    }
}

#[cfg(all(test, feature = "hiding-whir"))]
mod hiding_whir_codec_tests {
    use std::io::Cursor;

    use super::{
        decode_postcard_payload, decode_v9_trace_metadata, DecodeLimits, V9_MAX_WIRE_BYTES,
    };
    use crate::{codec::Encode, hiding_whir::HidingProof};

    #[test]
    fn v9_pcs_payload_length_is_rejected_before_allocation() {
        let mut wire = ((V9_MAX_WIRE_BYTES + 1) as u32).to_le_bytes().to_vec();
        let mut reader = Cursor::new(&mut wire);
        let limits = DecodeLimits::for_dimension(26, V9_MAX_WIRE_BYTES);
        assert!(decode_postcard_payload::<HidingProof, _>(&mut reader, limits).is_err());
        assert_eq!(reader.position(), 4);
    }

    #[test]
    fn v9_codec_reads_all_air_bitmap_bytes_before_trace_metadata() {
        let mut wire = Vec::new();
        9usize.encode(&mut wire).unwrap();
        0xffu8.encode(&mut wire).unwrap();
        0x01u8.encode(&mut wire).unwrap();
        for log_height in 1..=9 {
            (log_height as usize).encode(&mut wire).unwrap();
            0usize.encode(&mut wire).unwrap();
        }

        let mut reader = Cursor::new(&wire[4..]);
        let metadata = decode_v9_trace_metadata(&mut reader, 9).unwrap();
        assert_eq!(reader.position(), (wire.len() - 4) as u64);
        assert_eq!(metadata, (1..=9).map(Some).collect::<Vec<_>>());
    }
}

impl<SC: DecodableConfig> Decode for GkrProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let logup_pow_witness = SC::decode_base_field(reader)?;
        let q0_claim = SC::decode_extension_field(reader)?;
        let claims_per_layer = Vec::<GkrLayerClaims<SC>>::decode(reader)?;

        let num_sumcheck_polys = claims_per_layer.len().saturating_sub(1);
        let mut sumcheck_polys = vec_with_capped_capacity(num_sumcheck_polys);
        for round_idx_minus_one in 0..num_sumcheck_polys {
            let n = round_idx_minus_one + 1;
            let mut round = vec_with_capped_capacity(n);
            for _ in 0..n {
                round.push([
                    SC::decode_extension_field(reader)?,
                    SC::decode_extension_field(reader)?,
                    SC::decode_extension_field(reader)?,
                ]);
            }
            sumcheck_polys.push(round);
        }

        Ok(Self {
            logup_pow_witness,
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
        })
    }
}

impl<SC: DecodableConfig> Decode for BatchConstraintProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let numerator_term_per_air = SC::decode_extension_field_vec(reader)?;
        let num_present_airs = numerator_term_per_air.len();
        let denominator_term_per_air = SC::decode_extension_field_n(reader, num_present_airs)?;

        let univariate_round_coeffs = SC::decode_extension_field_vec(reader)?;

        let n_max = usize::decode(reader)?;
        let mut sumcheck_round_polys = vec_with_capped_capacity(n_max);
        if n_max > 0 {
            let max_degree_plus_one = usize::decode(reader)?;
            for _ in 0..n_max {
                sumcheck_round_polys
                    .push(SC::decode_extension_field_n(reader, max_degree_plus_one)?);
            }
        }

        let mut column_openings = vec_with_capped_capacity(num_present_airs);
        for _ in 0..num_present_airs {
            // Vec<Vec<SC::EF>>: length-prefixed outer, then each inner
            let num_parts = usize::decode(reader)?;
            let mut parts = vec_with_capped_capacity(num_parts);
            for _ in 0..num_parts {
                parts.push(SC::decode_extension_field_vec(reader)?);
            }
            column_openings.push(parts);
        }

        Ok(Self {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        })
    }
}

impl<SC: DecodableConfig> Decode for StackingProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let univariate_round_coeffs = SC::decode_extension_field_vec(reader)?;
        // sumcheck_round_polys: Vec<[SC::EF; 2]>
        let num_rounds = usize::decode(reader)?;
        let mut sumcheck_round_polys = vec_with_capped_capacity(num_rounds);
        for _ in 0..num_rounds {
            sumcheck_round_polys.push([
                SC::decode_extension_field(reader)?,
                SC::decode_extension_field(reader)?,
            ]);
        }
        // stacking_openings: Vec<Vec<SC::EF>>
        let num_openings = usize::decode(reader)?;
        let mut stacking_openings = vec_with_capped_capacity(num_openings);
        for _ in 0..num_openings {
            stacking_openings.push(SC::decode_extension_field_vec(reader)?);
        }
        Ok(Self {
            univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings,
        })
    }
}

impl<SC: DecodableConfig> Decode for WhirProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let mu_pow_witness = SC::decode_base_field(reader)?;
        // whir_sumcheck_polys: Vec<[SC::EF; 2]>
        let num_whir_sumcheck_rounds = usize::decode(reader)?;
        let mut whir_sumcheck_polys = vec_with_capped_capacity(num_whir_sumcheck_rounds);
        for _ in 0..num_whir_sumcheck_rounds {
            whir_sumcheck_polys.push([
                SC::decode_extension_field(reader)?,
                SC::decode_extension_field(reader)?,
            ]);
        }

        let codeword_commits = SC::decode_digest_vec(reader)?;
        let num_whir_rounds = codeword_commits.len() + 1;
        if num_whir_sumcheck_rounds % num_whir_rounds != 0 {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "num_whir_sumcheck_rounds must be a multiple of num_whir_rounds",
            ));
        }
        let k_whir = num_whir_sumcheck_rounds / num_whir_rounds;
        let ood_values = SC::decode_extension_field_n(reader, num_whir_rounds - 1)?;
        let folding_pow_witnesses = SC::decode_base_field_n(reader, num_whir_sumcheck_rounds)?;
        let query_phase_pow_witnesses = SC::decode_base_field_n(reader, num_whir_rounds)?;

        let num_commits = usize::decode(reader)?;
        if num_commits == 0 {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "num_commits must be nonzero",
            ));
        }
        let initial_num_whir_queries = usize::decode(reader)?;
        let k_whir_exp = 1 << k_whir;
        let mut merkle_depth = 0;
        if initial_num_whir_queries > 0 {
            merkle_depth = usize::decode(reader)?;
        }

        let mut widths = vec_with_capped_capacity(num_commits);
        if initial_num_whir_queries > 0 {
            for _ in 0..num_commits {
                widths.push(usize::decode(reader)?);
            }
        }

        let decoded_widths = widths.len();
        let mut initial_round_opened_rows = vec_with_capped_capacity(num_commits);
        for width in widths.into_iter().chain(std::iter::repeat_n(
            0,
            num_commits.saturating_sub(decoded_widths),
        )) {
            let mut opened_rows = vec_with_capped_capacity(initial_num_whir_queries);
            for _ in 0..initial_num_whir_queries {
                // Each query has k_whir_exp rows. Each row is a fixed-width list of F elements.
                let mut rows = vec_with_capped_capacity(k_whir_exp);
                for _ in 0..k_whir_exp {
                    rows.push(SC::decode_base_field_n(reader, width)?);
                }
                opened_rows.push(rows);
            }
            initial_round_opened_rows.push(opened_rows);
        }

        let mut initial_round_merkle_proofs = vec_with_capped_capacity(num_commits);
        for _ in 0..num_commits {
            let mut merkle_proofs = vec_with_capped_capacity(initial_num_whir_queries);
            for _ in 0..initial_num_whir_queries {
                merkle_proofs.push(SC::decode_digest_n(reader, merkle_depth)?);
            }
            initial_round_merkle_proofs.push(merkle_proofs);
        }

        let mut codeword_opened_values = vec_with_capped_capacity(num_whir_rounds - 1);
        for _ in 0..num_whir_rounds - 1 {
            let num_queries = usize::decode(reader)?;
            let mut opened_values = vec_with_capped_capacity(num_queries);
            for _ in 0..num_queries {
                opened_values.push(SC::decode_extension_field_n(reader, k_whir_exp)?);
            }
            codeword_opened_values.push(opened_values);
        }

        merkle_depth = usize::decode(reader)?;
        let mut codeword_merkle_proofs = vec_with_capped_capacity(num_whir_rounds - 1);
        for opened_values in codeword_opened_values.iter() {
            let num_queries = opened_values.len();
            let mut merkle_proof: Vec<_> = vec_with_capped_capacity(num_queries);
            for _ in 0..num_queries {
                merkle_proof.push(SC::decode_digest_n(reader, merkle_depth)?);
            }
            codeword_merkle_proofs.push(merkle_proof);
            merkle_depth -= 1;
        }

        let final_poly = SC::decode_extension_field_vec(reader)?;

        Ok(Self {
            mu_pow_witness,
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            folding_pow_witnesses,
            query_phase_pow_witnesses,
            initial_round_opened_rows,
            initial_round_merkle_proofs,
            codeword_opened_values,
            codeword_merkle_proofs,
            final_poly,
        })
    }
}
