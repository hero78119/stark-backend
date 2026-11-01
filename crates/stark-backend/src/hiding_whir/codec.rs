//! Size- and allocation-bounded postcard helpers for concrete V9 PCS payloads.

use core::{fmt, mem::size_of};

use serde::{
    de::{
        self, DeserializeOwned, DeserializeSeed, EnumAccess, Error as _, MapAccess, SeqAccess,
        VariantAccess, Visitor,
    },
    Deserializer, Serialize,
};
use thiserror::Error;

use super::{HidingBase, HidingExtension, HidingMerkleMmcs};

const MAX_DECODED_ALLOCATION: usize = 256 * 1024 * 1024;
const SEQUENCE_FLOOR: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub max_wire_bytes: usize,
    pub max_sequence_len: usize,
    pub max_total_elements: usize,
    pub max_total_allocation_bytes: usize,
    max_element_size: usize,
}

impl DecodeLimits {
    /// Derives conservative per-container limits from the VK-bound PCS dimension.
    pub fn for_dimension(num_variables: usize, max_wire_bytes: usize) -> Self {
        let table_len = 1usize
            .checked_shl(num_variables as u32)
            .unwrap_or(usize::MAX);
        let max_sequence_len = table_len.max(SEQUENCE_FLOOR);
        let max_total_elements = max_sequence_len.saturating_mul(64);
        let max_element_size = proof_max_element_size();
        Self {
            max_wire_bytes,
            max_sequence_len,
            max_total_elements,
            max_total_allocation_bytes: max_wire_bytes
                .saturating_mul(8)
                .min(MAX_DECODED_ALLOCATION),
            max_element_size,
        }
    }

    pub fn for_commitment(num_variables: usize, max_wire_bytes: usize) -> Self {
        let mut limits = Self::for_dimension(num_variables, max_wire_bytes);
        limits.max_element_size = size_of::<[HidingBase; 8]>();
        limits
    }
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("hiding-WHIR payload is {actual} bytes; limit is {limit}")]
    TooLarge { actual: usize, limit: usize },
    #[error("hiding-WHIR decode allocation budget exceeded: requested {requested} bytes; remaining {remaining}")]
    AllocationLimit { requested: usize, remaining: usize },
    #[error("invalid hiding-WHIR payload: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("trailing bytes after hiding-WHIR payload")]
    TrailingBytes,
}

/// Serializes into caller-owned bounded storage; no output allocation is performed here.
pub fn encode_into<'a, T: Serialize>(
    value: &T,
    storage: &'a mut [u8],
) -> Result<&'a [u8], postcard::Error> {
    postcard::to_slice(value, storage).map(|used| &*used)
}

/// Checks every postcard sequence length before Serde can reserve its vector.
pub fn decode_bounded<T: DeserializeOwned>(
    bytes: &[u8],
    limits: DecodeLimits,
) -> Result<T, CodecError> {
    if bytes.len() > limits.max_wire_bytes {
        return Err(CodecError::TooLarge {
            actual: bytes.len(),
            limit: limits.max_wire_bytes,
        });
    }

    let mut budget = DecodeBudget {
        limits,
        total_elements: 0,
        allocation_bytes: size_of::<T>(),
        rejected: None,
    };
    if budget.allocation_bytes > limits.max_total_allocation_bytes {
        return Err(CodecError::AllocationLimit {
            requested: budget.allocation_bytes,
            remaining: limits.max_total_allocation_bytes,
        });
    }
    let mut decoder = postcard::Deserializer::from_bytes(bytes);
    let value = T::deserialize(BoundedDeserializer {
        inner: &mut decoder,
        budget: &mut budget,
    });
    if let Some((requested, remaining)) = budget.rejected {
        return Err(CodecError::AllocationLimit {
            requested,
            remaining,
        });
    }
    let value = value?;
    let rest = decoder.finalize()?;
    if !rest.is_empty() {
        return Err(CodecError::TrailingBytes);
    }
    Ok(value)
}

fn proof_max_element_size() -> usize {
    use p3_sumcheck_08::zk::ZkSumcheckData;
    use p3_whir_08::pcs::{
        proof::{QueryOpenings, SharedProofOpening},
        zk::{BlindedMask, MaskOpeningPair, ZkRoundProof},
    };

    type Round = ZkRoundProof<HidingBase, HidingExtension, HidingMerkleMmcs>;
    type Openings = QueryOpenings<
        HidingBase,
        HidingExtension,
        <HidingMerkleMmcs as p3_commit_08::Mmcs<HidingBase>>::MultiProof,
    >;
    type Shared = SharedProofOpening<
        HidingBase,
        <HidingMerkleMmcs as p3_commit_08::Mmcs<HidingBase>>::MultiProof,
    >;

    // These cover every element type stored in a Vec in the pinned proof schema.
    [
        size_of::<Round>(),
        size_of::<ZkSumcheckData<HidingBase, HidingExtension>>(),
        size_of::<Openings>(),
        size_of::<Shared>(),
        size_of::<BlindedMask<HidingExtension>>(),
        size_of::<
            MaskOpeningPair<
                HidingExtension,
                <HidingMerkleMmcs as p3_commit_08::Mmcs<HidingBase>>::MultiProof,
            >,
        >(),
        size_of::<[HidingBase; 8]>(),
        size_of::<HidingExtension>(),
        size_of::<HidingBase>(),
        size_of::<Vec<HidingBase>>(),
        size_of::<Vec<HidingExtension>>(),
    ]
    .into_iter()
    .max()
    .unwrap()
}

struct DecodeBudget {
    limits: DecodeLimits,
    total_elements: usize,
    allocation_bytes: usize,
    rejected: Option<(usize, usize)>,
}

impl DecodeBudget {
    fn reserve(&mut self, sequence_len: usize) -> Result<(), postcard::Error> {
        let Some(total_elements) = self.total_elements.checked_add(sequence_len) else {
            self.reject(usize::MAX);
            return Err(de::Error::custom("sequence element budget overflow"));
        };
        let Some(requested) = sequence_len.checked_mul(self.limits.max_element_size) else {
            self.reject(usize::MAX);
            return Err(de::Error::custom("sequence allocation budget overflow"));
        };
        let remaining = self
            .limits
            .max_total_allocation_bytes
            .saturating_sub(self.allocation_bytes);
        if sequence_len > self.limits.max_sequence_len
            || total_elements > self.limits.max_total_elements
            || requested > remaining
        {
            self.reject(requested);
            return Err(de::Error::custom("hiding-WHIR allocation budget exceeded"));
        }
        self.total_elements = total_elements;
        self.allocation_bytes += requested;
        Ok(())
    }

    fn reject(&mut self, requested: usize) {
        let remaining = self
            .limits
            .max_total_allocation_bytes
            .saturating_sub(self.allocation_bytes);
        self.rejected = Some((requested, remaining));
    }
}

struct BoundedDeserializer<'a, D> {
    inner: D,
    budget: &'a mut DecodeBudget,
}

struct BoundedSeed<'a, S> {
    seed: S,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, S> DeserializeSeed<'de> for BoundedSeed<'a, S>
where
    S: DeserializeSeed<'de>,
{
    type Value = S::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.seed.deserialize(BoundedDeserializer {
            inner: deserializer,
            budget: self.budget,
        })
    }
}

struct BoundedVisitor<'a, V> {
    visitor: V,
    budget: &'a mut DecodeBudget,
    known_len: Option<usize>,
}

impl<'de, 'a, V> Visitor<'de> for BoundedVisitor<'a, V>
where
    V: Visitor<'de>,
{
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.visitor.expecting(formatter)
    }

    fn visit_seq<A>(self, access: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if self.known_len.is_none() {
            let len = access.size_hint().ok_or_else(|| {
                self.budget
                    .reject(self.budget.limits.max_total_allocation_bytes);
                A::Error::custom("sequence length cannot be bounded from input")
            })?;
            self.budget
                .reserve(len)
                .map_err(|error| A::Error::custom(error.to_string()))?;
        }
        self.visitor.visit_seq(BoundedSeqAccess {
            inner: access,
            budget: self.budget,
        })
    }

    fn visit_map<A>(self, access: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let len = access
            .size_hint()
            .ok_or_else(|| de::Error::custom("map has no checked length"))?;
        self.budget
            .reserve(len)
            .map_err(|error| A::Error::custom(error.to_string()))?;
        self.visitor.visit_map(BoundedMapAccess {
            inner: access,
            budget: self.budget,
        })
    }
}

struct BoundedSeqAccess<'a, A> {
    inner: A,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, A> SeqAccess<'de> for BoundedSeqAccess<'a, A>
where
    A: SeqAccess<'de>,
{
    type Error = A::Error;

    fn next_element_seed<S>(&mut self, seed: S) -> Result<Option<S::Value>, Self::Error>
    where
        S: DeserializeSeed<'de>,
    {
        self.inner.next_element_seed(BoundedSeed {
            seed,
            budget: self.budget,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

struct BoundedMapAccess<'a, A> {
    inner: A,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, A> MapAccess<'de> for BoundedMapAccess<'a, A>
where
    A: MapAccess<'de>,
{
    type Error = A::Error;

    fn next_key_seed<S>(&mut self, seed: S) -> Result<Option<S::Value>, Self::Error>
    where
        S: DeserializeSeed<'de>,
    {
        self.inner.next_key_seed(BoundedSeed {
            seed,
            budget: self.budget,
        })
    }

    fn next_value_seed<S>(&mut self, seed: S) -> Result<S::Value, Self::Error>
    where
        S: DeserializeSeed<'de>,
    {
        self.inner.next_value_seed(BoundedSeed {
            seed,
            budget: self.budget,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

struct BoundedEnumAccess<'a, A> {
    inner: A,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, A> EnumAccess<'de> for BoundedEnumAccess<'a, A>
where
    A: EnumAccess<'de>,
{
    type Error = A::Error;
    type Variant = BoundedVariantAccess<'a, A::Variant>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        let (variant, access) = self.inner.variant_seed(BoundedSeed {
            seed,
            budget: self.budget,
        })?;
        Ok((
            variant,
            BoundedVariantAccess {
                inner: access,
                budget: self.budget,
            },
        ))
    }
}

struct BoundedVariantAccess<'a, A> {
    inner: A,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, A> VariantAccess<'de> for BoundedVariantAccess<'a, A>
where
    A: VariantAccess<'de>,
{
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), Self::Error> {
        self.inner.unit_variant()
    }

    fn newtype_variant_seed<S>(self, seed: S) -> Result<S::Value, Self::Error>
    where
        S: DeserializeSeed<'de>,
    {
        self.inner.newtype_variant_seed(BoundedSeed {
            seed,
            budget: self.budget,
        })
    }

    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.tuple_variant(
            len,
            BoundedVisitor {
                visitor,
                budget: self.budget,
                known_len: Some(len),
            },
        )
    }

    fn struct_variant<V>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.struct_variant(
            fields,
            BoundedVisitor {
                visitor,
                budget: self.budget,
                known_len: Some(fields.len()),
            },
        )
    }
}

macro_rules! forward {
    ($method:ident $(, $arg:ident : $ty:ty)*) => {
        fn $method<V>(self, $($arg: $ty,)* visitor: V) -> Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            self.inner.$method($($arg,)* visitor)
        }
    };
}

impl<'de, D> Deserializer<'de> for BoundedDeserializer<'_, D>
where
    D: Deserializer<'de>,
{
    type Error = D::Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_any(visitor)
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }

    forward!(deserialize_bool);
    forward!(deserialize_i8);
    forward!(deserialize_i16);
    forward!(deserialize_i32);
    forward!(deserialize_i64);
    forward!(deserialize_i128);
    forward!(deserialize_u8);
    forward!(deserialize_u16);
    forward!(deserialize_u32);
    forward!(deserialize_u64);
    forward!(deserialize_u128);
    forward!(deserialize_f32);
    forward!(deserialize_f64);
    forward!(deserialize_char);
    forward!(deserialize_str);
    forward!(deserialize_string);
    forward!(deserialize_bytes);
    forward!(deserialize_byte_buf);
    forward!(deserialize_option);
    forward!(deserialize_unit);
    forward!(deserialize_newtype_struct, name: &'static str);
    forward!(deserialize_identifier);
    forward!(deserialize_ignored_any);

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_seq(BoundedVisitor {
            visitor,
            budget: self.budget,
            known_len: None,
        })
    }

    fn deserialize_unit_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_unit(visitor)
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_tuple(
            len,
            BoundedVisitor {
                visitor,
                budget: self.budget,
                known_len: Some(len),
            },
        )
    }

    fn deserialize_tuple_struct<V>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_tuple_struct(
            name,
            len,
            BoundedVisitor {
                visitor,
                budget: self.budget,
                known_len: Some(len),
            },
        )
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_struct(
            name,
            fields,
            BoundedVisitor {
                visitor,
                budget: self.budget,
                known_len: Some(fields.len()),
            },
        )
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_map(BoundedVisitor {
            visitor,
            budget: self.budget,
            known_len: None,
        })
    }

    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        self.inner.deserialize_enum(
            name,
            variants,
            BoundedEnumVisitor {
                visitor,
                budget: self.budget,
            },
        )
    }
}

struct BoundedEnumVisitor<'a, V> {
    visitor: V,
    budget: &'a mut DecodeBudget,
}

impl<'de, 'a, V> Visitor<'de> for BoundedEnumVisitor<'a, V>
where
    V: Visitor<'de>,
{
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.visitor.expecting(formatter)
    }

    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
    where
        A: EnumAccess<'de>,
    {
        self.visitor.visit_enum(BoundedEnumAccess {
            inner: data,
            budget: self.budget,
        })
    }
}
