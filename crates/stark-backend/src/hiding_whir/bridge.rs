//! Checked mapping between the existing SWIRL field/MLE convention and P3 0.8.

use p3_field::{
    BasedVectorSpace as OuterBasedVectorSpace, Field as OuterField,
    PrimeCharacteristicRing as OuterRing, PrimeField64 as OuterPrimeField64,
};
use p3_field_08::{
    BasedVectorSpace as HidingBasedVectorSpace, PrimeCharacteristicRing as HidingRing,
    PrimeField64 as HidingPrimeField64,
};
use p3_multilinear_util_08::{point::Point, poly::Poly};
use thiserror::Error;

use super::{HidingBase, HidingExtension, OuterBase, OuterExtension};

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum BridgeError {
    #[error("the stacked matrix must have at least one column")]
    EmptyColumns,
    #[error("stacked columns must be non-empty power-of-two tables of equal length")]
    InvalidColumnShape,
    #[error("stacked width is too large")]
    WidthOverflow,
    #[error("cube point length does not match the stacked table")]
    PointArityMismatch,
    #[error("selector denominator is zero for this batching challenge")]
    ZeroSelectorDenominator,
}

#[inline]
pub fn base_to_hiding(value: OuterBase) -> HidingBase {
    debug_assert_eq!(
        <OuterBase as OuterPrimeField64>::ORDER_U64,
        <HidingBase as HidingPrimeField64>::ORDER_U64
    );
    HidingRing::from_u64(value.as_canonical_u64())
}

#[inline]
pub fn base_to_outer(value: HidingBase) -> OuterBase {
    debug_assert_eq!(
        <OuterBase as OuterPrimeField64>::ORDER_U64,
        <HidingBase as HidingPrimeField64>::ORDER_U64
    );
    OuterRing::from_u64(value.as_canonical_u64())
}

/// Embeds `a0 + a1*α + a2*α² + a3*α³` as `a0 + a1*β² + a2*β⁴ + a3*β⁶`.
pub fn extension_to_hiding(value: OuterExtension) -> HidingExtension {
    let coefficients = OuterBasedVectorSpace::as_basis_coefficients_slice(&value);
    <HidingExtension as HidingBasedVectorSpace<HidingBase>>::from_basis_coefficients_fn(|i| {
        if i % 2 == 0 {
            base_to_hiding(coefficients[i / 2])
        } else {
            HidingBase::ZERO
        }
    })
}

/// Packs stacked column tables in P3's lexicographic order; missing columns are zero.
pub fn pack_columns(columns: &[Vec<OuterBase>]) -> Result<Poly<HidingBase>, BridgeError> {
    let first = columns.first().ok_or(BridgeError::EmptyColumns)?;
    if first.is_empty() || !first.len().is_power_of_two() {
        return Err(BridgeError::InvalidColumnShape);
    }
    if columns.iter().any(|column| column.len() != first.len()) {
        return Err(BridgeError::InvalidColumnShape);
    }
    let padded_width = columns
        .len()
        .checked_next_power_of_two()
        .ok_or(BridgeError::WidthOverflow)?;
    let mut table = vec![HidingBase::ZERO; first.len() * padded_width];
    for (index, column) in columns.iter().enumerate() {
        let start = index * first.len();
        table[start..start + first.len()]
            .iter_mut()
            .zip(column)
            .for_each(|(out, &value)| *out = base_to_hiding(value));
    }
    Ok(Poly::new(table))
}

/// Maps the OpenVM `μ` batch claim to P3's selector point and scaled opening value.
pub fn packed_opening(
    mu: OuterExtension,
    width: usize,
    cube_point_little_endian: &[OuterExtension],
    expected_value: OuterExtension,
) -> Result<(Point<HidingExtension>, HidingExtension), BridgeError> {
    if width == 0 {
        return Err(BridgeError::EmptyColumns);
    }
    let padded_width = width
        .checked_next_power_of_two()
        .ok_or(BridgeError::WidthOverflow)?;
    let selector_variables = padded_width.trailing_zeros() as usize;
    let mut theta = Vec::with_capacity(selector_variables);
    let mut scale = OuterExtension::ONE;

    for coordinate in 0..selector_variables {
        let exponent = 1u64 << (selector_variables - coordinate - 1);
        let power = mu.exp_u64(exponent);
        let denominator = OuterExtension::ONE + power;
        let inverse = denominator
            .try_inverse()
            .ok_or(BridgeError::ZeroSelectorDenominator)?;
        theta.push(power * inverse);
        scale *= denominator;
    }

    let mut point = theta
        .into_iter()
        .map(extension_to_hiding)
        .collect::<Vec<_>>();
    point.extend(
        cube_point_little_endian
            .iter()
            .rev()
            .copied()
            .map(extension_to_hiding),
    );
    let scaled_claim = expected_value
        * scale
            .try_inverse()
            .ok_or(BridgeError::ZeroSelectorDenominator)?;
    Ok((Point::new(point), extension_to_hiding(scaled_claim)))
}
