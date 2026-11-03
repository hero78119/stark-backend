//! Checked mapping between the existing SWIRL field/MLE convention and P3 0.8.

use p3_field::{
    BasedVectorSpace as OuterBasedVectorSpace, Field as OuterField,
    PrimeCharacteristicRing as OuterRing, PrimeField64 as OuterPrimeField64,
};
use p3_field_08::{
    BasedVectorSpace as HidingBasedVectorSpace, PrimeCharacteristicRing as HidingRing,
    PrimeField64 as HidingPrimeField64,
};
use p3_maybe_rayon::prelude::*;
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
    #[error("column converter returned {actual} coefficients; expected {expected}")]
    InvalidConvertedColumnLength { expected: usize, actual: usize },
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
    if columns.is_empty() {
        return Err(BridgeError::EmptyColumns);
    }
    let padded_width = columns
        .len()
        .checked_next_power_of_two()
        .ok_or(BridgeError::WidthOverflow)?;
    pack_columns_at_width(columns, padded_width)
}

/// Packs columns into a fixed selector width, zero-padding missing columns.
pub fn pack_columns_at_width(
    columns: &[Vec<OuterBase>],
    padded_width: usize,
) -> Result<Poly<HidingBase>, BridgeError> {
    let first = columns.first().ok_or(BridgeError::EmptyColumns)?;
    if first.is_empty()
        || !first.len().is_power_of_two()
        || !padded_width.is_power_of_two()
        || padded_width < columns.len()
    {
        return Err(BridgeError::InvalidColumnShape);
    }
    if columns.iter().any(|column| column.len() != first.len()) {
        return Err(BridgeError::InvalidColumnShape);
    }
    let table_len = first
        .len()
        .checked_mul(padded_width)
        .ok_or(BridgeError::WidthOverflow)?;
    let mut table = vec![HidingBase::ZERO; table_len];
    for (index, column) in columns.iter().enumerate() {
        let start = index * first.len();
        table[start..start + first.len()]
            .iter_mut()
            .zip(column)
            .for_each(|(out, &value)| *out = base_to_hiding(value));
    }
    Ok(Poly::new(table))
}

/// Packs the zero polynomial when a fixed-width selector chunk has no source columns.
pub fn pack_zero_columns_at_width(
    column_height: usize,
    padded_width: usize,
) -> Result<Poly<HidingBase>, BridgeError> {
    if column_height == 0
        || !column_height.is_power_of_two()
        || padded_width == 0
        || !padded_width.is_power_of_two()
    {
        return Err(BridgeError::InvalidColumnShape);
    }
    let table_len = column_height
        .checked_mul(padded_width)
        .ok_or(BridgeError::WidthOverflow)?;
    Ok(Poly::new(vec![HidingBase::ZERO; table_len]))
}

/// Converts selected evaluation columns directly into one P3 table.
///
/// The coefficient vector for each column is released as soon as it is copied into the
/// destination, avoiding a second full-size `Vec<Vec<_>>` beside the packed witness.
pub fn pack_eval_columns_at_width(
    values: &[OuterBase],
    column_height: usize,
    column_count: usize,
    first_column: usize,
    l_skip: usize,
    padded_width: usize,
    convert: impl Fn(&[OuterBase]) -> Vec<OuterBase> + Sync,
) -> Result<Poly<HidingBase>, BridgeError> {
    if column_height == 0
        || !column_height.is_power_of_two()
        || padded_width == 0
        || !padded_width.is_power_of_two()
        || column_count == 0
        || values.len()
            != column_height
                .checked_mul(column_count)
                .ok_or(BridgeError::WidthOverflow)?
        || l_skip > column_height.trailing_zeros() as usize
    {
        return Err(BridgeError::InvalidColumnShape);
    }

    let end_column = first_column.saturating_add(padded_width).min(column_count);
    let table_len = column_height
        .checked_mul(padded_width)
        .ok_or(BridgeError::WidthOverflow)?;
    let mut table = vec![HidingBase::ZERO; table_len];

    table
        .par_chunks_exact_mut(column_height)
        .take(end_column.saturating_sub(first_column))
        .enumerate()
        .try_for_each(|(destination, output)| {
            let source_column = first_column + destination;
            let start = source_column * column_height;
            let coeffs = convert(&values[start..start + column_height]);
            if coeffs.len() != column_height {
                return Err(BridgeError::InvalidConvertedColumnLength {
                    expected: column_height,
                    actual: coeffs.len(),
                });
            }
            output
                .iter_mut()
                .zip(coeffs)
                .for_each(|(dst, value)| *dst = base_to_hiding(value));
            Ok(())
        })?;

    Ok(Poly::new(table))
}

/// Adds leading zero selector coordinates, embedding the table in the first padded subcube.
pub fn pad_opening_point(
    point: &Point<HidingExtension>,
    added_variables: usize,
) -> Point<HidingExtension> {
    Point::new(
        std::iter::repeat_n(HidingExtension::ZERO, added_variables)
            .chain(point.as_slice().iter().copied())
            .collect::<Vec<_>>(),
    )
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
