//! One Plonky3 0.8 challenger shared by the V9 outer IOP and hiding PCS.

use p3_challenger_08::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_field::{PrimeCharacteristicRing as OuterRing, PrimeField64 as OuterPrimeField64};
use p3_field_08::{PrimeCharacteristicRing as HidingRing, PrimeField64 as HidingPrimeField64};

#[cfg(test)]
use std::sync::{Arc, Mutex};

use crate::{FiatShamirTranscript, StarkProtocolConfig};

use super::{HidingBase, OuterBase, OuterExtension, RawHidingChallenger};

#[derive(Clone, Debug)]
pub struct V9Transcript {
    inner: RawHidingChallenger,
    #[cfg(test)]
    events: Option<Arc<Mutex<Vec<TranscriptEvent>>>>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TranscriptEvent {
    ObserveBase(u64),
    ObserveP3Base(u64),
    ObserveP3Array(Vec<u64>),
    SampleBase(u64),
    SampleBits {
        bits: usize,
        value: usize,
    },
    SampleUniformBits {
        bits: usize,
        resample: bool,
        value: Option<usize>,
    },
    ObserveCommitment(Vec<u64>),
    Grind(u64),
}

impl V9Transcript {
    pub fn new(inner: RawHidingChallenger) -> Self {
        Self {
            inner,
            #[cfg(test)]
            events: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_events(inner: RawHidingChallenger) -> Self {
        Self {
            inner,
            events: Some(Arc::new(Mutex::new(Vec::new()))),
        }
    }

    #[cfg(test)]
    pub(super) fn clone_untraced(&self) -> Self {
        Self::new(self.inner.clone())
    }

    #[cfg(test)]
    fn record(&self, event: TranscriptEvent) {
        if let Some(events) = &self.events {
            events.lock().unwrap().push(event);
        }
    }

    #[cfg(test)]
    pub(super) fn events(&self) -> Vec<TranscriptEvent> {
        self.events
            .as_ref()
            .map(|events| events.lock().unwrap().clone())
            .unwrap_or_default()
    }

    pub fn challenger(&mut self) -> &mut RawHidingChallenger {
        &mut self.inner
    }

    pub fn into_challenger(self) -> RawHidingChallenger {
        self.inner
    }
}

#[inline]
fn to_hiding(value: OuterBase) -> HidingBase {
    debug_assert_eq!(
        <OuterBase as OuterPrimeField64>::ORDER_U64,
        <HidingBase as HidingPrimeField64>::ORDER_U64
    );
    HidingRing::from_u64(value.as_canonical_u64())
}

#[inline]
fn to_outer(value: HidingBase) -> OuterBase {
    debug_assert_eq!(
        <OuterBase as OuterPrimeField64>::ORDER_U64,
        <HidingBase as HidingPrimeField64>::ORDER_U64
    );
    OuterRing::from_u64(value.as_canonical_u64())
}

impl<SC> FiatShamirTranscript<SC> for V9Transcript
where
    SC: StarkProtocolConfig<F = OuterBase, EF = OuterExtension, Digest = [OuterBase; 8]>,
{
    fn observe(&mut self, value: OuterBase) {
        #[cfg(test)]
        self.record(TranscriptEvent::ObserveBase(value.as_canonical_u64()));
        CanObserve::observe(&mut self.inner, to_hiding(value));
    }

    fn sample(&mut self) -> OuterBase {
        let value = CanSample::<HidingBase>::sample(&mut self.inner);
        #[cfg(test)]
        self.record(TranscriptEvent::SampleBase(value.as_canonical_u64()));
        to_outer(value)
    }

    fn observe_commit(&mut self, digest: [OuterBase; 8]) {
        for value in digest {
            FiatShamirTranscript::<SC>::observe(self, value);
        }
    }

    fn sample_bits(&mut self, bits: usize) -> u64 {
        assert!(bits < u32::BITS as usize);
        assert!((1u64 << bits) < OuterBase::ORDER_U64);
        let sample = FiatShamirTranscript::<SC>::sample(self).as_canonical_u64();
        let value = sample & ((1u64 << bits) - 1);
        #[cfg(test)]
        self.record(TranscriptEvent::SampleBits {
            bits,
            value: value as usize,
        });
        value
    }
}

impl CanObserve<HidingBase> for V9Transcript {
    fn observe(&mut self, value: HidingBase) {
        #[cfg(test)]
        self.record(TranscriptEvent::ObserveP3Base(value.as_canonical_u64()));
        CanObserve::observe(&mut self.inner, value);
    }
}

impl<const N: usize> CanObserve<[HidingBase; N]> for V9Transcript {
    fn observe(&mut self, values: [HidingBase; N]) {
        #[cfg(test)]
        self.record(TranscriptEvent::ObserveP3Array(
            values
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect(),
        ));
        CanObserve::observe(&mut self.inner, values);
    }
}

impl CanObserve<super::HidingCommitment> for V9Transcript {
    fn observe(&mut self, commitment: super::HidingCommitment) {
        #[cfg(test)]
        self.record(TranscriptEvent::ObserveCommitment(
            commitment
                .as_ref()
                .iter()
                .flat_map(|digest| digest.iter())
                .map(|value| value.as_canonical_u64())
                .collect(),
        ));
        CanObserve::observe(&mut self.inner, commitment);
    }
}

impl CanSample<HidingBase> for V9Transcript {
    fn sample(&mut self) -> HidingBase {
        let value = CanSample::<HidingBase>::sample(&mut self.inner);
        #[cfg(test)]
        self.record(TranscriptEvent::SampleBase(value.as_canonical_u64()));
        value
    }
}

impl CanSampleBits<usize> for V9Transcript {
    fn sample_bits(&mut self, bits: usize) -> usize {
        let value = CanSampleBits::<usize>::sample_bits(&mut self.inner, bits);
        #[cfg(test)]
        self.record(TranscriptEvent::SampleBits { bits, value });
        value
    }
}

impl CanSampleUniformBits<HidingBase> for V9Transcript {
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, p3_challenger_08::ResamplingError> {
        let result = CanSampleUniformBits::<HidingBase>::sample_uniform_bits::<RESAMPLE>(
            &mut self.inner,
            bits,
        );
        #[cfg(test)]
        self.record(TranscriptEvent::SampleUniformBits {
            bits,
            resample: RESAMPLE,
            value: result.as_ref().ok().copied(),
        });
        result
    }
}

impl FieldChallenger<HidingBase> for V9Transcript {}

impl GrindingChallenger for V9Transcript {
    type Witness = HidingBase;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = GrindingChallenger::grind(&mut self.inner, bits);
        #[cfg(test)]
        self.record(TranscriptEvent::Grind(witness.as_canonical_u64()));
        witness
    }
}
