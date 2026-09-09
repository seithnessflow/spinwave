//! Deferred deallocation for the audio thread: everything a patch swap
//! replaces (per-kernel params with their line-generator buffers,
//! connection lists, samples, multisample instruments) is moved into a
//! bounded channel and dropped by a collector thread, so the audio thread
//! never frees heap memory.
//!
//! When the channel is full the item waits in a small preallocated
//! audio-side ring ([`OVERFLOW_CAPACITY`] entries, flushed at the next
//! block); only when that is full too does the audio thread drop it
//! itself (last resort, logged in debug builds).

use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

use spinwave_dsp::oscillator::{Multisample, Sample};
use spinwave_dsp::wavetable::Wavetable;
use spinwave_engine::kernel::mod_matrix::Connection;
use spinwave_engine::kernel::KernelParams;

use crate::patch::BuiltPatch;

/// Channel depth between the audio thread and the collector.
pub const CHANNEL_CAPACITY: usize = 64;
/// Audio-side overflow ring depth (preallocated, never grows).
pub const OVERFLOW_CAPACITY: usize = 32;

/// Something the audio thread replaced and must not drop itself.
pub enum Garbage {
    /// A whole applied patch: after the swap it holds the PREVIOUS kernel
    /// params, connection lists, effect params and global samples.
    Patch(Box<BuiltPatch>),
    /// Fallback path only (kernel pool larger than the prebuilt set).
    Kernel(Box<KernelParams>),
    Connections(Vec<Connection>),
    Sample(Arc<Sample>),
    Wavetable(Arc<Wavetable>),
    GlobalSample(Sample),
    Multisamples(Vec<Multisample>),
}

/// Audio-thread side: pushes garbage toward the collector without
/// blocking or allocating.
pub struct GarbageChute {
    sender: SyncSender<Garbage>,
    overflow: Vec<Garbage>,
    /// Items dropped on the audio thread because everything was full.
    pub dropped_inline: u64,
}

impl GarbageChute {
    /// Creates the chute and its receiving end.
    #[must_use]
    pub fn new() -> (GarbageChute, Receiver<Garbage>) {
        let (sender, receiver) = sync_channel(CHANNEL_CAPACITY);
        let chute = GarbageChute {
            sender,
            overflow: Vec::with_capacity(OVERFLOW_CAPACITY),
            dropped_inline: 0,
        };
        (chute, receiver)
    }

    /// Hands an item to the collector (or parks it in the overflow ring).
    pub fn discard(&mut self, item: Garbage) {
        self.flush();
        match self.sender.try_send(item) {
            Ok(()) => {}
            Err(TrySendError::Full(item)) | Err(TrySendError::Disconnected(item)) => {
                if self.overflow.len() < self.overflow.capacity() {
                    self.overflow.push(item);
                } else {
                    // Last resort: the drop happens here.
                    self.dropped_inline += 1;
                    drop(item);
                }
            }
        }
    }

    /// Retries the parked items (call once per block).
    pub fn flush(&mut self) {
        while let Some(item) = self.overflow.last() {
            let _ = item;
            let item = self.overflow.pop().expect("checked non-empty");
            match self.sender.try_send(item) {
                Ok(()) => {}
                Err(TrySendError::Full(item)) | Err(TrySendError::Disconnected(item)) => {
                    self.overflow.push(item);
                    break;
                }
            }
        }
    }

    /// Items waiting in the overflow ring.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.overflow.len()
    }
}

/// Spawns the collector: drops everything received until the chute (every
/// sender) is gone.
pub fn spawn_collector(receiver: Receiver<Garbage>) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("spinwave-gc".into())
        .spawn(move || {
            while let Ok(item) = receiver.recv() {
                drop(item);
            }
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_ring_holds_items_until_the_collector_catches_up() {
        let (mut chute, receiver) = GarbageChute::new();
        // Fill the channel and the ring, then one more.
        for _ in 0..(CHANNEL_CAPACITY + OVERFLOW_CAPACITY) {
            chute.discard(Garbage::Kernel(Box::default()));
        }
        assert_eq!(chute.pending(), OVERFLOW_CAPACITY);
        assert_eq!(chute.dropped_inline, 0);
        chute.discard(Garbage::Kernel(Box::default()));
        assert_eq!(chute.dropped_inline, 1);

        // Draining the channel lets the ring flush at the next block.
        for _ in 0..CHANNEL_CAPACITY {
            receiver.recv().unwrap();
        }
        chute.flush();
        assert_eq!(chute.pending(), 0);
        drop(chute);
        assert_eq!(receiver.iter().count(), OVERFLOW_CAPACITY);
    }

    #[test]
    fn collector_drops_and_exits_when_the_chute_is_gone() {
        let (mut chute, receiver) = GarbageChute::new();
        let handle = spawn_collector(receiver).unwrap();
        let sample = Arc::new(Sample::from_mono("s", &[0.0; 64], 44100));
        let weak = Arc::downgrade(&sample);
        chute.discard(Garbage::Sample(sample));
        drop(chute);
        handle.join().unwrap();
        assert!(weak.upgrade().is_none(), "the collector dropped the last Arc");
    }
}
