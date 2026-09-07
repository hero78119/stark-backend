use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use bytesize::ByteSize;

use crate::{
    common::set_device_by_id,
    error::{check, MemoryError},
    stream::{cudaStream_t, device_synchronize, GpuDeviceCtx, StreamGuard},
};

mod cuda;
mod vm_pool;
use vm_pool::VirtualMemoryPool;

#[cfg(test)]
mod tests;

#[link(name = "cudart")]
extern "C" {
    fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: cudaStream_t) -> i32;
    fn cudaFreeAsync(dev_ptr: *mut c_void, stream: cudaStream_t) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

type DeviceManagers = HashMap<u32, Arc<Mutex<MemoryManager>>>;

static MEMORY_MANAGERS: OnceLock<Mutex<DeviceManagers>> = OnceLock::new();

pub fn device_memory_used() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe { cudaMemGetInfo(&mut free, &mut total) };
    total - free
}

fn managers() -> &'static Mutex<DeviceManagers> {
    MEMORY_MANAGERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn manager_for_device(device_id: u32) -> Result<Arc<Mutex<MemoryManager>>, MemoryError> {
    let mut managers = managers().lock().map_err(|_| MemoryError::LockError)?;
    Ok(managers
        .entry(device_id)
        .or_insert_with(|| Arc::new(Mutex::new(MemoryManager::new(device_id))))
        .clone())
}

/// Allocation record for the small-allocation path (`cudaMallocAsync`).
struct AllocRecord {
    size: usize,
    stream: StreamGuard,
}

pub struct MemoryManager {
    device_id: u32,
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, AllocRecord>,
    current_size: usize,
    max_used_size: usize,
}

/// # Safety
/// `MemoryManager` is not internally synchronized. These impls are safe because
/// the singleton instance is wrapped in `Mutex` via `MEMORY_MANAGER`.
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub fn new(device_id: u32) -> Self {
        set_device_by_id(device_id as i32).expect("failed to bind CUDA memory-manager device");
        let pool = VirtualMemoryPool::new(device_id as i32, Default::default());

        Self {
            device_id,
            pool,
            allocated_ptrs: HashMap::new(),
            current_size: 0,
            max_used_size: 0,
        }
    }

    fn d_malloc_on(
        &mut self,
        size: usize,
        stream: &StreamGuard,
    ) -> Result<*mut c_void, MemoryError> {
        set_device_by_id(self.device_id as i32)?;
        assert!(size != 0, "Requested size must be non-zero");

        let mut tracked_size = size;
        let ptr = if size < self.pool.page_size {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check(unsafe { cudaMallocAsync(&mut ptr, size, stream.as_raw()) }).map_err(|e| {
                tracing::error!("cudaMallocAsync failed: size={}: {:?}", size, e);
                MemoryError::from(e)
            })?;
            self.allocated_ptrs.insert(
                NonNull::new(ptr).expect("BUG: cudaMallocAsync returned null"),
                AllocRecord {
                    size,
                    stream: stream.clone(),
                },
            );
            ptr
        } else {
            tracked_size = size.next_multiple_of(self.pool.page_size);
            self.pool.malloc_internal(tracked_size, stream)?
        };

        self.current_size += tracked_size;
        if self.current_size > self.max_used_size {
            self.max_used_size = self.current_size;
        }
        Ok(ptr)
    }

    /// Two-stage free: first resolves the record under the lock, returning the
    /// `StreamGuard` that must be dropped AFTER the lock is released.
    ///
    /// # Safety
    /// - The pointer `ptr` must be a valid, previously allocated device pointer.
    /// - The caller must ensure that `ptr` is not used after this function is called.
    /// - The caller must hold the `MEMORY_MANAGER` lock before calling this method.
    unsafe fn d_free_under_lock(&mut self, ptr: *mut c_void) -> Result<StreamGuard, MemoryError> {
        set_device_by_id(self.device_id as i32)?;
        let nn = NonNull::new(ptr).ok_or(MemoryError::NullPointer)?;

        if let Some(record) = self.allocated_ptrs.remove(&nn) {
            let size = record.size;
            self.current_size -= size;
            check(unsafe { cudaFreeAsync(ptr, record.stream.as_raw()) }).map_err(|e| {
                tracing::error!("cudaFreeAsync failed: ptr={:p}: {:?}", ptr, e);
                MemoryError::from(e)
            })?;
            Ok(record.stream)
        } else {
            let (freed_size, guard) = self.pool.free_internal(ptr)?;
            self.current_size -= freed_size;
            Ok(guard)
        }
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        set_device_by_id(self.device_id as i32).unwrap();
        device_synchronize().unwrap();
        let ptrs: Vec<*mut c_void> = self.allocated_ptrs.keys().map(|nn| nn.as_ptr()).collect();
        for &ptr in &ptrs {
            match unsafe { self.d_free_under_lock(ptr) } {
                Ok(guard) => drop(guard),
                Err(e) => tracing::error!("MemoryManager drop: failed to free {:p}: {:?}", ptr, e),
            }
        }
    }
}

pub fn d_malloc_on(size: usize, device_ctx: &GpuDeviceCtx) -> Result<*mut c_void, MemoryError> {
    if device_ctx.device_id != device_ctx.stream.device_id() {
        return Err(MemoryError::DeviceMismatch {
            context_device: device_ctx.device_id,
            stream_device: device_ctx.stream.device_id(),
        });
    }
    set_device_by_id(device_ctx.device_id as i32)?;
    let manager = manager_for_device(device_ctx.device_id)?;
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    manager.d_malloc_on(size, &device_ctx.stream)
}

/// # Safety
/// The pointer `ptr` must be a valid, previously allocated device pointer.
/// The caller must ensure that `ptr` is not used after this function is called.
pub unsafe fn d_free_on(ptr: *mut c_void, device_id: u32) -> Result<(), MemoryError> {
    set_device_by_id(device_id as i32)?;
    let manager = manager_for_device(device_id)?;
    let mut manager = manager.lock().map_err(|_| MemoryError::LockError)?;
    let guard = manager.d_free_under_lock(ptr)?;
    drop(manager);
    drop(guard);
    Ok(())
}

/// Synchronize and release all cached allocation state for an idle CUDA device.
pub fn synchronize_and_trim_device(device_id: u32) -> Result<(), MemoryError> {
    set_device_by_id(device_id as i32)?;
    device_synchronize()?;
    let manager = {
        let mut managers = managers().lock().map_err(|_| MemoryError::LockError)?;
        let Some(manager) = managers.get(&device_id) else {
            return Ok(());
        };
        let allocated_bytes = manager
            .lock()
            .map_err(|_| MemoryError::LockError)?
            .current_size;
        if allocated_bytes != 0 {
            return Err(MemoryError::ActiveAllocations {
                device_id,
                allocated_bytes,
            });
        }
        managers.remove(&device_id).unwrap()
    };
    drop(manager);
    device_synchronize()?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct MemTracker {
    current: usize,
    device_id: u32,
    label: &'static str,
}

impl MemTracker {
    pub fn start(label: &'static str) -> Self {
        let device_id = crate::common::get_device().unwrap_or_default() as u32;
        let current = managers()
            .lock()
            .ok()
            .and_then(|managers| managers.get(&device_id).cloned())
            .and_then(|manager| manager.lock().ok().map(|manager| manager.current_size))
            .unwrap_or_default();

        Self {
            current,
            device_id,
            label,
        }
    }

    pub fn start_and_reset_peak(label: &'static str) -> Self {
        let mut mem = Self::start(label);
        mem.reset_peak();
        mem
    }

    pub fn emit_metrics(&self) {
        self.emit_metrics_with_label(self.label);
    }

    pub fn emit_metrics_with_label(&self, label: &'static str) {
        let Some(manager) = managers()
            .lock()
            .ok()
            .and_then(|managers| managers.get(&self.device_id).cloned())
        else {
            return;
        };
        let Ok(manager) = manager.lock() else { return };

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            * 1000.0;
        let current = manager.current_size;
        // local_peak is local maximum memory size, as observed by the manager, since the last
        // reset_peak call
        let local_peak = manager.max_used_size;
        let reserved = manager.pool.memory_usage();
        metrics::gauge!("gpu_mem.timestamp_ms", "module" => label).set(ts);
        metrics::gauge!("gpu_mem.current_bytes", "module" => label).set(current as f64);
        metrics::gauge!("gpu_mem.local_peak_bytes", "module" => label).set(local_peak as f64);
        metrics::gauge!("gpu_mem.reserved_bytes", "module" => label).set(reserved as f64);
    }

    #[inline]
    pub fn tracing_info(&self, msg: impl Into<Option<&'static str>>) {
        let Some(manager) = managers()
            .lock()
            .ok()
            .and_then(|managers| managers.get(&self.device_id).cloned())
        else {
            tracing::error!("Memory manager not available");
            return;
        };
        let Ok(manager) = manager.lock() else { return };
        let current = manager.current_size;
        let peak = manager.max_used_size;
        let used = current as isize - self.current as isize;
        let sign = if used >= 0 { "+" } else { "-" };
        let pool_usage = manager.pool.memory_usage();
        tracing::info!(
            "GPU mem: used={}{}, current={}, peak={}, in pool={} ({})",
            sign,
            ByteSize::b(used.unsigned_abs() as u64),
            ByteSize::b(current as u64),
            ByteSize::b(peak as u64),
            ByteSize::b(pool_usage as u64),
            msg.into()
                .map_or(self.label.to_string(), |m| format!("{}:{}", self.label, m))
        );
    }

    pub fn reset_peak(&mut self) {
        let manager = managers()
            .lock()
            .ok()
            .and_then(|managers| managers.get(&self.device_id).cloned());
        if let Some(manager) = manager {
            if let Ok(mut manager) = manager.lock() {
                manager.max_used_size = manager.current_size;
            }
        }
    }
}

impl Drop for MemTracker {
    fn drop(&mut self) {
        self.tracing_info(None);
    }
}
