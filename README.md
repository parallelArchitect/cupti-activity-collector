# `cupti-activity-collector`

**GB10-aware CUPTI Activity collector**
Runtime kind detection, phase management, and JSON output for hardware-coherent UMA platforms

---

## Overview

`cupti-activity-collector` is a reusable CUPTI Activity instrumentation library for NVIDIA GB10 (DGX Spark) and other platforms.

It provides:

- Runtime GB10 detection via SM version check
- Automatic skip list for kinds confirmed absent on GB10
- Named measurement phases with per-phase record counts
- Per-kind record counters for all working kinds
- JSON output compatible with `timeline.json` and `events.json`
- Soft-fail per kind — never aborts if one kind fails

---

## Background

On GB10 (SM 12.1, hardware-coherent UMA), CUPTI Activity is largely functional. Three kinds are structurally absent:

```
UNIFIED_MEMORY_COUNTER  — CUPTI_ERROR_NOT_READY
                          no UVM faults to count on hardware-coherent UMA
CONCURRENT_KERNEL       — CUPTI_ERROR_NOT_COMPATIBLE
INSTRUCTION_EXECUTION   — CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED
```

All other kinds collect records normally. This library enables only the confirmed working kinds and skips the three absent ones automatically at runtime.

The confirmed kind map was established by dustin1925's `cupti_kind_sweep` and validated independently on two GB10 units (CUDA 13.0, driver 580.142).

---

## Confirmed Kind Map — GB10

| KIND | STATUS | dustin1925 | azampatti |
|------|--------|-----------|-----------|
| KERNEL | OK | 1 record | 1 record |
| MEMCPY | OK | 2 records | 2 records |
| MEMSET | OK | 1 record | 0 records |
| DEVICE | OK | 0 records | 0 records |
| CONTEXT | OK | 1 record | 0 records |
| RUNTIME | OK | 7 records | 9 records |
| DRIVER | OK | 4 records | 3 records |
| OVERHEAD | OK | 11 records | 14 records |
| SYNCHRONIZATION | OK | 1 record | 2 records |
| MEMORY2 | OK | 2 records | 4 records |
| NVLINK | OK | 0 records* | 0 records* |
| PCIE | OK | 0 records | 0 records |
| ENVIRONMENT | OK | 33 records | 0 records |
| UNIFIED_MEMORY_COUNTER | SKIPPED | — | — |
| CONCURRENT_KERNEL | SKIPPED | — | — |
| INSTRUCTION_EXECUTION | SKIPPED | — | — |

*NVLINK records 0 on synthetic workload — validation under real inference load pending.

Record count variance between units is expected — workload timing and system state differ. The kind map is identical across both units.

---

## API

```c
/* Initialize collector — detects GB10, enables confirmed kinds */
int  cupti_collector_init(CuptiCollector* col);

/* Begin a named measurement phase */
void cupti_collector_phase_begin(CuptiCollector* col, const char* name);

/* End current phase — flushes and records counts */
void cupti_collector_phase_end(CuptiCollector* col);

/* Flush all pending CUPTI activity buffers */
void cupti_collector_flush(CuptiCollector* col);

/* Print kind status table and phase summary */
void cupti_collector_print(const CuptiCollector* col);

/* Emit phases as JSON */
int  cupti_collector_write_json(const CuptiCollector* col, const char* path);

/* Teardown */
void cupti_collector_destroy(CuptiCollector* col);
```

---

## Build

**aarch64 (GB10 DGX Spark) — CUDA 13.0 required:**

```bash
/usr/local/cuda-13.0/bin/nvcc -O2 -std=c++17 -I./include \
    src/cupti_collector.cu src/cupti_collector_test.cu \
    -o cupti_collector_test -lcudart -lcupti
```

> **Note:** CUDA 13.1 produces broken event timing on GB10. Always use 13.0 on GB10 systems.

---

## Run

```bash
./cupti_collector_test
```

Expected output on GB10:

```
=== cupti_collector smoke test ===

=== CUPTI Collector — Kind Status ===
Platform : HARDWARE_COHERENT_UMA (GB10)
KIND                           STATUS     RECORDS
------------------------------ ---------- --------
KERNEL                         OK                1
MEMCPY                         OK                2
RUNTIME                        OK                9
DRIVER                         OK                3
OVERHEAD                       OK               14
SYNCHRONIZATION                OK                2
MEMORY2                        OK                4
NVLINK                         OK                0
UNIFIED_MEMORY_COUNTER         SKIPPED           0
CONCURRENT_KERNEL              SKIPPED           0
INSTRUCTION_EXECUTION          SKIPPED           0

=== Phase Summary ===
Phase: synthetic_kernel  (183.74 ms)
  kernels=1  memcpy_bytes=0  runtime=4  driver=3  sync=1  memory2=2
Phase: memcpy_test  (0.73 ms)
  kernels=0  memcpy_bytes=8192  runtime=5  driver=0  sync=1  memory2=2
```

---

## Known GB10 API Gaps

| API | Status |
|-----|--------|
| CUPTI UNIFIED_MEMORY_COUNTER | Structurally absent — no UVM faults on HW-coherent UMA |
| NVML memory clock | Not exposed (returns N/A) |
| Nsight Systems UVM trace | Unsupported on GB10 |
| NVLINK CUPTI records | 0 on synthetic — inference validation pending |

---

## Attribution

- dustin1925 — reviewed output from [cupti-uma-probe issue #1](https://github.com/parallelArchitect/cupti-uma-probe/issues/1) and built `cupti_kind_sweep` to resolve whether CUPTI failure on GB10 was broad or specific to UVM counters. His sweep established the confirmed GB10 Activity kind map this library is built on.
- azampatti — independent hardware validation on a second GB10 unit
- parallelArchitect — production library implementation based on the confirmed kind map

---

## Author

parallelArchitect
Human-directed GPU engineering with AI assistance

---

## Related

- [cupti-uma-probe](https://github.com/parallelArchitect/cupti-uma-probe) — UVM event collection diagnostic, original CUPTI_ERROR_NOT_READY finding
- [gb10-uma-diagnostics](https://github.com/parallelArchitect/gb10-uma-diagnostics) — full GB10 diagnostic suite, integrates this collector
- [cuda-unified-memory-analyzer](https://github.com/parallelArchitect/cuda-unified-memory-analyzer) — CUDA Unified Memory Analyzer: a hardware-aware CUDA diagnostic tool for analyzing Unified Memory migration behavior, residency stability, and transport performance on NVIDIA GPUs

---

## License

MIT
