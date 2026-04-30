#pragma once
/**
 * cupti_collector.h — GB10-aware CUPTI Activity collector
 *
 * Four-layer hybrid model:
 *   CUPTI  → what CUDA did (kernel timeline, memcpy, runtime activity)
 *   PTX    → what memory felt like (latency, bandwidth, contention)
 *   SPBM   → what the platform reacted with (power, thermal, PROCHOT)
 *   PSI    → whether the system stalled
 *
 * GB10 confirmed kind map (dustin1925, CUDA 13.0, driver 580.142):
 *   ENABLED:  KERNEL, MEMCPY, MEMSET, DEVICE, CONTEXT, RUNTIME, DRIVER,
 *             OVERHEAD, SYNCHRONIZATION, MEMORY2, NVLINK, PCIE, ENVIRONMENT
 *   SKIP:     UNIFIED_MEMORY_COUNTER  (CUPTI_ERROR_NOT_READY — structural,
 *                                      no UVM faults on hardware-coherent UMA)
 *             CONCURRENT_KERNEL       (CUPTI_ERROR_NOT_COMPATIBLE)
 *             INSTRUCTION_EXECUTION   (CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)
 *
 * Design: soft-fail per kind — never abort if one kind fails.
 * NVLINK enables cleanly but records 0 on synthetic workload.
 * Rerun under real inference load to determine if C2C traffic populates it.
 */

#include <cupti.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Kind status ──────────────────────────────────────────────────────────── */
typedef enum {
    KIND_STATUS_ENABLED   = 0,
    KIND_STATUS_SKIPPED   = 1,   /* known unsupported on GB10 */
    KIND_STATUS_FAILED    = 2,   /* unexpected failure */
    KIND_STATUS_DISABLED  = 3,   /* not attempted */
} CuptiKindStatus;

typedef struct {
    CUpti_ActivityKind  kind;
    CuptiKindStatus     status;
    CUptiResult         result;
    uint64_t            records;
} CuptiKindInfo;

/* ── Phase record ─────────────────────────────────────────────────────────── */
typedef struct {
    const char*  name;
    uint64_t     ts_start_ns;
    uint64_t     ts_end_ns;
    uint64_t     kernel_count;
    uint64_t     memcpy_bytes;
    uint64_t     memset_count;
    uint64_t     runtime_count;
    uint64_t     driver_count;
    uint64_t     memory2_count;
    uint64_t     nvlink_records;   /* 0 on synthetic — needs inference validation */
    uint64_t     sync_count;
    uint64_t     overhead_ns;
} CuptiPhase;

/* ── Collector context ────────────────────────────────────────────────────── */
typedef struct {
    bool            initialized;
    bool            hw_coherent_uma;    /* GB10 — skip UNIFIED_MEMORY_COUNTER */
    CuptiKindInfo   kinds[32];
    int             kind_count;
    CuptiPhase      phases[16];
    int             phase_count;
    CuptiPhase*     current_phase;
} CuptiCollector;

/* ── API ──────────────────────────────────────────────────────────────────── */

/**
 * Initialize collector. Detects GB10 HW-coherent UMA automatically.
 * Enables all supported kinds. Soft-fails on unsupported kinds.
 * Returns 0 on success, -1 on fatal init failure.
 */
int  cupti_collector_init(CuptiCollector* col);

/**
 * Begin a named measurement phase (e.g. "uma_bw_run1", "contention_sweep").
 * Flushes pending records from previous phase first.
 */
void cupti_collector_phase_begin(CuptiCollector* col, const char* name);

/**
 * End current phase. Flushes and records counts.
 */
void cupti_collector_phase_end(CuptiCollector* col);

/**
 * Flush all pending CUPTI activity buffers.
 */
void cupti_collector_flush(CuptiCollector* col);

/**
 * Print per-kind status table and per-phase summary to stdout.
 */
void cupti_collector_print(const CuptiCollector* col);

/**
 * Emit phases as JSON to file. Integrates with timeline.json.
 */
int  cupti_collector_write_json(const CuptiCollector* col, const char* path);

/**
 * Teardown — disable kinds, free buffers.
 */
void cupti_collector_destroy(CuptiCollector* col);

#ifdef __cplusplus
}
#endif
