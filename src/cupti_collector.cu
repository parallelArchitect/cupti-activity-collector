/**
 * cupti_collector.cu — GB10-aware CUPTI Activity collector implementation
 *
 * Confirmed GB10 kind map: dustin1925, CUDA 13.0, driver 580.142
 * See include/cupti_collector.h for full design notes.
 */

#include "cupti_collector.h"
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Buffer config ────────────────────────────────────────────────────────── */
#define BUFFER_SIZE        (32 * 1024)
#define BUFFER_ALIGN       (8)
#define MAX_KINDS          32

/* ── Internal state ───────────────────────────────────────────────────────── */
static CuptiCollector* g_col = NULL;

/* ── Helpers ──────────────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static const char* kind_name(CUpti_ActivityKind k) {
    switch (k) {
        case CUPTI_ACTIVITY_KIND_KERNEL:           return "KERNEL";
        case CUPTI_ACTIVITY_KIND_MEMCPY:           return "MEMCPY";
        case CUPTI_ACTIVITY_KIND_MEMSET:           return "MEMSET";
        case CUPTI_ACTIVITY_KIND_DEVICE:           return "DEVICE";
        case CUPTI_ACTIVITY_KIND_CONTEXT:          return "CONTEXT";
        case CUPTI_ACTIVITY_KIND_RUNTIME:          return "RUNTIME";
        case CUPTI_ACTIVITY_KIND_DRIVER:           return "DRIVER";
        case CUPTI_ACTIVITY_KIND_OVERHEAD:         return "OVERHEAD";
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:  return "SYNCHRONIZATION";
        case CUPTI_ACTIVITY_KIND_MEMORY2:          return "MEMORY2";
        case CUPTI_ACTIVITY_KIND_NVLINK:           return "NVLINK";
        case CUPTI_ACTIVITY_KIND_PCIE:             return "PCIE";
        case CUPTI_ACTIVITY_KIND_ENVIRONMENT:      return "ENVIRONMENT";
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: return "UNIFIED_MEMORY_COUNTER";
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "CONCURRENT_KERNEL";
        case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION: return "INSTRUCTION_EXECUTION";
        default:                                   return "UNKNOWN";
    }
}

/* ── GB10 skip list ───────────────────────────────────────────────────────── */
/* These are structurally absent or incompatible on GB10 HW-coherent UMA.     */
/* Confirmed: dustin1925 sweep, CUDA 13.0, driver 580.142                     */
static const CUpti_ActivityKind GB10_SKIP[] = {
    CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,  /* NOT_READY — no UVM faults */
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,        /* NOT_COMPATIBLE */
    CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION,    /* LEGACY_PROFILER_NOT_SUPPORTED */
};
#define GB10_SKIP_COUNT (sizeof(GB10_SKIP) / sizeof(GB10_SKIP[0]))

/* ── Enabled kind list ────────────────────────────────────────────────────── */
static const CUpti_ActivityKind ENABLED_KINDS[] = {
    CUPTI_ACTIVITY_KIND_KERNEL,
    CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET,
    CUPTI_ACTIVITY_KIND_DEVICE,
    CUPTI_ACTIVITY_KIND_CONTEXT,
    CUPTI_ACTIVITY_KIND_RUNTIME,
    CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_OVERHEAD,
    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
    CUPTI_ACTIVITY_KIND_MEMORY2,
    CUPTI_ACTIVITY_KIND_NVLINK,   /* enables OK, 0 records on synthetic */
    CUPTI_ACTIVITY_KIND_PCIE,
    CUPTI_ACTIVITY_KIND_ENVIRONMENT,
};
#define ENABLED_KIND_COUNT (sizeof(ENABLED_KINDS) / sizeof(ENABLED_KINDS[0]))

static bool is_gb10_skip(CUpti_ActivityKind k) {
    for (int i = 0; i < (int)GB10_SKIP_COUNT; i++)
        if (GB10_SKIP[i] == k) return true;
    return false;
}

/* ── CUPTI buffer callbacks ───────────────────────────────────────────────── */
static void CUPTIAPI buffer_requested(uint8_t** buf, size_t* size,
                                       size_t* max_num_records) {
    uint8_t* raw = (uint8_t*)malloc(BUFFER_SIZE + BUFFER_ALIGN);
    if (!raw) { *buf = NULL; *size = 0; return; }
    *buf = (uint8_t*)(((uintptr_t)raw + BUFFER_ALIGN - 1) & ~((uintptr_t)(BUFFER_ALIGN - 1)));
    *size = BUFFER_SIZE;
    *max_num_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id,
                                       uint8_t* buf, size_t size,
                                       size_t valid_size) {
    CUpti_Activity* rec = NULL;
    CUptiResult status;

    if (!g_col || !g_col->current_phase) goto done;
    do {
        status = cuptiActivityGetNextRecord(buf, valid_size, &rec);
        if (status != CUPTI_SUCCESS) break;

        CuptiPhase* p = g_col->current_phase;

        switch (rec->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                p->kernel_count++;
                /* update kind record count */
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_KERNEL)
                        g_col->kinds[i].records++;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy* m = (CUpti_ActivityMemcpy*)rec;
                p->memcpy_bytes += m->bytes;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_MEMCPY)
                        g_col->kinds[i].records++;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET:
                p->memset_count++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_MEMSET)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_RUNTIME:
                p->runtime_count++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_RUNTIME)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_DRIVER:
                p->driver_count++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_DRIVER)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_MEMORY2:
                p->memory2_count++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_MEMORY2)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_NVLINK:
                p->nvlink_records++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_NVLINK)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
                p->sync_count++;
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)
                        g_col->kinds[i].records++;
                break;
            case CUPTI_ACTIVITY_KIND_OVERHEAD: {
                CUpti_ActivityOverhead* o = (CUpti_ActivityOverhead*)rec;
                p->overhead_ns += (o->end - o->start);
                for (int i = 0; i < g_col->kind_count; i++)
                    if (g_col->kinds[i].kind == CUPTI_ACTIVITY_KIND_OVERHEAD)
                        g_col->kinds[i].records++;
                break;
            }
            default:
                break;
        }
    } while (1);

done:
    free(buf);
}

/* ── Init ─────────────────────────────────────────────────────────────────── */
int cupti_collector_init(CuptiCollector* col) {
    memset(col, 0, sizeof(*col));
    g_col = col;

    /* Detect GB10 hardware-coherent UMA */
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        /* GB10 = SM 12.1, hardware coherent */
        col->hw_coherent_uma = (prop.major == 12 && prop.minor == 1);
    }

    /* Register buffer callbacks */
    CUptiResult r = cuptiActivityRegisterCallbacks(buffer_requested,
                                                   buffer_completed);
    if (r != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti_collector] registerCallbacks failed: %d\n", r);
        return -1;
    }

    /* Enable kinds — soft fail per kind */
    col->kind_count = 0;
    for (int i = 0; i < (int)ENABLED_KIND_COUNT; i++) {
        CUpti_ActivityKind k = ENABLED_KINDS[i];
        CuptiKindInfo* info = &col->kinds[col->kind_count++];
        info->kind    = k;
        info->records = 0;

        if (col->hw_coherent_uma && is_gb10_skip(k)) {
            info->status = KIND_STATUS_SKIPPED;
            info->result = CUPTI_SUCCESS;
            continue;
        }

        info->result = cuptiActivityEnable(k);
        if (info->result == CUPTI_SUCCESS) {
            info->status = KIND_STATUS_ENABLED;
        } else {
            info->status = KIND_STATUS_FAILED;
            fprintf(stderr, "[cupti_collector] %s failed: %d (non-fatal)\n",
                    kind_name(k), info->result);
        }
    }

    /* Also record the known-skip kinds for documentation */
    if (col->hw_coherent_uma) {
        for (int i = 0; i < (int)GB10_SKIP_COUNT; i++) {
            CuptiKindInfo* info = &col->kinds[col->kind_count++];
            info->kind    = GB10_SKIP[i];
            info->status  = KIND_STATUS_SKIPPED;
            info->result  = CUPTI_SUCCESS;
            info->records = 0;
        }
    }

    col->initialized = true;
    return 0;
}

/* ── Phase management ─────────────────────────────────────────────────────── */
void cupti_collector_phase_begin(CuptiCollector* col, const char* name) {
    if (!col->initialized) return;
    cupti_collector_flush(col);

    if (col->phase_count >= 16) return;
    CuptiPhase* p = &col->phases[col->phase_count++];
    memset(p, 0, sizeof(*p));
    p->name         = name;
    p->ts_start_ns  = now_ns();
    col->current_phase = p;
}

void cupti_collector_phase_end(CuptiCollector* col) {
    if (!col->initialized || !col->current_phase) return;
    cupti_collector_flush(col);
    col->current_phase->ts_end_ns = now_ns();
    col->current_phase = NULL;
}

void cupti_collector_flush(CuptiCollector* col) {
    if (!col->initialized) return;
    cuptiActivityFlushAll(0);
}

/* ── Print ────────────────────────────────────────────────────────────────── */
void cupti_collector_print(const CuptiCollector* col) {
    printf("\n=== CUPTI Collector — Kind Status ===\n");
    printf("Platform : %s\n",
           col->hw_coherent_uma ? "HARDWARE_COHERENT_UMA (GB10)" : "standard");
    printf("%-30s %-10s %-8s\n", "KIND", "STATUS", "RECORDS");
    printf("%-30s %-10s %-8s\n",
           "------------------------------", "----------", "--------");
    for (int i = 0; i < col->kind_count; i++) {
        const CuptiKindInfo* k = &col->kinds[i];
        const char* st = k->status == KIND_STATUS_ENABLED  ? "OK" :
                         k->status == KIND_STATUS_SKIPPED  ? "SKIPPED" :
                         k->status == KIND_STATUS_FAILED   ? "FAILED" : "OFF";
        printf("%-30s %-10s %8llu\n", kind_name(k->kind), st,
               (unsigned long long)k->records);
    }

    if (col->phase_count > 0) {
        printf("\n=== Phase Summary ===\n");
        for (int i = 0; i < col->phase_count; i++) {
            const CuptiPhase* p = &col->phases[i];
            double dur_ms = (p->ts_end_ns - p->ts_start_ns) / 1e6;
            printf("Phase: %s  (%.2f ms)\n", p->name, dur_ms);
            printf("  kernels=%-6llu  memcpy_bytes=%-12llu  memset=%-4llu\n",
                   (unsigned long long)p->kernel_count,
                   (unsigned long long)p->memcpy_bytes,
                   (unsigned long long)p->memset_count);
            printf("  runtime=%-6llu  driver=%-6llu  sync=%-4llu  "
                   "memory2=%-4llu  nvlink=%-4llu\n",
                   (unsigned long long)p->runtime_count,
                   (unsigned long long)p->driver_count,
                   (unsigned long long)p->sync_count,
                   (unsigned long long)p->memory2_count,
                   (unsigned long long)p->nvlink_records);
            printf("  overhead=%.3f ms\n", p->overhead_ns / 1e6);
        }
    }
}

/* ── JSON output ──────────────────────────────────────────────────────────── */
int cupti_collector_write_json(const CuptiCollector* col, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"platform\": \"%s\",\n",
            col->hw_coherent_uma ? "HARDWARE_COHERENT_UMA" : "standard");
    fprintf(f, "  \"hw_coherent_uma\": %s,\n",
            col->hw_coherent_uma ? "true" : "false");

    fprintf(f, "  \"kinds\": [\n");
    for (int i = 0; i < col->kind_count; i++) {
        const CuptiKindInfo* k = &col->kinds[i];
        const char* st = k->status == KIND_STATUS_ENABLED  ? "enabled"  :
                         k->status == KIND_STATUS_SKIPPED  ? "skipped"  :
                         k->status == KIND_STATUS_FAILED   ? "failed"   : "disabled";
        fprintf(f, "    {\"kind\": \"%s\", \"status\": \"%s\", "
                   "\"records\": %llu}%s\n",
                kind_name(k->kind), st,
                (unsigned long long)k->records,
                i < col->kind_count - 1 ? "," : "");
    }
    fprintf(f, "  ],\n");

    fprintf(f, "  \"phases\": [\n");
    for (int i = 0; i < col->phase_count; i++) {
        const CuptiPhase* p = &col->phases[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"name\": \"%s\",\n", p->name);
        fprintf(f, "      \"ts_start_ns\": %llu,\n",
                (unsigned long long)p->ts_start_ns);
        fprintf(f, "      \"ts_end_ns\": %llu,\n",
                (unsigned long long)p->ts_end_ns);
        fprintf(f, "      \"kernel_count\": %llu,\n",
                (unsigned long long)p->kernel_count);
        fprintf(f, "      \"memcpy_bytes\": %llu,\n",
                (unsigned long long)p->memcpy_bytes);
        fprintf(f, "      \"memset_count\": %llu,\n",
                (unsigned long long)p->memset_count);
        fprintf(f, "      \"runtime_count\": %llu,\n",
                (unsigned long long)p->runtime_count);
        fprintf(f, "      \"driver_count\": %llu,\n",
                (unsigned long long)p->driver_count);
        fprintf(f, "      \"memory2_count\": %llu,\n",
                (unsigned long long)p->memory2_count);
        fprintf(f, "      \"nvlink_records\": %llu,\n",
                (unsigned long long)p->nvlink_records);
        fprintf(f, "      \"sync_count\": %llu,\n",
                (unsigned long long)p->sync_count);
        fprintf(f, "      \"overhead_ns\": %llu\n",
                (unsigned long long)p->overhead_ns);
        fprintf(f, "    }%s\n", i < col->phase_count - 1 ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

/* ── Destroy ──────────────────────────────────────────────────────────────── */
void cupti_collector_destroy(CuptiCollector* col) {
    if (!col->initialized) return;
    cuptiActivityFlushAll(0);
    for (int i = 0; i < col->kind_count; i++) {
        if (col->kinds[i].status == KIND_STATUS_ENABLED)
            cuptiActivityDisable(col->kinds[i].kind);
    }
    memset(col, 0, sizeof(*col));
    g_col = NULL;
}
