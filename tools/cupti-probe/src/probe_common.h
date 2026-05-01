// probe_common.h â€” shared scaffolding for cupti-probe subcommands.
//
// What lives here:
//  - error-handling macros (CUDA_CHECK, CUPTI_LOG_SOFT)
//  - hardware introspection banner (printed at the top of every subcommand)
//  - activity-kind name table + lookup
//  - buffer callback skeleton (subcommands customize the record handler)
//  - JSON helpers (output dir, write_results, etc.)
//
// Subcommand source files include this header and implement an `int run(...)`
// function. main.cu dispatches based on argv[1].

#pragma once

#include <cuda_runtime.h>
#include <cupti.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,           \
                         __LINE__, cudaGetErrorString(_err));                  \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

#define CUPTI_LOG_SOFT(call, label)                                           \
    do {                                                                      \
        CUptiResult _r = (call);                                               \
        const char *_msg = nullptr;                                           \
        cuptiGetResultString(_r, &_msg);                                      \
        std::fprintf(stderr, "[%s] %s -> %s (code=%d)\n", label,              \
                     #call, _msg ? _msg : "?", (int)_r);                      \
    } while (0)

// All activity kinds we know how to talk about by name.
struct KindInfo {
    CUpti_ActivityKind kind;
    const char *name;
};

// Keep this list in sync with the CUPTI version we target. Order is
// stable so subcommand output formatting can rely on it.
inline const std::vector<KindInfo> &all_kinds() {
    static const std::vector<KindInfo> kinds = {
        {CUPTI_ACTIVITY_KIND_KERNEL,                 "KERNEL"},
        {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,      "CONCURRENT_KERNEL"},
        {CUPTI_ACTIVITY_KIND_MEMCPY,                 "MEMCPY"},
        {CUPTI_ACTIVITY_KIND_MEMSET,                 "MEMSET"},
        {CUPTI_ACTIVITY_KIND_DEVICE,                 "DEVICE"},
        {CUPTI_ACTIVITY_KIND_CONTEXT,                "CONTEXT"},
        {CUPTI_ACTIVITY_KIND_RUNTIME,                "RUNTIME"},
        {CUPTI_ACTIVITY_KIND_DRIVER,                 "DRIVER"},
        {CUPTI_ACTIVITY_KIND_OVERHEAD,               "OVERHEAD"},
        {CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,        "SYNCHRONIZATION"},
        {CUPTI_ACTIVITY_KIND_MEMORY2,                "MEMORY2"},
        {CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER, "UNIFIED_MEMORY_COUNTER"},
        {CUPTI_ACTIVITY_KIND_NVLINK,                 "NVLINK"},
        {CUPTI_ACTIVITY_KIND_PCIE,                   "PCIE"},
        {CUPTI_ACTIVITY_KIND_ENVIRONMENT,            "ENVIRONMENT"},
        {CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION,  "INSTRUCTION_EXECUTION"},
    };
    return kinds;
}

inline CUpti_ActivityKind kind_from_name(const std::string &name,
                                          bool *ok) {
    for (const auto &k : all_kinds()) {
        if (name == k.name) {
            if (ok) *ok = true;
            return k.kind;
        }
    }
    if (ok) *ok = false;
    return CUPTI_ACTIVITY_KIND_INVALID;
}

inline const char *name_from_kind(CUpti_ActivityKind k) {
    for (const auto &ki : all_kinds()) {
        if (ki.kind == k) return ki.name;
    }
    return "UNKNOWN";
}

// Hardware + library info printed at the top of every subcommand for
// reproducibility. Also returned as a JSON object for inclusion in result
// files.
struct PlatformInfo {
    std::string gpu_name;
    int sm_major{0};
    int sm_minor{0};
    bool hw_coherent_uma{false};
    int driver_version{0};         // e.g., 13000 for CUDA 13.0
    int runtime_version{0};
    uint32_t cupti_version{0};
};

inline PlatformInfo gather_platform() {
    PlatformInfo p;
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    p.gpu_name = prop.name;
    p.sm_major = prop.major;
    p.sm_minor = prop.minor;
    p.hw_coherent_uma = prop.pageableMemoryAccessUsesHostPageTables != 0;
    cudaDriverGetVersion(&p.driver_version);
    cudaRuntimeGetVersion(&p.runtime_version);
    cuptiGetVersion(&p.cupti_version);
    return p;
}

inline void print_banner(const char *subcommand, const PlatformInfo &p) {
    std::printf("=== cupti-probe %s ===\n", subcommand);
    std::printf("GPU      : %s (SM %d.%d)\n", p.gpu_name.c_str(), p.sm_major,
                p.sm_minor);
    std::printf("Coherent : %s\n",
                p.hw_coherent_uma ? "yes (hardware)" : "no/software");
    std::printf("CUDA     : driver=%d runtime=%d (e.g., 13000 = 13.0)\n",
                p.driver_version, p.runtime_version);
    std::printf("CUPTI    : %u\n", p.cupti_version);
    std::printf("\n");
}

// Pretty-print a small "did the system look stuck-low-power?" hint.
// True power-stuck detection is out of scope here; we emit a heuristic
// note to flag obvious cases and link to the upstream forum thread.
inline void print_health_hint() {
    // Nothing automatic right now â€” this is a placeholder. The full
    // power-watchdog tool is its own project (see SPARK_UMA_TRACE_SPEC.md
    // sibling crate spark-power-watchdog). For now, just remind the user.
    std::printf("Health   : if subsequent runs deviate sharply, check\n"
                "           nvidia-smi power draw â€” GB10 has a known\n"
                "           PD-stuck-low-power state requiring AC unplug.\n"
                "           See forums.developer.nvidia.com/t/.../361294\n\n");
}

// Plain-English explanation for common CUPTI / NVPW result codes.
// Returns a multi-line string with leading newline if there's anything
// helpful to say; empty string otherwise (caller can suppress).
//
// The goal: turn raw API codes like `CUPTI_ERROR_NOT_COMPATIBLE` into
// "this means X, here's whether you should care, here's what to try
// instead." Less mystery for non-experts, faster diagnosis for everyone.
inline std::string explain_cupti(CUptiResult code) {
    switch (code) {
        case CUPTI_SUCCESS:
            return "";
        case CUPTI_ERROR_NOT_READY:
            return "    why: the API recognizes the request but cannot fulfill\n"
                   "         it on this hardware. On GB10, this happens for\n"
                   "         UNIFIED_MEMORY_COUNTER because hardware-coherent\n"
                   "         UMA produces no UVM page faults â€” there's nothing\n"
                   "         to count. Not a config bug, not your fault.\n"
                   "    fix: use a custom bandwidth probe (uma_bw, sparkview)\n"
                   "         or the planned spark-uma-trace tool instead.";
        case CUPTI_ERROR_NOT_COMPATIBLE:
            return "    why: this CUPTI version doesn't support enabling this\n"
                   "         specific kind in combination with whatever's\n"
                   "         already enabled. Many platforms support either\n"
                   "         KERNEL or CONCURRENT_KERNEL but not both at once.\n"
                   "    fix: try the alternate kind (e.g. KERNEL instead of\n"
                   "         CONCURRENT_KERNEL). Not Spark-specific.";
        case CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED:
            return "    why: this kind belongs to the deprecated 'legacy\n"
                   "         profiler' subset. NVIDIA replaced it with the\n"
                   "         modern Profiler API (cuptiProfilerInitialize +\n"
                   "         NVPW). The Activity API path is intentionally\n"
                   "         broken for these kinds.\n"
                   "    fix: use the Profiler API. See `cupti-probe profiler-*`\n"
                   "         subcommands.";
        case CUPTI_ERROR_INVALID_PARAMETER:
            return "    why: a function call had invalid arguments. Usually a\n"
                   "         struct size or device-index issue.\n"
                   "    fix: check the call site â€” most often the struct's\n"
                   "         _STRUCT_SIZE field needs initializing.";
        case CUPTI_ERROR_NOT_INITIALIZED:
            return "    why: an API was called before its initializer.\n"
                   "    fix: ensure cuptiProfilerInitialize / similar is\n"
                   "         called first.";
        case CUPTI_ERROR_INSUFFICIENT_PRIVILEGES:
            return "    why: this driver requires elevated privileges to\n"
                   "         enable the requested counter access.\n"
                   "    fix: run with sudo, or set the kernel module option\n"
                   "         NVreg_RestrictProfilingToAdminUsers=0.";
        default:
            return "";
    }
}

// NVPA_Status values are not easily stringified by NVPW; we map the
// common ones manually.
inline std::string explain_nvpa(int nvpa_status) {
    switch (nvpa_status) {
        case 0: // NVPA_STATUS_SUCCESS
            return "";
        case 1: // NVPA_STATUS_ERROR (generic)
            return "    why: a generic NVPW host or target error.\n"
                   "    fix: check the call's argument struct sizes are\n"
                   "         initialized via the matching _STRUCT_SIZE\n"
                   "         constant. Usually a 0-init bug.";
        case 8: // NVPA_STATUS_NOT_SUPPORTED (approximate; verify against header)
            return "    why: NVPW does not support this operation on this\n"
                   "         chip. Often means NVPW's metrics database has\n"
                   "         no entry for the chip â€” common on brand-new\n"
                   "         silicon, NVPW lags hardware by ~1 release.\n"
                   "    fix: wait for NVPW update, or use Activity API kinds\n"
                   "         (KERNEL, MEMCPY, etc.) which are independent.";
        default: {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "    why: NVPA_Status=%d. Check nvperf_host.h for\n"
                          "         the exact meaning.\n", nvpa_status);
            return std::string(buf);
        }
    }
}

inline std::string platform_json(const PlatformInfo &p) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "  \"platform\": {\n"
        "    \"gpu_name\": \"%s\",\n"
        "    \"sm_major\": %d,\n"
        "    \"sm_minor\": %d,\n"
        "    \"hw_coherent_uma\": %s,\n"
        "    \"driver_version\": %d,\n"
        "    \"runtime_version\": %d,\n"
        "    \"cupti_version\": %u\n"
        "  }",
        p.gpu_name.c_str(), p.sm_major, p.sm_minor,
        p.hw_coherent_uma ? "true" : "false",
        p.driver_version, p.runtime_version, p.cupti_version);
    return std::string(buf);
}
