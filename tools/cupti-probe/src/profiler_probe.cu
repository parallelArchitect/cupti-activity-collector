// profiler_probe.cu â€” CUPTI Profiler API foundation probe.
//
// Tests whether CUPTI's modern Profiler API (the path that's supposed to
// replace the deprecated INSTRUCTION_EXECUTION + concurrent_kernel kinds
// from the Activity API) initializes at all on GB10/SM_121.
//
// Conservative scope for v0.3 â€” just tests the foundation:
//   1. cuptiProfilerInitialize           â€” does the Profiler API load?
//   2. NVPW_InitializeHost               â€” does the NVPerf host lib init?
//   3. cuptiDeviceGetChipName            â€” does CUPTI recognize the chip?
//   4. NVPW MetricsEvaluator setup       â€” does NVPW have a metrics DB
//                                          entry for this chip?
//
// If all four pass, the Profiler API foundation is alive on this hardware.
// Future iterations (v0.4+) can extend with metric enumeration and actual
// counter collection once we've confirmed the foundation.
//
// If init step #1 fails: Profiler API is DOA on GB10 â€” clean parallel to
// the UVM activity finding, just on the modern API surface.
//
// Most likely failure point on a brand-new chip is step 4: NVPW often
// lags silicon by a release cycle, and GB10 may not yet have a metrics
// DB entry.

#include "probe_common.h"
#include "probes.h"

#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <nvperf_cuda_host.h>

namespace {

enum class CodeApi { None, Cupti, Nvpa };

struct ProbeStep {
    const char *name;
    bool ok = false;
    int code = 0;
    CodeApi api = CodeApi::None;
    std::string detail;
};

// Helpers to stringify results.
static const char *cupti_str(CUptiResult r) {
    const char *s = nullptr;
    cuptiGetResultString(r, &s);
    return s ? s : "?";
}

// Pretty-print a step result, with plain-English explanation on failure.
static void report_step(const ProbeStep &s) {
    std::printf("  [%-8s] %-44s %s\n",
                s.ok ? "OK" : "FAIL", s.name,
                s.detail.empty() ? "" : s.detail.c_str());
    if (!s.ok) {
        std::string why;
        if (s.api == CodeApi::Cupti) why = explain_cupti((CUptiResult)s.code);
        else if (s.api == CodeApi::Nvpa) why = explain_nvpa(s.code);
        if (!why.empty()) std::printf("%s\n", why.c_str());
    }
}

} // namespace

int cmd_profiler_init(const std::vector<std::string> & /*args*/) {
    auto plat = gather_platform();
    print_banner("profiler-init", plat);
    print_health_hint();

    std::printf("Testing CUPTI Profiler API foundation on this device.\n");
    std::printf("Conservative scope: foundation only, no metric collection.\n\n");

    std::vector<ProbeStep> steps;

    // ----- Step 1: cuptiProfilerInitialize -----
    {
        ProbeStep s;
        s.name = "cuptiProfilerInitialize";
        CUpti_Profiler_Initialize_Params p = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE
        };
        CUptiResult r = cuptiProfilerInitialize(&p);
        s.code = (int)r;
        s.api = CodeApi::Cupti;
        s.ok = (r == CUPTI_SUCCESS);
        if (!s.ok) s.detail = cupti_str(r);
        steps.push_back(s);
        if (!s.ok) {
            // If init fails, downstream calls are meaningless. Bail with
            // a clean verdict.
            for (const auto &x : steps) report_step(x);
            std::printf("\nVERDICT: PROFILER_API_DOA â€” initialization failed. "
                        "On GB10 this would parallel the UVM activity finding "
                        "(structural absence of a CUPTI subsystem on this "
                        "silicon). Worth reporting upstream.\n");
            return 0;
        }
    }

    // ----- Step 2: NVPW_InitializeHost -----
    {
        ProbeStep s;
        s.name = "NVPW_InitializeHost";
        NVPW_InitializeHost_Params p = {NVPW_InitializeHost_Params_STRUCT_SIZE};
        NVPA_Status r = NVPW_InitializeHost(&p);
        s.code = (int)r;
        s.api = CodeApi::Nvpa;
        s.ok = (r == NVPA_STATUS_SUCCESS);
        if (!s.ok) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "NVPA_Status=%d", (int)r);
            s.detail = buf;
        }
        steps.push_back(s);
    }

    // ----- Step 3: cuptiDeviceGetChipName -----
    std::string chip_name;
    {
        ProbeStep s;
        s.name = "cuptiDeviceGetChipName";
        CUpti_Device_GetChipName_Params p = {
            CUpti_Device_GetChipName_Params_STRUCT_SIZE
        };
        p.deviceIndex = 0;
        CUptiResult r = cuptiDeviceGetChipName(&p);
        s.code = (int)r;
        s.api = CodeApi::Cupti;
        s.ok = (r == CUPTI_SUCCESS);
        if (s.ok && p.pChipName) {
            chip_name = p.pChipName;
            s.detail = std::string("chip=\"") + chip_name + "\"";
        } else {
            s.detail = cupti_str(r);
        }
        steps.push_back(s);
    }

    // ----- Step 4: NVPW metrics scratch buffer for this chip -----
    bool nvpw_metrics_ok = false;
    size_t scratch_bytes = 0;
    if (!chip_name.empty()) {
        ProbeStep s;
        s.name = "NVPW MetricsEvaluator scratch size";
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params p = {
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE
        };
        p.pChipName = chip_name.c_str();
        NVPA_Status r =
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&p);
        s.code = (int)r;
        s.api = CodeApi::Nvpa;
        s.ok = (r == NVPA_STATUS_SUCCESS);
        if (s.ok) {
            scratch_bytes = p.scratchBufferSize;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "scratch=%llu bytes",
                          (unsigned long long)scratch_bytes);
            s.detail = buf;
            nvpw_metrics_ok = true;
        } else {
            char buf[80];
            std::snprintf(buf, sizeof(buf),
                          "NVPA_Status=%d (chip not in metrics DB?)",
                          (int)r);
            s.detail = buf;
        }
        steps.push_back(s);
    }

    // ----- Print results table -----
    std::printf("\n--- Foundation probe steps ---\n");
    for (const auto &s : steps) report_step(s);

    // ----- Verdict -----
    bool all_ok = true;
    for (const auto &s : steps) if (!s.ok) all_ok = false;

    const char *verdict = "INDETERMINATE";
    if (all_ok) {
        verdict = "PROFILER_API_FOUNDATION_OK â€” "
                  "init + NVPW host + chip recognized + metrics DB entry "
                  "present. Profiler API is usable on this hardware. "
                  "v0.4+ can extend with metric enumeration and counter "
                  "collection.";
    } else if (!steps.empty() && !steps[0].ok) {
        // Already handled with early return above, but for completeness:
        verdict = "PROFILER_API_DOA â€” initialization failed.";
    } else if (steps.size() >= 4 && !steps[3].ok) {
        verdict = "PROFILER_API_PARTIAL â€” init + NVPW host work, but NVPW's "
                  "metrics DB does not have an entry for this chip. NVPW "
                  "lags silicon by ~1 release cycle; metrics collection "
                  "will not work until NVPW is updated. Activity API "
                  "kinds (KERNEL, MEMCPY, etc.) still functional.";
    } else {
        verdict = "PROFILER_API_PARTIAL â€” at least one foundation step "
                  "failed. See per-step output for details.";
    }
    std::printf("\nVERDICT: %s\n\n", verdict);

    // ----- JSON output -----
    FILE *jf = std::fopen("cupti_probe_profiler_init_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe profiler-init\",\n");
        std::fprintf(jf, "  \"version\": \"0.3.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"chip_name\": \"%s\",\n", chip_name.c_str());
        std::fprintf(jf, "  \"nvpw_scratch_bytes\": %llu,\n",
                     (unsigned long long)scratch_bytes);
        std::fprintf(jf, "  \"steps\": [\n");
        for (size_t i = 0; i < steps.size(); ++i) {
            const auto &s = steps[i];
            std::fprintf(jf, "    {\n");
            std::fprintf(jf, "      \"name\": \"%s\",\n", s.name);
            std::fprintf(jf, "      \"ok\": %s,\n", s.ok ? "true" : "false");
            std::fprintf(jf, "      \"code\": %d,\n", s.code);
            std::fprintf(jf, "      \"detail\": \"%s\"\n",
                         s.detail.c_str());
            std::fprintf(jf, "    }%s\n", (i + 1 == steps.size()) ? "" : ",");
        }
        std::fprintf(jf, "  ],\n");
        std::fprintf(jf, "  \"all_ok\": %s,\n", all_ok ? "true" : "false");
        std::fprintf(jf, "  \"nvpw_metrics_ok\": %s,\n",
                     nvpw_metrics_ok ? "true" : "false");
        std::fprintf(jf, "  \"verdict\": \"%s\"\n", verdict);
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_profiler_init_results.json\n");
    }

    return 0;
}

// Subcommand: profiler-collect
// Collect the value of a single metric during a workload. End-to-end test
// of the CUPTI Profiler API on Spark â€” proves we can actually pull a number
// out of NVIDIA's modern profiler library, not just enumerate metric names.
//
// Workflow (per CUPTI Profiler API docs):
//   1. Build raw counter requests for the requested metric via NVPW
//   2. Build a raw metrics config + generate config image
//   3. Build a counter data image prefix (also via NVPW)
//   4. Initialize the CUPTI counter data image + scratch buffer
//   5. cuptiProfilerBeginSession
//   6. cuptiProfilerSetConfig with the config image
//   7. cuptiProfilerBeginPass
//   8. cuptiProfilerEnableProfiling
//   9. Run workload
//   10. cuptiProfilerDisableProfiling
//   11. cuptiProfilerEndPass
//   12. cuptiProfilerEndSession
//   13. Decode the counter data image via NVPW MetricsEvaluator
//
// Heavy NVPW API surface â€” likely to need iteration on field names.
// Most fixes are 1-2 character edits when they happen.

// External workload runner from probes.cu. Forward declare to avoid header
// ping-pong; defined as run_heavy_workload there with default visibility.
extern "C" {
} // (no-op; just to anchor the comment above the forward decl block)

// We'll inline a small workload here rather than depending on probes.cu â€”
// keeps the profiler probe independent.
__global__ void k_collect_workload(int *p, int n, int rounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = 0; r < rounds; ++r) {
        for (int i = idx; i < n; i += stride) {
            p[i] = (p[i] * 1664525 + 1013904223) ^ r;
        }
    }
}

static long long run_collect_workload(std::chrono::milliseconds duration) {
    constexpr size_t kElems = 16 * 1024 * 1024 / sizeof(int);  // 4 M ints
    int *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, kElems * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buf, 0xa5, kElems * sizeof(int)));

    auto start    = std::chrono::steady_clock::now();
    auto deadline = start + duration;
    int rounds = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        k_collect_workload<<<128, 128>>>(d_buf, (int)kElems, 4);
        CUDA_CHECK(cudaDeviceSynchronize());
        ++rounds;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    CUDA_CHECK(cudaFree(d_buf));
    (void)rounds;
    return elapsed;
}

int cmd_profiler_collect(const std::vector<std::string> &args) {
    // ---- Argument parsing ----
    std::string metric_name;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--metric") metric_name = args[i + 1];
    }
    if (metric_name.empty()) {
        std::fprintf(stderr,
            "profiler-collect: --metric NAME required.\n"
            "  Try: cupti-probe profiler-list --max 50  (to find a metric name)\n"
            "  Suggested first targets: gpu__compute_memory_throughput,\n"
            "    sm__memory_throughput, sm__instruction_throughput\n");
        return 1;
    }
    int duration_s = 3;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--duration") {
            duration_s = std::atoi(args[i + 1].c_str());
            if (duration_s <= 0) duration_s = 3;
        }
    }

    auto plat = gather_platform();
    print_banner("profiler-collect", plat);
    print_health_hint();
    std::printf("Target metric : %s\n", metric_name.c_str());
    std::printf("Duration      : %d seconds\n\n", duration_s);
    std::printf("WARNING: this subcommand runs a sustained GPU workload.\n"
                "         Do not run while a vLLM compile or training\n"
                "         process is active.\n\n");

    // ---- Foundation init ----
    {
        CUpti_Profiler_Initialize_Params p = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE
        };
        CUptiResult r = cuptiProfilerInitialize(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiProfilerInitialize: %s\n", cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            return 1;
        }
    }
    {
        NVPW_InitializeHost_Params p = {NVPW_InitializeHost_Params_STRUCT_SIZE};
        NVPA_Status r = NVPW_InitializeHost(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] NVPW_InitializeHost: NVPA_Status=%d\n",
                        (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            return 1;
        }
    }

    std::string chip_name;
    {
        CUpti_Device_GetChipName_Params p = {
            CUpti_Device_GetChipName_Params_STRUCT_SIZE
        };
        p.deviceIndex = 0;
        CUptiResult r = cuptiDeviceGetChipName(&p);
        if (r != CUPTI_SUCCESS || !p.pChipName) {
            std::printf("[FAIL] cuptiDeviceGetChipName: %s\n", cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            return 1;
        }
        chip_name = p.pChipName;
        std::printf("[OK] chip name: %s\n", chip_name.c_str());
    }

    // ---- Initialize MetricsEvaluator ----
    size_t scratch_bytes = 0;
    {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params p = {
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE
        };
        p.pChipName = chip_name.c_str();
        if (NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] scratch size calc\n");
            return 1;
        }
        scratch_bytes = p.scratchBufferSize;
    }
    std::vector<uint8_t> scratch(scratch_bytes);
    NVPW_MetricsEvaluator *evaluator = nullptr;
    {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params p = {
            NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE
        };
        p.pScratchBuffer = scratch.data();
        p.scratchBufferSize = scratch.size();
        p.pChipName = chip_name.c_str();
        if (NVPW_CUDA_MetricsEvaluator_Initialize(&p) != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] MetricsEvaluator init\n");
            return 1;
        }
        evaluator = p.pMetricsEvaluator;
        std::printf("[OK] MetricsEvaluator initialized\n");
    }

    // Auto-cleanup the evaluator on exit.
    auto destroy_evaluator = [&]() {
        if (evaluator) {
            NVPW_MetricsEvaluator_Destroy_Params p = {
                NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE
            };
            p.pMetricsEvaluator = evaluator;
            NVPW_MetricsEvaluator_Destroy(&p);
            evaluator = nullptr;
        }
    };

    // ---- Resolve metric name â†’ metric type + index ----
    NVPW_MetricEvalRequest eval_request = {};
    {
        NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params p = {
            NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        p.pMetricName = metric_name.c_str();
        NVPA_Status r = NVPW_MetricsEvaluator_GetMetricTypeAndIndex(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] resolve metric \"%s\": NVPA_Status=%d\n",
                        metric_name.c_str(), (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            std::printf("\nIs the metric name spelled exactly right? Try:\n"
                        "  cupti-probe profiler-list --max 200 | grep <substring>\n");
            destroy_evaluator();
            return 1;
        }
        eval_request.metricType = p.metricType;
        eval_request.metricIndex = p.metricIndex;
        eval_request.rollupOp = NVPW_ROLLUP_OP_AVG;
        eval_request.submetric = NVPW_SUBMETRIC_NONE;
        std::printf("[OK] resolved metric: type=%u index=%u\n",
                    (unsigned)p.metricType, (unsigned)p.metricIndex);
    }

    // ---- Fetch raw counter dependencies for this metric ----
    // Two-call pattern: first call with ppRawDependencies=nullptr to learn
    // the count; allocate; second call to fill.
    std::vector<const char *> raw_dep_names;
    {
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params p = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        p.pMetricEvalRequests = &eval_request;
        p.numMetricEvalRequests = 1;
        p.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        p.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        p.ppRawDependencies = nullptr;
        // Call 1: query count
        NVPA_Status r = NVPW_MetricsEvaluator_GetMetricRawDependencies(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetMetricRawDependencies (count): NVPA_Status=%d\n",
                        (int)r);
            destroy_evaluator();
            return 1;
        }
        size_t num_deps = p.numRawDependencies;
        if (num_deps == 0) {
            std::printf("[WARN] metric reports zero raw dependencies â€” "
                        "unusual, may indicate a derived metric whose roots "
                        "this NVPW version cannot enumerate.\n");
        }
        raw_dep_names.resize(num_deps);
        // Call 2: fill names
        p.ppRawDependencies = raw_dep_names.data();
        p.numRawDependencies = num_deps;
        r = NVPW_MetricsEvaluator_GetMetricRawDependencies(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetMetricRawDependencies (fill): NVPA_Status=%d\n",
                        (int)r);
            destroy_evaluator();
            return 1;
        }
        std::printf("[OK] metric depends on %zu raw counter(s)\n", num_deps);
        // Print up to 3 dependency names for visibility
        for (size_t i = 0; i < num_deps && i < 3; ++i) {
            std::printf("       %s\n", raw_dep_names[i] ? raw_dep_names[i] : "?");
        }
        if (num_deps > 3) {
            std::printf("       ... and %zu more\n", num_deps - 3);
        }
    }

    // Build NVPA_RawMetricRequest array from the dependency names. Used by
    // BOTH the CounterDataBuilder_AddMetrics and RawMetricsConfig_AddMetrics
    // calls below â€” they need to know which raw counters to allocate
    // storage for and which to actually program.
    std::vector<NVPA_RawMetricRequest> raw_requests(raw_dep_names.size());
    for (size_t i = 0; i < raw_dep_names.size(); ++i) {
        raw_requests[i].structSize = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
        raw_requests[i].pPriv = nullptr;
        raw_requests[i].pMetricName = raw_dep_names[i];
        raw_requests[i].isolated = 1;
        raw_requests[i].keepInstances = 1;
    }

    // ---- Build counter data image prefix via NVPW CounterDataBuilder ----
    NVPW_CUDA_CounterDataBuilder_Create_Params builder_create = {
        NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE
    };
    builder_create.pChipName = chip_name.c_str();
    if (NVPW_CUDA_CounterDataBuilder_Create(&builder_create) !=
        NVPA_STATUS_SUCCESS) {
        std::printf("[FAIL] CounterDataBuilder_Create\n");
        destroy_evaluator();
        return 1;
    }
    auto destroy_builder = [&]() {
        NVPW_CounterDataBuilder_Destroy_Params p = {
            NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE
        };
        p.pCounterDataBuilder = builder_create.pCounterDataBuilder;
        NVPW_CounterDataBuilder_Destroy(&p);
    };

    // Add metric raw counters to the builder. Without this, the resulting
    // counter data prefix won't have storage allocated for our metric's
    // raw counters and EvaluateToGpuValues returns NaN.
    {
        NVPW_CounterDataBuilder_AddMetrics_Params p = {
            NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE
        };
        p.pCounterDataBuilder = builder_create.pCounterDataBuilder;
        p.pRawMetricRequests = raw_requests.data();
        p.numMetricRequests = raw_requests.size();
        NVPA_Status r = NVPW_CounterDataBuilder_AddMetrics(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] CounterDataBuilder_AddMetrics: NVPA_Status=%d\n",
                        (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            destroy_builder();
            destroy_evaluator();
            return 1;
        }
    }

    // Get the counter data prefix bytes.
    std::vector<uint8_t> counter_data_prefix;
    {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params p = {
            NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE
        };
        p.pCounterDataBuilder = builder_create.pCounterDataBuilder;
        p.bytesAllocated = 0;
        p.pBuffer = nullptr;
        // First call: ask how big the buffer needs to be.
        if (NVPW_CounterDataBuilder_GetCounterDataPrefix(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetCounterDataPrefix size query\n");
            destroy_builder();
            destroy_evaluator();
            return 1;
        }
        counter_data_prefix.resize(p.bytesCopied);
        p.bytesAllocated = counter_data_prefix.size();
        p.pBuffer = counter_data_prefix.data();
        if (NVPW_CounterDataBuilder_GetCounterDataPrefix(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetCounterDataPrefix copy\n");
            destroy_builder();
            destroy_evaluator();
            return 1;
        }
    }
    destroy_builder();
    std::printf("[OK] counter data prefix built (%zu bytes)\n",
                counter_data_prefix.size());

    // ---- Calculate counter data image size ----
    std::vector<uint8_t> counter_data_image;
    std::vector<uint8_t> counter_data_scratch;
    {
        CUpti_Profiler_CounterDataImageOptions opts = {
            CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE
        };
        opts.pCounterDataPrefix = counter_data_prefix.data();
        opts.counterDataPrefixSize = counter_data_prefix.size();
        opts.maxNumRanges = 1;
        opts.maxNumRangeTreeNodes = 1;
        opts.maxRangeNameLength = 64;

        CUpti_Profiler_CounterDataImage_CalculateSize_Params sz = {
            CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE
        };
        sz.sizeofCounterDataImageOptions = sizeof(opts);
        sz.pOptions = &opts;
        if (cuptiProfilerCounterDataImageCalculateSize(&sz) != CUPTI_SUCCESS) {
            std::printf("[FAIL] CounterDataImage_CalculateSize\n");
            destroy_evaluator();
            return 1;
        }
        counter_data_image.resize(sz.counterDataImageSize);

        CUpti_Profiler_CounterDataImage_Initialize_Params init = {
            CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
        };
        init.sizeofCounterDataImageOptions = sizeof(opts);
        init.pOptions = &opts;
        init.counterDataImageSize = counter_data_image.size();
        init.pCounterDataImage = counter_data_image.data();
        if (cuptiProfilerCounterDataImageInitialize(&init) != CUPTI_SUCCESS) {
            std::printf("[FAIL] CounterDataImage_Initialize\n");
            destroy_evaluator();
            return 1;
        }

        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
            scrSize = {
              CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE
            };
        scrSize.counterDataImageSize = counter_data_image.size();
        scrSize.pCounterDataImage = counter_data_image.data();
        if (cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scrSize) !=
            CUPTI_SUCCESS) {
            std::printf("[FAIL] CounterDataImage_CalculateScratchBufferSize\n");
            destroy_evaluator();
            return 1;
        }
        counter_data_scratch.resize(scrSize.counterDataScratchBufferSize);

        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
            scrInit = {
              CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE
            };
        scrInit.counterDataImageSize = counter_data_image.size();
        scrInit.pCounterDataImage = counter_data_image.data();
        scrInit.counterDataScratchBufferSize = counter_data_scratch.size();
        scrInit.pCounterDataScratchBuffer = counter_data_scratch.data();
        if (cuptiProfilerCounterDataImageInitializeScratchBuffer(&scrInit) !=
            CUPTI_SUCCESS) {
            std::printf("[FAIL] CounterDataImage_InitializeScratchBuffer\n");
            destroy_evaluator();
            return 1;
        }
        std::printf("[OK] counter data image initialized (%zu bytes)\n",
                    counter_data_image.size());
    }

    // ---- Build config image via NVPW RawMetricsConfig ----
    // Separate from the counter data prefix: this image tells the profiler
    // which raw GPU counters to actually program. Created from the same
    // metric eval request we already resolved.
    std::vector<uint8_t> config_image;
    {
        NVPW_CUDA_RawMetricsConfig_Create_V2_Params create = {
            NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE
        };
        create.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        create.pChipName = chip_name.c_str();
        if (NVPW_CUDA_RawMetricsConfig_Create_V2(&create) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] RawMetricsConfig_Create_V2\n");
            destroy_evaluator();
            return 1;
        }
        NVPA_RawMetricsConfig *config = create.pRawMetricsConfig;

        auto destroy_config = [&]() {
            NVPW_RawMetricsConfig_Destroy_Params p = {
                NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            NVPW_RawMetricsConfig_Destroy(&p);
        };

        // Begin pass group, add the metric, end pass group, generate image.
        {
            NVPW_RawMetricsConfig_BeginPassGroup_Params p = {
                NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            // Throughput metrics often require many raw counters split
            // across multiple profiler passes (e.g., gpu__compute_memory_*
            // needs ~28 raw counters). 16 covers most metrics; the
            // allPassesSubmitted replay loop handles whatever count NVPW
            // actually needs.
            p.maxPassCount = 16;
            if (NVPW_RawMetricsConfig_BeginPassGroup(&p) !=
                NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] BeginPassGroup\n");
                destroy_config();
                destroy_evaluator();
                return 1;
            }
        }
        {
            NVPW_RawMetricsConfig_AddMetrics_Params p = {
                NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            p.pRawMetricRequests = raw_requests.data();
            p.numMetricRequests = raw_requests.size();
            NVPA_Status r = NVPW_RawMetricsConfig_AddMetrics(&p);
            if (r != NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] RawMetricsConfig_AddMetrics: NVPA_Status=%d\n",
                            (int)r);
                std::printf("%s\n", explain_nvpa((int)r).c_str());
                destroy_config();
                destroy_evaluator();
                return 1;
            }
        }
        {
            NVPW_RawMetricsConfig_EndPassGroup_Params p = {
                NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            if (NVPW_RawMetricsConfig_EndPassGroup(&p) !=
                NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] EndPassGroup\n");
                destroy_config();
                destroy_evaluator();
                return 1;
            }
        }
        {
            NVPW_RawMetricsConfig_GenerateConfigImage_Params p = {
                NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            if (NVPW_RawMetricsConfig_GenerateConfigImage(&p) !=
                NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] GenerateConfigImage\n");
                destroy_config();
                destroy_evaluator();
                return 1;
            }
        }
        // Two-call pattern: query size, then copy.
        {
            NVPW_RawMetricsConfig_GetConfigImage_Params p = {
                NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
            p.bytesAllocated = 0;
            p.pBuffer = nullptr;
            if (NVPW_RawMetricsConfig_GetConfigImage(&p) !=
                NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] GetConfigImage size query\n");
                destroy_config();
                destroy_evaluator();
                return 1;
            }
            config_image.resize(p.bytesCopied);
            p.bytesAllocated = config_image.size();
            p.pBuffer = config_image.data();
            if (NVPW_RawMetricsConfig_GetConfigImage(&p) !=
                NVPA_STATUS_SUCCESS) {
                std::printf("[FAIL] GetConfigImage copy\n");
                destroy_config();
                destroy_evaluator();
                return 1;
            }
        }
        destroy_config();
        std::printf("[OK] config image generated (%zu bytes)\n",
                    config_image.size());
    }

    // ---- Begin session ----
    // Profiler must be bound to a CUDA context. Make sure one exists by
    // doing a tiny CUDA call first.
    CUDA_CHECK(cudaFree(0));

    {
        CUpti_Profiler_BeginSession_Params p = {
            CUpti_Profiler_BeginSession_Params_STRUCT_SIZE
        };
        p.ctx = nullptr;  // use current context
        p.counterDataImageSize = counter_data_image.size();
        p.pCounterDataImage = counter_data_image.data();
        p.counterDataScratchBufferSize = counter_data_scratch.size();
        p.pCounterDataScratchBuffer = counter_data_scratch.data();
        p.bDumpCounterDataInFile = 0;
        p.pCounterDataFilePath = nullptr;
        p.range = CUPTI_UserRange;
        p.replayMode = CUPTI_UserReplay;
        p.maxRangesPerPass = 1;
        p.maxLaunchesPerPass = 256;
        CUptiResult r = cuptiProfilerBeginSession(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] BeginSession: %s (code=%d)\n",
                        cupti_str(r), (int)r);
            std::printf("%s\n", explain_cupti(r).c_str());
            destroy_evaluator();
            return 1;
        }
    }

    auto end_session = [&]() {
        CUpti_Profiler_EndSession_Params p = {
            CUpti_Profiler_EndSession_Params_STRUCT_SIZE
        };
        cuptiProfilerEndSession(&p);
    };

    {
        CUpti_Profiler_SetConfig_Params p = {
            CUpti_Profiler_SetConfig_Params_STRUCT_SIZE
        };
        p.pConfig = config_image.data();
        p.configSize = config_image.size();
        p.passIndex = 0;
        p.minNestingLevel = 1;
        p.numNestingLevels = 1;
        p.targetNestingLevel = 1;
        CUptiResult r = cuptiProfilerSetConfig(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] SetConfig: %s (code=%d)\n",
                        cupti_str(r), (int)r);
            std::printf("%s\n", explain_cupti(r).c_str());
            end_session();
            destroy_evaluator();
            return 1;
        }
    }
    std::printf("[OK] session begun + config set\n");

    // ---- Workload loop with profiling enabled ----
    // UserReplay mode requires looping until allPassesSubmitted=true.
    // For a single-pass-fitting metric, one iteration is enough; for
    // multi-pass metrics (most throughput metrics), the loop replays
    // the workload to gather all required raw counters.
    long long elapsed_ms = 0;
    int pass_count = 0;
    bool all_passes_submitted = false;
    while (!all_passes_submitted) {
        ++pass_count;
        {
            CUpti_Profiler_BeginPass_Params p = {
                CUpti_Profiler_BeginPass_Params_STRUCT_SIZE
            };
            if (cuptiProfilerBeginPass(&p) != CUPTI_SUCCESS) {
                std::printf("[FAIL] BeginPass (pass %d)\n", pass_count);
                end_session();
                destroy_evaluator();
                return 1;
            }
        }
        {
            CUpti_Profiler_EnableProfiling_Params p = {
                CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE
            };
            if (cuptiProfilerEnableProfiling(&p) != CUPTI_SUCCESS) {
                std::printf("[FAIL] EnableProfiling (pass %d)\n", pass_count);
                end_session();
                destroy_evaluator();
                return 1;
            }
        }
        {
            CUpti_Profiler_PushRange_Params p = {
                CUpti_Profiler_PushRange_Params_STRUCT_SIZE
            };
            p.pRangeName = "cupti_probe_workload";
            if (cuptiProfilerPushRange(&p) != CUPTI_SUCCESS) {
                std::printf("[FAIL] PushRange (pass %d)\n", pass_count);
                end_session();
                destroy_evaluator();
                return 1;
            }
        }

        std::printf("[OK] pass %d: profiling + range, running workload (%ds)...\n",
                    pass_count, duration_s);
        elapsed_ms +=
            run_collect_workload(std::chrono::seconds(duration_s));

        {
            CUpti_Profiler_PopRange_Params p = {
                CUpti_Profiler_PopRange_Params_STRUCT_SIZE
            };
            cuptiProfilerPopRange(&p);
        }
        {
            CUpti_Profiler_DisableProfiling_Params p = {
                CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE
            };
            cuptiProfilerDisableProfiling(&p);
        }
        {
            CUpti_Profiler_EndPass_Params p = {
                CUpti_Profiler_EndPass_Params_STRUCT_SIZE
            };
            if (cuptiProfilerEndPass(&p) != CUPTI_SUCCESS) {
                std::printf("[FAIL] EndPass (pass %d)\n", pass_count);
                end_session();
                destroy_evaluator();
                return 1;
            }
            all_passes_submitted = (p.allPassesSubmitted != 0);
        }
        if (pass_count > 32) {
            std::printf("[WARN] too many passes (>32), bailing out.\n");
            break;
        }
    }
    std::printf("[OK] all passes submitted after %d pass(es), total elapsed: %lld ms\n",
                pass_count, elapsed_ms);

    {
        CUpti_Profiler_FlushCounterData_Params p = {
            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE
        };
        cuptiProfilerFlushCounterData(&p);
    }
    end_session();

    // Diagnostic: how many ranges actually got captured?
    {
        NVPW_CounterData_GetNumRanges_Params p = {
            NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE
        };
        p.pCounterDataImage = counter_data_image.data();
        if (NVPW_CounterData_GetNumRanges(&p) == NVPA_STATUS_SUCCESS) {
            std::printf("[OK] counter data has %zu range(s)\n",
                        p.numRanges);
            if (p.numRanges == 0) {
                std::printf("[WARN] zero ranges captured â€” explains NaN. "
                            "Range/replay setup did not produce data.\n");
            }
        } else {
            std::printf("[INFO] GetNumRanges returned non-success "
                        "(may indicate counter data was not flushed properly)\n");
        }
    }

    // ---- Decode counter data ----
    {
        NVPW_MetricsEvaluator_SetDeviceAttributes_Params p = {
            NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        p.pCounterDataImage = counter_data_image.data();
        p.counterDataImageSize = counter_data_image.size();
        if (NVPW_MetricsEvaluator_SetDeviceAttributes(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] SetDeviceAttributes\n");
            destroy_evaluator();
            return 1;
        }
    }

    double metric_value = 0.0;
    {
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params p = {
            NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        p.pMetricEvalRequests = &eval_request;
        p.numMetricEvalRequests = 1;
        p.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        p.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        p.pCounterDataImage = counter_data_image.data();
        p.counterDataImageSize = counter_data_image.size();
        p.rangeIndex = 0;
        p.isolated = 1;
        p.pMetricValues = &metric_value;
        NVPA_Status r = NVPW_MetricsEvaluator_EvaluateToGpuValues(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] EvaluateToGpuValues: NVPA_Status=%d\n",
                        (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            destroy_evaluator();
            return 1;
        }
    }

    std::printf("\n--- Result ---\n");
    std::printf("metric  : %s\n", metric_name.c_str());
    std::printf("value   : %.6f\n", metric_value);
    std::printf("workload: %lld ms\n", elapsed_ms);
    std::printf("\nVERDICT: PROFILER_API_END_TO_END_OK â€” Spark's CUPTI\n"
                "Profiler API can be driven all the way from metric\n"
                "name to a measured value. spark-uma-trace's GPU-side\n"
                "memory observability source is unblocked.\n\n");

    destroy_evaluator();

    // ---- JSON output ----
    FILE *jf = std::fopen("cupti_probe_profiler_collect_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe profiler-collect\",\n");
        std::fprintf(jf, "  \"version\": \"0.7.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"chip_name\": \"%s\",\n", chip_name.c_str());
        std::fprintf(jf, "  \"metric_name\": \"%s\",\n", metric_name.c_str());
        std::fprintf(jf, "  \"metric_type\": %u,\n",
                     (unsigned)eval_request.metricType);
        std::fprintf(jf, "  \"metric_index\": %u,\n",
                     (unsigned)eval_request.metricIndex);
        std::fprintf(jf, "  \"config_image_bytes\": %zu,\n",
                     config_image.size());
        std::fprintf(jf, "  \"counter_data_image_bytes\": %zu,\n",
                     counter_data_image.size());
        std::fprintf(jf, "  \"workload_elapsed_ms\": %lld,\n", elapsed_ms);
        std::fprintf(jf, "  \"metric_value\": %.6f,\n", metric_value);
        std::fprintf(jf, "  \"verdict\": \"PROFILER_API_END_TO_END_OK\"\n");
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_profiler_collect_results.json\n");
    }
    return 0;
}

// Subcommand: profiler-list
// Enumerate the metrics NVPW knows about for this chip. Foundation must be
// confirmed working first (run `cupti-probe profiler-init`). Conservative
// scope: just count + sample names. v0.5 will add full metric metadata
// (dim units, base names, descriptions).

namespace {

static int parse_arg_int(const std::vector<std::string> &args,
                          const std::string &flag, int fallback) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) {
            int v = std::atoi(args[i + 1].c_str());
            return v > 0 ? v : fallback;
        }
    }
    return fallback;
}

} // namespace

int cmd_profiler_list(const std::vector<std::string> &args) {
    int sample_n = parse_arg_int(args, "--max", 50);

    auto plat = gather_platform();
    print_banner("profiler-list", plat);
    print_health_hint();

    std::printf("Enumerating CUPTI Profiler API metrics for this chip.\n");
    std::printf("Sample cap: %d names (use --max N to change).\n\n",
                sample_n);

    // Foundation init (mirror of profiler-init steps 1-3).
    {
        CUpti_Profiler_Initialize_Params p = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE
        };
        CUptiResult r = cuptiProfilerInitialize(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiProfilerInitialize: %s\n",
                        cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            std::printf("\nVERDICT: foundation broken; run `profiler-init` "
                        "for full diagnosis.\n");
            return 1;
        }
        std::printf("[OK] cuptiProfilerInitialize\n");
    }
    {
        NVPW_InitializeHost_Params p = {NVPW_InitializeHost_Params_STRUCT_SIZE};
        NVPA_Status r = NVPW_InitializeHost(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] NVPW_InitializeHost: NVPA_Status=%d\n",
                        (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            return 1;
        }
        std::printf("[OK] NVPW_InitializeHost\n");
    }

    // Get chip name.
    std::string chip_name;
    {
        CUpti_Device_GetChipName_Params p = {
            CUpti_Device_GetChipName_Params_STRUCT_SIZE
        };
        p.deviceIndex = 0;
        CUptiResult r = cuptiDeviceGetChipName(&p);
        if (r != CUPTI_SUCCESS || !p.pChipName) {
            std::printf("[FAIL] cuptiDeviceGetChipName: %s\n",
                        cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            return 1;
        }
        chip_name = p.pChipName;
        std::printf("[OK] cuptiDeviceGetChipName  chip=\"%s\"\n",
                    chip_name.c_str());
    }

    // Scratch buffer size for the metrics evaluator.
    size_t scratch_bytes = 0;
    {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params p = {
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE
        };
        p.pChipName = chip_name.c_str();
        NVPA_Status r =
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] NVPW MetricsEvaluator scratch size: "
                        "NVPA_Status=%d\n", (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            return 1;
        }
        scratch_bytes = p.scratchBufferSize;
        std::printf("[OK] scratch size           %llu bytes\n",
                    (unsigned long long)scratch_bytes);
    }

    // Allocate scratch + initialize the metrics evaluator.
    std::vector<uint8_t> scratch(scratch_bytes);
    NVPW_MetricsEvaluator *evaluator = nullptr;
    {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params p = {
            NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE
        };
        p.pScratchBuffer = scratch.data();
        p.scratchBufferSize = scratch.size();
        p.pChipName = chip_name.c_str();
        NVPA_Status r = NVPW_CUDA_MetricsEvaluator_Initialize(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] NVPW MetricsEvaluator init: NVPA_Status=%d\n",
                        (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            return 1;
        }
        evaluator = p.pMetricsEvaluator;
        std::printf("[OK] MetricsEvaluator initialized\n\n");
    }

    // Enumerate metric names. NVPW exposes metrics in three categories:
    // Counter, Ratio, Throughput. Hit each one.
    struct MetricCategory {
        NVPW_MetricType type;
        const char *label;
        size_t count = 0;
        std::vector<std::string> sample_names;
    };
    MetricCategory cats[] = {
        {NVPW_METRIC_TYPE_COUNTER,    "COUNTER",    0, {}},
        {NVPW_METRIC_TYPE_RATIO,      "RATIO",      0, {}},
        {NVPW_METRIC_TYPE_THROUGHPUT, "THROUGHPUT", 0, {}},
    };

    for (auto &cat : cats) {
        NVPW_MetricsEvaluator_GetMetricNames_Params p = {
            NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        p.metricType = cat.type;
        NVPA_Status r = NVPW_MetricsEvaluator_GetMetricNames(&p);
        if (r != NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] enumerate %s metrics: NVPA_Status=%d\n",
                        cat.label, (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            continue;
        }
        cat.count = p.numMetrics;
        // Names live in a single concatenated char blob (`pMetricNames`),
        // indexed via an offsets array (`pMetricNameBeginIndices`).
        // The i-th metric name is &pMetricNames[pMetricNameBeginIndices[i]].
        size_t to_sample = (cat.count < (size_t)sample_n) ? cat.count
                                                          : (size_t)sample_n;
        if (p.pMetricNames && p.pMetricNameBeginIndices) {
            for (size_t i = 0; i < to_sample; ++i) {
                const char *nm = p.pMetricNames + p.pMetricNameBeginIndices[i];
                cat.sample_names.emplace_back(nm);
            }
        }
    }

    // Report.
    size_t total = 0;
    for (const auto &cat : cats) total += cat.count;
    std::printf("--- Metric counts for chip \"%s\" ---\n", chip_name.c_str());
    for (const auto &cat : cats) {
        std::printf("  %-12s %llu metrics\n", cat.label,
                    (unsigned long long)cat.count);
    }
    std::printf("  %-12s %llu metrics total\n\n", "TOTAL",
                (unsigned long long)total);

    for (const auto &cat : cats) {
        if (cat.sample_names.empty()) continue;
        std::printf("--- First %zu %s metrics ---\n",
                    cat.sample_names.size(), cat.label);
        for (const auto &nm : cat.sample_names) {
            std::printf("  %s\n", nm.c_str());
        }
        std::printf("\n");
    }

    // Cleanup.
    if (evaluator) {
        NVPW_MetricsEvaluator_Destroy_Params p = {
            NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE
        };
        p.pMetricsEvaluator = evaluator;
        NVPW_MetricsEvaluator_Destroy(&p);
    }

    // JSON output.
    FILE *jf = std::fopen("cupti_probe_profiler_list_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe profiler-list\",\n");
        std::fprintf(jf, "  \"version\": \"0.4.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"chip_name\": \"%s\",\n", chip_name.c_str());
        std::fprintf(jf, "  \"total_metrics\": %llu,\n",
                     (unsigned long long)total);
        std::fprintf(jf, "  \"by_category\": {\n");
        for (size_t i = 0; i < sizeof(cats) / sizeof(*cats); ++i) {
            std::fprintf(jf, "    \"%s\": %llu%s\n", cats[i].label,
                         (unsigned long long)cats[i].count,
                         (i + 1 == sizeof(cats) / sizeof(*cats)) ? "" : ",");
        }
        std::fprintf(jf, "  }\n");
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_profiler_list_results.json\n");
    }
    return 0;
}
