// range_profiler_probe.cu â€” CUPTI Range Profiler API end-to-end probe (v0.8).
//
// Equivalent of profiler-collect, but driven through the modern Range
// Profiler API (cupti_range_profiler.h, NVIDIA "recommended replacement"
// for the deprecated Profiler API used by profiler-collect in v0.7).
//
// The two APIs share the NVPW host side (config image build, metric
// evaluator decode). Only the *collection* surface differs:
//
//   v0.7 (deprecated)                    v0.8 (modern Range Profiler)
//   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//   cuptiProfilerBeginSession            cuptiRangeProfilerEnable
//   cuptiProfilerCounterDataImage*       cuptiRangeProfilerGetCounterDataSize
//                                        cuptiRangeProfilerCounterDataImageInitialize
//   cuptiProfilerSetConfig               cuptiRangeProfilerSetConfig
//   cuptiProfilerBeginPass               cuptiRangeProfilerStart
//   cuptiProfilerEnableProfiling         (folded into Start)
//   cuptiProfilerPushRange               cuptiRangeProfilerPushRange
//   cuptiProfilerPopRange                cuptiRangeProfilerPopRange
//   cuptiProfilerDisableProfiling        (folded into Stop)
//   cuptiProfilerEndPass                 cuptiRangeProfilerStop
//   cuptiProfilerFlushCounterData        cuptiRangeProfilerDecodeData
//   cuptiProfilerEndSession              cuptiRangeProfilerDisable
//
// What we keep from v0.7: NVPW MetricsEvaluator (resolve + raw-deps),
// NVPW RawMetricsConfig (build the config image), NVPW
// EvaluateToGpuValues (decode result). The Range Profiler API replaces
// the collection dance only.
//
// Why this exists: cupti_profiler_target.h (and therefore v0.7's
// cmd_profiler_collect) is deprecated as of CUDA 13.0 and slated for
// removal. v0.8 future-proofs cupti-probe against CUDA 14+.

#include "probe_common.h"
#include "probes.h"

#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <nvperf_cuda_host.h>

#include <cuda.h>

namespace {

static const char *cupti_str(CUptiResult r) {
    const char *s = nullptr;
    cuptiGetResultString(r, &s);
    return s ? s : "?";
}

// Inline workload â€” matches profiler_probe.cu's k_collect_workload so the
// numbers v0.7 and v0.8 produce are directly comparable.
__global__ void k_rp_workload(int *p, int n, int rounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = 0; r < rounds; ++r) {
        for (int i = idx; i < n; i += stride) {
            p[i] = (p[i] * 1664525 + 1013904223) ^ r;
        }
    }
}

static long long run_rp_workload(std::chrono::milliseconds duration) {
    constexpr size_t kElems = 16 * 1024 * 1024 / sizeof(int);  // 4 M ints
    int *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, kElems * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_buf, 0xa5, kElems * sizeof(int)));

    auto start    = std::chrono::steady_clock::now();
    auto deadline = start + duration;
    while (std::chrono::steady_clock::now() < deadline) {
        k_rp_workload<<<128, 128>>>(d_buf, (int)kElems, 4);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    CUDA_CHECK(cudaFree(d_buf));
    return elapsed;
}

} // namespace

int cmd_profiler_collect_rp(const std::vector<std::string> &args) {
    // ---- Argument parsing ----
    std::string metric_name;
    int duration_s = 3;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--metric")   metric_name = args[i + 1];
        if (args[i] == "--duration") {
            duration_s = std::atoi(args[i + 1].c_str());
            if (duration_s <= 0) duration_s = 3;
        }
    }
    if (metric_name.empty()) {
        std::fprintf(stderr,
            "profiler-collect-rp: --metric NAME required.\n"
            "  Try: cupti-probe profiler-list --max 50  (to find a metric name)\n"
            "  Suggested first targets: gpu__compute_memory_throughput,\n"
            "    sm__cycles_active, lts__t_bytes, lts__d_sectors_fill_sysmem\n");
        return 1;
    }

    auto plat = gather_platform();
    print_banner("profiler-collect-rp", plat);
    print_health_hint();
    std::printf("API           : CUPTI Range Profiler (modern, replaces "
                "deprecated Profiler API in v0.7).\n");
    std::printf("Target metric : %s\n", metric_name.c_str());
    std::printf("Duration      : %d seconds per pass\n\n", duration_s);
    std::printf("WARNING: this subcommand runs a sustained GPU workload.\n"
                "         Do not run while a vLLM compile or training\n"
                "         process is active.\n\n");

    // ---- Foundation init: ProfilerInitialize + NVPW host ----
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
            std::printf("[FAIL] NVPW_InitializeHost: NVPA_Status=%d\n", (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            return 1;
        }
    }

    // ---- Device support check (Range Profiler-specific) ----
    {
        CUdevice cu_dev;
        if (cuDeviceGet(&cu_dev, 0) != CUDA_SUCCESS) {
            std::printf("[FAIL] cuDeviceGet(0)\n");
            return 1;
        }
        CUpti_Profiler_DeviceSupported_Params p = {
            CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE
        };
        p.cuDevice = cu_dev;
        p.api = CUPTI_PROFILER_RANGE_PROFILING;
        CUptiResult r = cuptiProfilerDeviceSupported(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiProfilerDeviceSupported: %s\n",
                        cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            return 1;
        }
        if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
            std::printf("[FAIL] device does NOT support Range Profiler API\n");
            std::printf("       arch=%d sli=%d vGpu=%d cc=%d cmp=%d wsl=%d\n",
                        (int)p.architecture, (int)p.sli, (int)p.vGpu,
                        (int)p.confidentialCompute, (int)p.cmp, (int)p.wsl);
            std::printf("\nVERDICT: RP_API_UNSUPPORTED â€” this configuration\n"
                        "is rejected by the Range Profiler API. On Spark this\n"
                        "should not normally happen; investigate before pivot.\n\n");
            return 2;
        }
        std::printf("[OK] Range Profiler API supported on this device\n");
    }

    // ---- Get chip name (needed for NVPW) ----
    std::string chip_name;
    {
        CUpti_Device_GetChipName_Params p = {
            CUpti_Device_GetChipName_Params_STRUCT_SIZE
        };
        p.deviceIndex = 0;
        CUptiResult r = cuptiDeviceGetChipName(&p);
        if (r != CUPTI_SUCCESS || !p.pChipName) {
            std::printf("[FAIL] cuptiDeviceGetChipName: %s\n", cupti_str(r));
            return 1;
        }
        chip_name = p.pChipName;
        std::printf("[OK] chip name: %s\n", chip_name.c_str());
    }

    // ---- MetricsEvaluator setup (same as v0.7) ----
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
    }
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
    std::printf("[OK] MetricsEvaluator initialized (%llu byte scratch)\n",
                (unsigned long long)scratch_bytes);

    // ---- Resolve metric name â†’ eval request (same as v0.7) ----
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

    // ---- Get raw counter dependencies (same as v0.7) ----
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
        if (NVPW_MetricsEvaluator_GetMetricRawDependencies(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetMetricRawDependencies (count)\n");
            destroy_evaluator();
            return 1;
        }
        size_t num_deps = p.numRawDependencies;
        raw_dep_names.resize(num_deps);
        p.ppRawDependencies = raw_dep_names.data();
        p.numRawDependencies = num_deps;
        if (NVPW_MetricsEvaluator_GetMetricRawDependencies(&p) !=
            NVPA_STATUS_SUCCESS) {
            std::printf("[FAIL] GetMetricRawDependencies (fill)\n");
            destroy_evaluator();
            return 1;
        }
        std::printf("[OK] metric depends on %zu raw counter(s)\n", num_deps);
    }

    std::vector<NVPA_RawMetricRequest> raw_requests(raw_dep_names.size());
    for (size_t i = 0; i < raw_dep_names.size(); ++i) {
        raw_requests[i].structSize = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
        raw_requests[i].pPriv = nullptr;
        raw_requests[i].pMetricName = raw_dep_names[i];
        raw_requests[i].isolated = 1;
        raw_requests[i].keepInstances = 1;
    }

    // ---- Build config image via NVPW RawMetricsConfig (same as v0.7) ----
    // Config image scheduling info is API-agnostic â€” Range Profiler consumes
    // the same NVPW-built config image as the deprecated Profiler API.
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

        {
            NVPW_RawMetricsConfig_BeginPassGroup_Params p = {
                NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE
            };
            p.pRawMetricsConfig = config;
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

    // ---- Make sure a CUDA context exists, then capture it ----
    CUDA_CHECK(cudaFree(0));
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
        std::printf("[FAIL] cuCtxGetCurrent â€” no current CUDA context\n");
        destroy_evaluator();
        return 1;
    }

    // ---- Range Profiler: Enable ----
    CUpti_RangeProfiler_Object *rp_obj = nullptr;
    {
        CUpti_RangeProfiler_Enable_Params p = {
            CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE
        };
        p.ctx = ctx;
        CUptiResult r = cuptiRangeProfilerEnable(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiRangeProfilerEnable: %s (code=%d)\n",
                        cupti_str(r), (int)r);
            std::printf("%s\n", explain_cupti(r).c_str());
            destroy_evaluator();
            return 1;
        }
        rp_obj = p.pRangeProfilerObject;
    }
    auto disable_rp = [&]() {
        if (rp_obj) {
            CUpti_RangeProfiler_Disable_Params p = {
                CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            cuptiRangeProfilerDisable(&p);
            rp_obj = nullptr;
        }
    };
    std::printf("[OK] cuptiRangeProfilerEnable\n");

    // ---- Range Profiler: build counter data image (replaces v0.7's
    //      cuptiProfilerCounterDataImage_Calculate/Initialize/Scratch dance) ----
    // The new API takes metric NAMES directly â€” internally it does what
    // v0.7's NVPW_CUDA_CounterDataBuilder_AddMetrics + GetCounterDataPrefix
    // + cuptiProfilerCounterDataImage_CalculateSize did. Major simplification.
    std::vector<uint8_t> counter_data_image;
    {
        // GetCounterDataSize wants metric name pointers as const char**; the
        // strings live in the metric_name std::string captured below.
        const char *metric_name_cstr = metric_name.c_str();

        CUpti_RangeProfiler_GetCounterDataSize_Params sz = {
            CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE
        };
        sz.pRangeProfilerObject = rp_obj;
        sz.pMetricNames = &metric_name_cstr;
        sz.numMetrics = 1;
        sz.maxNumOfRanges = 1;
        sz.maxNumRangeTreeNodes = 1;
        CUptiResult r = cuptiRangeProfilerGetCounterDataSize(&sz);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiRangeProfilerGetCounterDataSize: %s\n",
                        cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            disable_rp();
            destroy_evaluator();
            return 1;
        }
        counter_data_image.resize(sz.counterDataSize, 0);
        std::printf("[OK] counter data size: %zu bytes\n", sz.counterDataSize);

        CUpti_RangeProfiler_CounterDataImage_Initialize_Params init = {
            CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
        };
        init.pRangeProfilerObject = rp_obj;
        init.counterDataSize = counter_data_image.size();
        init.pCounterData = counter_data_image.data();
        r = cuptiRangeProfilerCounterDataImageInitialize(&init);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] cuptiRangeProfilerCounterDataImageInitialize: %s\n",
                        cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            disable_rp();
            destroy_evaluator();
            return 1;
        }
        std::printf("[OK] counter data image initialized\n");
    }

    // ---- Range Profiler: pass loop ----
    // SetConfig is called once initially, then re-called per pass with the
    // updated passIndex and targetNestingLevel returned from Stop. This
    // mirrors the sample's RangeProfiler::SetConfig pattern.
    long long elapsed_ms = 0;
    int pass_count = 0;
    bool all_passes_submitted = false;
    size_t pass_index = 0;
    uint16_t target_nesting_level = 1;

    while (!all_passes_submitted) {
        ++pass_count;

        // SetConfig for this pass.
        {
            CUpti_RangeProfiler_SetConfig_Params p = {
                CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            p.configSize = config_image.size();
            p.pConfig = config_image.data();
            p.counterDataImageSize = counter_data_image.size();
            p.pCounterDataImage = counter_data_image.data();
            p.range = CUPTI_UserRange;
            p.replayMode = CUPTI_UserReplay;
            p.maxRangesPerPass = 1;
            p.numNestingLevels = 1;
            p.minNestingLevel = 1;
            p.passIndex = pass_index;
            p.targetNestingLevel = target_nesting_level;
            CUptiResult r = cuptiRangeProfilerSetConfig(&p);
            if (r != CUPTI_SUCCESS) {
                std::printf("[FAIL] SetConfig pass %d: %s\n", pass_count,
                            cupti_str(r));
                std::printf("%s\n", explain_cupti(r).c_str());
                disable_rp();
                destroy_evaluator();
                return 1;
            }
        }

        // Start (replaces BeginPass + EnableProfiling).
        {
            CUpti_RangeProfiler_Start_Params p = {
                CUpti_RangeProfiler_Start_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            CUptiResult r = cuptiRangeProfilerStart(&p);
            if (r != CUPTI_SUCCESS) {
                std::printf("[FAIL] Start pass %d: %s\n", pass_count,
                            cupti_str(r));
                disable_rp();
                destroy_evaluator();
                return 1;
            }
        }

        // PushRange.
        {
            CUpti_RangeProfiler_PushRange_Params p = {
                CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            p.pRangeName = "cupti_probe_workload";
            CUptiResult r = cuptiRangeProfilerPushRange(&p);
            if (r != CUPTI_SUCCESS) {
                std::printf("[FAIL] PushRange pass %d: %s\n", pass_count,
                            cupti_str(r));
                disable_rp();
                destroy_evaluator();
                return 1;
            }
        }

        std::printf("[OK] pass %d (passIndex=%zu, nesting=%u): running workload (%ds)...\n",
                    pass_count, pass_index, (unsigned)target_nesting_level,
                    duration_s);
        elapsed_ms += run_rp_workload(std::chrono::seconds(duration_s));

        // PopRange.
        {
            CUpti_RangeProfiler_PopRange_Params p = {
                CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            cuptiRangeProfilerPopRange(&p);
        }

        // Stop (replaces DisableProfiling + EndPass).
        {
            CUpti_RangeProfiler_Stop_Params p = {
                CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE
            };
            p.pRangeProfilerObject = rp_obj;
            CUptiResult r = cuptiRangeProfilerStop(&p);
            if (r != CUPTI_SUCCESS) {
                std::printf("[FAIL] Stop pass %d: %s\n", pass_count,
                            cupti_str(r));
                disable_rp();
                destroy_evaluator();
                return 1;
            }
            pass_index = p.passIndex;
            target_nesting_level = (uint16_t)p.targetNestingLevel;
            all_passes_submitted = (p.isAllPassSubmitted != 0);
        }

        if (pass_count > 32) {
            std::printf("[WARN] too many passes (>32), bailing out.\n");
            break;
        }
    }
    std::printf("[OK] all passes submitted after %d pass(es), total elapsed: %lld ms\n",
                pass_count, elapsed_ms);

    // ---- DecodeData (replaces FlushCounterData) ----
    {
        CUpti_RangeProfiler_DecodeData_Params p = {
            CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE
        };
        p.pRangeProfilerObject = rp_obj;
        CUptiResult r = cuptiRangeProfilerDecodeData(&p);
        if (r != CUPTI_SUCCESS) {
            std::printf("[FAIL] DecodeData: %s\n", cupti_str(r));
            std::printf("%s\n", explain_cupti(r).c_str());
            disable_rp();
            destroy_evaluator();
            return 1;
        }
        if (p.numOfRangeDropped > 0) {
            std::printf("[WARN] %zu range(s) dropped during decode\n",
                        p.numOfRangeDropped);
        } else {
            std::printf("[OK] DecodeData (no ranges dropped)\n");
        }
    }

    // ---- Disable Range Profiler (must happen before evaluation;
    //      counter data image is fully populated by DecodeData above) ----
    disable_rp();
    std::printf("[OK] cuptiRangeProfilerDisable\n");

    // ---- Sanity check: how many ranges in the counter data image ----
    {
        NVPW_CounterData_GetNumRanges_Params p = {
            NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE
        };
        p.pCounterDataImage = counter_data_image.data();
        if (NVPW_CounterData_GetNumRanges(&p) == NVPA_STATUS_SUCCESS) {
            std::printf("[OK] counter data has %zu range(s)\n", p.numRanges);
            if (p.numRanges == 0) {
                std::printf("[WARN] zero ranges captured â€” explains NaN.\n");
            }
        }
    }

    // ---- Evaluate (same NVPW path as v0.7) ----
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
            std::printf("[FAIL] EvaluateToGpuValues: NVPA_Status=%d\n", (int)r);
            std::printf("%s\n", explain_nvpa((int)r).c_str());
            destroy_evaluator();
            return 1;
        }
    }

    std::printf("\n--- Result ---\n");
    std::printf("api     : Range Profiler (cupti_range_profiler.h)\n");
    std::printf("metric  : %s\n", metric_name.c_str());
    std::printf("value   : %.6f\n", metric_value);
    std::printf("workload: %lld ms across %d pass(es)\n",
                elapsed_ms, pass_count);
    std::printf("\nVERDICT: RP_API_END_TO_END_OK â€” Spark's modern CUPTI\n"
                "Range Profiler API drives metric collection from name to\n"
                "measured value. v0.8 path is parity with v0.7's deprecated\n"
                "Profiler API path. Compare numbers directly: same metric\n"
                "+ same workload via both paths should agree within noise.\n\n");

    destroy_evaluator();

    // ---- JSON output ----
    FILE *jf = std::fopen("cupti_probe_profiler_collect_rp_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe profiler-collect-rp\",\n");
        std::fprintf(jf, "  \"version\": \"0.8.0\",\n");
        std::fprintf(jf, "  \"api\": \"range_profiler\",\n");
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
        std::fprintf(jf, "  \"pass_count\": %d,\n", pass_count);
        std::fprintf(jf, "  \"workload_elapsed_ms\": %lld,\n", elapsed_ms);
        std::fprintf(jf, "  \"metric_value\": %.6f,\n", metric_value);
        std::fprintf(jf, "  \"verdict\": \"RP_API_END_TO_END_OK\"\n");
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_profiler_collect_rp_results.json\n");
    }
    return 0;
}
