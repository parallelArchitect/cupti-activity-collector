// main.cu â€” cupti-probe CLI dispatcher.
//
// Usage:
//   cupti-probe sweep
//   cupti-probe single --kind KERNEL --duration 10s
//   cupti-probe records-dump --kind KERNEL --max 20
//   cupti-probe nvlink-load
//
// All subcommands print a hardware/library banner first for reproducibility.
// All subcommands write a JSON result file alongside terminal output.

#include "probe_common.h"
#include "probes.h"

static void print_usage() {
    std::fprintf(stderr,
        "cupti-probe â€” CUPTI capability probe for GB10/DGX Spark and friends\n\n"
        "Usage:\n"
        "  cupti-probe sweep\n"
        "      Try cuptiActivityEnable across many kinds, run a small\n"
        "      synthetic workload, report enable + record counts per kind.\n\n"
        "  cupti-probe single --kind <NAME> [--duration 10s]\n"
        "      Focus on one kind. Sustained workload, record count + sample.\n\n"
        "  cupti-probe records-dump --kind <NAME> [--max 20]\n"
        "      Like single, but print the actual contents of records â€” not\n"
        "      just the count. Validates that records are meaningful.\n\n"
        "  cupti-probe nvlink-load\n"
        "      Sustained C2C-heavy workload (~10s, 64MB managed buffer +\n"
        "      H<->D copies). Tests whether NVLINK kind produces records on\n"
        "      GB10 under realistic load. WARNING: heavy GPU work â€” do not\n"
        "      run while a vLLM compile or training process is active.\n\n"
        "  cupti-probe profiler-init\n"
        "      Foundation test for CUPTI Profiler API (the modern path that\n"
        "      replaces deprecated Activity API kinds). Checks init, NVPW\n"
        "      host, chip recognition, and metrics DB entry for this chip.\n"
        "      Lightweight (no GPU workload), safe to run during vLLM build.\n\n"
        "  cupti-probe profiler-list [--max N]\n"
        "      Enumerate metrics NVPW knows for this chip. Reports counts\n"
        "      per category (COUNTER, RATIO, THROUGHPUT) and prints the\n"
        "      first N names per category. --max defaults to 50.\n"
        "      Lightweight, safe during vLLM build.\n\n"
        "  cupti-probe profiler-collect --metric NAME [--duration 3]\n"
        "      Collect a value for one metric during a workload. End-to-end\n"
        "      test of CUPTI Profiler API on this hardware (DEPRECATED API\n"
        "      as of CUDA 13.0; kept for A/B against profiler-collect-rp).\n"
        "      Suggested metrics: gpu__compute_memory_throughput,\n"
        "      sm__memory_throughput, sm__instruction_throughput.\n"
        "      WARNING: runs a sustained GPU workload â€” do not run during\n"
        "      a vLLM build or training process.\n\n"
        "  cupti-probe profiler-collect-rp --metric NAME [--duration 3]\n"
        "      Same end-to-end metric collection as profiler-collect, but\n"
        "      driven through the modern Range Profiler API (cupti_range\n"
        "      _profiler.h). Future-proof path; CUDA 13.0+ recommends this\n"
        "      over the deprecated cupti_profiler_target.h surface.\n"
        "      Compare numbers directly with profiler-collect â€” same metric\n"
        "      + same workload should agree within noise.\n"
        "      WARNING: same workload as profiler-collect â€” heavy GPU work.\n\n"
        "Output: terminal table + cupti_probe_<subcommand>_results.json\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string sub = argv[1];
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) rest.emplace_back(argv[i]);

    if (sub == "sweep") return cmd_sweep(rest);
    if (sub == "single") return cmd_single(rest);
    if (sub == "records-dump") return cmd_records_dump(rest);
    if (sub == "nvlink-load") return cmd_nvlink_load(rest);
    if (sub == "profiler-init") return cmd_profiler_init(rest);
    if (sub == "profiler-list") return cmd_profiler_list(rest);
    if (sub == "profiler-collect") return cmd_profiler_collect(rest);
    if (sub == "profiler-collect-rp") return cmd_profiler_collect_rp(rest);

    if (sub == "-h" || sub == "--help") {
        print_usage();
        return 0;
    }

    std::fprintf(stderr, "unknown subcommand: %s\n\n", sub.c_str());
    print_usage();
    return 1;
}
