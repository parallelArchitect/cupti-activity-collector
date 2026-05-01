// probes.cu â€” implementations of the cupti-probe subcommands.

#include "probe_common.h"
#include "probes.h"

#include <cxxabi.h>
#include <sstream>

// Demangle a C++ symbol name. Returns the demangled form if possible,
// or the original input on failure. Example:
//   _Z7k_touchPii -> "k_touch(int*, int)"
static std::string demangle(const char *name) {
    if (!name || !*name) return std::string();
    int status = 0;
    char *dem = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string out = (status == 0 && dem) ? std::string(dem) : std::string(name);
    if (dem) std::free(dem);
    return out;
}

// Shared CUPTI buffer plumbing.

static constexpr size_t kBufferSize = 16 * 1024 * 1024;

// Per-record callback installed by each subcommand. Lets a subcommand
// decide what to do with each record (count it, dump it, etc.) without
// duplicating the buffer boilerplate.
using RecordCallback = void (*)(const CUpti_Activity *);

namespace {
    RecordCallback g_record_cb = nullptr;
}

extern "C" {
static void CUPTIAPI g_buffer_requested(uint8_t **buffer, size_t *size,
                                         size_t *max_records) {
    *buffer = (uint8_t *)std::aligned_alloc(8, kBufferSize);
    *size = kBufferSize;
    *max_records = 0;
}

static void CUPTIAPI g_buffer_completed(CUcontext, uint32_t,
                                         uint8_t *buffer,
                                         size_t /*size*/,
                                         size_t valid_size) {
    if (!buffer) return;
    if (valid_size > 0 && g_record_cb) {
        CUpti_Activity *record = nullptr;
        for (;;) {
            CUptiResult r = cuptiActivityGetNextRecord(buffer, valid_size,
                                                        &record);
            if (r == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            if (r != CUPTI_SUCCESS) break;
            g_record_cb(record);
        }
    }
    std::free(buffer);
}
} // extern "C"

static void install_record_callback(RecordCallback cb) {
    g_record_cb = cb;
    CUPTI_LOG_SOFT(
        cuptiActivityRegisterCallbacks(g_buffer_requested, g_buffer_completed),
        "register_callbacks");
}

// Synthetic workloads.

__global__ void k_touch(int *p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = idx;
}

__global__ void k_stream(int *p, int n, int rounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = 0; r < rounds; ++r) {
        for (int i = idx; i < n; i += stride) {
            p[i] = (p[i] * 1664525 + 1013904223) ^ r;
        }
    }
}

static void run_light_workload() {
    int *d_p = nullptr;
    int *h_p = (int *)std::malloc(1024 * sizeof(int));
    for (int i = 0; i < 1024; ++i) h_p[i] = i;
    CUDA_CHECK(cudaMalloc(&d_p, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, 1024 * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_p, 0, 1024 * sizeof(int)));
    k_touch<<<8, 128>>>(d_p, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_p, d_p, 1024 * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_p));
    std::free(h_p);
}

// Sustained C2C-heavy workload. Returns elapsed milliseconds.
// WARNING: heavy GPU work â€” do not call while another GPU workload is active.
static long long run_heavy_workload(std::chrono::milliseconds duration) {
    constexpr size_t kManagedElems = 64 * 1024 * 1024 / sizeof(int);
    constexpr size_t kCopyBytes    = 32 * 1024 * 1024;
    int *managed = nullptr;
    int *h_pinned = nullptr;
    int *d_buf    = nullptr;
    CUDA_CHECK(cudaMallocManaged(&managed, kManagedElems * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_pinned, kCopyBytes));
    CUDA_CHECK(cudaMalloc(&d_buf, kCopyBytes));
    std::memset(h_pinned, 0xa5, kCopyBytes);

    auto start    = std::chrono::steady_clock::now();
    auto deadline = start + duration;
    int rounds = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        for (size_t i = 0; i < kManagedElems; i += 4096) {
            managed[i] = (int)i + rounds;
        }
        k_stream<<<256, 256>>>(managed, (int)kManagedElems, 4);
        for (int i = 0; i < 8; ++i) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, kCopyBytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, kCopyBytes,
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        ++rounds;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(managed));
    (void)rounds;
    return elapsed;
}

// Argument helpers.

static std::string get_arg(const std::vector<std::string> &args,
                            const std::string &flag,
                            const std::string &fallback = "") {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) return args[i + 1];
    }
    return fallback;
}

static int parse_duration_seconds(const std::string &s, int fallback) {
    if (s.empty()) return fallback;
    int n = std::atoi(s.c_str());
    if (n <= 0) return fallback;
    if (s.back() == 's') return n;
    if (s.back() == 'm') return n * 60;
    return n;
}

// Subcommand: sweep

namespace sweep_state {
struct PerKind {
    CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_INVALID;
    const char *name = nullptr;
    std::atomic<uint64_t> records{0};
    bool enable_ok = false;
    int enable_code = 0;
    std::string enable_msg;

    PerKind() = default;
    PerKind(CUpti_ActivityKind k, const char *n) : kind(k), name(n) {}

    // std::atomic deletes copy/move by default, which makes PerKind
    // non-emplaceable into std::vector. Provide an explicit move
    // constructor that snapshots the atomic value.
    PerKind(PerKind &&other) noexcept
        : kind(other.kind),
          name(other.name),
          records(other.records.load()),
          enable_ok(other.enable_ok),
          enable_code(other.enable_code),
          enable_msg(std::move(other.enable_msg)) {}
};

static std::vector<PerKind> g_kinds;

static PerKind *find(CUpti_ActivityKind k) {
    for (auto &pk : g_kinds) {
        if (pk.kind == k) return &pk;
    }
    return nullptr;
}

static void on_record(const CUpti_Activity *record) {
    if (auto *pk = find(record->kind)) {
        pk->records.fetch_add(1, std::memory_order_relaxed);
    }
}
} // namespace sweep_state

int cmd_sweep(const std::vector<std::string> & /*args*/) {
    auto plat = gather_platform();
    print_banner("sweep", plat);
    print_health_hint();

    sweep_state::g_kinds.clear();
    sweep_state::g_kinds.reserve(all_kinds().size());
    for (const auto &k : all_kinds()) {
        sweep_state::g_kinds.emplace_back(k.kind, k.name);
    }

    install_record_callback(sweep_state::on_record);

    std::printf("--- Phase 1: cuptiActivityEnable per kind ---\n");
    std::printf("%-26s  %-22s  %s\n", "KIND", "RESULT", "MSG");
    for (auto &pk : sweep_state::g_kinds) {
        CUptiResult r = cuptiActivityEnable(pk.kind);
        const char *msg = nullptr;
        cuptiGetResultString(r, &msg);
        pk.enable_code = (int)r;
        pk.enable_msg = msg ? msg : "?";
        pk.enable_ok = (r == CUPTI_SUCCESS);
        std::printf("%-26s  %-22s  %s\n", pk.name,
                    pk.enable_ok ? "OK" : "FAILED",
                    pk.enable_ok ? "" : pk.enable_msg.c_str());
        if (!pk.enable_ok) {
            std::string why = explain_cupti(r);
            if (!why.empty()) std::printf("%s\n", why.c_str());
        }
    }
    std::printf("\n");

    std::printf("--- Phase 2: light synthetic workload ---\n");
    run_light_workload();
    std::printf("workload completed.\n\n");

    CUPTI_LOG_SOFT(cuptiActivityFlushAll(0), "flush_all");

    std::printf("--- Phase 3: records collected per kind ---\n");
    std::printf("%-26s  %-9s  %-9s\n", "KIND", "ENABLE", "RECORDS");
    for (const auto &pk : sweep_state::g_kinds) {
        std::printf("%-26s  %-9s  %llu\n", pk.name,
                    pk.enable_ok ? "OK" : "FAIL",
                    (unsigned long long)pk.records.load());
    }
    std::printf("\n");

    FILE *jf = std::fopen("cupti_probe_sweep_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe sweep\",\n");
        std::fprintf(jf, "  \"version\": \"0.2.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"kinds\": [\n");
        for (size_t i = 0; i < sweep_state::g_kinds.size(); ++i) {
            const auto &pk = sweep_state::g_kinds[i];
            std::fprintf(jf, "    {\n");
            std::fprintf(jf, "      \"name\": \"%s\",\n", pk.name);
            std::fprintf(jf, "      \"enable_ok\": %s,\n",
                         pk.enable_ok ? "true" : "false");
            std::fprintf(jf, "      \"enable_code\": %d,\n", pk.enable_code);
            std::fprintf(jf, "      \"enable_msg\": \"%s\",\n",
                         pk.enable_msg.c_str());
            std::fprintf(jf, "      \"records\": %llu\n",
                         (unsigned long long)pk.records.load());
            std::fprintf(jf, "    }%s\n",
                         (i + 1 == sweep_state::g_kinds.size()) ? "" : ",");
        }
        std::fprintf(jf, "  ]\n}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_sweep_results.json\n");
    }
    return 0;
}

// Subcommand: single

namespace single_state {
static std::atomic<uint64_t> g_count{0};
static CUpti_ActivityKind g_kind = CUPTI_ACTIVITY_KIND_INVALID;

static void on_record(const CUpti_Activity *record) {
    if (record->kind == g_kind) {
        g_count.fetch_add(1, std::memory_order_relaxed);
    }
}
} // namespace single_state

int cmd_single(const std::vector<std::string> &args) {
    std::string kind_name = get_arg(args, "--kind");
    if (kind_name.empty()) {
        std::fprintf(stderr, "single: --kind required\n");
        return 1;
    }
    bool ok = false;
    CUpti_ActivityKind k = kind_from_name(kind_name, &ok);
    if (!ok) {
        std::fprintf(stderr, "single: unknown kind '%s'\n", kind_name.c_str());
        return 1;
    }
    int duration_s = parse_duration_seconds(get_arg(args, "--duration", "5s"), 5);

    auto plat = gather_platform();
    print_banner("single", plat);
    print_health_hint();
    std::printf("Target kind  : %s\n", kind_name.c_str());
    std::printf("Duration     : %d seconds\n\n", duration_s);

    single_state::g_kind = k;
    install_record_callback(single_state::on_record);

    CUptiResult r = cuptiActivityEnable(k);
    const char *msg = nullptr;
    cuptiGetResultString(r, &msg);
    bool enable_ok = (r == CUPTI_SUCCESS);
    std::printf("enable %s: %s (%s)\n", kind_name.c_str(),
                enable_ok ? "OK" : "FAILED", msg ? msg : "?");
    if (!enable_ok) {
        std::string why = explain_cupti(r);
        if (!why.empty()) std::printf("%s\n", why.c_str());
        std::printf("\nVERDICT: kind cannot be enabled on this platform.\n");
        return 0;
    }
    std::printf("\n");

    long long elapsed_ms = run_heavy_workload(
        std::chrono::seconds(duration_s));
    std::printf("workload elapsed: %lld ms\n", elapsed_ms);
    CUPTI_LOG_SOFT(cuptiActivityFlushAll(0), "flush_all");

    uint64_t records = single_state::g_count.load();
    std::printf("\n--- Result ---\n");
    std::printf("kind     : %s\n", kind_name.c_str());
    std::printf("records  : %llu\n", (unsigned long long)records);
    std::printf("verdict  : %s\n",
                records > 0 ? "FUNCTIONAL â€” kind produces records under load"
                            : "STRUCTURALLY ABSENT â€” kind enabled but never "
                              "recorded under sustained workload");
    std::printf("\n");

    char path[256];
    std::snprintf(path, sizeof(path),
                   "cupti_probe_single_%s_results.json", kind_name.c_str());
    FILE *jf = std::fopen(path, "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe single\",\n");
        std::fprintf(jf, "  \"version\": \"0.2.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"kind\": \"%s\",\n", kind_name.c_str());
        std::fprintf(jf, "  \"enable_ok\": true,\n");
        std::fprintf(jf, "  \"duration_s\": %d,\n", duration_s);
        std::fprintf(jf, "  \"workload_elapsed_ms\": %lld,\n", elapsed_ms);
        std::fprintf(jf, "  \"records\": %llu,\n",
                     (unsigned long long)records);
        std::fprintf(jf, "  \"verdict\": \"%s\"\n",
                     records > 0 ? "functional" : "structurally_absent");
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: %s\n", path);
    }
    return 0;
}

// Subcommand: records-dump
// Walks records of the requested kind and prints a human-readable summary
// of each. This validates that records aren't just "arriving" but are
// actually meaningful.

namespace dump_state {
static int g_max = 20;
static int g_collected = 0;
static CUpti_ActivityKind g_kind = CUPTI_ACTIVITY_KIND_INVALID;
static std::vector<std::string> g_lines;

static std::string describe_kernel(const CUpti_ActivityKernel9 *k) {
    std::string nm = k->name ? demangle(k->name) : "?";
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "  KERNEL: name=%s grid=(%d,%d,%d) block=(%d,%d,%d) "
                  "start=%llu end=%llu duration_us=%.2f",
                  nm.c_str(),
                  k->gridX, k->gridY, k->gridZ,
                  k->blockX, k->blockY, k->blockZ,
                  (unsigned long long)k->start,
                  (unsigned long long)k->end,
                  (k->end - k->start) / 1000.0);
    return std::string(buf);
}

static std::string describe_memcpy(const CUpti_ActivityMemcpy6 *m) {
    static const char *kind_str[] = {"UNKNOWN", "HtoD", "DtoH", "HtoA",
                                       "AtoH", "AtoA", "AtoD", "DtoA",
                                       "DtoD", "HtoH", "PtoP"};
    const char *ks = (m->copyKind < sizeof(kind_str) / sizeof(*kind_str))
                          ? kind_str[m->copyKind] : "?";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  MEMCPY: kind=%s bytes=%llu start=%llu end=%llu "
                  "duration_us=%.2f",
                  ks, (unsigned long long)m->bytes,
                  (unsigned long long)m->start,
                  (unsigned long long)m->end,
                  (m->end - m->start) / 1000.0);
    return std::string(buf);
}

static std::string describe_memset(const CUpti_ActivityMemset4 *m) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  MEMSET: bytes=%llu value=0x%x start=%llu end=%llu "
                  "duration_us=%.2f",
                  (unsigned long long)m->bytes, m->value,
                  (unsigned long long)m->start,
                  (unsigned long long)m->end,
                  (m->end - m->start) / 1000.0);
    return std::string(buf);
}

static std::string describe_api(const char *label, const CUpti_ActivityAPI *a) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  %s: cbid=%u start=%llu end=%llu duration_us=%.2f",
                  label, a->cbid,
                  (unsigned long long)a->start,
                  (unsigned long long)a->end,
                  (a->end - a->start) / 1000.0);
    return std::string(buf);
}

static std::string describe_overhead(const CUpti_ActivityOverhead *o) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  OVERHEAD: kind=%d start=%llu end=%llu duration_us=%.2f",
                  (int)o->overheadKind,
                  (unsigned long long)o->start,
                  (unsigned long long)o->end,
                  (o->end - o->start) / 1000.0);
    return std::string(buf);
}

static std::string describe_environment(const CUpti_ActivityEnvironment *e) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "  ENVIRONMENT: kind=%d device=%u",
                  (int)e->environmentKind, e->deviceId);
    return std::string(buf);
}

static std::string describe_generic(const CUpti_Activity *r) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "  %s record (kind=%d) â€” no specialized printer",
                  name_from_kind(r->kind), (int)r->kind);
    return std::string(buf);
}

static void on_record(const CUpti_Activity *record) {
    if (record->kind != g_kind) return;
    if (g_collected >= g_max) return;
    ++g_collected;
    std::string line;
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            line = describe_kernel(
                reinterpret_cast<const CUpti_ActivityKernel9 *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
            line = describe_memcpy(
                reinterpret_cast<const CUpti_ActivityMemcpy6 *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
            line = describe_memset(
                reinterpret_cast<const CUpti_ActivityMemset4 *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_RUNTIME:
            line = describe_api("RUNTIME",
                reinterpret_cast<const CUpti_ActivityAPI *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_DRIVER:
            line = describe_api("DRIVER ",
                reinterpret_cast<const CUpti_ActivityAPI *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
            line = describe_overhead(
                reinterpret_cast<const CUpti_ActivityOverhead *>(record));
            break;
        case CUPTI_ACTIVITY_KIND_ENVIRONMENT:
            line = describe_environment(
                reinterpret_cast<const CUpti_ActivityEnvironment *>(record));
            break;
        default:
            line = describe_generic(record);
            break;
    }
    g_lines.push_back(std::move(line));
}
} // namespace dump_state

int cmd_records_dump(const std::vector<std::string> &args) {
    std::string kind_name = get_arg(args, "--kind");
    if (kind_name.empty()) {
        std::fprintf(stderr, "records-dump: --kind required\n");
        return 1;
    }
    bool ok = false;
    CUpti_ActivityKind k = kind_from_name(kind_name, &ok);
    if (!ok) {
        std::fprintf(stderr, "records-dump: unknown kind '%s'\n",
                     kind_name.c_str());
        return 1;
    }
    int max_records = std::atoi(get_arg(args, "--max", "20").c_str());
    if (max_records <= 0) max_records = 20;

    auto plat = gather_platform();
    print_banner("records-dump", plat);
    print_health_hint();
    std::printf("Target kind  : %s\n", kind_name.c_str());
    std::printf("Max records  : %d\n\n", max_records);

    dump_state::g_kind = k;
    dump_state::g_max = max_records;
    dump_state::g_collected = 0;
    dump_state::g_lines.clear();

    install_record_callback(dump_state::on_record);

    CUptiResult r = cuptiActivityEnable(k);
    const char *msg = nullptr;
    cuptiGetResultString(r, &msg);
    if (r != CUPTI_SUCCESS) {
        std::printf("enable %s: FAILED (%s)\n", kind_name.c_str(),
                    msg ? msg : "?");
        std::string why = explain_cupti(r);
        if (!why.empty()) std::printf("%s\n", why.c_str());
        std::printf("\nVERDICT: kind cannot be enabled.\n");
        return 0;
    }
    std::printf("enable %s: OK\n\n", kind_name.c_str());

    std::printf("--- Workload ---\n");
    run_light_workload();
    std::printf("workload completed.\n");
    CUPTI_LOG_SOFT(cuptiActivityFlushAll(0), "flush_all");

    std::printf("\n--- Records (up to %d) ---\n", max_records);
    if (dump_state::g_lines.empty()) {
        std::printf("  (none)\n\n");
        std::printf("VERDICT: kind enabled but recorded zero on a light "
                    "workload. Try `cupti-probe single --kind %s --duration "
                    "30s` for sustained load before concluding structurally "
                    "absent.\n", kind_name.c_str());
    } else {
        for (const auto &line : dump_state::g_lines) {
            std::printf("%s\n", line.c_str());
        }
        std::printf("\nemitted %d records.\n",
                    (int)dump_state::g_lines.size());
    }
    return 0;
}

// Subcommand: nvlink-load

namespace nvlink_state {
static std::atomic<uint64_t> g_kernel{0};
static std::atomic<uint64_t> g_memcpy{0};
static std::atomic<uint64_t> g_nvlink{0};
static std::atomic<uint64_t> g_other{0};

static void on_record(const CUpti_Activity *record) {
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            g_kernel.fetch_add(1, std::memory_order_relaxed); break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
            g_memcpy.fetch_add(1, std::memory_order_relaxed); break;
        case CUPTI_ACTIVITY_KIND_NVLINK:
            g_nvlink.fetch_add(1, std::memory_order_relaxed); break;
        default:
            g_other.fetch_add(1, std::memory_order_relaxed); break;
    }
}
} // namespace nvlink_state

int cmd_nvlink_load(const std::vector<std::string> & /*args*/) {
    auto plat = gather_platform();
    print_banner("nvlink-load", plat);
    print_health_hint();
    std::printf("WARNING: this subcommand runs a sustained C2C-heavy workload\n"
                "         (~10s, 64MB managed buffer + 32MB H<->D copies). Do\n"
                "         not run while a vLLM compile or training process is\n"
                "         active â€” measurements will be invalid AND it'll\n"
                "         compete for resources.\n\n");

    install_record_callback(nvlink_state::on_record);

    auto try_enable = [](CUpti_ActivityKind k, const char *label) -> bool {
        CUptiResult r = cuptiActivityEnable(k);
        bool ok = (r == CUPTI_SUCCESS);
        std::printf("enable %s: %s\n", label, ok ? "OK" : "FAILED");
        if (!ok) {
            std::string why = explain_cupti(r);
            if (!why.empty()) std::printf("%s\n", why.c_str());
        }
        return ok;
    };

    bool kernel_ok = try_enable(CUPTI_ACTIVITY_KIND_KERNEL, "KERNEL");
    bool memcpy_ok = try_enable(CUPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY");
    bool nvlink_ok = try_enable(CUPTI_ACTIVITY_KIND_NVLINK, "NVLINK");
    std::printf("\n");

    long long elapsed = run_heavy_workload(std::chrono::seconds(10));
    std::printf("workload elapsed: %lld ms\n", elapsed);
    CUPTI_LOG_SOFT(cuptiActivityFlushAll(0), "flush_all");

    uint64_t kr = nvlink_state::g_kernel.load();
    uint64_t mr = nvlink_state::g_memcpy.load();
    uint64_t nr = nvlink_state::g_nvlink.load();
    uint64_t orec = nvlink_state::g_other.load();
    std::printf("\n--- Records ---\n");
    std::printf("KERNEL  records: %llu\n", (unsigned long long)kr);
    std::printf("MEMCPY  records: %llu\n", (unsigned long long)mr);
    std::printf("NVLINK  records: %llu\n", (unsigned long long)nr);
    std::printf("OTHER   records: %llu\n", (unsigned long long)orec);

    const char *verdict = "INDETERMINATE";
    if (kernel_ok && memcpy_ok && nvlink_ok) {
        if (kr > 0 && mr > 0 && nr == 0) {
            verdict = "NVLINK_STRUCTURALLY_ABSENT â€” KERNEL+MEMCPY worked, "
                      "NVLINK enabled but recorded zero across sustained "
                      "C2C-heavy workload";
        } else if (nr > 0) {
            verdict = "NVLINK_FUNCTIONAL â€” produced records under sustained "
                      "C2C-heavy workload";
        }
    }
    std::printf("\nVERDICT: %s\n\n", verdict);

    FILE *jf = std::fopen("cupti_probe_nvlink_load_results.json", "w");
    if (jf) {
        std::fprintf(jf, "{\n");
        std::fprintf(jf, "  \"tool\": \"cupti-probe nvlink-load\",\n");
        std::fprintf(jf, "  \"version\": \"0.2.0\",\n");
        std::fprintf(jf, "%s,\n", platform_json(plat).c_str());
        std::fprintf(jf, "  \"workload_elapsed_ms\": %lld,\n", elapsed);
        std::fprintf(jf, "  \"enable\": {\n");
        std::fprintf(jf, "    \"KERNEL\": %s,\n", kernel_ok ? "true" : "false");
        std::fprintf(jf, "    \"MEMCPY\": %s,\n", memcpy_ok ? "true" : "false");
        std::fprintf(jf, "    \"NVLINK\": %s\n",  nvlink_ok ? "true" : "false");
        std::fprintf(jf, "  },\n");
        std::fprintf(jf, "  \"records\": {\n");
        std::fprintf(jf, "    \"KERNEL\": %llu,\n", (unsigned long long)kr);
        std::fprintf(jf, "    \"MEMCPY\": %llu,\n", (unsigned long long)mr);
        std::fprintf(jf, "    \"NVLINK\": %llu,\n", (unsigned long long)nr);
        std::fprintf(jf, "    \"OTHER\":  %llu\n",  (unsigned long long)orec);
        std::fprintf(jf, "  },\n");
        std::fprintf(jf, "  \"verdict\": \"%s\"\n", verdict);
        std::fprintf(jf, "}\n");
        std::fclose(jf);
        std::printf("JSON written: cupti_probe_nvlink_load_results.json\n");
    }
    return 0;
}
