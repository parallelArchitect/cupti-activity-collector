# cupti-probe

CUPTI capability probe for NVIDIA GB10 (DGX Spark) and friends. Sister tool to [parallelArchitect/cupti-uma-probe](https://github.com/parallelArchitect/cupti-uma-probe).

Designed to answer: **what works, what doesn't, and what silently produces zero records on this hardware.**

## Why

NVIDIA's CUPTI library exposes ~16 different "activity kinds" — meters for kernel launches, memcpys, NVLink traffic, environment counters, etc. On normal NVIDIA hardware, most of these work out of the box. On GB10, parallelArchitect's probe demonstrated that `UNIFIED_MEMORY_COUNTER` is structurally unavailable (hardware-coherent UMA produces no UVM page faults, so the tracker has nothing to count).

Open question this tool answers: **which OTHER activity kinds are similarly broken on GB10?** Plus: do they silently fail (enable OK, record nothing) or fail loudly (enable returns an error)?

## Subcommands

```
cupti-probe sweep
    Try cuptiActivityEnable across many kinds, run a small synthetic
    workload, report enable + record counts per kind.

cupti-probe single --kind <NAME> [--duration 10s]
    Focus on one kind. Sustained workload, record count.

cupti-probe records-dump --kind <NAME> [--max 20]
    Like single, but print actual record contents — not just count.
    Validates that records are meaningful, not just arriving.

cupti-probe nvlink-load
    Sustained C2C-heavy workload (~10s, 64MB managed buffer + H<->D
    copies). Tests whether NVLINK kind produces records on GB10 under
    realistic load. WARNING: heavy GPU work — do not run while a
    vLLM compile or training process is active.
```

## Build

Requires CUDA toolkit (13.0 recommended for GB10).

```
make
```

Or with an explicit CUDA path:

```
make CUDA_HOME=/usr/local/cuda-13.0
```

## Run

```
./cupti-probe sweep
./cupti-probe single --kind KERNEL --duration 10s
./cupti-probe records-dump --kind MEMCPY
./cupti-probe nvlink-load
```

Each subcommand prints a hardware/library banner first (GPU, SM, CUDA driver, runtime, CUPTI version). Subcommands also write a JSON result file for sharing.

## Known findings on GB10 (from sweep, 2026-04-26)

| Kind | Enable | Records | Notes |
|---|---|---|---|
| KERNEL | OK | yes | works |
| CONCURRENT_KERNEL | FAIL: NOT_COMPATIBLE | — | CUPTI version mismatch, not Spark-specific |
| MEMCPY | OK | yes | works |
| MEMSET | OK | yes | works |
| RUNTIME | OK | yes | works |
| DRIVER | OK | yes | works |
| OVERHEAD | OK | yes | works |
| SYNCHRONIZATION | OK | yes | works |
| MEMORY2 | OK | yes | works |
| ENVIRONMENT | OK | yes | works |
| UNIFIED_MEMORY_COUNTER | FAIL: NOT_READY | — | structurally absent, silicon-level |
| INSTRUCTION_EXECUTION | FAIL: LEGACY_NOT_SUPPORTED | — | deprecated everywhere |
| NVLINK | OK | 0 (under light load) | open follow-up — needs `nvlink-load` |
| PCIE | OK | 0 | expected — GB10 GPU is on-package via C2C |
| DEVICE | OK | 0 | info-only |
| CONTEXT | OK | yes | works |

## Output files

- `cupti_probe_sweep_results.json`
- `cupti_probe_single_<KIND>_results.json`
- `cupti_probe_nvlink_load_results.json`

## Roadmap

- [x] Unified CLI with subcommands
- [x] Records-content inspection (`records-dump`)
- [x] Hardware introspection banner (CUDA/CUPTI version + GPU info)
- [x] CUPTI Profiler API parallel (`profiler-init`, `profiler-list`,
      `profiler-collect` — deprecated as of CUDA 13.0)
- [x] CUPTI Range Profiler API (`profiler-collect-rp` — modern path,
      future-proof against CUDA 14+)
- [ ] Cross-platform diff mode (run on Spark + B200, output side-by-side)
- [ ] Library API (header-only `cupti_capability.h` for downstream tools)

## License

MIT.
