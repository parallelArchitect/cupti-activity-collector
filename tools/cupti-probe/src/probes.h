// probes.h â€” subcommand entry points.

#pragma once

#include <string>
#include <vector>

// Each returns process exit code: 0 on success, nonzero on error.
int cmd_sweep(const std::vector<std::string> &args);
int cmd_single(const std::vector<std::string> &args);
int cmd_records_dump(const std::vector<std::string> &args);
int cmd_nvlink_load(const std::vector<std::string> &args);
int cmd_profiler_init(const std::vector<std::string> &args);
int cmd_profiler_list(const std::vector<std::string> &args);
int cmd_profiler_collect(const std::vector<std::string> &args);
int cmd_profiler_collect_rp(const std::vector<std::string> &args);
