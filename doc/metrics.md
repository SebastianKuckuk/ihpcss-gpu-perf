# General

A detailed overview is available [online](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure).

For counting metrics (e.g. bytes transferred) the structure is usually `metric.sum`.
For corresponding rates this can be extended to `metric.sum.per_second`.
This can also be recomputed in terms of percentage of theoretical peak performance with `metric.sum.pct_of_peak_sustained_elapsed`.

Available metrics can be queried with (example for A100)
* `ncu --query-metrics --chip ga100`
and with additional filters (e.g. only metrics starting with a certain string)
* `ncu --query-metrics-mode suffix --metrics sm__inst_executed --chip ga100`

# Time

`gpu__time_duration.sum`

# Compute

## Thread Level

Total number of instructions for *single precision* FMA, ADD, MUL
* `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`
* `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`
* `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum`

Total number of instructions for *double precision* FMA, ADD, MUL
* `smsp__sass_thread_inst_executed_op_dfma_pred_on.sum`
* `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum`
* `smsp__sass_thread_inst_executed_op_dmul_pred_on.sum`

The number of FLOPs in a given precision is 2 * FMA + ADD + MUL

Corresponding rates (instruction/second)
* `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second`
* `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second`
* `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second`

* `smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second`
* `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second`
* `smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second`

Total number of integer operations
* `sm__sass_thread_inst_executed_op_integer_pred_on.sum`

## Warp Level

Counts FMA warp level instructions
* `smsp__inst_executed_pipe_fma.sum`

# Memory

Total number of bytes transferred from/ to DRAM 
* `dram__bytes_read.sum`
* `dram__bytes_write.sum`

The corresponding rates/ bandwidths (bytes/second)
* `dram__bytes_read.sum.per_second`
* `dram__bytes_write.sum.per_second`

## L2 Cache

The number of bytes transferred for load and store operations
* `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum`
* `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum`
as well as for atomic updates/ additions
* `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom.sum`
* `lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_red.sum`

## Sectors

The corresponding number of sectors
* `dram__sectors_read.sum`
* `dram__sectors_write.sum`

The average bytes used per global memory sector accessed
* `smsp__sass_average_data_bytes_per_sector_mem_global.ratio`

# Launch statistics

* `smsp__warps_launched.sum`

# Atomics

* `smsp__inst_executed_op_global_red.sum`