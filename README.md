# International HPC Summer School - GPU Performance Analysis

## Prerequisites

We will obtain profiling data on Bridges. The generated report files will then be visualized and analyzed locally on each participants notebook. This requires a local installation of NVIDIA Nsight Compute and NVIDIA Nsight Systems. Having an NVIDIA GPU is _not_ required.
You can either install the tools separately ([nsight compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), [nsight systems](https://developer.nvidia.com/nsight-systems/get-started), might require a free NVIDIA developer account), or install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) which bundles both tools. In any case, please follow the instructions for the OS on you notebook.

A copy of all profiles obtained is also included in this repository.

## Course Content

All course material is collected and available at [https://github.com/SebastianKuckuk/ihpcss-gpu-perf](https://github.com/SebastianKuckuk/ihpcss-gpu-perf) (this repository).

Slides are availavle at [doc/ihpcss-2024-gpu-performance-engineering.pdf](doc/ihpcss-2024-gpu-performance-engineering.pdf).

## Instructions

Clone the repository on your target system (and on your notebook/ workstation to visualize the profiles locally)
```bash
git clone https://github.com/SebastianKuckuk/ihpcss-gpu-perf
```

On Bridges-2, load the module
```bash
module load nvhpc/21.7
```
and start and interactive job
```bash
interact -p GPU-shared --gres=gpu:v100-32:1 --time 2:00:00
```

If you want to use the runner, you additionally have to load a suitable Python module with
```bash
module load anaconda3
```

You can copy files to your local machine via the usual mean (scp, etc.) or by using the [on-demand web interface](https://ondemand.bridges2.psc.edu/) of Bridges-2.
