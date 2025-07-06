# International HPC Summer School - GPU Performance Analysis

## Prerequisites

All required profiling data files are available in the `profiles` folder.
They can be visualized and analyzed locally on each participants laptop.
This requires a local installation of NVIDIA Nsight Compute and NVIDIA Nsight Systems.
Having an NVIDIA GPU is _not_ required.
You can either install the tools separately ([nsight compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), [nsight systems](https://developer.nvidia.com/nsight-systems/get-started), might require a free NVIDIA developer account), or install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) which bundles both tools.
In any case, please follow the instructions for your OS.

## Course Content

All course material is collected and available at [https://github.com/SebastianKuckuk/ihpcss-gpu-perf](https://github.com/SebastianKuckuk/ihpcss-gpu-perf) (this repository).

The course material is organized as a [Jupyter notebook](doc/ihpcss-2025-gpu-analysis.ipynb).

## Instructions

Clone the repository on your notebook/ workstation to visualize the profiles locally.
```bash
git clone https://github.com/SebastianKuckuk/ihpcss-gpu-perf
```

To make your own measurements, e.g. on Bridges, also clone the repository on the target machine.

Then load required modules, e.g.
```bash
module load nvhpc
```
and start and interactive job, e.g.
```bash
interact -p GPU-shared --gres=gpu:v100-32:1 --time 2:00:00
```

You can copy files to your local machine via the usual mean (scp, etc.).
On Bridges, you can also use the [on-demand web interface](https://ondemand.bridges2.psc.edu/).
