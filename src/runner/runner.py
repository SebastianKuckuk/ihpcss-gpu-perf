import re
import subprocess

import fma
import stream
import strided_fma
import strided_stream

# evaluate GPU running on

out = subprocess.check_output(['nvidia-smi', '-L'])
out = out.decode('utf-8').strip()
gpu = re.findall(r'GPU 0: (.*) \(UUID: GPU', out)[0]
gpu_for_filename = gpu.replace(' ', '-').replace('NVIDIA-', '').replace('GeForce-', '')
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
out = out.decode('utf-8').strip()
gpu_cc = float(out)

print(f'Running on GPU {gpu} ({gpu_for_filename}), compute capability {gpu_cc}')

# specify configurations to be benchmarked -- longer list for quick (un-)commenting single cases

cases = []
# cases.append('base')
# cases.append('omp-host')
cases.append('cuda')
cases.append('omp-target')

# run benchmarks -- (un-)comment as needed

fma.run_fma(cases, gpu_for_filename)
strided_fma.run_strided_fma(cases, [1, 2, 4, 8, 16, 32, 64], gpu_for_filename)

stream.run_stream(cases, gpu_for_filename)
strided_stream.run_strided_stream(cases, [1, 2, 4, 8, 16, 32, 64], gpu_for_filename)
