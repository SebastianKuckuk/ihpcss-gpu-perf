import re
import subprocess

import pandas as pd
from matplotlib import pyplot as plt


def run_init(cases, gpu_for_filename, show_plot=False):
    benchmark = 'init'
    print(f'Running {benchmark} benchmark ...')

    # sizes_to_bench = [1024, 4096] # dummy test input for debugging
    sizes_to_bench = []
    for i in range(1_0, 30_0 + 1, 1):
        size = int(pow(2, 0.1 * i))
        if size not in sizes_to_bench:
            sizes_to_bench.append(size)

    # prepare data frame
    df = pd.DataFrame({'size': sizes_to_bench})
    df['size'] = df['size'].astype(float) * 8

    for case in cases:
        bandwidths = []

        for (i, size) in enumerate(sizes_to_bench):
            print(f'\r   ... with {case.ljust(16)} --- {round(100 * i / len(sizes_to_bench))}%', end='')

            if size * 8 <= 1024:
                n_warm, n_it = 8, 1000
            elif size * 8 <= 1024 * 1024:
                n_warm, n_it = 4, 100
            elif size * 8 <= 32 * 1024 * 1024:
                n_warm, n_it = 2, 10
            else:
                n_warm, n_it = 1, 1

            out = subprocess.check_output([f'../../build/{benchmark}/{benchmark}-{case}', f'{size}', f'{n_warm}', f'{n_it}'])
            out = out.decode("utf-8")
            bandwidths.append(re.findall(r'bandwidth: {5}(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+) GB/s', out)[0])

        print(f'\r   ... with {case.ljust(16)} --- done')

        df[case] = bandwidths
        df[case] = df[case].astype(float)

    # write to file
    results_file_name = f'../../measurements/{gpu_for_filename}---{benchmark}.csv'
    print(f'Writing results to \'{results_file_name}\'')

    df.to_csv(results_file_name)

    # plot resulting data frame
    plot_file_name = f'../../measurements/{gpu_for_filename}---{benchmark}.pdf'
    print(f'Plotting results to \'{plot_file_name}\'')

    df.plot(x='size', logx=True, figsize=[11.7, 8.3])
    plt.savefig(plot_file_name)

    if show_plot:
        plt.show()
