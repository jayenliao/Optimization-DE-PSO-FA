import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from args import init_arguments

def find_folders(N, t, ROOT='./output/'):
    lst = os.listdir('./output/')
    key = 'N='+str(N)+'_t='+str(t)
    out = []
    for x in lst:
        if '.' not in x:
            h = x.split('-')[3]
            if key in x and int(h) >= 21:
                out.append(ROOT + x)
    return out

def plot_algorithms(N, ROOT):

    d_globalv = {}
    roots = find_folders(N, 50, ROOT)

    for i, fn in enumerate(roots):
        fn = fn + '/' if fn[-1] != '/' else fn
        fn += 'lst_globalv.txt'
        for ax in ('DE', 'PSO', 'FA'):
            if ax in fn:
                a = ax
        d_globalv[a] = list(pd.read_table(fn, header=None).iloc[:,0])

    x = np.arange(50)
    plt.figure()
    axx = ['DE', 'PSO']
    if 'FA' in d_globalv.keys():
        axx += ['FA']
    for ax in axx:
        plt.plot(x, d_globalv[ax], label=ax)
    plt.xlabel('Run')
    plt.ylabel('Global Best Value')
    plt.legend()
    plt.grid(linestyle='--')

if __name__ == '__main__':
    ROOT = init_arguments().parse_args().savePATH
    print('The convergence plot are saved as:')
    for N in (10, 50, 100, 200):
        plot_algorithms(N, ROOT)
        fn = ROOT + 'convergence_plot_N=' + str(N) + '.png'
        plt.savefig(fn)
        print('-->', fn)