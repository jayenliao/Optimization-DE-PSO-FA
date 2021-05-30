import os, random
import matplotlib.pyplot as plt
from utils import save_txt, DE, PSO, FA
from datetime import datetime
from main import main
from args import init_arguments
if init_arguments().parse_args().gpu:
    import cupy as np
    np.cuda.Stream.null.synchronize()
else:
    import numpy as np

if __name__ == '__main__':
    args = init_arguments().parse_args()
    answer = np.array([[0.2288, 1.3886, 18.4168], [1/3, 1/3, 1/3]])
    A = args.algorithms[0]
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Training
    lsts_globalv = []
    lsts_time_cost = []
    for N in args.exp_N:
        args.N_DE = args.N_PSO = args.N_FA = N
        print()
        print(f'Algorithm: {A} (N={N})')
        lst_globalv, time_cost, folder_name = main(A, args, answer, True)
        lsts_globalv.append(lst_globalv)
        lsts_time_cost.append(time_cost)
    
    # Visualization
    x = np.arange(args.training_times)
    plt.figure()
    for i, N in enumerate(args.exp_N):
        plt.plot(x, lsts_globalv[i], label='N='+str(N))
    plt.title('Convergence Plot of ' + A + ' with Different Numbers of Points')
    plt.xlabel('Run')
    plt.ylabel('Global Best Value')
    plt.legend()
    plt.grid(linestyle='--')
    fn = folder_name + 'convergence_plot.png'
    plt.savefig(fn)
    print('The convergence plot is saved as')
    print('-->', fn)
    
    arr = np.array(lsts_globalv)
    plt.figure()
    plt.plot(args.exp_N, arr.min(axis=1))
    plt.title('The Smallest Global Best Value of ' + A + ' with Different Numbers of Points')
    plt.xlabel('Number of Points')
    plt.ylabel('The Smallest Global Best Value')
    plt.legend()
    plt.grid(linestyle='--')
    fn = folder_name + 'GBE_plot.png'
    plt.savefig(fn)
    print('The GBE plot is saved as')
    print('-->', fn)

    arr = np.array(lsts_time_cost)
    plt.figure()
    plt.errorbar(args.exp_N, arr.mean(axis=1), yerr=arr.std(axis=1))
    plt.title('Time Cost Plot of ' + A + ' with Different Numbers of Points')
    plt.xlabel('Numberof Points')
    plt.ylabel('Average Time Cost Per Run (s)')
    plt.grid(linestyle='--')
    fn = folder_name + 'time_cost.png'
    plt.savefig(fn)
    print('The time cost plot is saved as')
    print('-->', fn)