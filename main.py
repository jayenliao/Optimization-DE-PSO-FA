import os, time, random
from cupy._io.npz import save
from utils import save_txt, DE, PSO, FA
from datetime import datetime
from args import init_arguments
if init_arguments().parse_args().gpu:
    import cupy as np
    np.cuda.Stream.null.synchronize()
else:
    import numpy as np

def main(algorithm_name, args, answer, return_results=False):
    t0 = time.time()
    
    if algorithm_name == 'DE':
        N = args.N_DE
        algorithm = DE(
            theta=args.theta, F=args.F, N=N
        )
    elif algorithm_name == 'PSO':
        N = args.N_PSO
        algorithm = PSO(
            theta=args.theta, N=N,
            velocityP=args.velocityP, velocityPT=args.velocityPT,
            alpha=args.alpha_PSO, beta=args.beta_PSO, gamma=args.gamma_PSO
        )
    elif algorithm_name == 'FA':
        N = args.N_FA
        algorithm = FA(
            theta=args.theta, beta0=args.beta0_FA, gamma=args.gamma_FA,
            alpha=args.alpha_FA, alpha2=args.alpha2_FA, alphaP=args.alphaP_FA,
            lamb=args.lambda_FA, N=N
        )

    print('Initializing weights ...')
    algorithm.initW()
    
    print('Training ...')
    algorithm.train(args.training_times)
    dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    folder_name = args.savePATH + dt + '_' + algorithm_name + '_' + '_N=' + str(N) + '_t=' + str(args.training_times) + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    save_txt(algorithm.time_cost, folder_name+'time_cost')
    save_txt(algorithm.lst_globalv, folder_name+'lst_globalv')

    if algorithm_name == 'DE':
        x = algorithm.globalm
        out = [algorithm.vectexs[x], algorithm.values[x]]
    else:
        #algorithm.checkGlobal()
        out = [algorithm.globalx, algorithm.globalv]
    save_txt(out, folder_name+'gx_gv')
    save_txt([algorithm.getValue(answer)], folder_name+'getValue')
    print('gx:', out[0], ' gv:', out[1])
    print(algorithm.getValue(answer))

    print('Time cost of %s: %6.2f s.\n' % (algorithm_name, time.time() - t0))
    print('-'*40)

    if return_results:
        return algorithm.lst_globalv, algorithm.time_cost, folder_name

if __name__ == '__main__':
    args = init_arguments().parse_args()
    answer = np.array([[0.2288, 1.3886, 18.4168], [1/3, 1/3, 1/3]])
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    for a in args.algorithms:
        print()
        print(a)
        main(a, args, answer)