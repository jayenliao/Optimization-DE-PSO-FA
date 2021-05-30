import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Optimization - HW2: DE, PSO, FA')

    # General
    parser.add_argument('-a', '--algorithms', type=str, nargs='+', default=['DE', 'PSO', 'FA'])
    parser.add_argument('-s', '--seed', type=int, default=4028)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-t', '--training_times', type=int, default=50)
    parser.add_argument('-the', '--theta', type=float, nargs='+', default=[0.05884, 4.298, 21.8])
    parser.add_argument('-path', '--savePATH', type=str, default='./output/')
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('--exp_N', type=int, nargs='+', default=[10, 20, 50, 100, 200, 300, 400, 500, 750, 1000])

    # DE
    parser.add_argument('--F', type=float, default=.2)
    parser.add_argument('--N_DE', type=int, default=100)
    
    # PSO
    parser.add_argument('--velocityP', type=float, default=1.)
    parser.add_argument('--velocityPT', type=float, default=.01)
    parser.add_argument('--alpha_PSO', type=float, default=1.)
    parser.add_argument('--beta_PSO', type=float, default=.3)
    parser.add_argument('--gamma_PSO', type=float, default=1e-3)
    parser.add_argument('--N_PSO', type=int, default=100)

    # FA
    parser.add_argument('--beta0_FA', type=float, default=1.)
    parser.add_argument('--gamma_FA', type=float, default=.5)
    parser.add_argument('--alpha_FA', type=float, default=1.)
    parser.add_argument('--alpha2_FA', type=float, default=.001)
    parser.add_argument('--alphaP_FA', type=float, default=.99)
    parser.add_argument('--lambda_FA', type=float, default=1.)
    parser.add_argument('--N_FA', type=int, default=100)
    
    return parser