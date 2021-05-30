# Implement of Meta-Heuristic Algorithms: Differential Evolution, Particle Swarm Optimization, And Firefly Algorithm

## Codes

- `utils.py`: the implements of DE, PSO, and FA
- `main.py`: the main program to run DE, PSO, and FA
- `args.py` defines the arguments parser
- `exp_N.py` conducts the experiment of numbers of setting points

## Usage

```
python3 main.py -a 'DE'
python3 main.py -a 'DE' 'PSO' 'FA'
python3 exp_N.py -a 'DE' 'PSO'  # Only DE would be trained
python3 exp_N.py -a 'FA' 'PSO'  # Only FA would be trained
python3 exp_N.py -a 'PSO'
python3 exp_N.py -a 'FA' -t 50 --exp_N 300 400 500
```
