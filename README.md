# Implement of Meta-Heuristic Algorithms: Differential Evolution, Particle Swarm Optimization, And Firefly Algorithm

This project aimed to implement three well-known meta-heuristic algorithms: differential evolution (DE), particle swarm optimization (PSO), and firefly algorithm (FA). We set up the numerical experiments to compare 3 algorithms with different numbers of points. We found that FA performed well with only several points but take much higher time cost with more points. While DE and PSO had excellent performance in terms of precision and efficiency with more points (i.e., $N \geq 200$). K-means can be utilized to make the initialized points distribute more uniformly.

## Files

- `Optimization_hw2_RE6094028.pdf`: report file
- `utils.py`: the implements of DE, PSO, and FA
- `main.py`: the main program to run DE, PSO, and FA
- `args.py` defines the arguments parser.
- `exp_N.py` conducts the experiment of numbers of setting points.
- `exp_A.py` produces Fig 1 to Fig 4 in the report.
- `./output/` contains subfolders with results of different trainings.

## Usage examples

```
python3 main.py -a 'DE'
python3 main.py -a 'DE' -t 100        # Run DE 100 times
python3 main.py -a 'DE' --N_DE 500    # Initalize 500 points
python3 main.py -a 'DE' 'PSO' 'FA'
python3 exp_N.py -a 'DE' 'PSO'        # Only DE would be trained
python3 exp_N.py -a 'FA' 'DE'         # Only FA would be trained
python3 exp_N.py -a 'PSO'
python3 exp_N.py -a 'FA' -t 50 --exp_N 300 400 500
```
