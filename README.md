# AMT
Software accompanying the paper "Adaptive Monte Carlo Multiple Testing via Multi-Armed Bandits"

# Installation

- Go to the `/amt` directory
- Development installation `pip install -e .`
- Python version: `Python 3.6.3 :: Anaconda custom (64-bit)`

# Reproducing the experiments in the paper 
## Simulations
- Progression (Figure 2-3): `./amt/experiments/simulation_progression.ipynb`
- Reliability (Table 1): `./amt/experiments/simulation_progression.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have the number in Table 1.
- Scaling (Figure 4): `./amt/experiments/simulation_nMC.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 4.
- Varying nominal FDR (Figure 5a): `./amt/experiments/simulation_alpha.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5a.
- Varying alternative proportion (Figure 5b): `./amt/experiments/simulation_pi1.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5b.
- Varying effect size (Figure 5c): `./amt/experiments/simulation_effect.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5b.


## GWAS data on Parkinson's dicease
