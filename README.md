# AMT
Software accompanying the paper "Adaptive Monte Carlo Multiple Testing via Multi-Armed Bandits"

# Installation

- Go to the `/amt` directory
- Development installation `pip install -e .`
- Python version: `Python 3.6.3 :: Anaconda custom (64-bit)`

# Reproducing the experiments in the paper 
Relative file paths are used. 
## Simulations
- Progression (Figure 2-3): `./amt/experiments/simulation_progression.ipynb`
- Reliability (Table 1): `./amt/experiments/simulation_progression.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have the number in Table 1.
- Scaling (Figure 4): `./amt/experiments/simulation_nMC.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 4.
- Varying nominal FDR (Figure 5a): `./amt/experiments/simulation_alpha.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5a.
- Varying alternative proportion (Figure 5b): `./amt/experiments/simulation_pi1.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5b.
- Varying effect size (Figure 5c): `./amt/experiments/simulation_effect.ipynb` Skip the `Simulation` section and directly run the `Generate figures` section to have Figure 5b.


## GWAS data on Parkinson's disease
Unfortunately, the GWAS data is not included due to its large size. The data will be hosted online with the publication of the paper. Nonetheless, all results are recorded in corresponding notebooks. 
- Small GWAS (Table 2-3, Supp Table 1): `./amt/experiments/small_gwas_chr1.ipynb` Changing `chr1` to `chrx` (x=1,2,3,4) gives results on other chromosomes. `Compute fMC p-values` runs fMC and save the corresponding result, which also gives fMC running time. `Result analysis` prints fMC p-values for SNPs reported in the original paper. `Corresponding AMT result` runs `AMT` with the same MC samples. This is to check if AMT result is the same as fMC. This also gives the average number of MC samples `AMT` takes. `Directly run AMT` gives the running time of `AMT`, where the MC sampling is by actual permutation.  
- full GWAS: the experiment is done using python script `./amt/experiments/amt_gwas.py`. This may take an hour to run and the results are stored in `./results/GWAS/result_amt_alpha_0.1_rep0`, which gives running time and the average number of MC samples used. Analysis of the result is done using `./amt/experiments/parkinsons_analysis_full.ipynb`, which prints out all information of the SNPs reported in the original paper. 
