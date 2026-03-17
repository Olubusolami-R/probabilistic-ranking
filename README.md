# Probabilistic Ranking of ATP Tennis Players

Bayesian inference for ranking tennis players from match outcomes. I implemented and 
compared two inference approaches — Metropolis-Hastings MCMC and Expectation Propagation 
— on 2011 ATP season data (107 players, 1,801 matches).

## Repository Structure

- `ranking.ipynb` — main notebook: full analysis, convergence diagnostics, results and comparisons
- `MHrank.py` — Metropolis-Hastings sampler with acceptance rate tracking
- `eprank.py` — Expectation Propagation via message passing
- `barplot.py` — plotting utilities

## What I Implemented

Starting from a minimal scaffold (imports, data loading, a stub sampler, one ACF plot), 
I built out the full analysis:

- Single-site Random Walk Metropolis sampler with tuned proposal (σ=0.8, 36.8% acceptance)
- Convergence diagnostics: trace plots, ACF analysis, IPS autocorrelation time estimation
- Gelman-Rubin R-hat across three independent chains (R̂=1.01)
- EP convergence tracking via maximum absolute change in means and variances
- Skill comparison vs match outcome probability tables for top 4 players
- Three-method ranking comparison with Spearman rank correlation analysis

## Results

**MCMC convergence:** chains stabilise by iteration 50-100, burn-in set at 200. 
Autocorrelation time τ ≈ 30 (range 6.9-23.6 across players), giving ~60 effective 
samples from 2,000 iterations. Reliable inference requires ~30,200 iterations.

**EP convergence:** deterministic convergence at iteration 34 (ε = 10⁻⁴ threshold), 
~890× faster than MCMC with comparable accuracy (mean absolute difference 0.0125, 
correlation 0.9992).

**Skill vs match outcome:** Djokovic's 99% skill advantage over Murray translates to 
only 74% match win probability as match-level noise (ε ~ N(0,1)) causes outcome 
probabilities to regress towards 0.5.

**Rankings:** MCMC and EP agree perfectly on top-10 rankings. Both differ from empirical 
win rates (~0.47 Spearman correlation), primarily for players with uneven match schedules.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook ranking.ipynb
```

## Data
2011 ATP season: 107 players, 1,801 matches. Data file `tennis_data.mat` is not included in this repo.

---
*4F13 Probabilistic Machine Learning, Cambridge MPhil*