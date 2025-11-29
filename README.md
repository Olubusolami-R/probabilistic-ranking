# Probabilistic Ranking of ATP Tennis Players

Bayesian inference methods for ranking tennis players from match outcomes. Comparing MCMC sampling with Expectation Propagation on 2011 ATP season data.

## Overview

This project explores how to infer player skills from binary match results using probabilistic methods. Working with 107 players and 1,801 matches from the 2011 ATP season, I implemented and compared three approaches to ranking players based on latent skill estimates.

## Problem

Player skills aren't directly observable - we only see match outcomes. The challenge is to infer the underlying skill distribution from win/loss data whilst properly accounting for uncertainty and opponent strength.

## Approach

### Metropolis-Hastings MCMC
Implemented a single-site Random Walk Metropolis sampler to sample from the posterior distribution of player skills.

- Gaussian proposal with σ=0.8, achieving 36.8% acceptance rate
- Validated convergence using Gelman-Rubin diagnostic (R̂=1.01)
- Burn-in: 200 iterations, autocorrelation time τ ≈ 30
- Requires ~30,200 iterations for 1,000 effective samples

### Expectation Propagation
Used message-passing for fast approximate Bayesian inference.

- Converges to Gaussian approximation of posterior
- Reaches precision threshold (ε = 10⁻⁴) in 34 iterations
- ~890× faster than MCMC with comparable accuracy

### Empirical Baseline
Simple win/loss ratios for comparison - fast but doesn't account for opponent quality.

## Results

**Convergence**: MCMC chains stabilise quickly (50-100 iterations) but high autocorrelation means long runs needed. EP converges deterministically in 34 iterations.

**Method Agreement**: MCMC and EP show strong agreement (mean absolute difference 0.0125, correlation 0.9992). Both identify identical top-10 rankings.

**Skill vs Outcome**: There's an important distinction between skill comparison and match prediction. Match-level randomness (modelled as ε ~ N(0,1)) causes outcome probabilities to regress towards 0.5. For example, Djokovic's 99% skill advantage over Murray translates to only 74% match win probability.

**Inference Validation**: Tested three methods for computing skill comparisons from MCMC samples - direct sampling proved most reliable, though joint Gaussian approximation (accounting for correlation) performs reasonably well.

**Ranking Quality**: Model-based methods account for opponent strength, whilst empirical win rates treat all wins equally. This matters most for players with limited data or those who faced particularly strong/weak opponents.

## Technical Implementation

- Single-site Metropolis-Hastings with componentwise updates
- Convergence diagnostics: trace plots, Gelman-Rubin, autocorrelation analysis
- Message passing with moment matching for EP
- Computational complexity: O(T × M) where each match evaluated twice per iteration

## Key Takeaway

EP emerges as the practical choice for this problem: it accounts for opponent strength, provides uncertainty estimates, and achieves accuracy comparable to MCMC whilst being nearly 900× faster. The Gaussian approximation works well here, validated by the strong MCMC-EP agreement.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook tennis_ranking.ipynb
```

**Requirements**: numpy, scipy, matplotlib, pandas, jupyter

## Data

2011 ATP season match results: 107 players, 1,801 matches

---

*Originally completed as part of the Probabilistic Machine Learning course during my MPhil at Cambridge.*