# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)]
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to payer i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))
    print(f"Number of players: {num_players}")
    w = np.zeros(num_players)  # skill for each player


    # Track acceptance rates
    acceptance_rates = []
    for itr in tqdm(range(num_its)):
        total_proposals = 0
        accepted_moves = 0
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob
            lp1 = norm.logpdf(w[i]) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))
            # proposed new skill and log-prob
            w_prop_i = w[i] + np.random.normal(0, 0.8)
            lp2 = norm.logpdf(w_prop_i) + np.sum(norm.logcdf(outcome * (w_prop_i - w[j])))
            # accept or reject move:
            if np.log(np.random.rand()) < (lp2 - lp1):
                w[i] = w_prop_i
                accepted_moves += 1
            total_proposals += 1
        skill_samples[:, itr] = w
        acceptance_rates.append(accepted_moves / total_proposals)

    return skill_samples, acceptance_rates

