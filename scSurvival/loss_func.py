import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

def cox_loss_func(pred, label):
    '''
    Make sure the samples have been sorted in descending chronological order.
    Label should be the event indicators taking values in {0, 1}.
    '''
    epsilo = 1e-7
    gamma = pred.max()
    # print('pred-gamma:', (pred - gamma).abs().max())
    risk_set_sum = torch.cumsum(torch.exp(pred - gamma), dim=0) + epsilo
    log_risk_set_sum = torch.log(risk_set_sum) + gamma
    diff = pred - log_risk_set_sum
    n_observed = label.sum(0)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(label)
    loss = (-(sum_diff_in_observed / n_observed)).view(-1)
        
    return loss

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)

def sort_data(x, ytime, yevent):
    sort_ids = torch.argsort(ytime.view(-1), dim=0, descending=True)
    x = x[sort_ids]
    ytime = ytime[sort_ids]
    yevent = yevent[sort_ids]
    return x, ytime, yevent

def zero_inflated_gaussian_loss(x, recon_x, recon_logvar, recon_pi, eps=1e-8, reduction='none', gamma_beta_weight=(0.1, 0.1), gamma_dist=None, beta_dist=None):
    if gamma_dist is not None:
        gamma_prior = lambda var: -gamma_dist.log_prob(var.clamp(min=1e-6))
    if beta_dist is not None:
        beta_prior = lambda pi: -beta_dist.log_prob(pi.clamp(min=1e-2, max=1-1e-2))
    
    var = torch.exp(recon_logvar) + eps
    gamma_weight, beta_weight = gamma_beta_weight
    
    if gamma_weight == 0 and beta_weight == 0:
        loss_prior = 0
    else:
        if gamma_weight == 0:
            # loss_prior = beta_prior(recon_pi.mean(axis=1).view(-1, 1)) * beta_weight
            loss_prior = beta_prior(recon_pi.mean())* beta_weight
        elif beta_weight == 0:
            loss_prior = gamma_prior(var) * gamma_weight
        else:
            loss_prior = gamma_prior(var) * gamma_weight + beta_prior(recon_pi.mean()) * beta_weight

    sigma = torch.sqrt(var)
    normal_dist = torch.distributions.Normal(recon_x, sigma)
    log_gaussian = normal_dist.log_prob(x)  # shape: (batch, features)
    # log_gaussian = -0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(var) - 0.5 * ((x - recon_x) ** 2) / (var)
    
    # 对于 x==0 的情况：
    log_pi = torch.log(recon_pi + eps)
    log_1_minus_pi = torch.log(1 - recon_pi + eps)
    log_p_zero = torch.logaddexp(log_pi, log_1_minus_pi + log_gaussian)
    
    # 对于 x != 0 的情况：
    log_p_nonzero = log_1_minus_pi + log_gaussian
    
    # mask = (x == 0).float()
    # log_p = mask * log_p_zero + (1 - mask) * log_p_nonzero
    log_p= torch.where(x == 0, log_p_zero, log_p_nonzero)
    nll = -log_p

    nll = nll + loss_prior

    if reduction == 'none':
        return nll
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    
def mse_mae_loss(x, recon_x, reduction='none'):
    mask = (x == 0).float()
    mse = F.mse_loss(recon_x, x, reduction='none')
    mae = F.l1_loss(recon_x, x, reduction='none')
    loss = mask * mae * 2 + (1 - mask) * mae
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    
# metrics
def c_index(pred, ytime, yevent):
    # event_time = torch.Tensor(event_time).view(-1, 1)
    # label = torch.Tensor(label).view(-1, 1)
    # if torch.cuda.is_available():
    #     event_time = event_time.cuda()
    #     label = label.cuda()

    pred, ytime, yevent = sort_data(pred, ytime, yevent)
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    pred_matrix = torch.zeros_like(ytime_matrix)
    
    pred_matrix_tmp = pred.view(-1, 1) - pred.view(1, -1)
    pred_matrix[pred_matrix_tmp>0] = 1
    pred_matrix[pred_matrix_tmp==0] = 0.5
 
    concord_matrix = pred_matrix.mul(ytime_matrix)
    concord = torch.sum(concord_matrix)
    epsilon = torch.sum(ytime_matrix)
    concordance_index = torch.div(concord, epsilon)
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    return(concordance_index)

def conditional_cindex(
    event_times,
    event_indicators,
    risk_scores,
    subset
):
    """
    Compute the "conditional C-Index," excluding pairwise comparisons within a given subset.

    Parameters
    ----------
    event_times: 1D array-like
        Survival or follow-up times.
    event_indicators: 1D array-like
        Event indicators (1 = event occurred, 0 = censored).
    risk_scores: 1D array-like
        Model-predicted risk scores (higher values typically indicate higher risk).
    subset: 1D array-like or set
        A set (or list) of indices. Any pair (i, j) where both i and j are in this subset 
        will be excluded from the C-Index calculation.

    Returns
    ----------
    c_index: float
        The computed conditional C-Index.
    """

    event_times = np.asarray(event_times)
    event_indicators = np.asarray(event_indicators)
    risk_scores = np.asarray(risk_scores)
    
    # If subset is not already a set, convert it to a set for faster membership checks
    if not isinstance(subset, set):
        subset = set(subset)

    n = len(event_times)
    if not (len(event_indicators) == n and len(risk_scores) == n):
        raise ValueError("Input arrays must have the same length!")

    # Counters
    concordant = 0  # number of concordant (correct) pairs
    comparable = 0  # number of comparable pairs

    # Pairwise iteration (i, j) with i < j to avoid duplication
    for i in range(n):
        for j in range(i + 1, n):
            # If both i and j are in the excluded subset, skip them
            if i in subset and j in subset:
                continue

            # Determine if the pair is comparable: 
            # We need at least one event, and to see who occurred first.
            # Common approach:
            #    If T_i < T_j and E_i=1, then (i, j) is comparable
            #    If T_j < T_i and E_j=1, then (i, j) is comparable
            #
            # Adjust as needed for your specific requirements
            if event_times[i] < event_times[j] and event_indicators[i] == 1:
                # This is a comparable pair
                comparable += 1
                # Check for concordance: T_i < T_j => risk_scores[i] should be > risk_scores[j]
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    # If you want to handle ties by giving half credit
                    concordant += 0.5
            elif event_times[j] < event_times[i] and event_indicators[j] == 1:
                # This is also a comparable pair
                comparable += 1
                # Check for concordance: T_j < T_i => risk_scores[j] should be > risk_scores[i]
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] == risk_scores[i]:
                    # Half credit for ties
                    concordant += 0.5
            else:
                # Skip if the pair is not comparable
                continue

    if comparable == 0:
        # Return NaN, or alternatively 0, raise an exception, etc.
        return np.nan

    return concordant / comparable


# Example usage
if __name__ == "__main__":
    # Suppose we have the following simple data:
    times = np.random.randint(0, 100, 50)  # survival times
    events = np.random.choice([0, 1], 50)  # event indicators
    scores = np.random.rand(50)  # risk scores

    # Suppose subset = {1, 3}, and we want to exclude pairwise comparisons between id=1 and id=3
    for i in range(50):
        my_subset = set(range(50)) - {i}
        cidx = conditional_cindex(times, events, scores, my_subset)
        print("Conditional C-Index =", cidx)



