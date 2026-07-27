import numpy as np
import torch


def sample_first_thetas(
        N: int,
        gamma_range=(0.95, 0.9999),
        sp_range=(1e-4, 1e-3),
        device="cpu",
):
    '''
    1) Sample N thetas from the prior distribution. 
    '''
    g0, g1 = gamma_range
    p0, p1 = sp_range

    gamma = torch.rand(N, 1, device=device) * (g1 - g0) + g0
    sp = torch.rand(N, 1, device=device) * (p1 - p0) + p0

    # sample uniformly in log-space for sp 
    #log_sp = torch.rand(N, 1, device=device) * (torch.log(torch.tensor(p1)) - torch.log(torch.tensor(p0))) + torch.log(torch.tensor(p0))
    
    # exponentiate back to normal space
    #sp = torch.exp(log_sp)

    # stack parameters into theta
    theta = torch.cat([gamma, sp], dim=1)

    theta = theta.tolist()
    theta = np.array(theta)

    return theta

def sample_split_cost(
        #ratio_range,
        move_range=(0.00025, 0.004),
        turn_range=(0.00025, 0.004),
        n = 25,
        distribution="even_steps",
        seed=None,
):
    '''
    (not implemented: Ratio range determines the relative size of each parameter with respect to the other.)
    n is the number of samples for each parameter which will result in a n*n total samples.
    Move and turn ranges determine the absolute lower and upper ceilings for each parameter.
    Distribution argument determines how the parameter values are sampled:
        at even ratio steps,
        randomly via uniform distribution,
        randomly via log distribution
    '''

    rng = np.random.default_rng(seed)

    #r0, r1 = ratio_range
    m0, m1 = move_range
    t0, t1 = turn_range

    if m0 <= 0 or m1 <= 0 or t0 <= 0 or t1 <= 0:
        raise ValueError("Cost ranges must be positive.")

    if m0 >= m1 or t0 >= t1:
        raise ValueError("Each range must be ordered as (lower, upper).")

    if distribution=="even_steps":

        n_per_dim = int(np.sqrt(n))

        if n_per_dim ** 2 != n:
            raise ValueError(
                "For even steps, n must be a perfect square"
            )

        translation_costs = np.linspace(m0, m1, n_per_dim)
        turning_costs = np.linspace(t0, t1, n_per_dim)
        
        theta = np.array([
        [tc,rc]
        for tc in translation_costs
        for rc in turning_costs
        ])
    
    elif distribution=="uniform":

        translation_costs = rng.uniform(m0, m1, n)
        turning_costs = rng.uniform(t0, t1, n)

        theta = np.column_stack([
            translation_costs,
            turning_costs
        ])
    
    elif distribution=="log_uniform":

        translation_costs = np.exp(
            rng.uniform(np.log(m0), np.log(m1), n)
        )

        turning_costs = np.exp(
            rng.uniform(np.log(t0), np.log(t1), n)
        )

        theta = np.column_stack([
            translation_costs,
            turning_costs
        ])
    
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: even_steps, uniform, log_uniform"
        )

    return theta