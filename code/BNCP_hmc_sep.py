# This is the code file for paper "Modeling Contingent Decision Behavior:
# A Bayesian Nonparametric Preference Learning Approach"

import pyro
import torch
import pyro.distributions as dist
from pyro import poutine
from pyro.ops.indexing import Vindex
import torch.nn.functional as F


def model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K_plate = pyro.plate("model", K, dim=-1)
    with K_plate:
        u = pyro.sample("u", dist.Dirichlet(torch.ones((N_DIM_U, ), device=DEVICE)))  # (K, N_DIM_U)
        sigma = pyro.sample("sigma", dist.Dirichlet(torch.ones((Q, ), device=DEVICE)))  # (K, Q)
        phi = pyro.sample("phi", dist.Dirichlet(torch.ones((N_WORD, ), device=DEVICE)))  # (K, V)
    F_matrix_batched = F_MATRIX.repeat(K, 1, 1)
    xi = torch.bmm(F_matrix_batched, sigma.unsqueeze(-1))
    xi = xi.squeeze(-1)  # (K, Q+1)

    with pyro.plate("DM", R, dim=-2):
        tau = pyro.sample("tau",
                          dist.Gamma(
                              torch.tensor(ETA, device=DEVICE).float(),
                              torch.tensor(MU, device=DEVICE).float()))  # (R, 1)
        with K_plate:
            pi_prime = pyro.sample(
                "pi_prime", dist.Beta(torch.tensor(1., device=DEVICE),
                                      torch.tensor(ALPHA_0, device=DEVICE).float()))
        cumprod = torch.cumprod(1 - pi_prime, dim=1)  # (R, K)
        pi = pi_prime * torch.cat((torch.ones((R, 1), device=DEVICE), cumprod[:, :-1]), dim=1)
        pi_last_col = 1 - pi[:, :-1].sum(dim=1)
        pi = torch.cat((pi[:, :-1], pi_last_col.unsqueeze(-1)), dim=1)  # (R, K)

    with pyro.plate("R_2", R, dim=-1):
        with pyro.plate("ALT", N_ALT, dim=-2):
            with pyro.plate("WORD", N_WORD, dim=-3):
                z = pyro.sample("z", dist.Categorical(pi))
                p_w = Vindex(phi)[z]
                w = pyro.sample("w", dist.Categorical(p_w))  # (N_WORD, N_ALT, R)

    z_probs = torch.zeros((N_ALT, R, K), device=DEVICE)  # (N_ALT, R, K)
    for r in range(R):
        for a in range(N_ALT):
            z_count = z[:, a, r].bincount(minlength=K)
            z_prob = z_count / z_count.sum()
            z_probs[a, r, :] = z_prob

    with pyro.plate("R_3", R):
        with pyro.plate("ALT_2", N_ALT):
            y = pyro.sample("y", dist.Categorical(z_probs))  # (N_ALT, R)

    U = u @ V.T  # (K, N_ALT)
    theta = torch.zeros((K, N_ALT, Q), device=DEVICE)
    for k in range(K):
        for i in range(N_ALT):
            theta_ki = torch.zeros((Q, ), device=DEVICE)
            theta_ki[0] = xi[k, 0] - EPSILON - U[k, i]
            for q in range(1, Q - 1):
                theta_ki[q] = torch.min(U[k, i] - xi[k, q - 1], xi[k, q] - EPSILON - U[k, i])
            theta_ki[Q - 1] = U[k, i] - xi[k, Q - 2]
            theta[k, i, :] = theta_ki
    assignment_prob = tau.unsqueeze(-1).unsqueeze(-1) * theta.unsqueeze(0)
    assignment_prob = F.softmax(assignment_prob, dim=-1)  # (R, K, N_ALT, Q)
    assignment_prob = assignment_prob.permute(2, 0, 1, 3)  # (N_ALT, R, K, Q)
    for i in range(N_ALT):
        for r in range(R):
            assignment_prob_ir = assignment_prob[i, r]  # (K, Q)
            y_ir = y[i, r]
            x_probs = Vindex(assignment_prob_ir)[y_ir]
            x = pyro.sample(f"x_{r}_{i}", dist.Categorical(x_probs))


def trace_model_shape():
    trace = poutine.trace(model).get_trace()
    trace.compute_log_prob()
    with open("./model_shape.log", "w") as f:
        f.write(trace.format_shapes())
    print("Trace log save to ./model_shape.log")
    
def infer_hmc_cont(model, args, data):
    init_values = heuristic_init(args, data)
    return _infer_hmc(args, data, model, init_values=init_values)
    
def _infer_hmc(args, data, model, init_values={}):
    logging.info("Running inference...")
    kernel = NUTS(
        model,
        max_tree_depth=args.max_tree_depth,
        init_strategy=init_to_value(values=init_values),
        jit_compile=args.jit,
        ignore_jit_warnings=True,
    )

    energies = []

    def hook_fn(kernel, *unused):
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            logging.info("potential = {:0.6g}".format(e))

    mcmc = MCMC(
        kernel,
        hook_fn=hook_fn,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
    )
    mcmc.run(args, data)
    mcmc.summary()

    samples = mcmc.get_samples()
    return samples


if __name__ == "__main__":
    trace_model_shape()
    
    pyro.set_rng_seed(args.rng_seed)

    dataset = "../data"
    obs = dataset["obs"][: args.HDP]
    
    if args.enum:
        samples = infer_hmc_enum(args, obs)
    elif args.sequential:
        samples = infer_hmc_cont(abstract_model, args, obs)
    else:
        samples = infer_hmc_cont(vectorized_model, args, obs)
        
    evaluate(args, samples)

    if args.forecast:
        samples = predict(args, obs, samples, truth=dataset["sichong"])

    return samples
