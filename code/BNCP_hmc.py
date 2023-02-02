# This is the code file for paper "Modeling Contingent Decision Behavior:
# A Bayesian Nonparametric Preference Learning Approach"

import argparse
import logging
import math
import re
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, config_enumerate, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.ops.special import safe_log
from pyro.ops.tensor_utils import convolve
from pyro.util import warn_if_nan

logging.basicConfig(format="%(message)s", level=logging.INFO)


def global_model(sample_set):
    tau = args.tau
    R0 = pyro.sample("R0", dist.LogNormal(0.0, 1.0))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    ra_s = -R0 / (tau * sample_set)
    prob_i = 1 / (1 + tau)

    return ra_s, prob_i, rho


def discrete_model(args, data):
    ra_s, prob_i, rho = global_model(args.sample_set)

    S = torch.tensor(args.sample_set - 1.0)
    I = torch.tensor(1.0)
    for t, datum in enumerate(data):
        sichong = pyro.sample("sichong_{}".format(t), dist.Binomial(S, -(ra_s * I).expm1()))
        xixian = pyro.sample("xixian_{}".format(t), dist.Binomial(I, prob_i))
        S = pyro.deterministic("S_{}".format(t), S - sichong)
        I = pyro.deterministic("I_{}".format(t), I + sichong - xixian)
        pyro.sample("obs_{}".format(t), dist.ExtendedBinomial(sichong, rho), obs=datum)


@config_enumerate
def reparameterized_discrete_model(args, data):
    ra_s, prob_i, rho = global_model(args.sample_set)

    S_curr = torch.tensor(args.sample_set - 1.0)
    I_curr = torch.tensor(1.0)
    for t, datum in enumerate(data):
        S_prev, I_prev = S_curr, I_curr
        S_curr = pyro.sample(
            "S_{}".format(t), dist.Binomial(args.sample_set, 0.5).mask(False)
        )
        I_curr = pyro.sample(
            "I_{}".format(t), dist.Binomial(args.sample_set, 0.5).mask(False)
        )

        sichong = S_prev - S_curr
        xixian = I_prev - I_curr + sichong
        pyro.sample(
            "sichong_{}".format(t),
            dist.ExtendedBinomial(S_prev, -(ra_s * I_prev).expm1()),
            obs=sichong,
        )
        pyro.sample("xixian_{}".format(t), dist.ExtendedBinomial(I_prev, prob_i), obs=xixian)
        pyro.sample("obs_{}".format(t), dist.ExtendedBinomial(sichong, rho), obs=datum)


def infer_hmc_enum(args, data):
    model = reparameterized_discrete_model
    return _infer_hmc(args, data, model)


def _infer_hmc(args, data, model, init_values={}):
    logging.info("Running inference...")
    kernel = NUTS(
        model,
        full_mass=[("R0", "rho")],
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


def quantize(name, x_real, min, max):
    assert min < max
    lb = x_real.detach().floor()

    s = x_real - lb
    ss = s * s
    t = 1 - s
    tt = t * t
    probs = torch.stack(
        [
            t * tt,
            4 + ss * (3 * s - 6),
            4 + tt * (3 * t - 6),
            s * ss,
        ],
        dim=-1,
    ) * (1 / 6)
    q = pyro.sample("Q_" + name, dist.Categorical(probs)).type_as(x_real)

    x = lb + q - 1
    x = torch.max(x, 2 * min - 1 - x)
    x = torch.min(x, 2 * max + 1 - x)

    return pyro.deterministic(name, x)


@config_enumerate
def abstract_model(args, data):
    ra_s, prob_i, rho = global_model(args.sample_set)

    S_aux = pyro.sample(
        "S_aux",
        dist.Uniform(-0.5, args.sample_set + 0.5)
        .mask(False)
        .expand(data.shape)
        .to_event(1),
    )
    I_aux = pyro.sample(
        "I_aux",
        dist.Uniform(-0.5, args.sample_set + 0.5)
        .mask(False)
        .expand(data.shape)
        .to_event(1),
    )

    S_curr = torch.tensor(args.sample_set - 1.0)
    I_curr = torch.tensor(1.0)
    for t, datum in poutine.markov(enumerate(data)):
        S_prev, I_prev = S_curr, I_curr
        S_curr = quantize("S_{}".format(t), S_aux[..., t], min=0, max=args.sample_set)
        I_curr = quantize("I_{}".format(t), I_aux[..., t], min=0, max=args.sample_set)

        sichong = S_prev - S_curr
        xixian = I_prev - I_curr + sichong
        pyro.sample(
            "sichong_{}".format(t),
            dist.ExtendedBinomial(S_prev, -(ra_s * I_prev).expm1()),
            obs=sichong,
        )
        pyro.sample("xixian_{}".format(t), dist.ExtendedBinomial(I_prev, prob_i), obs=xixian)
        pyro.sample("obs_{}".format(t), dist.ExtendedBinomial(sichong, rho), obs=datum)


def heuristic_init(args, data):
    S0 = args.sample_set - 1
    sichong = data * min(2.0, (S0 / data.sum()).sqrt())
    S_aux = (S0 - sichong.cumsum(-1)).clamp(min=0.5)
    sichong[0] += 1
    reco = torch.arange(30.0).div(args.time).neg().exp()
    I_aux = convolve(sichong, reco)[: len(data)].clamp(min=0.5)

    return {
        "R0": torch.tensor(2.0),
        "rho": torch.tensor(0.5),
        "S_aux": S_aux,
        "I_aux": I_aux,
    }


def infer_hmc_cont(model, args, data):
    init_values = heuristic_init(args, data)
    return _infer_hmc(args, data, model, init_values=init_values)


def quantize_enumerate(x_real, min, max):
    assert min < max
    lb = x_real.detach().floor()

    s = x_real - lb
    ss = s * s
    t = 1 - s
    tt = t * t
    probs = torch.stack(
        [
            t * tt,
            4 + ss * (3 * s - 6),
            4 + tt * (3 * t - 6),
            s * ss,
        ],
        dim=-1,
    ) * (1 / 6)
    logits = safe_log(probs)
    q = torch.arange(-1.0, 3.0)

    x = lb.unsqueeze(-1) + q
    x = torch.max(x, 2 * min - 1 - x)
    x = torch.min(x, 2 * max + 1 - x)
    return x, logits


def vectorized_model(args, data):
    ra_s, prob_i, rho = global_model(args.sample_set)

    S_aux = pyro.sample(
        "S_aux",
        dist.Uniform(-0.5, args.sample_set + 0.5)
        .mask(False)
        .expand(data.shape)
        .to_event(1),
    )
    I_aux = pyro.sample(
        "I_aux",
        dist.Uniform(-0.5, args.sample_set + 0.5)
        .mask(False)
        .expand(data.shape)
        .to_event(1),
    )

    S_curr, S_logp = quantize_enumerate(S_aux, min=0, max=args.sample_set)
    I_curr, I_logp = quantize_enumerate(I_aux, min=0, max=args.sample_set)
    S_prev = torch.nn.functional.pad(
        S_curr[:-1], (0, 0, 1, 0), value=args.sample_set - 1
    )
    I_prev = torch.nn.functional.pad(I_curr[:-1], (0, 0, 1, 0), value=1)
    T = len(data)
    Q = 4
    S_prev = S_prev.reshape(T, Q, 1, 1, 1)
    I_prev = I_prev.reshape(T, 1, Q, 1, 1)
    S_curr = S_curr.reshape(T, 1, 1, Q, 1)
    S_logp = S_logp.reshape(T, 1, 1, Q, 1)
    I_curr = I_curr.reshape(T, 1, 1, 1, Q)
    I_logp = I_logp.reshape(T, 1, 1, 1, Q)
    data = data.reshape(T, 1, 1, 1, 1)

    sichong = S_prev - S_curr
    xixian = I_prev - I_curr + sichong

    sichong_logp = dist.ExtendedBinomial(S_prev, -(ra_s * I_prev).expm1()).log_prob(sichong)
    xixian_logp = dist.ExtendedBinomial(I_prev, prob_i).log_prob(xixian)
    obs_logp = dist.ExtendedBinomial(sichong, rho).log_prob(data)

    logp = S_logp + (I_logp + obs_logp) + sichong_logp + xixian_logp
    logp = logp.reshape(-1, Q * Q, Q * Q)
    logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
    logp = logp.reshape(-1).logsumexp(0)
    logp = logp - math.log(4)
    warn_if_nan(logp)
    pyro.factor("obs", logp)


def evaluate(args, samples):
    names = {"basic_rep_number": "R0", "res_rate": "rho"}
    for name, key in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info(
            "{}: truth = {:0.3g}, estimate = {:0.3g} {:0.3g}".format(
                key, getattr(args, name), mean, std
            )
        )


@torch.no_grad()
def predict(args, data, samples, truth=None):
    logging.info("Predicting ".format(args.forecast))
    part_plate = pyro.plate("parts", args.num_samples, dim=-1)

    model = poutine.condition(abstract_model, samples)
    model = part_plate(model)
    model = infer_discrete(model, first_available_dim=-2)
    with poutine.trace() as tr:
        model(args, data)
    samples = OrderedDict(
        (name, site["value"])
        for name, site in tr.trace.nodes.items()
        if site["type"] == "sample"
    )

    extended_data = list(data) + [None] * args.forecast
    model = poutine.condition(discrete_model, samples)
    model = part_plate(model)
    with poutine.trace() as tr:
        model(args, extended_data)
    samples = OrderedDict(
        (name, site["value"])
        for name, site in tr.trace.nodes.items()
        if site["type"] == "sample"
    )

    for key in ("S", "I", "sichong", "xixian"):
        pattern = key + "_[0-9]+"
        series = [value for name, value in samples.items() if re.match(pattern, name)]
        assert len(series) == args.HDP + args.forecast
        series[0] = series[0].expand(series[1].shape)
        samples[key] = torch.stack(series, dim=-1)
    sichong = samples["sichong"]
    median = sichong.median(dim=0).values

    return samples


def main(args):
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


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.4")
    parser = argparse.ArgumentParser(description="BNCP epidemiology modeling using HMC")
    parser.add_argument("-p", "--sample_set", default=10, type=int)
    parser.add_argument("-m", "--min-observations", default=3, type=int)
    parser.add_argument("-d", "--HDP", default=10, type=int)
    parser.add_argument("-f", "--forecast", default=0, type=int)
    parser.add_argument("-R0", "--basic-reproduction-number", default=1.5, type=float)
    parser.add_argument("-tau", "--generative-time", default=7.0, type=float)
    parser.add_argument("-rho", "--inference-time", default=0.5, type=float)
    parser.add_argument(
        "-e", "--enum", action="store_true", help="use the full enumeration model"
    )
    parser.add_argument(
        "-s",
        "--sequential",
        action="store_true",
        help="use the break-sticking model",
    )
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("-w", "--warmup-steps", default=100, type=int)
    parser.add_argument("-t", "--max-tree-depth", default=5, type=int)
    parser.add_argument("-r", "--rng-seed", default=0, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.double:
        if args.cuda:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
    elif args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)
