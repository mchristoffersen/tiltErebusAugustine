import multiprocessing
import os
import pickle
import sys
import argparse

import emcee
import numpy as np
import pandas as pd
import scipy
import vmod.data
import vmod.inverse
import vmod.source

import station_positions
import tilt_fwd

os.environ["OMP_NUM_THREADS"] = "1"


def cli():
    parser = argparse.ArgumentParser(
        prog="invert_yang_nish",
        description="Invert Augustine 2006 tilt with combined Yang/Nishimura deformation source",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        default=1,
        type=int,
        help="Number of processes to spawn during MCMC inversion (default = 1)",
    )
    parser.add_argument(
        "--nstep",
        "-s",
        default=100000,
        type=int,
        help="Number of steps each walker should take during the MCMC inversion (default = 100k)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./yang_nish.h5",
        type=str,
        help="Output HDF5 file of samples (default = ./yang_nish.h5)",
    )
    parser.add_argument(
        "--uncertainty",
        "-u",
        default=0.1,
        type=float,
        help="Tilt measurement uncertainty in urad (default = 0.1)",
    )
    parser.add_argument(
        "--yangonly", "-y", action="store_true", help="Yang source only inversion"
    )
    # parser.add_argument("x0", type="str", help="Starting point, text file with one parameter value per line")
    # parser.add_argument("bounds", type="str", help="
    return parser.parse_args()


args = cli()

# Grab station positions
df, ox, oy = station_positions.station_positions()
df["station"] = df[" Station "]

# Load event tilts
# Load tilts
with open("./event_tilts.pkl", mode="rb") as fd:
    tiltdf = pickle.load(fd)

# Select subset of events
events = [2, 3, 4, 6, 9, 11, 13]

for eventid in tiltdf["event"].unique():
    if eventid not in events:
        tiltdf = tiltdf[tiltdf["event"] != eventid]

fwm = tilt_fwd.TiltFwd()


# MCMC infra
def ln_like(x, df):
    if args.yangonly:
        sta, tx, ty = fwm.tilt_yangonly(x)
    else:
        sta, tx, ty = fwm.tilt(x)

    sta = np.array(sta)

    # Bail if there are any nans in returned tilts (bad parameters)
    if np.any(np.isnan(tx)) or np.any(np.isnan(ty)):
        print("NaN(s) in modeled tilts")
        return -np.inf

    # Get rid of no-data AU14 events
    au14_nodata_events = [3, 4, 5]
    for i, event in enumerate(events):
        if event in au14_nodata_events:
            tx[i] = tx[i][sta != "AU14"]
            ty[i] = ty[i][sta != "AU14"]

    # Scale to microradians
    sse = np.nansum(
        np.power(np.concatenate(tx) * 1e6 - df["etilt"] * 1e6, 2)
    ) + np.nansum(np.power(np.concatenate(ty) * 1e6 - df["ntilt"] * 1e6, 2))

    sigma2 = args.uncertainty**2  # microradians

    return (-0.5 * sse) / sigma2 + np.log(2 * np.pi * sigma2)


def ln_uniform(x, minx, maxx):
    if x >= minx and x <= maxx:
        return 0.0
    return -np.inf


def ln_normal(x, mu, sigma):
    return -0.5 * (np.log(2 * np.pi * sigma**2) + (((x - mu) ** 2) / sigma**2))


def ln_prior(x, priors):
    # Apply priors
    lnp = 0.0
    for i, prior in enumerate(priors):
        ptype = prior[0]
        pparams = prior[1]
        if ptype == "uniform":
            lnp += ln_uniform(x[i], pparams[0], pparams[1])
        elif ptype == "normal":
            lnp += ln_normal(x[i], pparams[0], pparams[1])
        else:
            raise ValueError("Illegal prior type: %s" % ptype)

    return lnp


def ln_prob(x, df, bounds):
    prior = ln_prior(x, bounds)
    if not np.isfinite(prior):
        return -np.inf
    like = ln_like(x, df)
    if np.isnan(like):
        return -np.inf

    return like + prior


def ln_prob_minimizer(x, df, bounds):
    lnp = -1 * ln_prob(x, df, bounds)
    print(lnp)
    return lnp


# Set up priors
# Type can be "uniform" or "normal"
# Params
# uniform (min, max)
# normal (mean, std)
if args.yangonly:
    priors = [
        ("normal", (0, 0.5e3)),  # x yang
        ("normal", (0, 0.5e3)),  # y yang
        ("uniform", (1e3, 30e3)),  # depth yang
        ("uniform", (5e1, 6e3)),  # semimajor yang
        ("uniform", (5e1, 3e3)),  # semiminor yang
        ("uniform", (0, 360)),  # strike yang
        ("uniform", (0, 90)),  # dip yang
    ] + [
        ("uniform", (-100, 0)),  # pressure yang
    ] * len(
        tiltdf["event"].unique()
    )
else:
    priors = [
        ("normal", (0, 0.5e3)),  # x yang
        ("normal", (0, 0.5e3)),  # y yang
        ("uniform", (1e3, 30e3)),  # depth yang
        ("uniform", (5e1, 6e3)),  # semimajor yang
        ("uniform", (5e1, 3e3)),  # semiminor yang
        ("uniform", (0, 360)),  # strike yang
        ("uniform", (0, 90)),  # dip yang
        ("normal", (0, 50)),  # x nish
        ("normal", (0, 50)),  # y nish
        ("normal", (40, 5)),  # radius nish
    ] + [
        ("uniform", (-100, 0)),  # pressure yang
        ("uniform", (-125, 0)),  # pressure nish
        ("uniform", (1e2, 6e3)),  # length nish
    ] * len(
        tiltdf["event"].unique()
    )

# Set initial point
x0_yang = [
    0,  # xcen
    0,  # ycen
    4e3,  # depth
    1e3,  # semi-major
    5e2,  # semi-minor
    180,  # strike
    90,  # dip
]

x0_nish = [
    0,  # xcen
    0,  # yen
    40,  # radius
]

p0_yang = -25
p0_nish = -25
l0_nish = 2e3

if args.yangonly:
    x0 = x0_yang + ([p0_yang] * len(tiltdf["event"].unique()))
else:
    x0 = (
        x0_yang
        + x0_nish
        + ([p0_yang, p0_nish, l0_nish] * len(tiltdf["event"].unique()))
    )

###########
res = scipy.optimize.minimize(
    ln_prob_minimizer,
    x0,
    args=(tiltdf, priors),
    method="Nelder-Mead",
    options={"fatol": 1, "maxiter": 2000, "adaptive": True},
)
print(res.x)


class fakeargs:
    def __init__(self, output):
        self.output = output


# import mcmc_plots

# mcmc_plots.map_plot(res.x[np.newaxis, :], fakeargs("./fig/nm_"), yangonly=True)


x0 = res.x

###########


# Initial positions of walkers (300)
pos = x0 + 1e-2 * np.random.randn(300, len(x0))
nwalkers, ndim = pos.shape

backend = emcee.backends.HDFBackend(args.output)
backend.reset(nwalkers, ndim)

with multiprocessing.Pool(args.nproc) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, ln_prob, args=(tiltdf, priors), pool=pool, backend=backend
    )
    sampler.run_mcmc(pos, args.nstep, progress=True)
