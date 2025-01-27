# Fixed geometry inversion - use mean geometry solution and only invert for pressure changes for all explosions

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


sys.path.append("../")
import station_positions
import tilt_fwd

os.environ["OMP_NUM_THREADS"] = "1"


def cli():
    parser = argparse.ArgumentParser(
        prog="invert_yang_nish",
        description="Invert Augustine 2006 tilt with combined Yang/Nishimura deformation source - fixed geometry",
    )
    parser.add_argument(
        "samples", type=str, help="MCMC samples to determine source geometry"
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
        default=1000,
        type=int,
        help="Number of steps each walker should take during the MCMC inversion (default = 1k)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./yang_nish_fixgeom.h5",
        type=str,
        help="Output HDF5 file of samples (default = ./yang_nish_fixgeom_[n].h5)",
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

fwm = tilt_fwd.TiltFwd()

# Load mcmc samples and get geometry
sampler = emcee.backends.HDFBackend(args.samples)
samps = sampler.get_chain(discard=150000, thin=10, flat=True)
xmed = np.median(samps, axis=0)

if args.yangonly:
    xgeom = xmed[:7]
else:
    xgeom = xmed[:10]


# MCMC infra
def ln_like(x, xgeom, df):
    x = np.append(xgeom, x)
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
    if eventid in au14_nodata_events:
        tx[0] = tx[0][sta != "AU14"]
        ty[0] = ty[0][sta != "AU14"]

    # Get rid of likely erroneous tilt on AU13 during event 12
    if eventid == 12:
        tx[0] = tx[0][sta != "AU13"]
        ty[0] = ty[0][sta != "AU13"]
        df = df[df["station"] != "AU13"]

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


def ln_prob(x, xgeom, df, bounds):
    prior = ln_prior(x, bounds)
    if not np.isfinite(prior):
        return -np.inf
    like = ln_like(x, xgeom, df)
    if np.isnan(like):
        return -np.inf

    return like + prior


def ln_prob_minimizer(x, xgeom, df, bounds):
    lnp = -1 * ln_prob(x, xgeom, df, bounds)
    print(lnp)
    return lnp


# Set up priors
# Type can be "uniform" or "normal"
# Params
# uniform (min, max)
# normal (mean, std)
if args.yangonly:
    priors = [("uniform", (-100, 0))]  # pressure yang
else:
    priors = [
        ("uniform", (-100, 0)),  # pressure yang
        ("uniform", (-125, 0)),  # pressure nish
        ("uniform", (1e2, 6e3)),  # length nish
    ]

# Set initial point
p0_yang = -50
p0_nish = -50
l0_nish = 2e3

if args.yangonly:
    x0 = p0_yang
else:
    x0 = (p0_yang, p0_nish, l0_nish)

###########

# Select subset of events

for eventid in tiltdf["event"].unique():
    if eventid != 12:
        continue
    tiltdf_sub = tiltdf[tiltdf["event"] == eventid]

    # res = scipy.optimize.minimize(
    #    ln_prob_minimizer,
    #    x0,
    #    args=(xgeom, tiltdf_sub, priors),
    #    method="Nelder-Mead",
    #    options={"fatol": 0.1, "maxiter": 2000},
    # )
    # print(res)

    # class fakeargs:
    #    def __init__(self, output):
    #        self.output = output

    # import mcmc_plots

    # mcmc_plots.map_plot(res.x[np.newaxis, :], fakeargs("./fig/nm_"), yangonly=True)

    # x0 = res.x

    ###########

    # Initial positions of walkers (300)
    if args.yangonly:
        pos = np.linspace(priors[0][1][0], priors[0][1][1], 300)[:, np.newaxis]
    else:
        grid = [np.linspace(p[1][0], p[1][1], 7) for p in priors]
        grid = np.meshgrid(*grid)
        pos = np.vstack([g.flatten() for g in grid]).T
    # pos = x0 + 1e-2 * np.random.randn(300, len(x0))
    nwalkers, ndim = pos.shape

    backend = emcee.backends.HDFBackend(args.output.replace(".h5", "_%d.h5" % eventid))
    backend.reset(nwalkers, ndim)

    with multiprocessing.Pool(args.nproc) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            ln_prob,
            args=(xgeom, tiltdf_sub, priors),
            pool=pool,
            backend=backend,
        )
        sampler.run_mcmc(pos, args.nstep, progress=True)

    print(np.median(sampler.get_chain(discard=500, thin=10, flat=True), axis=0))
