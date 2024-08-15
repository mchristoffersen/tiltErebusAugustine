# Plots of mcmc results
import argparse
import pickle

import corner
import emcee
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import fiona
import rasterio

import sys
sys.path.append("..")
import station_positions
import tilt_fwd


def cli():
    parser = argparse.ArgumentParser(
        prog="mcmc_plots",
        description="Plot MCMC results",
    )
    parser.add_argument(
        "samples", help="HDF5 file withs samples from posterior", type=str
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./fig/",
        type=str,
        help="Output directory (default = ./fig/)",
    )
    parser.add_argument(
        "--thin",
        "-t",
        default=10,
        type=int,
        help="Factor to decimate samples by (default = 10)",
    )
    parser.add_argument(
        "--burnin",
        "-b",
        default=150000,
        type=int,
        help="Number of samples to discard from each walker (default = 150k)",
    )
    parser.add_argument(
        "--yangonly", "-y", action="store_true", help="Yang source only inversion"
    )
    return parser.parse_args()


def walkers(samps, args, yangonly=False):
    if yangonly:
        labels = [
            "YangX",
            "YangY",
            "YangZ",
            "YangMax",
            "YangMin",
            "YangStr",
            "YangDip",
        ] + (["YangPres"] * 7)
    else:
        labels = [
            "YangX",
            "YangY",
            "YangZ",
            "YangMax",
            "YangMin",
            "YangStr",
            "YangDip",
            "NishX",
            "NishY",
            "NishRad",
        ] + (["YangPres", "NishPres", "NishLen"] * 7)

    for i in range(samps.shape[2]):
        if yangonly:
            if i > 6:
                name = labels[i] + str(i - 5)
            else:
                name = labels[i]
        else:
            if i > 9:
                name = labels[i] + str((i - 9) // 3 + 1)
            else:
                name = labels[i]
        plt.plot(samps[:, :, i], "k-", alpha=0.1)
        plt.title("%s" % name)
        plt.xlabel("Step (x10)")
        plt.ylabel("%s" % name)
        plt.savefig(args.output + "%s_walker.png" % name, dpi=300, bbox_inches="tight")
        plt.close()


def corner_plot_yangonly(samps, args):
    psamps = np.copy(samps)
    # Scale parameters for corner plot
    psamps[:, :5] /= 1e3
    psamps[:, 7:] /= 1e3

    fig, axs = plt.subplots(7, 9, figsize=(0.85 * (9 / 7) * 6, 0.85 * 6))

    for ax in axs.flatten():
        ax.axis("off")

    # Plot yang geometry
    labels = [
        "Yang X\n(km)",
        "Yang Y\n(km)",
        "Yang Z\n(km)",
        "Yang Maj\n(km)",
        "Yang Min\n(km)",
        "Yang Str\n(deg)",
        "Yang Dip\n(deg)",
    ]
    for i in range(0, 7):
        for j in range(0, 7):
            axs[i, j].tick_params(
                axis="both",
                left=False,
                labelleft=False,
                labelbottom=False,
                direction="in",
                right=False,
                top=False,
            )
            axs[i, j].grid(False)
            if i == j:
                axs[i, j].axis("on")
                axs[i, j].set_yticks([])
                axs[i, j].hist(psamps[:, i], histtype="step", color="k", linewidth=0.5)
                axs[i, j].tick_params(
                    axis="both", direction="in", labelrotation=90, labelbottom=False
                )
                q_lo, q_mid, q_hi = corner.quantile(psamps[:, i], [0.16, 0.5, 0.84])
                q_m, q_p = q_mid - q_lo, q_hi - q_mid
                fmt = "{{0:{0}}}".format(".2f").format
                fmtsup = "{{0:{0}}}".format(".1f").format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmtsup(q_m), fmtsup(q_p))
                axs[i, j].set_title(title, fontsize=8)

            if i > j:
                axs[i, j].axis("on")
                corner.hist2d(
                    psamps[:, j],
                    psamps[:, i],
                    ax=axs[i, j],
                    plot_datapoints=False,
                    contour_kwargs={"linewidths": 0.5},
                )
                axs[i, j].tick_params(
                    axis="both",
                    direction="in",
                    labelbottom=False,
                    labelleft=False,
                    bottom=False,
                    left=False,
                )
            if i == 6:
                axs[i, j].tick_params(
                    axis="x",
                    labelbottom=True,
                    bottom=True,
                    labelrotation=90,
                    labelsize=8,
                )
                # if(j != 6):
                axs[i, j].set_xlabel(labels[j], fontsize=8)

            if j == 0:
                axs[i, j].tick_params(axis="y", labelleft=True, left=True, labelsize=8)
                if i != 0:
                    axs[i, j].set_ylabel(labels[i], fontsize=8)

    # Plot pressures and nishimura length
    events = [2, 3, 4, 6, 9, 11, 13]
    labels = ["Yang Pres.\n(GPa)", "Nish. Pres.\n(GPa)", "Nish. Len.\n(km)"]

    for i in range(7):
        axs[i, 8].grid(False)
        r = (-0.1, 0)
        axs[i, 8].axis("on")
        axs[i, 8].hist(
            psamps[:, 7 + i],
            histtype="step",
            color="k",
            linewidth=0.5,
            range=r,
        )
        axs[i, 8].set_xlim(r)
        axs[i, 8].tick_params(
            axis="both",
            left=False,
            labelleft=False,
            labelbottom=False,
            direction="in",
            right=False,
            top=False,
        )
        if i == 6:
            axs[i, 8].tick_params(
                axis="x", labelbottom=True, labelrotation=90, labelsize=8
            )
            axs[i, 8].set_xlabel(labels[j - 9], fontsize=8)

        axs[i, 8].set_ylabel(events[i], rotation=0, labelpad=10, fontsize=8)
        axs[i, 8].yaxis.set_label_position("right")

    fig.supylabel("Event", fontsize=8, x=0.93, rotation=90)

    fig.align_ylabels(axs=axs[:, 0])
    fig.align_ylabels(axs=axs[:, 6])
    fig.align_xlabels(axs=axs[6, :])
    fig.align_xlabels(axs=axs[0, 4:7])

    # Add subfig labels
    axs[6, 0].annotate(
        "",
        xy=(0, -1.4),
        xycoords="axes fraction",
        xytext=(8.3, -1.4),
        arrowprops=dict(arrowstyle="-", color="k"),
    )
    axs[6, 0].annotate("a", xy=(4, -1.65), xycoords="axes fraction", fontsize=8)

    axs[6, 8].annotate(
        "",
        xy=(0, -1.4),
        xycoords="axes fraction",
        xytext=(1.1, -1.4),
        arrowprops=dict(arrowstyle="-", color="k"),
    )
    axs[6, 8].annotate("b", xy=(0.5, -1.65), xycoords="axes fraction", fontsize=8)

    fig.savefig(args.output + "./mcmc_yang_fmt.png", dpi=300, bbox_inches="tight")


def corner_plot(samps, args):
    psamps = np.copy(samps)
    # Scale parameters for corner plot
    psamps[:, :5] /= 1e3
    # psamps[:, 7:9] /= 1e3
    psamps[:, 10:] /= 1e3

    fig, axs = plt.subplots(7, 12, figsize=(0.85 * (12 / 7) * 6, 0.85 * 6))

    for ax in axs.flatten():
        ax.axis("off")

    # Plot nish geometry
    labels = ["Nish. X\n(m)", "Nish. Y\n(m)", "Nish. Rad\n(m)"]
    for i in range(0, 3):
        for j in range(4, 7):
            axs[i, j].grid(False)
            if i == j - 4:
                axs[i, j].axis("on")
                axs[i, j].set_yticks([])
                axs[i, j].hist(
                    psamps[:, 9 - i], histtype="step", color="k", linewidth=0.5
                )
                axs[i, j].tick_params(
                    axis="both", direction="in", labelrotation=90, labelbottom=False
                )

                q_lo, q_mid, q_hi = corner.quantile(psamps[:, 9 - i], [0.16, 0.5, 0.84])
                q_m, q_p = q_mid - q_lo, q_hi - q_mid
                fmt = "{{0:{0}}}".format(".2f").format
                fmtsup = "{{0:{0}}}".format(".1f").format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmtsup(q_m), fmtsup(q_p))
                axs[i, j].set_title(title, fontsize=8, y=-0.45)

            if i < j - 4:
                axs[i, j].axis("on")
                corner.hist2d(
                    psamps[:, 9 - (j - 4)],
                    psamps[:, 9 - i],
                    ax=axs[i, j],
                    plot_datapoints=False,
                    contour_kwargs={"linewidths": 0.5},
                )
                axs[i, j].tick_params(
                    axis="both",
                    direction="in",
                    labelbottom=False,
                    labelleft=False,
                    bottom=False,
                    left=False,
                )

            if i == 0:
                axs[i, j].tick_params(
                    axis="x", labeltop=True, top=True, labelrotation=90, labelsize=8
                )
                # if(j != 4):
                axs[i, j].set_xlabel(labels[2 - (j - 4)], fontsize=8)
                axs[i, j].xaxis.set_label_position("top")

            if j == 6:
                axs[i, j].tick_params(
                    axis="y", labelright=True, right=True, labelsize=8
                )
                if i != 2:
                    axs[i, j].set_ylabel(labels[2 - i], fontsize=8, rotation=90)
                    axs[i, j].yaxis.set_label_position("right")

    # Plot yang geometry
    labels = [
        "Yang X\n(km)",
        "Yang Y\n(km)",
        "Yang Z\n(km)",
        "Yang Maj\n(km)",
        "Yang Min\n(km)",
        "Yang Str\n(deg)",
        "Yang Dip\n(deg)",
    ]
    for i in range(0, 7):
        for j in range(0, 7):
            axs[i, j].tick_params(
                axis="both",
                left=False,
                labelleft=False,
                labelbottom=False,
                direction="in",
                right=False,
                top=False,
            )
            axs[i, j].grid(False)
            if i == j:
                axs[i, j].axis("on")
                axs[i, j].set_yticks([])
                axs[i, j].hist(psamps[:, i], histtype="step", color="k", linewidth=0.5)
                axs[i, j].tick_params(
                    axis="both", direction="in", labelrotation=90, labelbottom=False
                )

                q_lo, q_mid, q_hi = corner.quantile(psamps[:, i], [0.16, 0.5, 0.84])
                q_m, q_p = q_mid - q_lo, q_hi - q_mid
                fmt = "{{0:{0}}}".format(".2f").format
                fmtsup = "{{0:{0}}}".format(".1f").format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmtsup(q_m), fmtsup(q_p))
                axs[i, j].set_title(title, fontsize=8)

            if i > j:
                axs[i, j].axis("on")
                corner.hist2d(
                    psamps[:, j],
                    psamps[:, i],
                    ax=axs[i, j],
                    plot_datapoints=False,
                    contour_kwargs={"linewidths": 0.5},
                )
                axs[i, j].tick_params(
                    axis="both",
                    direction="in",
                    labelbottom=False,
                    labelleft=False,
                    bottom=False,
                    left=False,
                )
            if i == 6:
                axs[i, j].tick_params(
                    axis="x",
                    labelbottom=True,
                    bottom=True,
                    labelrotation=90,
                    labelsize=8,
                )
                # if(j != 6):
                axs[i, j].set_xlabel(labels[j], fontsize=8)

            if j == 0:
                axs[i, j].tick_params(axis="y", labelleft=True, left=True, labelsize=8)
                if i != 0:
                    axs[i, j].set_ylabel(labels[i], fontsize=8)

    # Plot pressures and nishimura length
    events = [2, 3, 4, 6, 9, 11, 13]
    labels = ["Yang Pres.\n(GPa)", "Nish. Pres.\n(GPa)", "Nish. Len.\n(km)"]

    for i in range(7):
        for j in range(9, 12):
            axs[i, j].grid(False)
            if j == 11:
                r = (0, 6)
            elif j == 10:
                r = (-0.125, 0)
            elif j == 9:
                r = (-0.1, 0)
            axs[i, j].axis("on")
            axs[i, j].hist(
                psamps[:, 10 + (i * 3) + (j - 9)],
                histtype="step",
                color="k",
                linewidth=0.5,
                range=r,
            )
            axs[i, j].tick_params(
                axis="both",
                left=False,
                labelleft=False,
                labelbottom=False,
                direction="in",
                right=False,
                top=False,
            )
            if i == 6:
                axs[i, j].tick_params(
                    axis="x", labelbottom=True, labelrotation=90, labelsize=8
                )
                axs[i, j].set_xlabel(labels[j - 9], fontsize=8)
            if j == 11:
                axs[i, j].set_ylabel(events[i], rotation=0, labelpad=10, fontsize=8)
                axs[i, j].yaxis.set_label_position("right")

    fig.supylabel("Event", fontsize=8, x=0.93, rotation=90)

    fig.align_ylabels(axs=axs[:, 0])
    fig.align_ylabels(axs=axs[:, 6])
    fig.align_xlabels(axs=axs[6, :])
    fig.align_xlabels(axs=axs[0, 4:7])

    # Add subfig labels
    axs[6, 0].annotate(
        "",
        xy=(0, -1.3),
        xycoords="axes fraction",
        xytext=(8.3, -1.3),
        arrowprops=dict(arrowstyle="-", color="k"),
    )
    axs[6, 0].annotate("a", xy=(4, -1.55), xycoords="axes fraction", fontsize=8)

    axs[0, 4].annotate(
        "",
        xy=(0, 2.25),
        xycoords="axes fraction",
        xytext=(3.4, 2.25),
        arrowprops=dict(arrowstyle="-", color="k"),
    )
    axs[0, 4].annotate("b", xy=(1.6, 2.4), xycoords="axes fraction", fontsize=8)

    axs[6, 9].annotate(
        "",
        xy=(0, -1.3),
        xycoords="axes fraction",
        xytext=(3.4, -1.3),
        arrowprops=dict(arrowstyle="-", color="k"),
    )
    axs[6, 9].annotate("c", xy=(1.6, -1.55), xycoords="axes fraction", fontsize=8)

    fig.savefig(args.output + "./mcmc_yang_nish_fmt.png", dpi=300, bbox_inches="tight")


def map_plot(samps, args, yangonly=False):
    ## Map plot of tilt solutions
    with open("./event_tilts.pkl", mode="rb") as fd:
        dfmeas = pickle.load(fd)

    events = [2, 3, 4, 6, 9, 11, 13]

    for eventid in dfmeas["event"].unique():
        if eventid not in events:
            dfmeas = dfmeas[dfmeas["event"] != eventid]

    # Grab station positions
    df, ox, oy = station_positions.station_positions()
    df["station"] = df[" Station "]
    dfmeas = pd.merge(dfmeas, df[["x", "y", "station"]], on="station")

    contours = []
    with fiona.open("./plotting/augustine_ifsar_contour.gpkg") as fd:
        for line in fd:
            contours.append(list(zip(*line["geometry"]["coordinates"])))

    with rasterio.open("./plotting/augustine_ifsar_hillshade.tif", "r") as src:
        left, bottom, right, top = src.bounds
        hillshade = src.read(1)

    with rasterio.open("./plotting/augustine_ifsar.tif", "r") as src:
        hgt_aug = src.read(1)

    # Make randomly drawn tilt scenarios
    fwm = tilt_fwd.TiltFwd()
    n = 500
    indxs = (np.random.rand(n) * len(samps)).astype(np.int32)
    modtilts_e = []
    modtilts_n = []
    for i in indxs:
        if yangonly:
            sta, tx, ty = fwm.tilt_yangonly(samps[i, :])
        else:
            sta, tx, ty = fwm.tilt(samps[i, :])
        modtilts_e.append(tx)
        modtilts_n.append(ty)
        # print(fwm.tilt(samps[i, :])[1])

    # Get median tilt
    median_tilts = {}

    for i in range(len(events)):
        ntil = np.zeros((len(modtilts_n), len(modtilts_n[0][i])))
        etil = np.zeros((len(modtilts_e), len(modtilts_e[0][i])))
        for j in range(n):
            ntil[j, :] = modtilts_n[j][i]
            etil[j, :] = modtilts_e[j][i]

        median_tilts[events[i]] = (
            np.nanmedian(ntil, axis=0),
            np.nanmedian(etil, axis=0),
        )

    fig, axs = plt.subplots(3, 3, figsize=(6, 5.5), constrained_layout=True)

    axs = axs.flatten()

    letters = ["a", "b", "c", "d", "e", "f"]

    hillshade = hillshade.astype(np.float32)
    hillshade[hgt_aug == 0] = np.nan

    df, ox, oy = station_positions.station_positions()
    df = df[df[" Station "] != "AU15"]

    for i, event in enumerate(events):
        axs[i].grid(False)
        axs[i].set_facecolor("lightblue")

        # Hillshade
        axs[i].imshow(
            hillshade,
            cmap="Greys_r",
            extent=[left, right, bottom, top],
            vmin=50,
            vmax=255,
            zorder=0,
        )
        axs[i].set(
            xlim=(left + 12e3, right - 5e3),
            ylim=(bottom + 8e3, top - 10.5e3),
            xticks=[],
            yticks=[],
        )
        axs[i].set_aspect("equal")

        # Contours
        for contour in contours:
            axs[i].plot(
                contour[0], contour[1], "k-", alpha=0.5, linewidth=0.5, zorder=1
            )

        # Stations
        axs[i].scatter(
            df["x"],
            df["y"],
            marker="v",
            s=50,
            edgecolor="k",
            facecolor="w",
        )

        # Measured tilt
        dfmeas_sub = dfmeas[dfmeas["event"] == event]

        scale = 8e-10

        # Add scale arrow
        axs[i].quiver([35e3], [1036.5e3], [2e-6], [0], scale=scale, color="k")

        # Add label
        axs[i].annotate(
            "2 $\\mu$rad",
            (36e3, 1037e3),
            fontsize=8,
            horizontalalignment="center",
            bbox=dict(
                facecolor="white", edgecolor="none", boxstyle="round,pad=0.1", alpha=0.5
            ),
        )

        # Add uncertainty ellipse
        for j in range(len(dfmeas_sub)):
            err = 0.1e-6
            if j == 0:
                axs[i].arrow(
                    dfmeas_sub["x"].iloc[j],
                    dfmeas_sub["y"].iloc[j],
                    dfmeas_sub["etilt"].iloc[j] / scale,
                    dfmeas_sub["ntilt"].iloc[j] / scale,
                    head_width=300,
                    color="r",
                    label="Measured",
                    length_includes_head=True,
                )
            else:
                axs[i].arrow(
                    dfmeas_sub["x"].iloc[j],
                    dfmeas_sub["y"].iloc[j],
                    dfmeas_sub["etilt"].iloc[j] / scale,
                    dfmeas_sub["ntilt"].iloc[j] / scale,
                    head_width=300,
                    color="r",
                    length_includes_head=True,
                )
            cmt = """
            ell = matplotlib.patches.Ellipse(
                (
                    dfmeas_sub["x"].iloc[j] + (dfmeas_sub["etilt"].iloc[j] / scale),
                    dfmeas_sub["y"].iloc[j] + (dfmeas_sub["ntilt"].iloc[j] / scale),
                ),
                2 * err / scale,
                2 * err / scale,
                fill=False,
            )
            # axs[i].plot(dfmeas_sub["x"].iloc[j] + (dfmeas_sub["etilt"].iloc[j]/scale), dfmeas_sub["y"].iloc[j] + (dfmeas_sub["ntilt"].iloc[j]/scale), '.')
            axs[i].add_patch(ell)"""
        # print(dfmeas_sub)
        # axs[i].plot(dfmeas_sub["x"].to_numpy() + (dfmeas_sub["etilt"].to_numpy()/5e-10), dfmeas_sub["y"].to_numpy() + (dfmeas_sub["ntilt"].to_numpy()/5e-10), '.')

        # axs[i].quiver(df["x"], df["y"], median_tilts[event][0], median_tilts[event][1], scale=1e-5, label="Modeled", color="b")
        for j in range(len(df)):
            if j == 0:
                axs[i].arrow(
                    df["x"].iloc[j],
                    df["y"].iloc[j],
                    median_tilts[event][0][j] / scale,
                    median_tilts[event][1][j] / scale,
                    head_width=300,
                    color="b",
                    label="Modeled",
                    length_includes_head=True,
                )
            else:
                axs[i].arrow(
                    df["x"].iloc[j],
                    df["y"].iloc[j],
                    median_tilts[event][0][j] / scale,
                    median_tilts[event][1][j] / scale,
                    head_width=300,
                    color="b",
                    length_includes_head=True,
                )

        if i == 0:
            # Add legend
            axs[i].legend(loc="upper right", framealpha=0.5, fontsize=8)

        # Add label
        axs[i].annotate(
            event,
            (0.025, 0.925),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(
                facecolor="white", edgecolor="none", boxstyle="round,pad=0.1", alpha=0.5
            ),
        )

    axs[7].set_axis_off()
    axs[8].set_axis_off()

    fig.savefig(args.output + "augustine_inversion.png", bbox_inches="tight", dpi=300)


def volumes(samps, args, yangonly=False):
    # Yang source volume changes
    psamps = np.copy(samps)

    if yangonly:
        # Scale parameters for corner plot
        psamps[:, :5] /= 1e3
        psamps[:, 7:] /= 1e3
        # Calculate pressure chage for all models
        pres = 1e9 * psamps[:, 7:]
    else:
        # Scale parameters for corner plot
        psamps[:, :5] /= 1e3
        psamps[:, 7:9] /= 1e3
        psamps[:, 10:] /= 1e3
        # Calculate pressure chage for all models
        pres = 1e9 * psamps[:, 10::3]

    fig, axs = plt.subplots(7, 1, figsize=(1, 7))

    v = ((1e9) * psamps[:, 3] * psamps[:, 4] * psamps[:, 4]) * (4 / 3) * (np.pi)

    events = [2, 3, 4, 6, 9, 11, 13]

    r = 100

    for i in range(pres.shape[1]):
        axs[i].hist(
            1e-6 * (v * pres[:, i]) / 9.6e9,
            histtype="step",
            color="k",
            linewidth=0.5,
            range=(-r, 0),
        )
        if i != 6:
            axs[i].set_xticklabels([])
            axs[i].set_xticks([-r, -r / 2, 0])
        else:
            axs[i].set_xticks([-r, -r / 2, 0])
            axs[i].set_xlabel("dV (10$^6$ m$^3$)")
        axs[i].set_yticks([])
        axs[i].set_ylabel(events[i], rotation=0, labelpad=10)
        axs[i].grid(False)
        med = np.median(1e-6 * (v * pres[:, i]) / 9.6e9)
        axs[i].annotate(
            "%.2f $\\times 10^{6}$ m$^3$" % med,
            (1.1, 0.5),
            xycoords="axes fraction",
            va="center",
        )

    fig.supylabel("Event", x=-0.25)

    fig.savefig(args.output + "volumes.png", dpi=300, bbox_inches="tight")


def probabilities(probs, args):
    plt.plot(probs, "k-", alpha=0.1)
    plt.title("log probability")
    plt.xlabel("Step (x10)")
    plt.ylabel("log probability")
    plt.savefig(args.output + "probs.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = cli()
    sampler = emcee.backends.HDFBackend(args.samples)

    # Make plots of walker paths - load all samples for this
    #samps = sampler.get_chain(discard=0, thin=args.thin)

    # Plot all walkers
    #walkers(samps, args, yangonly=args.yangonly)

    # Plot log prob
    #probs = sampler.get_log_prob(discard=0, thin=args.thin)
    #probabilities(probs, args)

    # Discard burnin samples for the rest of the plots
    # then flatten
    samps = sampler.get_chain(discard=args.burnin, thin=args.thin, flat=True)
    if samps.shape[0] == 0:
        print("Too many burn in - no samples to plot.")
        exit()

    if args.yangonly:
        corner_plot_yangonly(samps, args)
    else:
        corner_plot(samps, args)

    # Make map plots
    map_plot(samps, args, yangonly=args.yangonly)

    # Yang volume changes
    volumes(samps, args, yangonly=args.yangonly)


if __name__ == "__main__":
    main()
