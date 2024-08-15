import numpy as np

import vmod.source
import vmod.data
import vmod.inverse

import station_positions

# Set up forward model for tilt


class TiltFwd:
    def __init__(self):
        df, ox, oy = station_positions.station_positions()

        # Exclude AU15 since not used
        df = df[df[" Station "] != "AU15"]

        # VMOD setup
        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()

        # Track station ordering
        self.stations = df[" Station "].to_list()

        tiltm = vmod.data.tilt.Tilt()

        tiltm.add_xs(xs - ox)
        tiltm.add_ys(ys - oy)

        # Filler values for measurements and errors
        tiltm.add_dx(np.zeros_like(xs))
        tiltm.add_dy(np.zeros_like(xs))

        tiltm.add_err(np.ones_like(xs), np.ones_like(ys))

        # Yang and Nishimura sources
        self.yang = vmod.source.yang.Yang(tiltm)
        self.nish = vmod.source.nish.Nish(tiltm)

    def tilt(self, x):
        # x holds parameters for yang and nishimura models
        # x[0] -> yang x; x[1] -> yang y; x[2] -> yang z
        # x[3] -> yang semi-major; x[4] -> yang semi-minor
        # x[5] -> yang strike; x[6] -> yang dip
        # x[7] -> nish x; x[8] -> nish y, x[9] -> nish radius
        # Then per event parameters
        # x[10] -> yang pressure; x[11] -> nish pressure
        # x[12] -> nish length
        # This repeats for each event

        nevent = (len(x) - 10) // 3

        tx = []
        ty = []
        for i in range(nevent):
            # Unpack parameters to vmod expected ordering
            # Pressure being multiplied by 1e6 to scale to MPa
            # xcen, ycen, depth, pressure, semimajor axis, semiminor axis, strike and dip angles in degrees
            xyang = np.concatenate((x[:3], [x[10 + (i * 3)] * 1e6], x[3:7]))
            # xcen, ycen, depth, radius, length and pressure
            xnish = np.concatenate(
                (
                    x[7:9],
                    [x[12 + (i * 3)] / 2],
                    [x[9]],
                    [x[12 + (i * 3)]],
                    [x[11 + (i * 3)] * 1e6],
                )
            )

            ytx, yty = self.yang.forward(xyang, unravel=False)
            ntx, nty = self.nish.forward(xnish, unravel=False)
            tx.append(ytx + ntx)
            ty.append(yty + nty)

        return (self.stations, tx, ty)

    def tilt_yangonly(self, x):
        # x holds parameters for yang models
        # x[0] -> yang x; x[1] -> yang y; x[2] -> yang z
        # x[3] -> yang semi-major; x[4] -> yang semi-minor
        # x[5] -> yang strike; x[6] -> yang dip
        # Then per event parameters
        # x[7+i] -> yang pressure
        # This repeats for each event

        nevent = len(x) - 7

        tx = []
        ty = []
        for i in range(nevent):
            # Unpack parameters to vmod expected ordering
            # Pressure being multiplied by 1e6 to scale to MPa
            # xcen, ycen, depth, pressure, semimajor axis, semiminor axis, strike and dip angles in degrees
            xyang = np.concatenate((x[:3], [x[7 + i] * 1e6], x[3:7]))

            ytx, yty = self.yang.forward(xyang, unravel=False)

            tx.append(ytx)
            ty.append(yty)

        return (self.stations, tx, ty)


cmt = """
fwm = TiltFwd()

x0_yang = [
    0,     # xcen
    0,     # ycen
    6e3,   # depth
    2e3,   # semi-major
    1e3,   # semi-minor
    215,     # strike
    80     # dip
]

x0_nish = [
    0,     # xcen
    0,     # yen
    10,    # radius
]

p0_yang = -10
p0_nish = -10
l0_nish = 2e3

# Set up x0
x0 = x0_yang + x0_nish + [p0_yang, p0_nish, l0_nish]*6

tilts = fwm.tilt(x0)

print(tilts)"""
