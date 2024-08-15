import pandas as pd
import pyproj
import numpy as np


def station_positions():
    # url to get station locations
    # Built here:
    # https://service.iris.edu/fdsnws/station/docs/1/builder/
    # url = "https://service.iris.edu/fdsnws/station/1/query?net=AV&sta=AU*&cha=HHE&starttime=2005-11-01&endtime=2006-05-01&level=station&format=text&includecomments=true&nodata=404"
    # df = pd.read_csv(url, sep="|")
    # df.to_pickle("au_stations.pkl")
    df = pd.read_pickle("/home/mchristo/proj/ms/tiltErebusAugustine/au_stations.pkl")

    # Convert station coordinates to 3338
    xform = pyproj.Transformer.from_crs("4326", "3338")
    df["x"], df["y"] = xform.transform(df[" Latitude "], df[" Longitude "])

    # Fixed coordinate offset (approx top of Augustine)
    ox, oy = xform.transform(59.3608498, -153.4298857)

    # Get north and east vectors at each station
    bx, by = xform.transform(df[" Latitude "], df[" Longitude "])
    nx, ny = xform.transform(df[" Latitude "] + 1e-4, df[" Longitude "])
    ex, ey = xform.transform(df[" Latitude "], df[" Longitude "] + 1e-4)

    # Add to dataframe
    nmag = np.linalg.norm(list(zip(nx - bx, ny - by)), axis=1)
    n = np.divide(list(zip(nx - bx, ny - by)), nmag[:, np.newaxis])

    emag = np.linalg.norm(list(zip(ex - bx, ey - by)), axis=1)
    e = np.divide(list(zip(ex - bx, ey - by)), emag[:, np.newaxis])

    df["nvec_x"] = n[:, 0]
    df["nvec_y"] = n[:, 1]

    df["evec_x"] = e[:, 0]
    df["evec_y"] = e[:, 1]

    return df, ox, oy
