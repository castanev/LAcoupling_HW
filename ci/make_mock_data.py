"""Write tiny NetCDF inputs under ci/data/ for 1_coupling.py CI."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ci" / "config_ci.yaml"


def save_nc_3d(path, var, lats, lons, dates_str, var_str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ncfile = Dataset(path, "w")
    ncfile.createDimension("lat", len(lats))
    ncfile.createDimension("lon", len(lons))
    ncfile.createDimension("time", None)
    var_lats = ncfile.createVariable("lat", "f", ("lat",))
    var_lons = ncfile.createVariable("lon", "f", ("lon",))
    var_time = ncfile.createVariable("time", "str", ("time",))
    var_lats[:] = lats
    var_lons[:] = lons
    var_time[:] = dates_str
    varr = ncfile.createVariable(var_str, "f", ("time", "lat", "lon"))
    varr[:, :, :] = var
    ncfile.close()


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    for key in (
        "path_case_land",
        "path_case",
        "path_outputs_land",
        "path_outputs",
        "path_outputs_case",
    ):
        Path(ROOT / cfg[key]).mkdir(parents=True, exist_ok=True)

    lats = np.array([30.0, 35.0, 40.0], dtype=np.float32)
    lons = np.array([250.0, 260.0, 270.0], dtype=np.float32)
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    dates_str = times.strftime("%Y%m%d").to_numpy()
    ntime, nlat, nlon = len(times), len(lats), len(lons)

    rng = np.random.default_rng(0)
    doy = times.dayofyear.values[:, None, None]
    lat_term = ((lats - lats.mean()) / 10.0)[None, :, None]
    lon_term = ((lons - lons.mean()) / 20.0)[None, None, :]

    sms = np.clip(
        0.22
        + 0.08 * np.sin(2 * np.pi * doy / 366.0)
        + 0.02 * lat_term
        + 0.02 * rng.standard_normal((ntime, nlat, nlon)),
        0.05,
        0.45,
    ).astype(np.float32)
    smrz = np.clip(sms + 0.03, 0.05, 0.50).astype(np.float32)
    e = np.clip(
        2.0
        + 1.0 * np.sin(2 * np.pi * doy / 366.0)
        + 0.3 * rng.standard_normal((ntime, nlat, nlon)),
        0.1,
        None,
    ).astype(np.float32)
    h = np.clip(
        80.0
        + 40.0 * np.cos(2 * np.pi * doy / 366.0)
        + 5.0 * rng.standard_normal((ntime, nlat, nlon)),
        1.0,
        None,
    ).astype(np.float32)

    save_nc_3d(ROOT / cfg["path_file_SMs"], sms, lats, lons, dates_str, "SMs")
    save_nc_3d(ROOT / cfg["path_file_SMrz"], smrz, lats, lons, dates_str, "SMrz")
    save_nc_3d(ROOT / cfg["path_file_E"], e, lats, lons, dates_str, "E")
    save_nc_3d(ROOT / cfg["path_file_H"], h, lats, lons, dates_str, "H")

    # 1_coupling.py expects SM anomalies to already exist under path_outputs_land.
    sms_anom = sms - sms.mean(axis=0, keepdims=True)
    out_dir = ROOT / cfg["path_outputs_land"]
    save_nc_3d(
        out_dir / f"daily_anoma_window_SMs_{cfg['name_land']}_US.nc",
        sms_anom.astype(np.float32),
        lats,
        lons,
        dates_str,
        "SMs",
    )
    print(f"Wrote mock inputs for {ntime} days, {nlat}x{nlon} grid")


if __name__ == "__main__":
    main()
