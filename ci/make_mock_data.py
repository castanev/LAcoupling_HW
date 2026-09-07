"""Write CI inputs: yearly GLEAM-like files for 0_data.py, plus E/SMs/SMrz for 1_coupling.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
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


def field(times, lats, lons, rng, base, amp, noise):
    ntime, nlat, nlon = len(times), len(lats), len(lons)
    doy = times.dayofyear.values[:, None, None]
    lat_term = ((lats - lats.mean()) / 10.0)[None, :, None]
    return np.clip(
        base
        + amp * np.sin(2 * np.pi * doy / 366.0)
        + 0.02 * lat_term
        + noise * rng.standard_normal((ntime, nlat, nlon)),
        0.05 if base < 1 else 0.1,
        None,
    ).astype(np.float32)


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    for key in (
        "path_data",
        "path_case_land",
        "path_case",
        "path_outputs_land",
        "path_outputs",
        "path_outputs_case",
        "path_temp",
    ):
        Path(ROOT / cfg[key]).mkdir(parents=True, exist_ok=True)

    # 0_data.py subsets lat 20–55 and lon −125–−70 (235/290 minus 360).
    lats = np.array([25.0, 35.0, 45.0], dtype=np.float32)
    lons = np.array([-120.0, -100.0, -80.0], dtype=np.float32)
    rng = np.random.default_rng(0)

    raw_dir = ROOT / cfg["path_data"]
    for year, nday in ((2000, 366), (2001, 365)):
        times = pd.date_range(f"{year}-01-01", periods=nday, freq="D")
        h = field(times, lats, lons, rng, base=80.0, amp=40.0, noise=5.0)
        xr.Dataset(
            {"H": (("time", "lat", "lon"), h)},
            coords={"time": np.arange(nday), "lat": lats, "lon": lons},
        ).to_netcdf(raw_dir / f"H_{year}_GLEAM.nc")

    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    dates_str = times.strftime("%Y%m%d").to_numpy()
    sms = np.clip(field(times, lats, lons, rng, 0.22, 0.08, 0.02), 0.05, 0.45)
    smrz = np.clip(sms + 0.03, 0.05, 0.50)
    e = field(times, lats, lons, rng, 2.0, 1.0, 0.3)
    save_nc_3d(ROOT / cfg["path_file_SMs"], sms, lats, lons, dates_str, "SMs")
    save_nc_3d(ROOT / cfg["path_file_SMrz"], smrz, lats, lons, dates_str, "SMrz")
    save_nc_3d(ROOT / cfg["path_file_E"], e, lats, lons, dates_str, "E")

    # 0_data.py later opens these with dask chunks (segfaults on tiny files).
    # Coupling does not need them; touching skips those blocks.
    out = ROOT / cfg["path_outputs_land"]
    name_land = cfg["name_land"]
    for stub in (
        f"monthly_anoma_SMs_{name_land}_US.nc",
        f"daily_anoma_window_smoothed15daysw_SMs_{name_land}_US.nc",
        f"daily_anoma_window_smoothed15daysw_SMs_{name_land}_AprMay_US.nc",
    ):
        (out / stub).touch()

    print(f"Wrote yearly H inputs in {raw_dir} and E/SMs/SMrz for coupling")


if __name__ == "__main__":
    main()
