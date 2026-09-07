#This file computes the sensitivity index of the evaporative fraction to the soil moisture 
# SENSITIVITY INDEX — Dirmeyer (2011)
# S = b_f * r_f * s_w  →  here we compute b_f * s_w

#1. Compute the evaporative fraction
#2. Computes anomalies of the soil moisture and the evaporative fraction
#3. For each month, it computes the slope of the linear regression of the anomalies of the evaporative fraction on the anomalies of the soil moisture
#4. For each month, it computes the standard deviation of the anomalies of the soil moisture
#5. Sensitivity index = slope * standard deviation

from Functions import *
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
import random 
import xarray as xr
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
case = cfg['case']
region = cfg['region']
path_case_land = cfg['path_case_land']
path_file_SMrz = cfg['path_file_SMrz']
path_file_SMs = cfg['path_file_SMs']
path_file_E = cfg['path_file_E']
path_file_H = cfg['path_file_H']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])


if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #-125 to -70
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat=45 #-125 to -100   WEST
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat=45 #-125 to -100   WEST
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{path_case_land}/Figures/'
path_outputs_case = f'{path_outputs}{case}/'
path_figures_all = f'{path_case_land}/../Figures/'

num_time_laps = 20 

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'

# ======================================================= E ====================================================================

def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)

variable_E = 'E'
variable_EF = 'EF'
ds = xr.open_dataset(path_file_E, decode_times=False, engine="netcdf4")
time = np.array(ds['time'][:])
dates = pd.to_datetime(time, format="%Y%m%d")
lats_US = np.array(ds['lat'])
lons_US = np.array(ds['lon'])

if not os.path.exists(path_file_EF):
    # vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
    # climatology_month = np.zeros([12, lats_US.shape[0], lons_US.shape[0]])
    # for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
    #     month_idx = np.where(dates.month == i)[0]
    #     clima_month = ds[variable_E].isel(time=month_idx).mean(dim='time')
    #     clima_month = evaporation_to_latent_heat(clima_month)
    #     climatology_month[i-1] = clima_month.values.reshape(lats_US.shape[0], lons_US.shape[0])
    #     vmin, vmax, colormap, bounds = colorm(0, 70, 6, 0, 'PuBuGn') 
    #     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable_E, colormap, f'{path_figures_all}{name_land}_{variable}_{mo}', topography = True)

    # Evaporative fraction
    LE = evaporation_to_latent_heat(ds['E'])
    ds_H = xr.open_dataset(path_file_H, decode_times=False, engine="netcdf4")
    H = ds_H['H']

    # Set LE < 0 or H < 0 to np.nan
    LE = np.where(LE < 0, np.nan, LE)
    H = np.where(H < 0, np.nan, H)

    # Calculate evaporative fraction, then mask invalid ranges
    EF = LE / (H + LE)
    EF = np.where((EF >= 0) & (EF <= 1), EF, np.nan)

    save_nc_3d(path_file_EF, EF, lats_US, lons_US, time, variable_EF)





# # ======================================================= SM ====================================================================
variable_SMs = 'SMs'
# ds = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
# time = np.array(ds['time'][:])
# dates = pd.to_datetime(time, format="%Y%m%d")
# lats_US = np.array(ds['lat'])
# lons_US = np.array(ds['lon'])

# vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
# climatology_month = np.zeros([12, lats_US.shape[0], lons_US.shape[0]])
# for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
#     month_idx = np.where(dates.month == i)[0]
#     clima_month = ds[variable_SMs].isel(time=month_idx).mean(dim='time', skipna=True)
#     climatology_month[i-1] = clima_month.values.reshape(lats_US.shape[0], lons_US.shape[0])
#     vmin, vmax, colormap, bounds = colorm(0, 0.35, 6, 0, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable_SMs, colormap, f'{path_figures_all}{name_land}_{variable}_{mo}', topography = True)




# # ======================================================= SM ====================================================================
# variable_SMrz = 'SMrz'
# ds = xr.open_dataset(path_file_SMrz, decode_times=False, engine="netcdf4")
# time = np.array(ds['time'][:])
# dates = pd.to_datetime(time, format="%Y%m%d")
# lats_US = np.array(ds['lat'])
# lons_US = np.array(ds['lon'])

# vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
# climatology_month = np.zeros([12, lats_US.shape[0], lons_US.shape[0]])
# for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
#     month_idx = np.where(dates.month == i)[0]
#     clima_month = ds[variable_SMrz].isel(time=month_idx).mean(dim='time', skipna=True)
#     climatology_month[i-1] = clima_month.values.reshape(lats_US.shape[0], lons_US.shape[0])
#     vmin, vmax, colormap, bounds = colorm(0, 0.35, 6, 0, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable_SMrz, colormap, f'{path_figures_all}{name_land}_{variable}_{mo}', topography = True)


# # ======================================================= H ====================================================================
variable_H = 'H'

# ======================================================= Calculating anomalies ====================================================================
# for var, path_file in zip([variable_SMs, variable_EF, variable_H], [path_file_SMs, path_file_EF, path_file_H]):
for var, path_file in zip([variable_H], [path_file_H]):
    variable = var

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])  
    timei = np.array(ncfile['time'][:]) 
    dates = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei])
    print(dates[:4])

    path_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable}_{name_land}_US.nc'  
    path_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable}_{name_land}_US.nc'


    if not os.path.exists(path_annual_cycle_window):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in time.values])
        ds['time'] = dates_d
        sm_daily = ds[variable]

        annual_cycle_window_0 = compute_annual_cycle_window(sm_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        save_nc_3d(path_annual_cycle_window, annual_cycle_window_0, lats, lons, np.arange(1, len(annual_cycle_window_0.time)+1).astype(str), variable)


    if not os.path.exists(path_daily_anoma_window):

        # 1) climatology (dayofyear, lat, lon)
        ds_clim = xr.open_dataset(path_annual_cycle_window, decode_times=False, engine="netcdf4")
        clim = ds_clim[variable]  # ideally has coord dayofyear=1..366
        cycle_times = pd.date_range(f"{2000}-01-01", f"{2000}-12-31", freq="D")
        clim = clim.assign_coords(time=("time", cycle_times))

        # 2) daily data
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4")  # keep time unchunked if possible
        # ds = ds.isel(time=slice(0,400))
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d_lwa = np.array([dt.datetime.strptime(iii, "%Y%m%d") for iii in time.values])
        ds = ds.assign_coords(time=("time", dates_d_lwa))
        
        expanded_clim = expand_annual_cycle_optimized(dates_d_lwa, len(lats), len(lons), clim)

        # 3) anomaly WITHOUT expanding
        anom = ds[variable].astype("float32") - expanded_clim
        del expanded_clim
        # 4) save (prefer to_netcdf over save_nc_3d if save_nc_3d materializes .values)
        save_nc_3d(path_daily_anoma_window, anom.values, lats, lons, time.values, variable)
        del anom, ds, ds_clim


# ======================================================= Calculating sensitivity index ====================================================================

def period_mean_pc(pc_daily, dates, month, day_lo, day_hi):
    """One mean PC value per year over the given calendar day window."""
    pc_daily = np.asarray(pc_daily)
    dates = pd.DatetimeIndex(dates)
    mask = (
        (dates.month == month)
        & (dates.day >= day_lo)
        & (dates.day <= day_hi)
    )
    df = pd.DataFrame({'year': dates.year[mask], 'pc': pc_daily[mask]})
    yearly = df.groupby('year', sort=True)['pc'].mean()
    return yearly.values.astype(float), yearly.index.values.astype(int)


def compute_sensitivity_index(
    sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
    year_set=None, MIN_N=20,
):
    """Dirmeyer-style SI = bf * sw for each calendar month.

    If year_set is given, only days in those years are used.
    """
    months_arr = np.array([d.month for d in dates])
    months_arr_raw = np.array([d.month for d in dates_raw])
    years_arr = np.array([d.year for d in dates])
    years_arr_raw = np.array([d.year for d in dates_raw])

    bf = np.full((12, nlat, nlon), np.nan, dtype=np.float32)
    sw = np.full((12, nlat, nlon), np.nan, dtype=np.float32)
    SI = np.full((12, nlat, nlon), np.nan, dtype=np.float32)

    for m in range(1, 13):
        mi = m - 1
        idx_anom = np.where(months_arr == m)[0]
        idx_raw = np.where(months_arr_raw == m)[0]
        if year_set is not None:
            year_set = np.asarray(year_set, dtype=int)
            idx_anom = idx_anom[np.isin(years_arr[idx_anom], year_set)]
            idx_raw = idx_raw[np.isin(years_arr_raw[idx_raw], year_set)]

        if len(idx_anom) < MIN_N or len(idx_raw) < MIN_N:
            print(f"Month {m:02d} | skipped (N_anom={len(idx_anom)}, N_raw={len(idx_raw)} < {MIN_N})")
            continue

        w_anom_m = sm_anom[idx_anom]
        f_anom_m = ef_anom[idx_anom]
        w_raw_m = sm_raw[idx_raw]

        sw[mi] = np.nanstd(w_raw_m, axis=0, ddof=1)

        valid_mask = np.isfinite(w_anom_m) & np.isfinite(f_anom_m)
        w_c = np.where(valid_mask, w_anom_m - np.nanmean(w_anom_m, axis=0, keepdims=True), 0.0)
        f_c = np.where(valid_mask, f_anom_m - np.nanmean(f_anom_m, axis=0, keepdims=True), 0.0)

        n_valid = valid_mask.sum(axis=0)
        cov_wf = np.sum(w_c * f_c, axis=0)
        var_w = np.sum(w_c ** 2, axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            bf_m = np.where(var_w > 0, cov_wf / var_w, np.nan)

        bf_m[n_valid < MIN_N] = np.nan
        bf[mi] = bf_m.astype(np.float32)
        SI[mi] = bf[mi] * sw[mi]

        print(
            f"Month {m:02d} | N_anom={len(idx_anom):4d}  N_raw={len(idx_raw):4d} "
            f"| bf  [{np.nanmin(bf[mi]):.3f}, {np.nanmax(bf[mi]):.3f}] "
            f"| sw  [{np.nanmin(sw[mi]):.3f}, {np.nanmax(sw[mi]):.3f}]"
        )

    return bf, sw, SI


def save_SI_dataset(bf, sw, SI, lats, lons, variable_flux, path_out, description, extra_attrs=None):
    month_coord = np.arange(1, 13)
    ds_out = xr.Dataset(
        {
            "bf": (
                ["month", "lat", "lon"], bf,
                {
                    "long_name": f"Regression slope {variable_flux} on SM (anomalies)",
                    "units": f"{variable_flux} / (m3 m-3)",
                },
            ),
            "sw": (
                ["month", "lat", "lon"], sw,
                {
                    "long_name": "Std dev of raw daily soil moisture",
                    "units": "m3 m-3",
                },
            ),
            "SI": (
                ["month", "lat", "lon"], SI,
                {
                    "long_name": "Sensitivity index bf*sw (Dirmeyer 2011)",
                    "units": f"{variable_flux} / (m3 m-3)",
                },
            ),
        },
        coords={"month": month_coord, "lat": lats, "lon": lons},
    )
    ds_out.attrs["description"] = description
    if extra_attrs:
        ds_out.attrs.update(extra_attrs)
    ds_out.to_netcdf(path_out)
    print(f"\nSaved → {path_out}")


# EOF3 negative-phase years (same recipe as 9_analysis_EOF_AprMay.py)
pc_thresh = 0.75
EOF_i_neg = 2  # EOF3 (0-based)
# 15-day PC windows used to define the phase
eof_periods = [
    ('lateApr', 4, 16, 30),
    ('lateMay', 5, 16, 31),
]

path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'SMrz_{name_land}_AprMay_US.nc'
)
ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
time_ymd = ds_eof['time_ymd']
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')
pc_daily = np.asarray(eof_ts[EOF_i_neg].values)

year_neg_by_period = {}
for eof_period_label, eof_sel_month, eof_day_lo, eof_day_hi in eof_periods:
    pc_yr, years_yr = period_mean_pc(
        pc_daily, time_eof_ts, eof_sel_month, eof_day_lo, eof_day_hi
    )
    pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()
    year_neg = years_yr[pc_std < -pc_thresh]
    year_neg_by_period[eof_period_label] = year_neg
    print(
        f'EOF{EOF_i_neg + 1} {eof_period_label} negative phase '
        f'(PC* < -{pc_thresh:g}): n={len(year_neg)} years → {year_neg.tolist()}'
    )

# for variable_flux in [variable_H, variable_EF]:
for variable_flux in [variable_H]:

    # ── 1. Load anomaly files ────────────────────────────────────
    ds_sm = xr.open_dataset(f'{path_outputs}daily_anoma_window_{variable_SMs}_{name_land}_US.nc', decode_times=False)
    ds_ef = xr.open_dataset(f'{path_outputs}daily_anoma_window_{variable_flux}_{name_land}_US.nc', decode_times=False)

    # Parse times — your save_nc_3d stores them as '%Y%m%d' strings
    times_raw = ds_sm["time"].values
    dates = np.array([dt.datetime.strptime(str(t), "%Y%m%d") for t in times_raw])

    sm_anom = ds_sm[variable_SMs].values   # (time, lat, lon)  — anomalies w'
    ef_anom = ds_ef[variable_flux].values    # (time, lat, lon)  — anomalies f'

    # ── 2. Load RAW soil moisture for s_w ───────────────────────
    ds_sm_raw = xr.open_dataset(path_file_SMs, decode_times=False)
    times_raw2  = ds_sm_raw["time"].values
    dates_raw   = np.array([dt.datetime.strptime(str(t), "%Y%m%d") for t in times_raw2])
    sm_raw      = ds_sm_raw[variable_SMs].values  # (time, lat, lon) — raw w

    lats = ds_sm["lat"].values
    lons = ds_sm["lon"].values
    nlat, nlon = len(lats), len(lons)

    # ── 3a. All years (climatological SI) ────────────────────────
    print(f'\n=== SI for all years | {variable_flux} ===')
    bf, sw, SI = compute_sensitivity_index(
        sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon, year_set=None
    )
    path_SI_out = f"{path_outputs}sensitivity_index_SI_{variable_flux}_{name_land}_US.nc"
    save_SI_dataset(
        bf, sw, SI, lats, lons, variable_flux, path_SI_out,
        description="Land-atmosphere coupling sensitivity index, Dirmeyer (2011)",
    )

    # ── 3b. Negative EOF3 years for each 15-day period ───────────
    for eof_period_label, year_neg_eof3 in year_neg_by_period.items():
        print(
            f'\n=== SI for EOF3 {eof_period_label} negative phase years | {variable_flux} ==='
        )
        bf_neg, sw_neg, SI_neg = compute_sensitivity_index(
            sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
            year_set=year_neg_eof3,
        )
        path_SI_neg = (
            f"{path_outputs}sensitivity_index_SI_{variable_flux}_{name_land}_US_"
            f"EOF{EOF_i_neg + 1}neg_{eof_period_label}_thresh{pc_thresh:g}.nc"
        )
        save_SI_dataset(
            bf_neg, sw_neg, SI_neg, lats, lons, variable_flux, path_SI_neg,
            description=(
                f"Sensitivity index restricted to Apr–May SMrz EOF{EOF_i_neg + 1} "
                f"negative phase years ({eof_period_label} mean PC* < -{pc_thresh:g})"
            ),
            extra_attrs={
                "eof_file": path_file_eof,
                "eof_index": EOF_i_neg + 1,
                "eof_period": eof_period_label,
                "pc_thresh": pc_thresh,
                "phase": "negative",
                "years": ",".join(str(y) for y in year_neg_eof3.tolist()),
            },
        )

