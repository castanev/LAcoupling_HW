# This code is for analyzing 
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import scipy as scp
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import os
import glob
import re
from scipy.ndimage import uniform_filter
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import argparse
import yaml
import xarray as xr
from scipy.stats import gaussian_kde
path_ncr = '/apps/spack/negishi/apps/nco/5.0.1-gcc-12.2.0-f3lr7i3/bin/ncrcat'


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
name_land = cfg['name_land']
case = cfg['case']
region = cfg['region']
path_case = cfg['path_case_land']
path_case_land = cfg['path_case_land']
path_file_SMs = cfg['path_file_SMs']
path_file_t = cfg['path_file_t']
path_file_t_anom = cfg['path_file_t_anom']
initial_year = cfg['initial_year']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])


if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #-125 to -70
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat=45 #-125 to -100   WEST
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat=45 #-125 to -100   WEST
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
elif region == 'westsouthUS':
    lat_minHW = 25; lat_maxHW = 39; lon_minHW = 252; lon_maxHW = 270; midlat=45 #-125 to -100   WEST
elif region == 'southcentralUS':
    lat_minHW = 25; lat_maxHW = 39; lon_minHW = 255; lon_maxHW = 270; midlat=45 #-125 to -100   WEST
elif region == 'midwestUS':
    lat_minHW = 35; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{path_case_land}/Figures/'
path_outputs_case = f'{path_outputs}{case}/'
path_figures_all = f'{path_case_land}/../Figures/'

num_time_laps = 20 

def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx_series(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]

def get_lon_idx_single(lons, lon_center):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_idx = np.where((lons - lon_center_adj) == np.min(abs(lons - lon_center_adj)))[0]

    return lon_idx


# ======================================================= T ====================================================================
variable_t = 'TS'

ncfile_t = Dataset(f'{path_file_t}')
dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(np.array(ncfile_t['time'][:])))])  #[1950-2021]
dates_d_str = np.array([dt.datetime.strftime(iii, 'Y%m%d') for iii in dates_d])
timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
lats_t = np.array(ncfile_t['lat'])
lons_t = np.array(ncfile_t['lon'])
pos_lats_midlat_t = np.where((lats_t >= 35) & (lats_t <= 65))[0]
pos_middle_HW_t = np.where(abs(lons_t - (lon_minHW+lon_maxHW)/2) == np.min(abs(lons_t - (lon_minHW+lon_maxHW)/2)))[0][0]

print("lats_t :", lats_t[:4])
print("lons_t :", lons_t[:4])

# File with the positions of heat waves 
name_file_posHW = f'Heat_waves_events_list.csv'
df_heatwaves = pd.read_csv(f'{path_case}{name_file_posHW}', index_col=0)
df_heatwaves.index = df_heatwaves.index.astype(int)

pos_HW_day0 = df_heatwaves.index
pos_HW_day0 = pos_HW_day0[pos_HW_day0 <= len(timei)]

duration_HW = df_heatwaves.loc[pos_HW_day0, 'Duration']
pos_HWdays = np.concatenate([
    pos_HW0 + np.arange(int(dur))
    for pos_HW0, dur in zip(pos_HW_day0, duration_HW)
])
pos_HWdays = pd.DataFrame(pos_HWdays)


dates_d_HWdays = dates_d[pos_HWdays].flatten()
dates_d_HWdays = np.unique([x.date() for x in dates_d_HWdays])
dates_d_HW_day0 = dates_d[pos_HW_day0].flatten()
dates_d_HW_day0 = np.unique([x.date() for x in dates_d_HW_day0])

pos_summer = np.where([i.month in [6,7,8] for i in dates_d])[0]
pos_all_non_HW_summer = []
for i in pos_summer:
    if i not in pos_HWdays.values:
        pos_all_non_HW_summer.append(i)
pos_all_non_HW_summer = np.array(pos_all_non_HW_summer)

maxT_events = df_heatwaves.iloc[:, 3]



# =======================================================
# Event centers
# =======================================================
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

duration_events = df_heatwaves.iloc[:, 1]



# ======================================================= Regimes ====================================================================
ncfilenm=path_outputs + '/BP_SMsxLE_' + name_land + '_JJA_optimized.nc'
# ncfilenm=path_outputs + '/BP_SMsxEF_' + name_land + '_JJA_optimized.nc'
ds = xr.open_dataset(ncfilenm)

# Open SM dataset once
variable_sm = "SMs"
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

# Best model map (time=0)
BIC = np.array(ds['BIC'])[0]  # (option, lat, lon)
BIC_clean = np.where(np.isclose(BIC, -9.99e8), np.nan, BIC)
best_model = np.full((len(lats_sm), len(lons_sm)), -1, dtype=int)
valid = np.sum(np.isfinite(BIC_clean), axis=0) > 0
best_model[valid] = np.nanargmin(BIC_clean[:, valid], axis=0)

# Align event arrays to the same valid subset used for onset indices
valid_event_mask = df_heatwaves.index <= len(timei)
df_events_valid = df_heatwaves.loc[valid_event_mask]
lats_events, lons_events = extract_latlons_from_coord_events(df_events_valid.iloc[:, 5])
duration_events = df_events_valid.iloc[:, 1].values
dates_d_HW_day0 = dates_d[df_events_valid.index.values].flatten()
dates_d_HW_day0 = np.array([x.date() for x in dates_d_HW_day0])

rows = []


miss_bp = -9.99e08

ncfilenm = path_outputs + f'/map_wet_regime_fraction_{name_land}_US.nc'
ds = xr.open_dataset(ncfilenm)


def _squeeze2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return np.squeeze(arr)


bp_rhs_field_2seg_plot = np.where(
    np.isclose(_squeeze2d(ds['BPx_2Seg_RHSflat']), miss_bp),
    np.nan,
    _squeeze2d(ds['BPx_2Seg_RHSflat']),
)
bp_rhs_field_3seg_plot = np.where(
    np.isclose(_squeeze2d(ds['BPx2_3Seg']), miss_bp),
    np.nan,
    _squeeze2d(ds['BPx2_3Seg']),
)
regime_field = _squeeze2d(ds['regime_map']).astype(int)

# Breakpoint critical SM only for regimes 3 and 4
critical_sm_bp = np.full(regime_field.shape, np.nan, dtype=float)
critical_sm_bp[regime_field == 3] = bp_rhs_field_2seg_plot[regime_field == 3]
critical_sm_bp[regime_field == 4] = bp_rhs_field_3seg_plot[regime_field == 4]


def dominant_regime_in_box(regime_box):
    """Most common regime_map value in the event box (ties -> higher regime index)."""
    vals = regime_box[(regime_box >= 0) & (regime_box <= 4)]
    if vals.size == 0:
        return -1
    counts = np.bincount(vals.astype(int), minlength=5)
    return int(np.argmax(counts))

# for event_idx, (lat_center_hw, lon_center_hw, onset_day, duration_event) in enumerate(
#     zip(lats_events, lons_events, dates_d_HW_day0, duration_events)
# ):
#     # Locate nearest SM grid point
#     ilat_sm = int(np.argmin(np.abs(lats_sm - lat_center_hw)))
#     ilon_sm = int(get_lon_idx_single(lons_sm, lon_center_hw)[0])

#     best_model_event = int(best_model[ilat_sm, ilon_sm])

#     # Save RHS breakpoint only for RHS regime (option 3)
#     bpx_rhs = np.nan
#     if best_model_event in [3, 4]:
#         bpx_rhs = float(bp_rhs_field[ilat_sm, ilon_sm])

#     # Count transitional days (SM < BPx) only when RHS breakpoint is available
#     n_days_transitional = np.nan
#     onset_idx = np.where(dates_sm == onset_day)[0]
#     if len(onset_idx) == 1 and np.isfinite(bpx_rhs):
#         sm_point = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
#         start = int(onset_idx[0])
#         duration_int = int(duration_event)
#         if duration_int > 0:
#             end = min(start + duration_int, len(sm_point))
#             sm_event = sm_point[start:end]
#             n_days_transitional = int(np.sum(sm_event < bpx_rhs))
#         else:
#             n_days_transitional = 0

#     rows.append(
#         {
#             "event_id": int(df_events_valid.index[event_idx]),
#             "onset_day": onset_day.isoformat(),
#             "duration_days": int(duration_event),
#             "lat_event": float(lat_center_hw),
#             "lon_event": float(lon_center_hw),
#             "ilat_sm": ilat_sm,
#             "ilon_sm": ilon_sm,
#             "best_model_option": best_model_event,
#             "BPx_rhs": bpx_rhs,
#             "n_days_transitional_sm_lt_BPx": n_days_transitional,
#         }
#     )

# df_out = pd.DataFrame(rows)
# out_csv = os.path.join(path_case_land, f"event_regime_rhs.csv")
# df_out.to_csv(out_csv, index=False)
# print(f"Saved event summary: {out_csv}")

# ds_sm.close()
# ds.close()




# ======================================================= DENSITY PLOT ====================================================================
lags_days = [5]
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "LE"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine="netcdf4")




lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])


def _plot_density_panel(
    all_sm_summer,
    all_LE_summer,
    all_sm_hw,
    all_LE_hw,
    all_t_hw,
    all_sm_hw3,
    all_LE_hw3,
    all_t_hw3,
    lag_days,
    xlabel,
    title,
    outfile,
    show_critical_vline=False,
):
    all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
    all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

    all_sm_hw = np.array(all_sm_hw)
    all_LE_hw = np.array(all_LE_hw)
    all_t_hw = np.array(all_t_hw)
    all_sm_hw3 = np.array(all_sm_hw3)
    all_LE_hw3 = np.array(all_LE_hw3)
    all_t_hw3 = np.array(all_t_hw3)

    if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
        all_t_hw = all_t_hw - 273.15
    if len(all_t_hw3) > 0 and np.nanmean(all_t_hw3) > 200:
        all_t_hw3 = all_t_hw3 - 273.15

    mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
    sm_summer_valid = all_sm_summer[mask]
    LE_summer_valid = all_LE_summer[mask]
    has_summer = sm_summer_valid.size > 0
    has_hw = len(all_sm_hw) > 0 or len(all_sm_hw3) > 0

    if not has_summer and not has_hw:
        print(f"Skipping plot (no events): {outfile}")
        return

    plt.figure(figsize=(8, 5))

    if has_summer and len(sm_summer_valid) > 5:
        xy = np.vstack([sm_summer_valid, LE_summer_valid])
        kde = gaussian_kde(xy)

        xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
        ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

        xgrid, ygrid = np.meshgrid(
            np.linspace(xmin, xmax, 45),
            np.linspace(ymin, ymax, 45),
        )

        z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        levels = np.linspace(0, 0.2, 21)

        cf = plt.contourf(
            xgrid,
            ygrid,
            z,
            levels=levels,
            cmap="Blues",
            vmin=0,
            vmax=0.2,
            alpha=0.8,
            extend="max",
        )
        plt.colorbar(cf, label="Density", extend="max")
    elif has_summer:
        plt.scatter(
            sm_summer_valid,
            LE_summer_valid,
            s=10,
            alpha=0.3,
            color="cornflowerblue",
            label="Summer",
        )

    if show_critical_vline:
        plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="Critical SM")

    if len(all_sm_hw) > 0 or len(all_sm_hw3) > 0:
        t_all = (
            np.concatenate([all_t_hw, all_t_hw3])
            if len(all_t_hw) > 0 and len(all_t_hw3) > 0
            else (all_t_hw if len(all_t_hw) > 0 else all_t_hw3)
        )
        t_min, t_max = np.nanmin(t_all), np.nanmax(t_all)
        s_min, s_max = 40, 100

        def _marker_sizes(t_vals):
            if t_max > t_min:
                return s_min + (s_max - s_min) * (t_vals - t_min) / (t_max - t_min)
            return np.full(len(t_vals), (s_min + s_max) / 2)

        if len(all_sm_hw) > 0:
            plt.scatter(
                all_sm_hw,
                all_LE_hw,
                s=_marker_sizes(all_t_hw),
                c="red",
                alpha=0.75,
                edgecolors="k",
                linewidths=0.4,
                label="Max local T (size = regional T)",
            )
        if len(all_sm_hw3) > 0:
            plt.scatter(
                all_sm_hw3,
                all_LE_hw3,
                s=_marker_sizes(all_t_hw3),
                c="gold",
                alpha=0.75,
                edgecolors="k",
                linewidths=0.4,
                label=f"Max local T-{lag_days}d (size = regional T)",
            )

    x_plot = [sm_summer_valid]
    y_plot = [LE_summer_valid]
    if len(all_sm_hw) > 0:
        x_plot.append(all_sm_hw)
        y_plot.append(all_LE_hw)
    if len(all_sm_hw3) > 0:
        x_plot.append(all_sm_hw3)
        y_plot.append(all_LE_hw3)
    x_parts = [np.asarray(a).ravel() for a in x_plot if np.size(a) > 0]
    y_parts = [np.asarray(a).ravel() for a in y_plot if np.size(a) > 0]
    if x_parts and y_parts:
        x_all = np.concatenate(x_parts)
        y_all = np.concatenate(y_parts)
        x_all = x_all[np.isfinite(x_all)]
        y_all = y_all[np.isfinite(y_all)]
        if x_all.size > 0 and y_all.size > 0:
            ax = plt.gca()
            ax.set_xlim(np.nanmin(x_all), np.nanmax(x_all))
            ax.set_ylim(np.nanmin(y_all), np.nanmax(y_all))
            ax.margins(0)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(r"LE (W m$^{-2}$)", fontsize=12)
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


for lag_days in lags_days:
    # Figure 1: weak coupling (regime 0)
    weak_sm_summer, weak_LE_summer = [], []
    weak_sm_hw, weak_LE_hw, weak_t_hw = [], [], []
    weak_sm_hw3, weak_LE_hw3, weak_t_hw3 = [], [], []

    # Figure 2: transitional (regime 1 strong opt1 + regime 2)
    trans_sm_summer, trans_LE_summer = [], []
    trans_sm_hw, trans_LE_hw, trans_t_hw = [], [], []
    trans_sm_hw3, trans_LE_hw3, trans_t_hw3 = [], [], []

    # Figure 3: moisture-limited (regime 3 + 4)
    ml_sm_summer, ml_LE_summer = [], []
    ml_sm_hw, ml_LE_hw, ml_t_hw = [], [], []
    ml_sm_hw3, ml_LE_hw3, ml_t_hw3 = [], [], []

    for lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
        lats_events, lons_events, dates_d_HW_day0, duration_events
    ):

        if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
            print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
            continue

        lat_min = lat_center_hw - 0.5
        lat_max = lat_center_hw + 0.5

        ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
        ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

        ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
        ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

        ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
        ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

        if (
            len(ilat_LE) == 0
            or len(ilon_LE) == 0
            or len(ilat_sm) == 0
            or len(ilon_sm) == 0
            or len(ilat_t) == 0
            or len(ilon_t) == 0
        ):
            continue

        regime_box = regime_field[np.ix_(ilat_sm, ilon_sm)]
        dom_regime = dominant_regime_in_box(regime_box)
        if dom_regime < 0:
            continue

        LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
        sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
        t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

        if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
            continue

        if (
            not np.isfinite(LE_region).any()
            or not np.isfinite(sm_region).any()
            or not np.isfinite(t_region).any()
        ):
            continue

        LE_point = np.nanmean(LE_region, axis=(1, 2))
        t_point = np.nanmean(t_region, axis=(1, 2))
        LE_point = np.where((LE_point >= 0) & np.isfinite(LE_point), LE_point, np.nan)

        if dom_regime == 0:
            sm_point = np.nanmean(sm_region, axis=(1, 2))
            sm_summer_list, sm_hw_list, sm_hw3_list = weak_sm_summer, weak_sm_hw, weak_sm_hw3
            le_summer_list, le_hw_list, le_hw3_list = weak_LE_summer, weak_LE_hw, weak_LE_hw3
            t_hw_list, t_hw3_list = weak_t_hw, weak_t_hw3
        elif dom_regime in (1, 2):
            sm_point = np.nanmean(sm_region, axis=(1, 2))
            sm_summer_list, sm_hw_list, sm_hw3_list = trans_sm_summer, trans_sm_hw, trans_sm_hw3
            le_summer_list, le_hw_list, le_hw3_list = trans_LE_summer, trans_LE_hw, trans_LE_hw3
            t_hw_list, t_hw3_list = trans_t_hw, trans_t_hw3
        elif dom_regime in (3, 4):
            bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
            if not np.isfinite(bp_box).any():
                continue
            sm_anom_region = sm_region - bp_box[np.newaxis, :, :]
            sm_point = np.nanmean(sm_anom_region, axis=(1, 2))
            sm_summer_list, sm_hw_list, sm_hw3_list = ml_sm_summer, ml_sm_hw, ml_sm_hw3
            le_summer_list, le_hw_list, le_hw3_list = ml_LE_summer, ml_LE_hw, ml_LE_hw3
            t_hw_list, t_hw3_list = ml_t_hw, ml_t_hw3
        else:
            continue

        common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

        LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
        sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
        t_aligned = t_point[np.isin(dates_t, common_dates)]

        summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])
        LE_summer = LE_aligned[summer_mask]
        sm_summer = sm_aligned[summer_mask]

        valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
        if np.any(valid):
            le_summer_list.append(LE_summer[valid])
            sm_summer_list.append(sm_summer[valid])

        heatwave_days = [onset_day + dt.timedelta(days=i) for i in range(int(duration_event))]
        heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
        if len(heatwave_idx) > 0:
            i_max = int(np.nanargmax(t_aligned[heatwave_idx]))
            idx_max = heatwave_idx[i_max]
            tmax_hw = t_aligned[idx_max]
            LE_max = LE_aligned[idx_max]
            sm_max = sm_aligned[idx_max]
            if np.isfinite(LE_max) and np.isfinite(sm_max) and np.isfinite(tmax_hw):
                le_hw_list.append(LE_max)
                sm_hw_list.append(sm_max)
                t_hw_list.append(float(tmax_hw))

            idx_max3 = idx_max - int(lag_days)
            if idx_max3 >= 0:
                tmax_hw3 = t_aligned[idx_max3]
                LE_max3 = LE_aligned[idx_max3]
                sm_max3 = sm_aligned[idx_max3]
                if np.isfinite(LE_max3) and np.isfinite(sm_max3) and np.isfinite(tmax_hw3):
                    le_hw3_list.append(LE_max3)
                    sm_hw3_list.append(sm_max3)
                    t_hw3_list.append(float(tmax_hw3))

    _plot_density_panel(
        weak_sm_summer,
        weak_LE_summer,
        weak_sm_hw,
        weak_LE_hw,
        weak_t_hw,
        weak_sm_hw3,
        weak_LE_hw3,
        weak_t_hw3,
        lag_days,
        xlabel=r"SMs (regional mean, ±0.5°)",
        title=(
            "SM vs LE — weak coupling (dominant regime 0 in ±0.5° box)\n"
            "Models 0 + weak option 1; summer + heatwave markers"
        ),
        outfile=(
            f"{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_HWcenterIntensity_"
            f"maxlocalTanom_{lag_days}daybefore_eachEvent_05deg_weak_{name_land}_{region}.png"
        ),
        show_critical_vline=False,
    )

    _plot_density_panel(
        trans_sm_summer,
        trans_LE_summer,
        trans_sm_hw,
        trans_LE_hw,
        trans_t_hw,
        trans_sm_hw3,
        trans_LE_hw3,
        trans_t_hw3,
        lag_days,
        xlabel=r"SMs (regional mean, ±0.5°)",
        title=(
            "SM vs LE — transitional (dominant regime 1 or 2 in ±0.5° box)\n"
            "Strong option 1 + LHS-flat; summer + heatwave markers"
        ),
        outfile=(
            f"{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_HWcenterIntensity_"
            f"maxlocalTanom_{lag_days}daybefore_eachEvent_05deg_trans_{name_land}_{region}.png"
        ),
        show_critical_vline=False,
    )

    _plot_density_panel(
        ml_sm_summer,
        ml_LE_summer,
        ml_sm_hw,
        ml_LE_hw,
        ml_t_hw,
        ml_sm_hw3,
        ml_LE_hw3,
        ml_t_hw3,
        lag_days,
        xlabel=r"SMs $-$ critical SM (regime 3/4 BP, regional mean, ±0.5°)",
        title=(
            "SM vs LE — moisture-limited (dominant regime 3 or 4 in ±0.5° box)\n"
            "RHS-flat / 3-segment breakpoints; summer + heatwave markers"
        ),
        outfile=(
            f"{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_HWcenterIntensity_"
            f"maxlocalTanom_{lag_days}daybefore_eachEvent_05deg_moistlim_{name_land}_{region}.png"
        ),
        show_critical_vline=True,
    )

    ds_LE.close()
    ds_sm.close()
    ds_t.close()

aaaaaa

# ======================================================= DENSITY PLOT ====================================================================
lags_days = [5]
path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine="netcdf4")




lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

for lag_days in lags_days:
    # =======================================================
    # Containers for all events
    # =======================================================
    all_sm_summer = []
    all_LE_summer = []

    all_sm_hw = []
    all_LE_hw = []
    all_t_hw = []

    all_sm_hw3 = []
    all_LE_hw3 = []
    all_t_hw3 = []

    all_duration_hw = []

    # =======================================================
    # Loop over events
    # =======================================================
    for lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
        lats_events, lons_events, dates_d_HW_day0, duration_events
    ):

        if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
            print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
            continue

        # --- local box ±0.5°
        lat_min = lat_center_hw - 0.5
        lat_max = lat_center_hw + 0.5

        ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
        ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

        ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
        ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

        ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
        ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

        if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
            continue

        bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
        if not np.isfinite(bpx_rhs_mean):
            continue

        # --- regional mean time series for this event-centered box
        LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
        sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
        t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values


        if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
            continue

        if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
            continue

        LE_point = np.nanmean(LE_region, axis=(1, 2))
        sm_point = np.nanmean(sm_region, axis=(1, 2))
        t_point = np.nanmean(t_region, axis=(1, 2))

        LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

        # --- align dates across EF, SM, and T
        common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

        LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
        sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
        t_aligned = t_point[np.isin(dates_t, common_dates)]

        # --- summer points from this event-centered box
        summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

        LE_summer = LE_aligned[summer_mask]
        sm_summer = sm_aligned[summer_mask]

        valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
        if np.any(valid):
            all_LE_summer.append(LE_summer[valid])
            all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

        # --- EF, SM, T on the day of maximum regional mean T during the event
        heatwave_days = [onset_day + dt.timedelta(days=i) for i in range(int(duration_event))]
        heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
        if len(heatwave_idx) > 0:
            i_max = int(np.nanargmax(t_aligned[heatwave_idx]))
            idx_max = heatwave_idx[i_max]
            tmax_hw = t_aligned[idx_max]
            LE_max = LE_aligned[idx_max]
            sm_max = sm_aligned[idx_max]
            if np.isfinite(LE_max) and np.isfinite(sm_max) and np.isfinite(tmax_hw):
                all_LE_hw.append(LE_max)
                all_sm_hw.append(sm_max - bpx_rhs_mean)
                all_t_hw.append(float(tmax_hw))
                all_duration_hw.append(float(duration_event))
            
            idx_max3 = idx_max - int(lag_days)
            tmax_hw3 = t_aligned[idx_max3]
            LE_max3 = LE_aligned[idx_max3]
            sm_max3 = sm_aligned[idx_max3]
            if np.isfinite(LE_max3) and np.isfinite(sm_max3) and np.isfinite(tmax_hw3):
                all_LE_hw3.append(LE_max3)
                all_sm_hw3.append(sm_max3 - bpx_rhs_mean)
                all_t_hw3.append(float(tmax_hw3))
    # =======================================================
    # Concatenate all summer points
    # =======================================================
    all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
    all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

    all_sm_hw = np.array(all_sm_hw)
    all_LE_hw = np.array(all_LE_hw)
    all_t_hw = np.array(all_t_hw)
    all_sm_hw3 = np.array(all_sm_hw3)
    all_LE_hw3 = np.array(all_LE_hw3)
    all_t_hw3 = np.array(all_t_hw3)

    if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
        all_t_hw = all_t_hw - 273.15
    if len(all_t_hw3) > 0 and np.nanmean(all_t_hw3) > 200:
        all_t_hw3 = all_t_hw3 - 273.15

    # =======================================================
    # Plot
    # =======================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
    sm_summer_valid = all_sm_summer[mask]
    LE_summer_valid = all_LE_summer[mask]

    xmin = xmax = ymin = ymax = None
    if len(sm_summer_valid) > 5:
        xy = np.vstack([sm_summer_valid, LE_summer_valid])
        kde = gaussian_kde(xy)

        xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
        ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

        xgrid, ygrid = np.meshgrid(
            np.linspace(xmin, xmax, 45),
            np.linspace(ymin, ymax, 45)
        )

        z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        zmax = np.nanmax(z)
        levels = np.linspace(0, zmax, 21) if zmax > 0 else np.linspace(0, 1, 21)

        cf = ax.contourf(
            xgrid, ygrid, z,
            levels=levels,
            cmap="Blues",
            vmin=0,
            vmax=levels[-1],
            alpha=0.8,
            extend="max")
        fig.colorbar(cf, ax=ax, label="Density", extend="max")
    else:
        ax.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs")

    if len(all_sm_hw) > 0 or len(all_sm_hw3) > 0:
        t_all = np.concatenate([all_t_hw, all_t_hw3]) if len(all_t_hw) > 0 and len(all_t_hw3) > 0 else (
            all_t_hw if len(all_t_hw) > 0 else all_t_hw3
        )
        t_min, t_max = np.nanmin(t_all), np.nanmax(t_all)
        s_min, s_max = 40, 100

        def _marker_sizes(t_vals):
            if t_max > t_min:
                return s_min + (s_max - s_min) * (t_vals - t_min) / (t_max - t_min)
            return np.full(len(t_vals), (s_min + s_max) / 2)
        
        d_min, d_max = np.min(all_duration_hw), np.max(all_duration_hw)
        alpha_min, alpha_max = 0.25, 0.95
        if d_max > d_min:
            marker_alphas = alpha_min + (alpha_max - alpha_min) * (all_duration_hw - d_min) / (d_max - d_min)
        else:
            marker_alphas = np.full(len(all_duration_hw), (alpha_min + alpha_max) / 2)
        marker_colors = np.column_stack([np.ones(len(all_sm_hw)), np.zeros(len(all_sm_hw)), np.zeros(len(all_sm_hw)), marker_alphas])

        if len(all_sm_hw) > 0:
            ax.scatter(
                all_sm_hw, all_LE_hw,
                s=_marker_sizes(all_t_hw), c=marker_colors,
                edgecolors="k", linewidths=0.4,
                label="Max local T (size = regional T, color = duration)",
                clip_on=True,
            )
        if len(all_sm_hw3) > 0:
            ax.scatter(
                all_sm_hw3, all_LE_hw3,
                s=_marker_sizes(all_t_hw3), c="gold", alpha=0.75,
                edgecolors="k", linewidths=0.4,
                label=f"Max local T-{lag_days}d (size = regional T)",
                clip_on=True,
            )

    x_plot = [sm_summer_valid]
    y_plot = [LE_summer_valid]
    if len(all_sm_hw) > 0:
        x_plot.append(all_sm_hw)
        y_plot.append(all_LE_hw)
    if len(all_sm_hw3) > 0:
        x_plot.append(all_sm_hw3)
        y_plot.append(all_LE_hw3)
    x_all = np.concatenate([np.asarray(a).ravel() for a in x_plot if np.size(a) > 0])
    y_all = np.concatenate([np.asarray(a).ravel() for a in y_plot if np.size(a) > 0])
    x_all = x_all[np.isfinite(x_all)]
    y_all = y_all[np.isfinite(y_all)]
    if x_all.size > 0 and y_all.size > 0:
        ax.set_xlim(np.nanmin(x_all), np.nanmax(x_all))
        ax.set_ylim(np.nanmin(y_all), np.nanmax(y_all))
        ax.margins(0)

    ax.set_xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
    ax.set_ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
    ax.set_title(
        "Density plot of Soil Moisture vs Evaporative Fraction\n"
        "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
        fontsize=12
    )
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWcenterIntensity_maxlocalTanom_duration_{lag_days}daybefore_eachEvent_05deg_relBPxEF_{name_land}_{region}.png")
    plt.close(fig)

    ds_LE.close()
    ds_sm.close()
    ds_t.close()







# ======================================================= DENSITY PLOT ====================================================================

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine="netcdf4")




lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_dT_hw = []


# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
    lats_events, lons_events, dates_d_HW_day0, duration_events
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    # centered difference of T': (T'(t+1) - T'(t-1)) / 2  [per day]
    t_region_dT = np.full_like(t_region, np.nan)
    t_region_dT[1:-1, :, :] = 0.5 * (t_region[2:, :, :] - t_region[:-2, :, :])
    t_region_dT[0, :, :] = t_region[1, :, :] - t_region[0, :, :]
    t_region_dT[-1, :, :] = t_region[-1, :, :] - t_region[-2, :, :]

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))
    t_dT_point = np.nanmean(t_region_dT, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]
    t_dT_aligned = t_dT_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- EF, SM at day of max regional mean T'; dot size = centered dT' that day
    heatwave_days = [onset_day + dt.timedelta(days=i) for i in range(int(duration_event))]
    heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
    if len(heatwave_idx) > 0:
        i_max = int(np.nanargmax(t_aligned[heatwave_idx]))
        idx_max = heatwave_idx[i_max]
        dT_hw = t_dT_aligned[idx_max]
        LE_max = LE_aligned[idx_max]
        sm_max = sm_aligned[idx_max]
        if np.isfinite(LE_max) and np.isfinite(sm_max) and np.isfinite(dT_hw):
            all_LE_hw.append(LE_max)
            all_sm_hw.append(sm_max - bpx_rhs_mean)
            all_dT_hw.append(float(dT_hw))

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_dT_hw = np.array(all_dT_hw)

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs")

if len(all_sm_hw) > 0:
    dT_min, dT_max = np.nanmin(all_dT_hw), np.nanmax(all_dT_hw)
    s_min, s_max = 40, 140
    if dT_max > dT_min:
        marker_sizes = s_min + (s_max - s_min) * (all_dT_hw - dT_min) / (dT_max - dT_min)
    else:
        marker_sizes = np.full(len(all_dT_hw), (s_min + s_max) / 2)
    plt.scatter(
        all_sm_hw, all_LE_hw,
        s=marker_sizes, c='red',
        edgecolors="k", linewidths=0.4,
        label=r"Max local T$'$ (size = centered dT$'$/dt)",
    )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWcenterIntensity_maxlocalTanom_centereddiff_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()
ds_t.close()
aaaaaa









# ======================================================= DENSITY PLOT ====================================================================

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []
all_duration_hw = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
    lats_events, lons_events, dates_d_HW_day0, duration_events
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- EF, SM, T on the day of maximum regional mean T during the event
    heatwave_days = [onset_day + dt.timedelta(days=i) for i in range(int(duration_event))]
    heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
    if len(heatwave_idx) > 0:
        i_max = int(np.nanargmax(t_aligned[heatwave_idx]))
        idx_max = heatwave_idx[i_max]
        tmax_hw = t_aligned[idx_max]
        LE_max = LE_aligned[idx_max]
        sm_max = sm_aligned[idx_max]
        if np.isfinite(LE_max) and np.isfinite(sm_max) and np.isfinite(tmax_hw):
            all_LE_hw.append(LE_max)
            all_sm_hw.append(sm_max - bpx_rhs_mean)
            all_t_hw.append(float(tmax_hw))
            all_duration_hw.append(int(duration_event))

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)
all_duration_hw = np.array(all_duration_hw)

if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs")

if len(all_sm_hw) > 0:
    t_min, t_max = np.nanmin(all_t_hw), np.nanmax(all_t_hw)
    s_min, s_max = 40, 140
    if t_max > t_min:
        marker_sizes = s_min + (s_max - s_min) * (all_t_hw - t_min) / (t_max - t_min)
    else:
        marker_sizes = np.full(len(all_t_hw), (s_min + s_max) / 2)
    d_min, d_max = np.min(all_duration_hw), np.max(all_duration_hw)
    alpha_min, alpha_max = 0.25, 0.95
    if d_max > d_min:
        marker_alphas = alpha_min + (alpha_max - alpha_min) * (all_duration_hw - d_min) / (d_max - d_min)
    else:
        marker_alphas = np.full(len(all_duration_hw), (alpha_min + alpha_max) / 2)
    marker_colors = np.column_stack([np.ones(len(all_sm_hw)), np.zeros(len(all_sm_hw)), np.zeros(len(all_sm_hw)), marker_alphas])
    plt.scatter(
        all_sm_hw, all_LE_hw,
        s=marker_sizes, c=marker_colors,
        edgecolors="k", linewidths=0.4,
        label="Max local T (size = regional T, color = duration)",
    )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWcenterIntensity_maxlocalT_duration_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()
ds_t.close()
aaaaaa

# ======================================================= DENSITY PLOT ====================================================================

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []

all_sm_hw_p3 = []
all_LE_hw_p3 = []
all_t_hw_p3 = []

# For pre-onset dots -- 3 days before onset (yellow dots)
all_sm_hw_minus3 = []
all_LE_hw_minus3 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day in zip(
    lats_events, lons_events, dates_d_HW_day0
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset (regional mean T at onset sets marker size)
    onset_idx = np.where(common_dates == onset_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        T0 = t_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0) and np.isfinite(T0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_t_hw.append(float(T0))



    # --- onset +3 (yellow dots: 3 days after onset)
    onsetp3_day = onset_day + dt.timedelta(days=3)
    onsetp3_idx = np.where(common_dates == onsetp3_day)[0]
    if len(onsetp3_idx) == 1:
        LEp3 = LE_aligned[onsetp3_idx[0]]
        SMp3 = sm_aligned[onsetp3_idx[0]]
        Tp3 = t_aligned[onsetp3_idx[0]]
        if np.isfinite(LEp3) and np.isfinite(SMp3):
            all_LE_hw_p3.append(LEp3)
            all_sm_hw_p3.append(SMp3 - bpx_rhs_mean)
            all_t_hw_p3.append(float(Tp3))

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)

all_sm_hw_p3 = np.array(all_sm_hw_p3)
all_LE_hw_p3 = np.array(all_LE_hw_p3)
all_t_hw_p3 = np.array(all_t_hw_p3)
# Convert to °C if stored in Kelvin
if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15
if len(all_t_hw_p3) > 0 and np.nanmean(all_t_hw_p3) > 200:
    all_t_hw_p3 = all_t_hw_p3 - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs")

if len(all_sm_hw) > 0 or len(all_sm_hw_p3) > 0:
    t_all = np.concatenate([all_t_hw, all_t_hw_p3]) if len(all_t_hw) > 0 and len(all_t_hw_p3) > 0 else (
        all_t_hw if len(all_t_hw) > 0 else all_t_hw_p3
    )
    t_min, t_max = np.nanmin(t_all), np.nanmax(t_all)
    s_min, s_max = 40, 140

    def _marker_sizes(t_vals):
        if t_max > t_min:
            return s_min + (s_max - s_min) * (t_vals - t_min) / (t_max - t_min)
        return np.full(len(t_vals), (s_min + s_max) / 2)

    if len(all_sm_hw) > 0:
        plt.scatter(
            all_sm_hw, all_LE_hw,
            s=_marker_sizes(all_t_hw), c="red", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Onset (dot size = regional T)",
        )
    if len(all_sm_hw_p3) > 0:
        plt.scatter(
            all_sm_hw_p3, all_LE_hw_p3,
            s=_marker_sizes(all_t_hw_p3), c="gold", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Onset+3d",
        )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWcenterIntensity_onsets_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()




# ======================================================= DENSITY PLOT ====================================================================
pos_maxTs = df_heatwaves.iloc[:, 6]
dates_d_maxTs = dates_d[pos_maxTs].flatten()
dates_d_maxTs = np.array([x.date() for x in dates_d_maxTs])

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []


# For pre-onset dots -- 3 days before onset (yellow dots)
all_sm_hw_minus3 = []
all_LE_hw_minus3 = []
all_t_hw_minus3 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, maxTs_day in zip(
    lats_events, lons_events, dates_d_maxTs
):

    if maxTs_day < dates_sm[0] or maxTs_day < dates_LE[0] or maxTs_day < dates_t[0]:
        print(f"Skipping event {maxTs_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset (regional mean T at onset sets marker size)
    onset_idx = np.where(common_dates == maxTs_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        T0 = t_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0) and np.isfinite(T0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_t_hw.append(float(T0))



    # --- onset -3 (yellow dots: 3 days before max T onset)
    onsetm5_day = maxTs_day - dt.timedelta(days=5)
    onsetm5_idx = np.where(common_dates == onsetm5_day)[0]
    if len(onsetm5_idx) == 1:
        LEm5 = LE_aligned[onsetm5_idx[0]]
        SMm5 = sm_aligned[onsetm5_idx[0]]
        Tm5 = t_aligned[onsetm5_idx[0]]
        if np.isfinite(LEm5) and np.isfinite(SMm5) and np.isfinite(Tm5):
            all_LE_hw_minus3.append(LEm5)
            all_sm_hw_minus3.append(SMm5 - bpx_rhs_mean)
            all_t_hw_minus3.append(float(Tm5))

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)

all_sm_hw_minus3 = np.array(all_sm_hw_minus3)
all_LE_hw_minus3 = np.array(all_LE_hw_minus3)
all_t_hw_minus3 = np.array(all_t_hw_minus3)

# Convert to °C if stored in Kelvin (both arrays — shared scale for marker sizes)
if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15
if len(all_t_hw_minus3) > 0 and np.nanmean(all_t_hw_minus3) > 200:
    all_t_hw_minus3 = all_t_hw_minus3 - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs (regional mean)")

if len(all_sm_hw) > 0 or len(all_sm_hw_minus3) > 0:
    t_all = np.concatenate([all_t_hw, all_t_hw_minus3]) if len(all_t_hw) > 0 and len(all_t_hw_minus3) > 0 else (
        all_t_hw if len(all_t_hw) > 0 else all_t_hw_minus3
    )
    t_min, t_max = np.nanmin(t_all), np.nanmax(t_all)
    s_min, s_max = 40, 140

    def _marker_sizes(t_vals):
        if t_max > t_min:
            return s_min + (s_max - s_min) * (t_vals - t_min) / (t_max - t_min)
        return np.full(len(t_vals), (s_min + s_max) / 2)

    if len(all_sm_hw) > 0:
        plt.scatter(
            all_sm_hw, all_LE_hw,
            s=_marker_sizes(all_t_hw), c="red", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Max T (dot size = regional T)",
        )
    if len(all_sm_hw_minus3) > 0:
        plt.scatter(
            all_sm_hw_minus3, all_LE_hw_minus3,
            s=_marker_sizes(all_t_hw_minus3), c="gold", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Max T -5d",
        )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWmaxT_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()





# ======================================================= DENSITY PLOT ====================================================================

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []

all_sm_hw_p3 = []
all_LE_hw_p3 = []
all_t_hw_p3 = []

# For pre-onset dots -- 3 days before onset (yellow dots)
all_sm_hw_minus3 = []
all_LE_hw_minus3 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day in zip(
    lats_events, lons_events, dates_d_HW_day0
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset (regional mean T at onset sets marker size)
    onset_idx = np.where(common_dates == onset_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        T0 = t_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0) and np.isfinite(T0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_t_hw.append(float(T0))



    # --- onset +3 (yellow dots: 3 days after onset)
    onsetp3_day = onset_day + dt.timedelta(days=3)
    onsetp3_idx = np.where(common_dates == onsetp3_day)[0]
    if len(onsetp3_idx) == 1:
        LEp3 = LE_aligned[onsetp3_idx[0]]
        SMp3 = sm_aligned[onsetp3_idx[0]]
        Tp3 = t_aligned[onsetp3_idx[0]]
        if np.isfinite(LEp3) and np.isfinite(SMp3):
            all_LE_hw_p3.append(LEp3)
            all_sm_hw_p3.append(SMp3 - bpx_rhs_mean)
            all_t_hw_p3.append(float(Tp3))

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)

all_sm_hw_p3 = np.array(all_sm_hw_p3)
all_LE_hw_p3 = np.array(all_LE_hw_p3)
all_t_hw_p3 = np.array(all_t_hw_p3)
# Convert to °C if stored in Kelvin
if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15
if len(all_t_hw_p3) > 0 and np.nanmean(all_t_hw_p3) > 200:
    all_t_hw_p3 = all_t_hw_p3 - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs")

if len(all_sm_hw) > 0 or len(all_sm_hw_p3) > 0:
    t_all = np.concatenate([all_t_hw, all_t_hw_p3]) if len(all_t_hw) > 0 and len(all_t_hw_p3) > 0 else (
        all_t_hw if len(all_t_hw) > 0 else all_t_hw_p3
    )
    t_min, t_max = np.nanmin(t_all), np.nanmax(t_all)
    s_min, s_max = 40, 140

    def _marker_sizes(t_vals):
        if t_max > t_min:
            return s_min + (s_max - s_min) * (t_vals - t_min) / (t_max - t_min)
        return np.full(len(t_vals), (s_min + s_max) / 2)

    if len(all_sm_hw) > 0:
        plt.scatter(
            all_sm_hw, all_LE_hw,
            s=_marker_sizes(all_t_hw), c="red", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Onset (dot size = regional T)",
        )
    if len(all_sm_hw_p3) > 0:
        plt.scatter(
            all_sm_hw_p3, all_LE_hw_p3,
            s=_marker_sizes(all_t_hw_p3), c="gold", alpha=0.75,
            edgecolors="k", linewidths=0.4,
            label="Onset+3d",
        )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWonset_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()

aaaaaaa









# ======================================================= DENSITY PLOT ====================================================================
pos_maxTs = df_heatwaves.iloc[:, 6]
dates_d_maxTs = dates_d[pos_maxTs].flatten()
dates_d_maxTs = np.array([x.date() for x in dates_d_maxTs])

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []

all_sm_hw5 = []
all_LE_hw5 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, maxTs_day in zip(
    lats_events, lons_events, dates_d_maxTs
):

    if maxTs_day < dates_sm[0] or maxTs_day < dates_LE[0] or maxTs_day < dates_t[0]:
        print(f"Skipping event {maxTs_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset (regional mean T at onset sets marker size)
    onset_idx = np.where(common_dates == maxTs_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        T0 = t_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0) and np.isfinite(T0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_t_hw.append(float(T0))

    # --- onset +3
    onset5_day = maxTs_day + dt.timedelta(days=3)
    onset5_idx = np.where(common_dates == onset5_day)[0]
    if len(onset5_idx) == 1:
        LE5 = LE_aligned[onset5_idx[0]]
        SM5 = sm_aligned[onset5_idx[0]]
        if np.isfinite(LE5) and np.isfinite(SM5):
            all_LE_hw5.append(LE5)
            all_sm_hw5.append(SM5 - bpx_rhs_mean)

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)

all_sm_hw5 = np.array(all_sm_hw5)
all_LE_hw5 = np.array(all_LE_hw5)

# Convert to °C if stored in Kelvin
if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs (regional mean)")

if len(all_sm_hw) > 0:
    t_min, t_max = np.nanmin(all_t_hw), np.nanmax(all_t_hw)
    if t_max > t_min:
        marker_sizes = 40 + 160 * (all_t_hw - t_min) / (t_max - t_min)
    else:
        marker_sizes = np.full(len(all_t_hw), 80.0)
    plt.scatter(
        all_sm_hw, all_LE_hw,
        s=marker_sizes, c="red", alpha=0.75,
        edgecolors="k", linewidths=0.4,
        label="Max T (dot size = regional T)",
    )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWmaxT_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()

aaaaaaa












# ======================================================= DENSITY PLOT ====================================================================
path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers
# =======================================================
coord_events = df_heatwaves.iloc[:, 5]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_intensity_hw = []

all_sm_hw5 = []
all_LE_hw5 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day, maxT_event in zip(
    lats_events, lons_events, dates_d_HW_day0, maxT_events
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values

    if LE_region.ndim != 3 or sm_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates
    common_dates = np.intersect1d(dates_LE, dates_sm)

    LE_mask = np.isin(dates_LE, common_dates)
    sm_mask = np.isin(dates_sm, common_dates)

    LE_aligned = LE_point[LE_mask]
    sm_aligned = sm_point[sm_mask]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset
    onset_idx = np.where(common_dates == onset_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_intensity_hw.append(float(maxT_event))

    # --- onset +3
    onset5_day = onset_day + dt.timedelta(days=3)
    onset5_idx = np.where(common_dates == onset5_day)[0]
    if len(onset5_idx) == 1:
        LE5 = LE_aligned[onset5_idx[0]]
        SM5 = sm_aligned[onset5_idx[0]]
        if np.isfinite(LE5) and np.isfinite(SM5):
            all_LE_hw5.append(LE5)
            all_sm_hw5.append(SM5 - bpx_rhs_mean)

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_intensity_hw = np.array(all_intensity_hw)

all_sm_hw5 = np.array(all_sm_hw5)
all_LE_hw5 = np.array(all_LE_hw5)

print("all_sm_summer shape:", all_sm_summer.shape)
print("all_LE_summer shape:", all_LE_summer.shape)
print("all_sm_hw:", all_sm_hw)
print("all_LE_hw:", all_LE_hw)

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask] 
LE_summer_valid = all_LE_summer[mask] 

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs (regional mean)")

if len(all_sm_hw) > 0:
    i_min, i_max = np.nanmin(all_intensity_hw), np.nanmax(all_intensity_hw)
    if i_max > i_min:
        marker_sizes = 40 + 160 * (all_intensity_hw - i_min) / (i_max - i_min)
    else:
        marker_sizes = np.full(len(all_intensity_hw), 80.0)
    plt.scatter(
        all_sm_hw, all_LE_hw,
        s=marker_sizes, c="red", alpha=0.75,
        edgecolors="k", linewidths=0.4,
        label="Onset (dot size = intensity)",
    )
# plt.scatter(all_sm_hw5, all_LE_hw5, marker="x", s=22, label="Onset +3 days", color="orange")

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWmaxT_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()
aaaaaa





# ======================================================= DENSITY PLOT ====================================================================
path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
miss_bp = -9.99e08
bp_rhs_field_plot = np.where(np.isclose(bp_rhs_field, miss_bp), np.nan, bp_rhs_field)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers (aligned with dates_d_HW_day0)
# =======================================================
coord_events = df_events_valid.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

dates_t = np.array([x.date() for x in dates_d])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []
all_t_hw = []

all_sm_hw5 = []
all_LE_hw5 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day in zip(
    lats_events, lons_events, dates_d_HW_day0
):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM/T data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_idx(lons_t, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0 or len(ilat_t) == 0 or len(ilon_t) == 0:
        continue

    bpx_rhs_mean = np.nanmean(bp_rhs_field_plot[np.ix_(ilat_sm, ilon_sm)])
    if not np.isfinite(bpx_rhs_mean):
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or t_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any() or not np.isfinite(t_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))
    t_point = np.nanmean(t_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates across EF, SM, and T
    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    t_aligned = t_point[np.isin(dates_t, common_dates)]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid] - bpx_rhs_mean)

    # --- onset (regional mean T at onset sets marker size)
    onset_idx = np.where(common_dates == onset_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        T0 = t_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0) and np.isfinite(T0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0 - bpx_rhs_mean)
            all_t_hw.append(float(T0))

    # --- onset +3
    onset5_day = onset_day + dt.timedelta(days=3)
    onset5_idx = np.where(common_dates == onset5_day)[0]
    if len(onset5_idx) == 1:
        LE5 = LE_aligned[onset5_idx[0]]
        SM5 = sm_aligned[onset5_idx[0]]
        if np.isfinite(LE5) and np.isfinite(SM5):
            all_LE_hw5.append(LE5)
            all_sm_hw5.append(SM5 - bpx_rhs_mean)

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)
all_t_hw = np.array(all_t_hw)

all_sm_hw5 = np.array(all_sm_hw5)
all_LE_hw5 = np.array(all_LE_hw5)

# Convert to °C if stored in Kelvin
if len(all_t_hw) > 0 and np.nanmean(all_t_hw) > 200:
    all_t_hw = all_t_hw - 273.15

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask]
LE_summer_valid = all_LE_summer[mask]

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="BPx_rhs (regional mean)")

if len(all_sm_hw) > 0:
    t_min, t_max = np.nanmin(all_t_hw), np.nanmax(all_t_hw)
    if t_max > t_min:
        marker_sizes = 40 + 160 * (all_t_hw - t_min) / (t_max - t_min)
    else:
        marker_sizes = np.full(len(all_t_hw), 80.0)
    plt.scatter(
        all_sm_hw, all_LE_hw,
        s=marker_sizes, c="red", alpha=0.75,
        edgecolors="k", linewidths=0.4,
        label="Onset (dot size = regional T)",
    )

plt.xlabel(r"SMs $-$ BPx$_{\mathrm{RHS}}$ (regional mean, ±0.5°)", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center (x relative to BPx_rhs)",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWonset_eachEvent_05deg_relBPx_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()

aaaaaaa











