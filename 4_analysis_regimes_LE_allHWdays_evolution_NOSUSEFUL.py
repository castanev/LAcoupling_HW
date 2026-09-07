# Aggregate SM–LE evolution around day of max T (onset location).
# Summer density + yellow (−3/−2/−1), red (max T), brown (+1/+2/+3) point clouds.
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import scipy as scp
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import scipy as scp
import os
import glob
import re
from scipy.ndimage import uniform_filter
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import argparse
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
path_ncr = '/apps/spack/negishi/apps/nco/5.0.1-gcc-12.2.0-f3lr7i3/bin/ncrcat'


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--name_land', type=str, required=True)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--path_case', type=str, required=True)
parser.add_argument('--path_case_land', type=str, required=True)
parser.add_argument('--path_file_SMs', type=str, required=True)
parser.add_argument('--path_file_t', type=str, required=True)
parser.add_argument('--path_file_t_anom', type=str, required=True)
parser.add_argument('--initial_year', type=int, required=True)
parser.add_argument('--path_outputs', type=str, required=True)
args = parser.parse_args()

name = args.name
name_land = args.name_land
case = args.case
region = args.region
path_case = args.path_case
path_case_land = args.path_case_land
path_file_SMs = args.path_file_SMs
path_file_t = args.path_file_t
path_file_t_anom = args.path_file_t_anom
initial_year = args.initial_year
path_outputs = args.path_outputs


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
# Spatial center = Onset Position (column 4)
lats_events, lons_events = extract_latlons_from_coord_events(df_events_valid.iloc[:, 4])
duration_events = df_events_valid.iloc[:, 1].values
dates_d_HW_day0 = dates_d[df_events_valid.index.values].flatten()
dates_d_HW_day0 = np.array([x.date() for x in dates_d_HW_day0])
# Temporal center = day of maximum Ts (column 6)
pos_max_Ts = df_events_valid.iloc[:, 6].astype(int).values
dates_d_HW_maxT = np.array([dates_d[i].date() for i in pos_max_Ts])

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




# ======================================================= EVOLUTION CLOUDS ====================================================================
# Summer SM–LE density by regime, with point clouds relative to day of max T:
# yellow (−3/−2/−1), red (max T), brown (+1/+2/+3). Location = Onset Position.
lags_days = [5]
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'

PRE_LAGS = (-3, -2, -1)
CENTER_LAG = 0
POST_LAGS = (1, 2, 3)
COLOR_PRE = 'gold'
COLOR_MAXT = 'red'
COLOR_POST = 'saddlebrown'


def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)
    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360
    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width
    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# Event centers: onset position (col 4); timing from max-T day (col 6)
coord_events = df_events_valid.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

variable_LE = 'LE'
variable_sm = 'SMs'
dates_t = np.array([x.date() for x in dates_d])

os.makedirs(path_outputs_case, exist_ok=True)
density_cache_path = (
    f'{path_outputs_case}density_data_{variable_LE}_{variable_sm}_'
    f'summer_maxTevol_05deg_{name_land}_{region}.npz'
)


def _as_1d_points(arr_or_list):
    if isinstance(arr_or_list, list):
        return np.concatenate(arr_or_list) if len(arr_or_list) > 0 else np.array([])
    arr = np.asarray(arr_or_list)
    return arr.ravel() if arr.size else np.array([])


def _append_lag_cloud(sm_aligned, LE_aligned, date_to_i, center_day, lags, sm_list, le_list):
    """Append SM/LE values for relative lags around center_day (one event)."""
    sm_vals, le_vals = [], []
    for lag in lags:
        day = center_day + dt.timedelta(days=int(lag))
        i = date_to_i.get(day)
        if i is None:
            continue
        smv, lev = float(sm_aligned[i]), float(LE_aligned[i])
        if np.isfinite(smv) and np.isfinite(lev):
            sm_vals.append(smv)
            le_vals.append(lev)
    if sm_vals:
        sm_list.append(np.asarray(sm_vals, dtype=float))
        le_list.append(np.asarray(le_vals, dtype=float))
        return True
    return False


def _plot_evolution_panel(
    all_sm_summer,
    all_LE_summer,
    sm_pre, LE_pre,
    sm_maxT, LE_maxT,
    sm_post, LE_post,
    xlabel,
    title,
    outfile,
    show_critical_vline=False,
    density_vmax=0.2,
    min_events=5,
    n_events=None,
    xmax_sm=None,
    show_clouds=True,
):
    if n_events is None:
        n_events = 0
    if show_clouds and n_events < min_events:
        print(f'Skipping plot (<{min_events} events, n={n_events}): {outfile}')
        return

    sm_summer = _as_1d_points(all_sm_summer)
    LE_summer = _as_1d_points(all_LE_summer)
    sm_pre = _as_1d_points(sm_pre)
    LE_pre = _as_1d_points(LE_pre)
    sm_maxT = _as_1d_points(sm_maxT)
    LE_maxT = _as_1d_points(LE_maxT)
    sm_post = _as_1d_points(sm_post)
    LE_post = _as_1d_points(LE_post)

    mask_s = np.isfinite(sm_summer) & np.isfinite(LE_summer)
    sm_s = sm_summer[mask_s]
    LE_s = LE_summer[mask_s]

    cloud_xy = []
    for sx, ly in (
        (sm_pre, LE_pre), (sm_maxT, LE_maxT), (sm_post, LE_post),
    ):
        m = np.isfinite(sx) & np.isfinite(ly)
        cloud_xy.append((sx[m], ly[m]))
    sm_pre_v, LE_pre_v = cloud_xy[0]
    sm_maxT_v, LE_maxT_v = cloud_xy[1]
    sm_post_v, LE_post_v = cloud_xy[2]

    has_summer = sm_s.size > 0
    has_clouds = any(a.size > 0 for a, _ in cloud_xy)
    if not has_summer and not (show_clouds and has_clouds):
        print(f'Skipping plot (no events): {outfile}')
        return

    x_parts = [sm_s] if has_summer else []
    y_parts = [LE_s] if has_summer else []
    if show_clouds:
        for sx, ly in cloud_xy:
            if sx.size:
                x_parts.append(sx)
                y_parts.append(ly)
    xmin = float(np.nanmin(np.concatenate(x_parts)))
    xmax = float(xmax_sm) if xmax_sm is not None else float(np.nanmax(np.concatenate(x_parts)))
    ymin = float(np.nanmin(np.concatenate(y_parts)))
    ymax_le = 150.0

    fig = plt.figure(figsize=(7.0, 5))
    ax = fig.add_axes([0.11, 0.12, 0.70, 0.78])
    cbar_fmt = FormatStrFormatter('%.2f')

    if has_summer and sm_s.size > 5:
        xgrid, ygrid = np.meshgrid(
            np.linspace(xmin, xmax, 45),
            np.linspace(ymin, ymax_le, 45),
        )
        levels = np.linspace(0, density_vmax, 21)
        kde_s = gaussian_kde(np.vstack([sm_s, LE_s]))
        z_s = kde_s(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cf = ax.contourf(
            xgrid, ygrid, z_s,
            levels=levels, cmap='Blues', vmin=0, vmax=density_vmax,
            alpha=0.8, extend='max',
        )
        ax_pos = ax.get_position()
        cax_s = fig.add_axes([ax_pos.x1 + 0.006, ax_pos.y0, 0.022, ax_pos.height])
        cbar_s = fig.colorbar(cf, cax=cax_s, extend='max')
        cbar_s.set_label(f'Summer density (0–{density_vmax:.2f})', fontsize=10)
        cbar_s.ax.yaxis.set_major_formatter(cbar_fmt)
        cbar_s.ax.tick_params(labelsize=9)
    elif has_summer:
        ax.scatter(sm_s, LE_s, s=10, alpha=0.3, color='cornflowerblue',
                   label='Summer (excl. HW)')

    if show_clouds:
        if sm_pre_v.size:
            ax.scatter(
                sm_pre_v, LE_pre_v,
                s=18, marker='o', color=COLOR_PRE, edgecolors='none',
                alpha=0.45, zorder=5, label=f'Day −3/−2/−1 (n={sm_pre_v.size})',
            )
        if sm_maxT_v.size:
            ax.scatter(
                sm_maxT_v, LE_maxT_v,
                s=28, marker='o', color=COLOR_MAXT, edgecolors='none',
                alpha=0.55, zorder=6, label=f'Day of max T (n={sm_maxT_v.size})',
            )
        if sm_post_v.size:
            ax.scatter(
                sm_post_v, LE_post_v,
                s=18, marker='o', color=COLOR_POST, edgecolors='none',
                alpha=0.45, zorder=5, label=f'Day +1/+2/+3 (n={sm_post_v.size})',
            )

    if show_critical_vline:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7, label='Critical SM')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax_le)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'LE (W m$^{-2}$)', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=10, loc='best')
    fig.savefig(outfile, dpi=500)
    plt.close(fig)
    print(f'Saved {outfile}')


for lag_days in lags_days:
    load_cache = False
    if os.path.exists(density_cache_path):
        cache = np.load(density_cache_path)
        needed = (
            'weak_sm_pre', 'weak_sm_maxT', 'weak_sm_post',
            'trans_sm_pre', 'ml_sm_pre', 'r3_sm_pre', 'r4_sm_pre',
        )
        if all(k in cache.files for k in needed):
            load_cache = True
        else:
            print(f'Cache missing evolution fields; recomputing ({density_cache_path})')
            cache.close()

    if load_cache:
        print(f'Loading cached evolution data from {density_cache_path}')
        def _load_bucket(prefix):
            return (
                cache[f'{prefix}_sm_summer'], cache[f'{prefix}_LE_summer'],
                cache[f'{prefix}_sm_pre'], cache[f'{prefix}_LE_pre'],
                cache[f'{prefix}_sm_maxT'], cache[f'{prefix}_LE_maxT'],
                cache[f'{prefix}_sm_post'], cache[f'{prefix}_LE_post'],
                int(cache[f'n_evt_{prefix}']),
            )
        (weak_sm_summer, weak_LE_summer, weak_sm_pre, weak_LE_pre,
         weak_sm_maxT, weak_LE_maxT, weak_sm_post, weak_LE_post, n_evt_weak) = _load_bucket('weak')
        (trans_sm_summer, trans_LE_summer, trans_sm_pre, trans_LE_pre,
         trans_sm_maxT, trans_LE_maxT, trans_sm_post, trans_LE_post, n_evt_trans) = _load_bucket('trans')
        (r3_sm_summer, r3_LE_summer, r3_sm_pre, r3_LE_pre,
         r3_sm_maxT, r3_LE_maxT, r3_sm_post, r3_LE_post, n_evt_r3) = _load_bucket('r3')
        (r4_sm_summer, r4_LE_summer, r4_sm_pre, r4_LE_pre,
         r4_sm_maxT, r4_LE_maxT, r4_sm_post, r4_LE_post, n_evt_r4) = _load_bucket('r4')
        (ml_sm_summer, ml_LE_summer, ml_sm_pre, ml_LE_pre,
         ml_sm_maxT, ml_LE_maxT, ml_sm_post, ml_LE_post, n_evt_ml) = _load_bucket('ml')
    else:
        print(f'Computing evolution data (will cache to {density_cache_path})')
        ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine='netcdf4')
        ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')
        ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine='netcdf4')

        lats_LE = ds_LE.lat.values
        lons_LE = ds_LE.lon.values
        dates_LE = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_LE.time.values])
        lats_sm = ds_sm.lat.values
        lons_sm = ds_sm.lon.values
        dates_sm = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_sm.time.values])

        weak_sm_summer, weak_LE_summer = [], []
        weak_sm_pre, weak_LE_pre = [], []
        weak_sm_maxT, weak_LE_maxT = [], []
        weak_sm_post, weak_LE_post = [], []
        n_evt_weak = 0

        trans_sm_summer, trans_LE_summer = [], []
        trans_sm_pre, trans_LE_pre = [], []
        trans_sm_maxT, trans_LE_maxT = [], []
        trans_sm_post, trans_LE_post = [], []
        n_evt_trans = 0

        r3_sm_summer, r3_LE_summer = [], []
        r3_sm_pre, r3_LE_pre = [], []
        r3_sm_maxT, r3_LE_maxT = [], []
        r3_sm_post, r3_LE_post = [], []
        n_evt_r3 = 0

        r4_sm_summer, r4_LE_summer = [], []
        r4_sm_pre, r4_LE_pre = [], []
        r4_sm_maxT, r4_LE_maxT = [], []
        r4_sm_post, r4_LE_post = [], []
        n_evt_r4 = 0

        for lat_center_hw, lon_center_hw, onset_day, maxT_day, duration_event in zip(
            lats_events, lons_events, dates_d_HW_day0, dates_d_HW_maxT, duration_events
        ):
            if maxT_day < dates_sm[0] or maxT_day < dates_LE[0] or maxT_day < dates_t[0]:
                print(f'Skipping event maxT {maxT_day}: before SM/LE/T start')
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
                len(ilat_LE) == 0 or len(ilon_LE) == 0
                or len(ilat_sm) == 0 or len(ilon_sm) == 0
                or len(ilat_t) == 0 or len(ilon_t) == 0
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
            LE_point = np.where((LE_point >= 0) & np.isfinite(LE_point), LE_point, np.nan)

            if dom_regime == 0:
                sm_point = np.nanmean(sm_region, axis=(1, 2))
                sm_s_l, le_s_l = weak_sm_summer, weak_LE_summer
                sm_pre_l, le_pre_l = weak_sm_pre, weak_LE_pre
                sm_c_l, le_c_l = weak_sm_maxT, weak_LE_maxT
                sm_post_l, le_post_l = weak_sm_post, weak_LE_post
                evt_counter = 'weak'
            elif dom_regime in (1, 2):
                sm_point = np.nanmean(sm_region, axis=(1, 2))
                sm_s_l, le_s_l = trans_sm_summer, trans_LE_summer
                sm_pre_l, le_pre_l = trans_sm_pre, trans_LE_pre
                sm_c_l, le_c_l = trans_sm_maxT, trans_LE_maxT
                sm_post_l, le_post_l = trans_sm_post, trans_LE_post
                evt_counter = 'trans'
            elif dom_regime == 3:
                bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
                if not np.isfinite(bp_box).any():
                    continue
                sm_point = np.nanmean(sm_region - bp_box[np.newaxis, :, :], axis=(1, 2))
                sm_s_l, le_s_l = r3_sm_summer, r3_LE_summer
                sm_pre_l, le_pre_l = r3_sm_pre, r3_LE_pre
                sm_c_l, le_c_l = r3_sm_maxT, r3_LE_maxT
                sm_post_l, le_post_l = r3_sm_post, r3_LE_post
                evt_counter = 'r3'
            elif dom_regime == 4:
                bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
                if not np.isfinite(bp_box).any():
                    continue
                sm_point = np.nanmean(sm_region - bp_box[np.newaxis, :, :], axis=(1, 2))
                sm_s_l, le_s_l = r4_sm_summer, r4_LE_summer
                sm_pre_l, le_pre_l = r4_sm_pre, r4_LE_pre
                sm_c_l, le_c_l = r4_sm_maxT, r4_LE_maxT
                sm_post_l, le_post_l = r4_sm_post, r4_LE_post
                evt_counter = 'r4'
            else:
                continue

            common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)
            date_to_i = {d: i for i, d in enumerate(common_dates)}
            LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
            sm_aligned = sm_point[np.isin(dates_sm, common_dates)]

            heatwave_days = [
                onset_day + dt.timedelta(days=i) for i in range(int(duration_event))
            ]
            hw_mask = np.array([d in set(heatwave_days) for d in common_dates])
            summer_mask = np.array([d.month in (6, 7, 8) for d in common_dates]) & (~hw_mask)
            LE_summer = LE_aligned[summer_mask]
            sm_summer = sm_aligned[summer_mask]
            valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
            if np.any(valid):
                sm_s_l.append(sm_summer[valid])
                le_s_l.append(LE_summer[valid])

            _append_lag_cloud(
                sm_aligned, LE_aligned, date_to_i, maxT_day, PRE_LAGS, sm_pre_l, le_pre_l
            )
            has_center = _append_lag_cloud(
                sm_aligned, LE_aligned, date_to_i, maxT_day, (CENTER_LAG,), sm_c_l, le_c_l
            )
            _append_lag_cloud(
                sm_aligned, LE_aligned, date_to_i, maxT_day, POST_LAGS, sm_post_l, le_post_l
            )
            if has_center:
                if evt_counter == 'weak':
                    n_evt_weak += 1
                elif evt_counter == 'trans':
                    n_evt_trans += 1
                elif evt_counter == 'r3':
                    n_evt_r3 += 1
                elif evt_counter == 'r4':
                    n_evt_r4 += 1

        def _flatten_bucket(sm_s, le_s, sm_pre, le_pre, sm_c, le_c, sm_post, le_post):
            return (
                _as_1d_points(sm_s), _as_1d_points(le_s),
                _as_1d_points(sm_pre), _as_1d_points(le_pre),
                _as_1d_points(sm_c), _as_1d_points(le_c),
                _as_1d_points(sm_post), _as_1d_points(le_post),
            )

        (weak_sm_summer, weak_LE_summer, weak_sm_pre, weak_LE_pre,
         weak_sm_maxT, weak_LE_maxT, weak_sm_post, weak_LE_post) = _flatten_bucket(
            weak_sm_summer, weak_LE_summer, weak_sm_pre, weak_LE_pre,
            weak_sm_maxT, weak_LE_maxT, weak_sm_post, weak_LE_post,
        )
        (trans_sm_summer, trans_LE_summer, trans_sm_pre, trans_LE_pre,
         trans_sm_maxT, trans_LE_maxT, trans_sm_post, trans_LE_post) = _flatten_bucket(
            trans_sm_summer, trans_LE_summer, trans_sm_pre, trans_LE_pre,
            trans_sm_maxT, trans_LE_maxT, trans_sm_post, trans_LE_post,
        )
        (r3_sm_summer, r3_LE_summer, r3_sm_pre, r3_LE_pre,
         r3_sm_maxT, r3_LE_maxT, r3_sm_post, r3_LE_post) = _flatten_bucket(
            r3_sm_summer, r3_LE_summer, r3_sm_pre, r3_LE_pre,
            r3_sm_maxT, r3_LE_maxT, r3_sm_post, r3_LE_post,
        )
        (r4_sm_summer, r4_LE_summer, r4_sm_pre, r4_LE_pre,
         r4_sm_maxT, r4_LE_maxT, r4_sm_post, r4_LE_post) = _flatten_bucket(
            r4_sm_summer, r4_LE_summer, r4_sm_pre, r4_LE_pre,
            r4_sm_maxT, r4_LE_maxT, r4_sm_post, r4_LE_post,
        )

        n_evt_ml = n_evt_r3 + n_evt_r4
        ml_sm_summer = _as_1d_points([r3_sm_summer, r4_sm_summer])
        ml_LE_summer = _as_1d_points([r3_LE_summer, r4_LE_summer])
        ml_sm_pre = _as_1d_points([r3_sm_pre, r4_sm_pre])
        ml_LE_pre = _as_1d_points([r3_LE_pre, r4_LE_pre])
        ml_sm_maxT = _as_1d_points([r3_sm_maxT, r4_sm_maxT])
        ml_LE_maxT = _as_1d_points([r3_LE_maxT, r4_LE_maxT])
        ml_sm_post = _as_1d_points([r3_sm_post, r4_sm_post])
        ml_LE_post = _as_1d_points([r3_LE_post, r4_LE_post])

        np.savez_compressed(
            density_cache_path,
            weak_sm_summer=weak_sm_summer, weak_LE_summer=weak_LE_summer,
            weak_sm_pre=weak_sm_pre, weak_LE_pre=weak_LE_pre,
            weak_sm_maxT=weak_sm_maxT, weak_LE_maxT=weak_LE_maxT,
            weak_sm_post=weak_sm_post, weak_LE_post=weak_LE_post,
            n_evt_weak=n_evt_weak,
            trans_sm_summer=trans_sm_summer, trans_LE_summer=trans_LE_summer,
            trans_sm_pre=trans_sm_pre, trans_LE_pre=trans_LE_pre,
            trans_sm_maxT=trans_sm_maxT, trans_LE_maxT=trans_LE_maxT,
            trans_sm_post=trans_sm_post, trans_LE_post=trans_LE_post,
            n_evt_trans=n_evt_trans,
            r3_sm_summer=r3_sm_summer, r3_LE_summer=r3_LE_summer,
            r3_sm_pre=r3_sm_pre, r3_LE_pre=r3_LE_pre,
            r3_sm_maxT=r3_sm_maxT, r3_LE_maxT=r3_LE_maxT,
            r3_sm_post=r3_sm_post, r3_LE_post=r3_LE_post,
            n_evt_r3=n_evt_r3,
            r4_sm_summer=r4_sm_summer, r4_LE_summer=r4_LE_summer,
            r4_sm_pre=r4_sm_pre, r4_LE_pre=r4_LE_pre,
            r4_sm_maxT=r4_sm_maxT, r4_LE_maxT=r4_LE_maxT,
            r4_sm_post=r4_sm_post, r4_LE_post=r4_LE_post,
            n_evt_r4=n_evt_r4,
            ml_sm_summer=ml_sm_summer, ml_LE_summer=ml_LE_summer,
            ml_sm_pre=ml_sm_pre, ml_LE_pre=ml_LE_pre,
            ml_sm_maxT=ml_sm_maxT, ml_LE_maxT=ml_LE_maxT,
            ml_sm_post=ml_sm_post, ml_LE_post=ml_LE_post,
            n_evt_ml=n_evt_ml,
        )
        print(f'Saved evolution data cache: {density_cache_path}')
        ds_LE.close()
        ds_sm.close()
        ds_t.close()

    def _panel(sm_s, le_s, sm_pre, le_pre, sm_c, le_c, sm_post, le_post,
               xlabel, title, outfile, n_events, density_vmax,
               show_critical_vline=False, xmax_sm=None, show_clouds=True):
        _plot_evolution_panel(
            sm_s, le_s, sm_pre, le_pre, sm_c, le_c, sm_post, le_post,
            xlabel=xlabel, title=title, outfile=outfile,
            show_critical_vline=show_critical_vline,
            density_vmax=density_vmax, n_events=n_events,
            xmax_sm=xmax_sm, show_clouds=show_clouds,
        )

    tag = (
        f'−3/−2/−1 (yellow), max T (red), +1/+2/+3 (brown); '
        f'onset location; relative to day of max T'
    )

    _panel(
        weak_sm_summer, weak_LE_summer, weak_sm_pre, weak_LE_pre,
        weak_sm_maxT, weak_LE_maxT, weak_sm_post, weak_LE_post,
        xlabel=r'SMs (regional mean, ±0.5°)',
        title=f'SM vs LE — weak coupling (regime 0)\n{tag}',
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_maxTevol_'
            f'05deg_weak_{name_land}_{region}.png'
        ),
        n_events=n_evt_weak, density_vmax=0.15,
    )
    _panel(
        trans_sm_summer, trans_LE_summer, trans_sm_pre, trans_LE_pre,
        trans_sm_maxT, trans_LE_maxT, trans_sm_post, trans_LE_post,
        xlabel=r'SMs (regional mean, ±0.5°)',
        title=f'SM vs LE — transitional (regime 1 or 2)\n{tag}',
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_maxTevol_'
            f'05deg_trans_{name_land}_{region}.png'
        ),
        n_events=n_evt_trans, density_vmax=0.6,
    )
    _panel(
        ml_sm_summer, ml_LE_summer, ml_sm_pre, ml_LE_pre,
        ml_sm_maxT, ml_LE_maxT, ml_sm_post, ml_LE_post,
        xlabel=r'SMs $-$ critical SM (regime 3/4 BP, regional mean, ±0.5°)',
        title=f'SM vs LE — moisture-limited (regime 3 or 4)\n{tag}',
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_maxTevol_'
            f'05deg_moistlim_{name_land}_{region}.png'
        ),
        n_events=n_evt_ml, density_vmax=0.35,
        show_critical_vline=True, xmax_sm=0.15,
    )
    _panel(
        r3_sm_summer, r3_LE_summer, r3_sm_pre, r3_LE_pre,
        r3_sm_maxT, r3_LE_maxT, r3_sm_post, r3_LE_post,
        xlabel=r'SMs $-$ critical SM (regime 3 BP, regional mean, ±0.5°)',
        title=f'SM vs LE — moisture-limited regime 3\n{tag}',
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_maxTevol_'
            f'05deg_regime3_{name_land}_{region}.png'
        ),
        n_events=n_evt_r3, density_vmax=0.35,
        show_critical_vline=True, xmax_sm=0.15,
    )
    _panel(
        r4_sm_summer, r4_LE_summer, r4_sm_pre, r4_LE_pre,
        r4_sm_maxT, r4_LE_maxT, r4_sm_post, r4_LE_post,
        xlabel=r'SMs $-$ critical SM (regime 4 BP, regional mean, ±0.5°)',
        title=f'SM vs LE — moisture-limited regime 4\n{tag}',
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_maxTevol_'
            f'05deg_regime4_{name_land}_{region}.png'
        ),
        n_events=n_evt_r4, density_vmax=0.35,
        show_critical_vline=True, xmax_sm=0.15,
    )
