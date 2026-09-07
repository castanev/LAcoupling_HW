# Per-event SM–LE trajectories around heatwave onset.
# Location: Max Anomaly Position (df_heatwaves.iloc[:, 5]).
# Time: onset as red; ±3 days yellow/brown.
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
import cartopy
import cartopy.crs as ccrs
from cartopy import crs as cartopy_crs
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
# Spatial center = point of maximum anomaly (column "Max Anomaly Position")
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




# ======================================================= PER-EVENT FIGURES ====================================================================
# One figure per HW event at Max Anomaly Position (iloc[:, 5]).
# Left: SM–LE summer density + onset±3 markers.
# Right: absolute T vs SM (same markers). T from path_file_t (TS_ERA5.nc).
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'
path_file_T_abs = path_file_t  # /depot/.../TS_ERA5.nc
path_event_figs = f'{path_figures}event_SM_LE_trajectories/'
os.makedirs(path_event_figs, exist_ok=True)

PRE_LAGS = (-3, -2, -1)
CENTER_LAG = 0  # onset
POST_LAGS = (1, 2, 3)
COLOR_PRE = 'gold'
COLOR_ONSET = 'red'
COLOR_POST = 'saddlebrown'

get_lon_idx = get_lon_idx_series

event_ids = df_events_valid.index.values
dates_onset = np.asarray(dates_d_HW_day0)

variable_LE = 'LE'
variable_sm = 'SMs'
variable_T_abs = 'TS'


def _scatter_lag_groups(ax, traj_pts, x_idx=1, y_idx=2, legend=True):
    """traj_pts: list of (lag, x, y, ...)."""
    pre = [p for p in traj_pts if p[0] in PRE_LAGS]
    center = [p for p in traj_pts if p[0] == CENTER_LAG]
    post = [p for p in traj_pts if p[0] in POST_LAGS]
    if pre:
        ax.scatter(
            [p[x_idx] for p in pre], [p[y_idx] for p in pre],
            s=55, marker='o', color=COLOR_PRE, edgecolors='k', linewidths=0.4,
            zorder=5, label='Day −3/−2/−1 vs onset' if legend else None,
        )
    if center:
        ax.scatter(
            [p[x_idx] for p in center], [p[y_idx] for p in center],
            s=90, marker='o', color=COLOR_ONSET, edgecolors='k', linewidths=0.5,
            zorder=6, label='Onset (day 0)' if legend else None,
        )
    if post:
        ax.scatter(
            [p[x_idx] for p in post], [p[y_idx] for p in post],
            s=55, marker='o', color=COLOR_POST, edgecolors='k', linewidths=0.4,
            zorder=5, label='Day +1/+2/+3 vs onset' if legend else None,
        )


def _plot_location_inset(fig, lat_center, lon_center, rect=(0.40, 0.78, 0.20, 0.14)):
    """Small US croquis with a red dot at the event location."""
    lon = float(lon_center)
    if lon < 0:
        lon += 360.0
    center_lon = 210
    axm = fig.add_axes(rect, projection=cartopy_crs.PlateCarree(central_longitude=center_lon))
    axm.add_feature(cartopy.feature.LAND, facecolor='0.92', edgecolor='none', zorder=0)
    axm.add_feature(cartopy.feature.OCEAN, facecolor='white', edgecolor='none', zorder=0)
    axm.add_feature(cartopy.feature.COASTLINE, lw=0.4, zorder=2)
    axm.add_feature(cartopy.feature.BORDERS, lw=0.3, zorder=2)
    axm.scatter(
        [lon], [float(lat_center)],
        s=36, c='red', marker='o', edgecolors='k', linewidths=0.5,
        transform=cartopy_crs.PlateCarree(), zorder=5,
    )
    axm.set_extent(
        [235 - center_lon, 290 - center_lon, 22, 52],
        crs=cartopy_crs.PlateCarree(central_longitude=center_lon),
    )
    axm.set_title(
        f'{lat_center:.1f}°N, {lon:.1f}°E',
        fontsize=8, pad=3,
    )
    for spine in axm.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('k')


def _plot_event_trajectory(
    sm_summer_LE,
    LE_summer,
    sm_summer_T,
    T_summer,
    traj,  # dict lag -> (sm, LE, T)
    xlabel,
    title,
    outfile,
    lat_center=None,
    lon_center=None,
    show_critical_vline=False,
    density_vmax=0.35,
    xmax_sm=None,
):
    sm_s = np.asarray(sm_summer_LE, dtype=float).ravel()
    LE_s = np.asarray(LE_summer, dtype=float).ravel()
    mask_le = np.isfinite(sm_s) & np.isfinite(LE_s)
    sm_s, LE_s = sm_s[mask_le], LE_s[mask_le]

    sm_st = np.asarray(sm_summer_T, dtype=float).ravel()
    T_st = np.asarray(T_summer, dtype=float).ravel()
    mask_t = np.isfinite(sm_st) & np.isfinite(T_st)
    sm_st, T_st = sm_st[mask_t], T_st[mask_t]

    traj_pts = []
    for lag, vals in traj.items():
        if vals is None or len(vals) < 3:
            continue
        smv, lev, tv = float(vals[0]), float(vals[1]), float(vals[2])
        traj_pts.append((lag, smv, lev, tv))

    traj_le = [(lag, sm, le) for lag, sm, le, tv in traj_pts
               if np.isfinite(sm) and np.isfinite(le)]
    traj_T = [(lag, sm, tv) for lag, sm, le, tv in traj_pts
              if np.isfinite(sm) and np.isfinite(tv)]

    if sm_s.size == 0 and len(traj_le) == 0 and sm_st.size == 0 and len(traj_T) == 0:
        print(f'Skipping (no data): {outfile}')
        return

    x_parts = []
    if sm_s.size:
        x_parts.append(sm_s)
    if sm_st.size:
        x_parts.append(sm_st)
    xs = [p[1] for p in traj_pts if np.isfinite(p[1])]
    if xs:
        x_parts.append(np.asarray(xs))
    xmin = float(np.nanmin(np.concatenate(x_parts)))
    xmax = float(xmax_sm) if xmax_sm is not None else float(np.nanmax(np.concatenate(x_parts)))

    # Taller figure: title (top) → croquis → gap → main panels
    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_axes([0.07, 0.08, 0.38, 0.58])
    ax2 = fig.add_axes([0.55, 0.08, 0.38, 0.58])

    # Croquis centered above panels (below title)
    if lat_center is not None and lon_center is not None:
        _plot_location_inset(fig, lat_center, lon_center, rect=(0.40, 0.74, 0.20, 0.16))

    # ===== Left: SM vs LE =====
    y_parts = [LE_s] if LE_s.size else []
    if traj_le:
        y_parts.append(np.array([p[2] for p in traj_le]))
    ymin = float(np.nanmin(np.concatenate(y_parts))) if y_parts else 0.0
    ymax_le = 150.0

    if sm_s.size > 0:
        ax.scatter(
            sm_s, LE_s, s=18, alpha=0.25, color='cornflowerblue',
            zorder=2, label='Summer (excl. HW)',
        )

    _scatter_lag_groups(ax, traj_le, legend=True)
    if show_critical_vline:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7, label='Critical SM')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax_le)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r'LE (W m$^{-2}$)', fontsize=11)
    ax.set_title('SM vs LE', fontsize=11)
    ax.tick_params(labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc='best')

    # ===== Right: SM vs absolute T =====
    if sm_st.size or traj_T:
        yT_parts = []
        if T_st.size:
            yT_parts.append(T_st)
        if traj_T:
            yT_parts.append(np.array([p[2] for p in traj_T]))
        ymin_T = float(np.nanmin(np.concatenate(yT_parts)))
        ymax_T = float(np.nanmax(np.concatenate(yT_parts)))
        pad = 0.05 * max(ymax_T - ymin_T, 1.0)
        ymin_T, ymax_T = ymin_T - pad, ymax_T + pad

        if sm_st.size > 0:
            ax2.scatter(
                sm_st, T_st, s=18, alpha=0.25, color='0.55',
                zorder=2, label='Summer (excl. HW)',
            )
        _scatter_lag_groups(ax2, traj_T, legend=True)
        if show_critical_vline:
            ax2.axvline(0, color='k', lw=1, ls='--', alpha=0.7)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin_T, ymax_T)
        ax2.margins(0)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(r'T (K)', fontsize=11)
        ax2.set_title('SM vs T (absolute)', fontsize=11)
        ax2.tick_params(labelsize=10)
        handles2, _ = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(fontsize=8, loc='best')
    else:
        ax2.set_title('SM vs T (no data)', fontsize=11)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(r'T (K)', fontsize=11)

    fig.suptitle(title, fontsize=11, y=0.97)
    fig.savefig(outfile, dpi=400)
    plt.close(fig)
    print(f'Saved {outfile}')


print(f'Per-event SM–LE / SM–T figures → {path_event_figs}')
try:
    ds_sm.close()
except Exception:
    pass

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine='netcdf4')
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')
# Absolute temperature (not anomaly)
ds_T = xr.open_dataset(path_file_T_abs, decode_times=False, engine='netcdf4')

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_LE.time.values])
lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_sm.time.values])
# Absolute T uses the same calendar constructed from path_file_t
lats_T = ds_T.lat.values if 'lat' in ds_T.coords or 'lat' in ds_T.dims else lats_t
lons_T = ds_T.lon.values if 'lon' in ds_T.coords or 'lon' in ds_T.dims else lons_t
dates_T = np.array([x.date() for x in dates_d])

n_saved = 0
n_skipped = 0

for event_id, lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
    event_ids, lats_events, lons_events, dates_onset, duration_events
):
    if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_T[0]:
        print(f'Skipping event {event_id} (onset {onset_day}): before SM/LE/T start')
        n_skipped += 1
        continue

    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)
    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)
    ilat_T = np.where((lats_T >= lat_min) & (lats_T <= lat_max))[0]
    ilon_T = get_lon_idx(lons_T, lon_center_hw, half_width=0.5)

    if (
        len(ilat_LE) == 0 or len(ilon_LE) == 0
        or len(ilat_sm) == 0 or len(ilon_sm) == 0
        or len(ilat_T) == 0 or len(ilon_T) == 0
    ):
        n_skipped += 1
        continue

    regime_box = regime_field[np.ix_(ilat_sm, ilon_sm)]
    dom_regime = dominant_regime_in_box(regime_box)
    if dom_regime < 0:
        n_skipped += 1
        continue

    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
    T_region = ds_T[variable_T_abs].isel(lat=ilat_T, lon=ilon_T).values

    if LE_region.ndim != 3 or sm_region.ndim != 3 or T_region.ndim != 3:
        n_skipped += 1
        continue
    if (
        not np.isfinite(LE_region).any()
        or not np.isfinite(sm_region).any()
        or not np.isfinite(T_region).any()
    ):
        n_skipped += 1
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    T_point = np.nanmean(T_region, axis=(1, 2))
    LE_point = np.where((LE_point >= 0) & np.isfinite(LE_point), LE_point, np.nan)
    # Absolute T: convert °C → K if needed
    if np.nanmedian(T_point) < 100:
        T_point = T_point + 273.15
    sm_abs = np.nanmean(sm_region, axis=(1, 2))

    use_rel_bp = False
    show_vline = False
    xmax_sm = None
    density_vmax = 0.35
    if dom_regime == 0:
        sm_point = sm_abs
        regime_label = 'weak (regime 0)'
        xlabel = r'SMs (regional mean, ±0.5°)'
        density_vmax = 0.15
    elif dom_regime in (1, 2):
        sm_point = sm_abs
        regime_label = f'transitional (regime {dom_regime})'
        xlabel = r'SMs (regional mean, ±0.5°)'
        density_vmax = 0.6
    elif dom_regime in (3, 4):
        bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
        if not np.isfinite(bp_box).any():
            n_skipped += 1
            continue
        sm_point = np.nanmean(sm_region - bp_box[np.newaxis, :, :], axis=(1, 2))
        use_rel_bp = True
        show_vline = True
        xmax_sm = 0.15
        regime_label = f'moisture-limited (regime {dom_regime})'
        xlabel = r'SMs $-$ critical SM (regional mean, ±0.5°)'
        density_vmax = 0.35
    else:
        n_skipped += 1
        continue

    common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_T)
    date_to_i = {d: i for i, d in enumerate(common_dates)}
    LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
    sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
    T_aligned = T_point[np.isin(dates_T, common_dates)]

    heatwave_days = [
        onset_day + dt.timedelta(days=i) for i in range(int(duration_event))
    ]
    hw_mask = np.array([d in set(heatwave_days) for d in common_dates])
    summer_mask = np.array([d.month in (6, 7, 8) for d in common_dates]) & (~hw_mask)
    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]
    T_summer = T_aligned[summer_mask]
    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    LE_summer_v = LE_summer[valid]
    sm_summer_v = sm_summer[valid]
    valid_T = np.isfinite(T_summer) & np.isfinite(sm_summer)
    sm_summer_T = sm_summer[valid_T]
    T_summer_v = T_summer[valid_T]

    # Trajectory: relative days around onset → (sm, LE, T)
    traj = {}
    for lag in list(PRE_LAGS) + [CENTER_LAG] + list(POST_LAGS):
        day = onset_day + dt.timedelta(days=int(lag))
        i = date_to_i.get(day)
        if i is None:
            traj[lag] = (np.nan, np.nan, np.nan)
        else:
            traj[lag] = (
                float(sm_aligned[i]), float(LE_aligned[i]), float(T_aligned[i])
            )

    onset_str = onset_day.strftime('%Y%m%d')
    sm_tag = 'relBP' if use_rel_bp else 'absSM'
    outfile = (
        f'{path_event_figs}event_{event_id}_onset{onset_str}_'
        f'{variable_LE}_{variable_sm}_T_{sm_tag}_regime{dom_regime}_'
        f'{name_land}_{region}.png'
    )
    title = (
        f'Event {event_id} | onset {onset_day.isoformat()} | {regime_label}\n'
        f'Max-anomaly location; −3/−2/−1 (yellow), onset (red), +1/+2/+3 (brown)'
    )
    _plot_event_trajectory(
        sm_summer_v, LE_summer_v, sm_summer_T, T_summer_v, traj,
        xlabel=xlabel, title=title, outfile=outfile,
        lat_center=lat_center_hw, lon_center=lon_center_hw,
        show_critical_vline=show_vline,
        density_vmax=density_vmax,
        xmax_sm=xmax_sm,
    )
    n_saved += 1

ds_LE.close()
ds_sm.close()
ds_T.close()
print(f'Done. Saved {n_saved} event figures; skipped {n_skipped}.')
