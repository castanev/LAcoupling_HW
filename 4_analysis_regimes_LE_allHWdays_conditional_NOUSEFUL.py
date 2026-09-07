# This code is for analyzing 
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
lats_events_all, lons_events_all = extract_latlons_from_coord_events(coord_events)
dates_onset_all = np.asarray(dates_d_HW_day0)
duration_events_all = np.asarray(duration_events)

variable_LE = "LE"
variable_sm = "SMs"
dates_t = np.array([x.date() for x in dates_d])

os.makedirs(path_outputs_case, exist_ok=True)

# =======================================================
# EOF3 negative-phase years (same recipe as 9_analysis_EOF_AprMay.py)
# Restrict HW events to onset years in each phase window.
# =======================================================
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


pc_thresh = 0.75
EOF_i_neg = 2  # EOF3 (0-based)
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
ds_eof.close()

# Tag each HW event: True if onset year is late-April EOF3 negative phase
hw_csv_path = f'{path_case}{name_file_posHW}'
year_neg_lateApr = set(int(y) for y in year_neg_by_period['lateApr'])
onset_idx = df_heatwaves.index.values.astype(int)
lateApr_eof3_neg = np.zeros(len(onset_idx), dtype=bool)
valid_onset = onset_idx < len(dates_d)
for j, day0 in enumerate(onset_idx):
    if not valid_onset[j]:
        continue
    lateApr_eof3_neg[j] = int(dates_d[int(day0)].year) in year_neg_lateApr
df_heatwaves['LateApri_EOF3'] = lateApr_eof3_neg
df_heatwaves.to_csv(hw_csv_path)
n_true = int(lateApr_eof3_neg.sum())
print(
    f"Updated {hw_csv_path} with LateApri_EOF3 "
    f"({n_true}/{len(lateApr_eof3_neg)} events True; "
    f"lateApr EOF3-neg years={sorted(year_neg_lateApr)})"
)


def _as_1d_points(arr_or_list):
    """Concatenate list-of-arrays or pass through a flat array."""
    if isinstance(arr_or_list, list):
        return np.concatenate(arr_or_list) if len(arr_or_list) > 0 else np.array([])
    arr = np.asarray(arr_or_list)
    return arr.ravel() if arr.size else np.array([])


def _plot_density_panel(
    all_sm_summer,
    all_LE_summer,
    all_sm_hw,
    all_LE_hw,
    xlabel,
    title,
    outfile,
    show_critical_vline=False,
    density_vmax=0.2,
    min_hw_events=5,
    n_hw_events=None,
    xmax_sm=None,
    show_hw_contours=True,
    spearman_corr=False,
):
    if n_hw_events is None:
        n_hw_events = len(all_sm_hw) if isinstance(all_sm_hw, list) else 0
    if show_hw_contours and n_hw_events < min_hw_events:
        print(
            f"Skipping plot (<{min_hw_events} HW events, n={n_hw_events}): {outfile}"
        )
        return

    all_sm_summer = _as_1d_points(all_sm_summer)
    all_LE_summer = _as_1d_points(all_LE_summer)
    all_sm_hw = _as_1d_points(all_sm_hw)
    all_LE_hw = _as_1d_points(all_LE_hw)

    mask_s = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
    sm_summer_valid = all_sm_summer[mask_s]
    LE_summer_valid = all_LE_summer[mask_s]
    mask_h = np.isfinite(all_sm_hw) & np.isfinite(all_LE_hw)
    sm_hw_valid = all_sm_hw[mask_h]
    LE_hw_valid = all_LE_hw[mask_h]

    has_summer = sm_summer_valid.size > 0
    has_hw = sm_hw_valid.size > 0
    if not has_summer and not (show_hw_contours and has_hw):
        print(f"Skipping plot (no events): {outfile}")
        return

    # Axis limits: use summer (+ HW if drawn) so companion plots share the same frame
    x_parts = [a for a in (sm_summer_valid, sm_hw_valid) if a.size > 0]
    y_parts = [a for a in (LE_summer_valid, LE_hw_valid) if a.size > 0]
    if not show_hw_contours:
        x_parts = [a for a in (sm_summer_valid,) if a.size > 0] or x_parts
        y_parts = [a for a in (LE_summer_valid,) if a.size > 0] or y_parts
    xmin = float(np.nanmin(np.concatenate(x_parts)))

    if xmax_sm is not None:
        xmax = xmax_sm
    else:
        xmax = float(np.nanmax(np.concatenate(x_parts)))
    ymin = float(np.nanmin(np.concatenate(y_parts)))
    ymax_le = 150.0
    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax_le, 45),
    )
    levels = np.linspace(0, density_vmax, 21)
    hw_lo = 0.5 * density_vmax
    # Upper-half contours only; ~2/3 as many lines as before (~11 → ~7)
    n_hw_contours = 7
    hw_levels = np.linspace(hw_lo, density_vmax, n_hw_contours)

    fig = plt.figure(figsize=(7.0, 5))
    # Slightly wider axes when only the blue colorbar is needed
    ax_w = 0.70 if not show_hw_contours else 0.63
    ax = fig.add_axes([0.11, 0.12, ax_w, 0.78])
    cbar_fmt = FormatStrFormatter("%.2f")
    ax_pos = ax.get_position()
    pad_blue = 0.006
    cbar_w = 0.022
    # Gap after blue (ticks/label), then red a bit farther right
    gap_blue_to_red = 0.10
    cax_s_box = [
        ax_pos.x1 + pad_blue,
        ax_pos.y0,
        cbar_w,
        ax_pos.height,
    ]
    # Red bar: thinner and longer (~68% of plot height)
    red_h_frac = 0.58
    red_cbar_w = 0.023
    cax_h_box = [
        ax_pos.x1 + pad_blue + cbar_w + gap_blue_to_red,
        ax_pos.y0 + ax_pos.height * (1.0 - red_h_frac),
        red_cbar_w,
        ax_pos.height * red_h_frac,
    ]

    # Summer climatology: filled blue density (full 0 → density_vmax)
    if has_summer and sm_summer_valid.size > 5:
        kde_s = gaussian_kde(np.vstack([sm_summer_valid, LE_summer_valid]))
        z_s = kde_s(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cf = ax.contourf(
            xgrid, ygrid, z_s,
            levels=levels, cmap="Blues", vmin=0, vmax=density_vmax, alpha=0.8, extend="max",
        )
        # Blue outline at lowest red HW level (only when HW contours are shown)
        if show_hw_contours:
            ax.contour(
                xgrid, ygrid, z_s,
                levels=[hw_lo], colors="navy", linewidths=1.6, linestyles="-",
            )
        cax_s = fig.add_axes(cax_s_box)
        cbar_s = fig.colorbar(cf, cax=cax_s, extend="max")
        cbar_s.set_label(f"Summer density (0–{density_vmax:.2f})", fontsize=10)
        cbar_s.ax.yaxis.set_major_formatter(cbar_fmt)
        cbar_s.ax.tick_params(labelsize=9)
    elif has_summer:
        ax.scatter(
            sm_summer_valid, LE_summer_valid,
            s=10, alpha=0.3, color="cornflowerblue", label="Summer (excl. HW)",
        )

    # Heatwave days: red contours for upper half only (colormap 0→vmax so lines start red)
    if show_hw_contours and has_hw and sm_hw_valid.size > 5:
        kde_h = gaussian_kde(np.vstack([sm_hw_valid, LE_hw_valid]))
        z_h = kde_h(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cs = ax.contour(
            xgrid, ygrid, z_h,
            levels=hw_levels, cmap="Reds", vmin=0, vmax=density_vmax,
            linewidths=1.4,
        )
        # Discrete red colorbar: one block + tick per contour level
        cax_h = fig.add_axes(cax_h_box)
        colors_hw = [mpl.cm.Reds(lvl / density_vmax) for lvl in hw_levels]
        if len(hw_levels) > 1:
            half = 0.5 * (hw_levels[1] - hw_levels[0])
            boundaries = np.concatenate([
                [hw_levels[0] - half],
                0.5 * (hw_levels[:-1] + hw_levels[1:]),
                [hw_levels[-1] + half],
            ])
        else:
            boundaries = np.array([hw_levels[0] - 0.01, hw_levels[0] + 0.01])
        cmap_hw = ListedColormap(colors_hw)
        norm_hw = BoundaryNorm(boundaries, ncolors=len(colors_hw))
        sm_hw = mpl.cm.ScalarMappable(cmap=cmap_hw, norm=norm_hw)
        sm_hw.set_array([])
        cbar_hw = fig.colorbar(sm_hw, cax=cax_h)
        cbar_hw.set_ticks(hw_levels)
        cbar_hw.ax.set_yticklabels([f"{v:.2f}" for v in hw_levels])
        cbar_hw.set_label("HW-day density", fontsize=10)
        cbar_hw.ax.tick_params(labelsize=9)
    elif show_hw_contours and has_hw:
        ax.scatter(
            sm_hw_valid, LE_hw_valid,
            s=12, alpha=0.5, color="salmon", label="Heatwave days",
        )

    if show_critical_vline:
        ax.axvline(0, color="k", lw=1, ls="--", alpha=0.7, label="Critical SM")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax_le)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"LE (W m$^{-2}$)", fontsize=12)
    if spearman_corr == True:
        spearman_corr_summer = spearmanr(sm_summer_valid, LE_summer_valid)[0]
        spearman_corr_hw = spearmanr(sm_hw_valid, LE_hw_valid)[0]
        ax.text(0.05, 0.95, f"Spearman correlation (non-HW): {spearman_corr_summer:.2f}", fontsize=11, transform=ax.transAxes, ha="left", va="top")
        ax.text(0.05, 0.90, f"Spearman correlation (HW): {spearman_corr_hw:.2f}", fontsize=11, transform=ax.transAxes, ha="left", va="top")
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11, loc="best")
    fig.savefig(outfile, dpi=500)
    plt.close(fig)


# Run density analysis separately for each EOF3-negative phase window
for eof_period_label, year_neg_eof3 in year_neg_by_period.items():
    year_neg_set = set(int(y) for y in np.asarray(year_neg_eof3))
    year_mask = np.array([int(d.year) in year_neg_set for d in dates_onset_all], dtype=bool)
    lats_events = np.asarray(lats_events_all)[year_mask]
    lons_events = np.asarray(lons_events_all)[year_mask]
    dates_d_HW_day0 = dates_onset_all[year_mask]
    duration_events = duration_events_all[year_mask]
    cond_tag = f'EOF{EOF_i_neg + 1}neg_{eof_period_label}_thresh{pc_thresh:g}'
    print(
        f'\n=== Regime density | {cond_tag}: '
        f'{year_mask.sum()} / {year_mask.size} HW events ==='
    )

    density_cache_path = (
        f'{path_outputs_case}density_data_{variable_LE}_{variable_sm}_'
        f'summer_HWdays_{cond_tag}_05deg_{name_land}_{region}.npz'
    )


    for lag_days in lags_days:
        load_cache = False
        if os.path.exists(density_cache_path):
            cache = np.load(density_cache_path)
            if 'r3_sm_summer' in cache.files and 'r4_sm_summer' in cache.files:
                load_cache = True
            else:
                print(
                    f'Cache missing regime-3/4 fields; recomputing '
                    f'({density_cache_path})'
                )
                cache.close()

        if load_cache:
            print(f'Loading cached density data from {density_cache_path}')
            weak_sm_summer = cache['weak_sm_summer']
            weak_LE_summer = cache['weak_LE_summer']
            weak_sm_hw = cache['weak_sm_hw']
            weak_LE_hw = cache['weak_LE_hw']
            n_hw_weak = int(cache['n_hw_weak'])
            trans_sm_summer = cache['trans_sm_summer']
            trans_LE_summer = cache['trans_LE_summer']
            trans_sm_hw = cache['trans_sm_hw']
            trans_LE_hw = cache['trans_LE_hw']
            n_hw_trans = int(cache['n_hw_trans'])
            ml_sm_summer = cache['ml_sm_summer']
            ml_LE_summer = cache['ml_LE_summer']
            ml_sm_hw = cache['ml_sm_hw']
            ml_LE_hw = cache['ml_LE_hw']
            n_hw_ml = int(cache['n_hw_ml'])
            r3_sm_summer = cache['r3_sm_summer']
            r3_LE_summer = cache['r3_LE_summer']
            r3_sm_hw = cache['r3_sm_hw']
            r3_LE_hw = cache['r3_LE_hw']
            n_hw_r3 = int(cache['n_hw_r3'])
            r4_sm_summer = cache['r4_sm_summer']
            r4_LE_summer = cache['r4_LE_summer']
            r4_sm_hw = cache['r4_sm_hw']
            r4_LE_hw = cache['r4_LE_hw']
            n_hw_r4 = int(cache['n_hw_r4'])
        else:
            print(f'Computing density data (will cache to {density_cache_path})')
            ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
            ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
            ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine="netcdf4")

            lats_LE = ds_LE.lat.values
            lons_LE = ds_LE.lon.values
            dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])
            lats_sm = ds_sm.lat.values
            lons_sm = ds_sm.lon.values
            dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

            # Figure 1: weak coupling (regime 0)
            weak_sm_summer, weak_LE_summer = [], []
            weak_sm_hw, weak_LE_hw = [], []

            # Figure 2: transitional (regime 1 strong opt1 + regime 2)
            trans_sm_summer, trans_LE_summer = [], []
            trans_sm_hw, trans_LE_hw = [], []

            # Moisture-limited split: regime 3 and regime 4 (also kept combined as ml_*)
            r3_sm_summer, r3_LE_summer = [], []
            r3_sm_hw, r3_LE_hw = [], []
            r4_sm_summer, r4_LE_summer = [], []
            r4_sm_hw, r4_LE_hw = [], []

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
                    sm_summer_list, sm_hw_list = weak_sm_summer, weak_sm_hw
                    le_summer_list, le_hw_list = weak_LE_summer, weak_LE_hw
                elif dom_regime in (1, 2):
                    sm_point = np.nanmean(sm_region, axis=(1, 2))
                    sm_summer_list, sm_hw_list = trans_sm_summer, trans_sm_hw
                    le_summer_list, le_hw_list = trans_LE_summer, trans_LE_hw
                elif dom_regime == 3:
                    bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
                    if not np.isfinite(bp_box).any():
                        continue
                    sm_anom_region = sm_region - bp_box[np.newaxis, :, :]
                    sm_point = np.nanmean(sm_anom_region, axis=(1, 2))
                    sm_summer_list, sm_hw_list = r3_sm_summer, r3_sm_hw
                    le_summer_list, le_hw_list = r3_LE_summer, r3_LE_hw
                elif dom_regime == 4:
                    bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
                    if not np.isfinite(bp_box).any():
                        continue
                    sm_anom_region = sm_region - bp_box[np.newaxis, :, :]
                    sm_point = np.nanmean(sm_anom_region, axis=(1, 2))
                    sm_summer_list, sm_hw_list = r4_sm_summer, r4_sm_hw
                    le_summer_list, le_hw_list = r4_LE_summer, r4_LE_hw
                else:
                    continue

                common_dates = np.intersect1d(np.intersect1d(dates_LE, dates_sm), dates_t)

                LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
                sm_aligned = sm_point[np.isin(dates_sm, common_dates)]
                t_aligned = t_point[np.isin(dates_t, common_dates)]

                heatwave_days = [onset_day + dt.timedelta(days=i) for i in range(int(duration_event))]
                heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
                hw_mask = np.zeros(len(common_dates), dtype=bool)
                hw_mask[heatwave_idx] = True

                # Summer climatology at this location, excluding this event's heatwave days
                summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates]) & (~hw_mask)
                LE_summer = LE_aligned[summer_mask]
                sm_summer = sm_aligned[summer_mask]

                valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
                if np.any(valid):
                    le_summer_list.append(LE_summer[valid])
                    sm_summer_list.append(sm_summer[valid])

                if len(heatwave_idx) > 0:
                    LE_hw = LE_aligned[heatwave_idx]
                    sm_hw = sm_aligned[heatwave_idx]
                    valid_hw = np.isfinite(LE_hw) & np.isfinite(sm_hw)
                    if np.any(valid_hw):
                        le_hw_list.append(LE_hw[valid_hw])
                        sm_hw_list.append(sm_hw[valid_hw])

            n_hw_weak = len(weak_sm_hw)
            n_hw_trans = len(trans_sm_hw)
            n_hw_r3 = len(r3_sm_hw)
            n_hw_r4 = len(r4_sm_hw)
            n_hw_ml = n_hw_r3 + n_hw_r4

            weak_sm_summer = _as_1d_points(weak_sm_summer)
            weak_LE_summer = _as_1d_points(weak_LE_summer)
            weak_sm_hw = _as_1d_points(weak_sm_hw)
            weak_LE_hw = _as_1d_points(weak_LE_hw)
            trans_sm_summer = _as_1d_points(trans_sm_summer)
            trans_LE_summer = _as_1d_points(trans_LE_summer)
            trans_sm_hw = _as_1d_points(trans_sm_hw)
            trans_LE_hw = _as_1d_points(trans_LE_hw)
            r3_sm_summer = _as_1d_points(r3_sm_summer)
            r3_LE_summer = _as_1d_points(r3_LE_summer)
            r3_sm_hw = _as_1d_points(r3_sm_hw)
            r3_LE_hw = _as_1d_points(r3_LE_hw)
            r4_sm_summer = _as_1d_points(r4_sm_summer)
            r4_LE_summer = _as_1d_points(r4_LE_summer)
            r4_sm_hw = _as_1d_points(r4_sm_hw)
            r4_LE_hw = _as_1d_points(r4_LE_hw)

            # Combined moisture-limited (regime 3 + 4) for the existing moistlim figure
            ml_sm_summer = _as_1d_points([r3_sm_summer, r4_sm_summer])
            ml_LE_summer = _as_1d_points([r3_LE_summer, r4_LE_summer])
            ml_sm_hw = _as_1d_points([r3_sm_hw, r4_sm_hw])
            ml_LE_hw = _as_1d_points([r3_LE_hw, r4_LE_hw])

            np.savez_compressed(
                density_cache_path,
                weak_sm_summer=weak_sm_summer,
                weak_LE_summer=weak_LE_summer,
                weak_sm_hw=weak_sm_hw,
                weak_LE_hw=weak_LE_hw,
                n_hw_weak=n_hw_weak,
                trans_sm_summer=trans_sm_summer,
                trans_LE_summer=trans_LE_summer,
                trans_sm_hw=trans_sm_hw,
                trans_LE_hw=trans_LE_hw,
                n_hw_trans=n_hw_trans,
                ml_sm_summer=ml_sm_summer,
                ml_LE_summer=ml_LE_summer,
                ml_sm_hw=ml_sm_hw,
                ml_LE_hw=ml_LE_hw,
                n_hw_ml=n_hw_ml,
                r3_sm_summer=r3_sm_summer,
                r3_LE_summer=r3_LE_summer,
                r3_sm_hw=r3_sm_hw,
                r3_LE_hw=r3_LE_hw,
                n_hw_r3=n_hw_r3,
                r4_sm_summer=r4_sm_summer,
                r4_LE_summer=r4_LE_summer,
                r4_sm_hw=r4_sm_hw,
                r4_LE_hw=r4_LE_hw,
                n_hw_r4=n_hw_r4,
            )
            print(f'Saved density data cache: {density_cache_path}')

            ds_LE.close()
            ds_sm.close()
            ds_t.close()

        _plot_density_panel(
            weak_sm_summer,
            weak_LE_summer,
            weak_sm_hw,
            weak_LE_hw,
            xlabel=r"SMs (regional mean, ±0.5°)",
            title=(
                f"SM vs LE — weak coupling (dominant regime 0 in ±0.5° box)\n"
                f"Models 0 + weak option 1 | HW events in {cond_tag}"
            ),
            outfile=(
                f"{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_HWdays_"
                f"05deg_weak_{cond_tag}_{name_land}_{region}.png"
            ),
            show_critical_vline=False,
            density_vmax=0.15,
            n_hw_events=n_hw_weak,
            spearman_corr=True,
        )


        _plot_density_panel(
            trans_sm_summer,
            trans_LE_summer,
            trans_sm_hw,
            trans_LE_hw,
            xlabel=r"SMs (regional mean, ±0.5°)",
            title=(
                f"SM vs LE — transitional (dominant regime 1 or 2 in ±0.5° box)\n"
                f"Strong option 1 + LHS-flat | HW events in {cond_tag}"
            ),
            outfile=(
                f"{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_HWdays_"
                f"05deg_trans_{cond_tag}_{name_land}_{region}.png"
            ),
            show_critical_vline=False,
            density_vmax=0.6,
            n_hw_events=n_hw_trans,
        )


        _plot_density_panel(
            ml_sm_summer,
            ml_LE_summer,
            ml_sm_hw,
            ml_LE_hw,
            xlabel=r"SMs $-$ critical SM (regime 3/4 BP, regional mean, ±0.5°)",
            title=(
                f"SM vs LE — moisture-limited (dominant regime 3 or 4 in ±0.5° box)\n"
                f"RHS-flat / 3-segment breakpoints | HW events in {cond_tag}"
            ),
            outfile=(
                f"{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_HWdays_"
                f"05deg_moistlim_{cond_tag}_{name_land}_{region}.png"
            ),
            show_critical_vline=True,
            density_vmax=0.35,
            n_hw_events=n_hw_ml,
            xmax_sm=0.15
        )


        _plot_density_panel(
            r3_sm_summer,
            r3_LE_summer,
            r3_sm_hw,
            r3_LE_hw,
            xlabel=r"SMs $-$ critical SM (regime 3 BP, regional mean, ±0.5°)",
            title=(
                f"SM vs LE — moisture-limited regime 3 (dominant in ±0.5° box)\n"
                f"RHS-flat (2-segment) breakpoint | HW events in {cond_tag}"
            ),
            outfile=(
                f"{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_HWdays_"
                f"05deg_regime3_{cond_tag}_{name_land}_{region}.png"
            ),
            show_critical_vline=True,
            density_vmax=0.35,
            n_hw_events=n_hw_r3,
            xmax_sm=0.15,
        )


        _plot_density_panel(
            r4_sm_summer,
            r4_LE_summer,
            r4_sm_hw,
            r4_LE_hw,
            xlabel=r"SMs $-$ critical SM (regime 4 BP, regional mean, ±0.5°)",
            title=(
                f"SM vs LE — moisture-limited regime 4 (dominant in ±0.5° box)\n"
                f"3-segment breakpoint | HW events in {cond_tag}"
            ),
            outfile=(
                f"{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_HWdays_"
                f"05deg_regime4_{cond_tag}_{name_land}_{region}.png"
            ),
            show_critical_vline=True,
            density_vmax=0.35,
            n_hw_events=n_hw_r4,
            xmax_sm=0.15,
        )

