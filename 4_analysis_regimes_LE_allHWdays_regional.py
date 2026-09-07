# Regional SM–LE density: west / central / east US by event longitude.
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
import yaml
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
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

path_figures = f'{path_case_land}/Figures/regional_analysis/'
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




# ======================================================= DENSITY PLOT (by lon region) ========================================================
# West: lon < 255; Central: 255 ≤ lon ≤ 270; East: lon > 270 (0–360°E).
# Regimes: weak (0), transitional (1–2), moisture-limited (3+4 combined). No min-N skip.
lags_days = [5]
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'


def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)
    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360
    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width
    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


def _lon360(lon):
    lon = float(lon)
    return lon + 360.0 if lon < 0 else lon


# Event centers = Max Anomaly Position
coord_events = df_events_valid.iloc[:, 4]
lats_events_all, lons_events_all = extract_latlons_from_coord_events(coord_events)
dates_onset_all = np.asarray(dates_d_HW_day0)
duration_events_all = np.asarray(duration_events)

variable_LE = 'LE'
variable_sm = 'SMs'
dates_t = np.array([x.date() for x in dates_d])

os.makedirs(path_outputs_case, exist_ok=True)

# Longitude bands (0–360°E)
US_LON_REGIONS = [
    ('westUS', None, 255.0),          # lon ≤ 255
    ('centralUS', 255.0, 270.0),      # 255 < lon ≤ 270
    ('eastUS', 270.0, None),          # lon > 270
]


def _as_1d_points(arr_or_list):
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
    n_hw_events=None,
    xmax_sm=None,
    show_hw_contours=True,
    spearman_corr=False,
):
    if n_hw_events is None:
        n_hw_events = len(all_sm_hw) if isinstance(all_sm_hw, list) else 0

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
        print(f'Skipping plot (no data): {outfile}')
        return

    x_parts = [a for a in (sm_summer_valid, sm_hw_valid) if a.size > 0]
    y_parts = [a for a in (LE_summer_valid, LE_hw_valid) if a.size > 0]
    if not show_hw_contours:
        x_parts = [a for a in (sm_summer_valid,) if a.size > 0] or x_parts
        y_parts = [a for a in (LE_summer_valid,) if a.size > 0] or y_parts
    xmin = float(np.nanmin(np.concatenate(x_parts)))
    xmax = float(xmax_sm) if xmax_sm is not None else float(np.nanmax(np.concatenate(x_parts)))
    ymin = float(np.nanmin(np.concatenate(y_parts)))
    ymax_le = 150.0
    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax_le, 45),
    )
    levels = np.linspace(0, density_vmax, 21)
    hw_lo = 0.5 * density_vmax
    n_hw_contours = 7
    hw_levels = np.linspace(hw_lo, density_vmax, n_hw_contours)

    fig = plt.figure(figsize=(7.0, 5))
    ax_w = 0.70 if not show_hw_contours else 0.63
    ax = fig.add_axes([0.11, 0.12, ax_w, 0.78])
    cbar_fmt = FormatStrFormatter('%.2f')
    ax_pos = ax.get_position()
    pad_blue = 0.006
    cbar_w = 0.022
    gap_blue_to_red = 0.10
    cax_s_box = [ax_pos.x1 + pad_blue, ax_pos.y0, cbar_w, ax_pos.height]
    red_h_frac = 0.58
    red_cbar_w = 0.023
    cax_h_box = [
        ax_pos.x1 + pad_blue + cbar_w + gap_blue_to_red,
        ax_pos.y0 + ax_pos.height * (1.0 - red_h_frac),
        red_cbar_w,
        ax_pos.height * red_h_frac,
    ]

    if has_summer and sm_summer_valid.size > 5:
        kde_s = gaussian_kde(np.vstack([sm_summer_valid, LE_summer_valid]))
        z_s = kde_s(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cf = ax.contourf(
            xgrid, ygrid, z_s,
            levels=levels, cmap='Blues', vmin=0, vmax=density_vmax,
            alpha=0.8, extend='max',
        )
        if show_hw_contours:
            ax.contour(
                xgrid, ygrid, z_s,
                levels=[hw_lo], colors='navy', linewidths=1.6, linestyles='-',
            )
        cax_s = fig.add_axes(cax_s_box)
        cbar_s = fig.colorbar(cf, cax=cax_s, extend='max')
        cbar_s.set_label(f'Summer density (0–{density_vmax:.2f})', fontsize=10)
        cbar_s.ax.yaxis.set_major_formatter(cbar_fmt)
        cbar_s.ax.tick_params(labelsize=9)
    elif has_summer:
        ax.scatter(
            sm_summer_valid, LE_summer_valid,
            s=10, alpha=0.3, color='cornflowerblue', label='Summer (excl. HW)',
        )

    if show_hw_contours and has_hw and sm_hw_valid.size > 5:
        kde_h = gaussian_kde(np.vstack([sm_hw_valid, LE_hw_valid]))
        z_h = kde_h(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        ax.contour(
            xgrid, ygrid, z_h,
            levels=hw_levels, cmap='Reds', vmin=0, vmax=density_vmax,
            linewidths=1.4,
        )
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
        cbar_hw.ax.set_yticklabels([f'{v:.2f}' for v in hw_levels])
        cbar_hw.set_label('HW-day density', fontsize=10)
        cbar_hw.ax.tick_params(labelsize=9)
    elif show_hw_contours and has_hw:
        ax.scatter(
            sm_hw_valid, LE_hw_valid,
            s=12, alpha=0.5, color='salmon', label='Heatwave days',
        )

    if show_critical_vline:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7, label='Critical SM')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax_le)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'LE (W m$^{-2}$)', fontsize=12)
    if spearman_corr and has_summer and has_hw:
        spearman_corr_summer = spearmanr(sm_summer_valid, LE_summer_valid)[0]
        spearman_corr_hw = spearmanr(sm_hw_valid, LE_hw_valid)[0]
        ax.text(
            0.05, 0.95,
            f'Spearman (non-HW): {spearman_corr_summer:.2f}',
            fontsize=11, transform=ax.transAxes, ha='left', va='top',
        )
        ax.text(
            0.05, 0.90,
            f'Spearman (HW): {spearman_corr_hw:.2f}',
            fontsize=11, transform=ax.transAxes, ha='left', va='top',
        )
    # Always annotate number of HW events included
    ax.text(
        0.98, 0.02,
        f'n events = {int(n_hw_events)}',
        fontsize=11, transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.9),
    )
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11, loc='best')
    fig.savefig(outfile, dpi=500)
    plt.close(fig)
    print(f'Saved {outfile} (n events = {n_hw_events})')


for region_tag, lon_lo, lon_hi in US_LON_REGIONS:
    lons360 = np.array([_lon360(lon) for lon in lons_events_all])
    if lon_lo is None and lon_hi is not None:
        # west: lon ≤ lon_hi
        region_mask = lons360 <= float(lon_hi)
        lon_desc = f'lon ≤ {lon_hi:g}'
    elif lon_lo is not None and lon_hi is None:
        # east: lon > lon_lo
        region_mask = lons360 > float(lon_lo)
        lon_desc = f'lon > {lon_lo:g}'
    else:
        # central: lon_lo < lon ≤ lon_hi
        region_mask = (lons360 > float(lon_lo)) & (lons360 <= float(lon_hi))
        lon_desc = f'{lon_lo:g} < lon ≤ {lon_hi:g}'

    lats_events = np.asarray(lats_events_all)[region_mask]
    lons_events = np.asarray(lons_events_all)[region_mask]
    dates_d_HW_day0 = dates_onset_all[region_mask]
    duration_events = duration_events_all[region_mask]
    n_region_events = int(region_mask.sum())

    density_cache_path = (
        f'{path_outputs_case}density_data_{variable_LE}_{variable_sm}_'
        f'aroundOnsetloc_summer_allHWdays_05deg_{name_land}_{region_tag}.npz'
    )
    print(
        f'\n=== Lon region {region_tag} ({lon_desc}): '
        f'{n_region_events} / {region_mask.size} HW events ==='
    )

    for lag_days in lags_days:
        load_cache = False
        if os.path.exists(density_cache_path):
            cache = np.load(density_cache_path)
            needed = (
                'weak_sm_summer', 'trans_sm_summer', 'ml_sm_summer',
                'n_hw_weak', 'n_hw_trans', 'n_hw_ml',
            )
            if all(k in cache.files for k in needed):
                load_cache = True
            else:
                print(f'Cache incomplete; recomputing ({density_cache_path})')
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
        else:
            print(f'Computing density data (will cache to {density_cache_path})')
            ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine='netcdf4')
            ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')
            ds_t = xr.open_dataset(path_file_t_anom, decode_times=False, engine='netcdf4')

            lats_LE = ds_LE.lat.values
            lons_LE = ds_LE.lon.values
            dates_LE = np.array([
                dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_LE.time.values
            ])
            lats_sm = ds_sm.lat.values
            lons_sm = ds_sm.lon.values
            dates_sm = np.array([
                dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_sm.time.values
            ])

            weak_sm_summer, weak_LE_summer = [], []
            weak_sm_hw, weak_LE_hw = [], []
            trans_sm_summer, trans_LE_summer = [], []
            trans_sm_hw, trans_LE_hw = [], []
            ml_sm_summer, ml_LE_summer = [], []
            ml_sm_hw, ml_LE_hw = [], []

            for lat_center_hw, lon_center_hw, onset_day, duration_event in zip(
                lats_events, lons_events, dates_d_HW_day0, duration_events
            ):
                if onset_day < dates_sm[0] or onset_day < dates_LE[0] or onset_day < dates_t[0]:
                    print(f'Skipping event {onset_day}: before SM/LE/T start')
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
                    sm_summer_list, sm_hw_list = weak_sm_summer, weak_sm_hw
                    le_summer_list, le_hw_list = weak_LE_summer, weak_LE_hw
                elif dom_regime in (1, 2):
                    sm_point = np.nanmean(sm_region, axis=(1, 2))
                    sm_summer_list, sm_hw_list = trans_sm_summer, trans_sm_hw
                    le_summer_list, le_hw_list = trans_LE_summer, trans_LE_hw
                elif dom_regime in (3, 4):
                    bp_box = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
                    if not np.isfinite(bp_box).any():
                        continue
                    sm_point = np.nanmean(
                        sm_region - bp_box[np.newaxis, :, :], axis=(1, 2)
                    )
                    sm_summer_list, sm_hw_list = ml_sm_summer, ml_sm_hw
                    le_summer_list, le_hw_list = ml_LE_summer, ml_LE_hw
                else:
                    continue

                common_dates = np.intersect1d(
                    np.intersect1d(dates_LE, dates_sm), dates_t
                )
                LE_aligned = LE_point[np.isin(dates_LE, common_dates)]
                sm_aligned = sm_point[np.isin(dates_sm, common_dates)]

                heatwave_days = [
                    onset_day + dt.timedelta(days=i)
                    for i in range(int(duration_event))
                ]
                heatwave_idx = np.where(np.isin(common_dates, heatwave_days))[0]
                hw_mask = np.zeros(len(common_dates), dtype=bool)
                hw_mask[heatwave_idx] = True

                summer_mask = (
                    np.array([d.month in (6, 7, 8) for d in common_dates]) & (~hw_mask)
                )
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
            n_hw_ml = len(ml_sm_hw)

            weak_sm_summer = _as_1d_points(weak_sm_summer)
            weak_LE_summer = _as_1d_points(weak_LE_summer)
            weak_sm_hw = _as_1d_points(weak_sm_hw)
            weak_LE_hw = _as_1d_points(weak_LE_hw)
            trans_sm_summer = _as_1d_points(trans_sm_summer)
            trans_LE_summer = _as_1d_points(trans_LE_summer)
            trans_sm_hw = _as_1d_points(trans_sm_hw)
            trans_LE_hw = _as_1d_points(trans_LE_hw)
            ml_sm_summer = _as_1d_points(ml_sm_summer)
            ml_LE_summer = _as_1d_points(ml_LE_summer)
            ml_sm_hw = _as_1d_points(ml_sm_hw)
            ml_LE_hw = _as_1d_points(ml_LE_hw)

            np.savez_compressed(
                density_cache_path,
                weak_sm_summer=weak_sm_summer, weak_LE_summer=weak_LE_summer,
                weak_sm_hw=weak_sm_hw, weak_LE_hw=weak_LE_hw, n_hw_weak=n_hw_weak,
                trans_sm_summer=trans_sm_summer, trans_LE_summer=trans_LE_summer,
                trans_sm_hw=trans_sm_hw, trans_LE_hw=trans_LE_hw, n_hw_trans=n_hw_trans,
                ml_sm_summer=ml_sm_summer, ml_LE_summer=ml_LE_summer,
                ml_sm_hw=ml_sm_hw, ml_LE_hw=ml_LE_hw, n_hw_ml=n_hw_ml,
            )
            print(f'Saved density data cache: {density_cache_path}')
            ds_LE.close()
            ds_sm.close()
            ds_t.close()


        _plot_density_panel(
            weak_sm_summer, weak_LE_summer, weak_sm_hw, weak_LE_hw,
            xlabel=r'SMs (regional mean, ±0.5°)',
            title=(
                f'SM vs LE — weak coupling (regime 0) | {region_tag}\n'
                f'{lon_desc} | Max-anomaly location'
            ),
            outfile=(
                f'{path_figures}density_{variable_LE}_{variable_sm}_aroundOnsetloc_absSM_summer_allHWdays_'
                f'05deg_weak_{name_land}_{region_tag}.png'
            ),
            show_critical_vline=False,
            density_vmax=0.2,
            n_hw_events=n_hw_weak,
            spearman_corr=True,
        )
        _plot_density_panel(
            trans_sm_summer, trans_LE_summer, trans_sm_hw, trans_LE_hw,
            xlabel=r'SMs (regional mean, ±0.5°)',
            title=(
                f'SM vs LE — transitional (regime 1 or 2) | {region_tag}\n'
                f'{lon_desc} | Max-anomaly location'
            ),
            outfile=(
                f'{path_figures}density_{variable_LE}_{variable_sm}_aroundOnsetloc_absSM_summer_allHWdays_'
                f'05deg_trans_{name_land}_{region_tag}.png'
            ),
            show_critical_vline=False,
            density_vmax=0.6,
            n_hw_events=n_hw_trans,
        )
        
        if region_tag == "centralUS": dens_mx = 0.15
        else: dens_mx = 0.6
        _plot_density_panel(
            ml_sm_summer, ml_LE_summer, ml_sm_hw, ml_LE_hw,
            xlabel=r'SMs $-$ critical SM (regime 3/4 BP, regional mean, ±0.5°)',
            title=(
                f'SM vs LE — moisture-limited (regime 3+4) | {region_tag}\n'
                f'{lon_desc} | Max-anomaly location'
            ),
            outfile=(
                f'{path_figures}density_{variable_LE}_{variable_sm}_aroundOnsetloc_relBP_summer_allHWdays_'
                f'05deg_moistlim_{name_land}_{region_tag}.png'
            ),
            show_critical_vline=True,
            density_vmax=dens_mx,
            n_hw_events=n_hw_ml,
            xmax_sm=0.15,
        )
