# Summer SM–LE density: conditioned years vs rest, all pixels in south-central US
# (lat 25–40°N, lon 255–270°E). Weak (0) and moistlim (3+4) only.
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




# ======================================================= SUMMER COND vs REST (all pixels) ====================================================
# Green filled density: JJA in years WITHOUT the EOF-neg condition
# Orange contours:     JJA in years WITH the EOF-neg condition
# All land pixels in each lon band; figures for weak (regime 0) and moistlim (3+4).
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_outputs_case, exist_ok=True)

variable_LE = 'LE'
variable_sm = 'SMs'


def _lon360_arr(lons):
    lons = np.asarray(lons, dtype=float)
    return np.where(lons < 0, lons + 360.0, lons)


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


# EOF2 / EOF3 negative-phase years (late April) — same recipe as conditional_regional
pc_thresh = 0.75
EOF_indices_neg = [1, 2]  # EOF2 and EOF3 (0-based)
eof_period_label = 'lateApr'
eof_sel_month, eof_day_lo, eof_day_hi = 4, 16, 30

path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'SMrz_{name_land}_AprMay_US.nc'
)
ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
time_ymd = ds_eof['time_ymd']
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')

year_neg_by_eof = {}
for EOF_i in EOF_indices_neg:
    pc_daily = np.asarray(eof_ts[EOF_i].values)
    pc_yr, years_yr = period_mean_pc(
        pc_daily, time_eof_ts, eof_sel_month, eof_day_lo, eof_day_hi
    )
    pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()
    year_neg = years_yr[pc_std < -pc_thresh]
    year_neg_by_eof[EOF_i + 1] = year_neg
    print(
        f'EOF{EOF_i + 1} {eof_period_label} negative phase '
        f'(PC* < -{pc_thresh:g}): n={len(year_neg)} years → {year_neg.tolist()}'
    )
ds_eof.close()

# Analysis domain: south-central US box only
ANALYSIS_LAT_MIN, ANALYSIS_LAT_MAX = 25.0, 40.0
ANALYSIS_LON_MIN, ANALYSIS_LON_MAX = 255.0, 270.0
REGION_TAG = 'southcentralUS'
US_LON_REGIONS = [
    (REGION_TAG, ANALYSIS_LON_MIN, ANALYSIS_LON_MAX),  # 255 ≤ lon ≤ 270
]


def _as_1d_points(arr_or_list):
    if isinstance(arr_or_list, list):
        return np.concatenate(arr_or_list) if len(arr_or_list) > 0 else np.array([])
    arr = np.asarray(arr_or_list)
    return arr.ravel() if arr.size else np.array([])


def _plot_cond_vs_rest(
    sm_rest, LE_rest,
    sm_cond, LE_cond,
    xlabel, title, outfile,
    show_critical_vline=False,
    density_vmax=0.2,
    xmax_sm=None,
    n_years_cond=None,
    n_years_rest=None,
    n_pixels=None,
):
    sm_rest = _as_1d_points(sm_rest)
    LE_rest = _as_1d_points(LE_rest)
    sm_cond = _as_1d_points(sm_cond)
    LE_cond = _as_1d_points(LE_cond)

    m_r = np.isfinite(sm_rest) & np.isfinite(LE_rest)
    m_c = np.isfinite(sm_cond) & np.isfinite(LE_cond)
    sm_r, LE_r = sm_rest[m_r], LE_rest[m_r]
    sm_c, LE_c = sm_cond[m_c], LE_cond[m_c]

    if sm_r.size == 0 and sm_c.size == 0:
        print(f'Skipping (no data): {outfile}')
        return

    x_parts = [a for a in (sm_r, sm_c) if a.size > 0]
    y_parts = [a for a in (LE_r, LE_c) if a.size > 0]
    xmin = float(np.nanmin(np.concatenate(x_parts)))
    xmax = float(xmax_sm) if xmax_sm is not None else float(np.nanmax(np.concatenate(x_parts)))
    ymin = float(np.nanmin(np.concatenate(y_parts)))
    ymax_le = 150.0

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax_le, 45),
    )
    levels = np.linspace(0, density_vmax, 21)
    cont_lo = 0.5 * density_vmax
    n_cont = 7
    cont_levels = np.linspace(cont_lo, density_vmax, n_cont)

    fig = plt.figure(figsize=(7.0, 5))
    ax = fig.add_axes([0.11, 0.12, 0.63, 0.78])
    cbar_fmt = FormatStrFormatter('%.2f')
    ax_pos = ax.get_position()
    cax_g = fig.add_axes([ax_pos.x1 + 0.006, ax_pos.y0, 0.022, ax_pos.height])
    red_h_frac = 0.58
    cax_o = fig.add_axes([
        ax_pos.x1 + 0.006 + 0.022 + 0.10,
        ax_pos.y0 + ax_pos.height * (1.0 - red_h_frac),
        0.023,
        ax_pos.height * red_h_frac,
    ])

    # Rest-of-years: green filled density
    if sm_r.size > 5:
        kde_r = gaussian_kde(np.vstack([sm_r, LE_r]))
        z_r = kde_r(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cf = ax.contourf(
            xgrid, ygrid, z_r,
            levels=levels, cmap='Greens', vmin=0, vmax=density_vmax,
            alpha=0.85, extend='max',
        )
        if sm_c.size > 5:
            ax.contour(
                xgrid, ygrid, z_r,
                levels=[cont_lo], colors='darkgreen', linewidths=1.4, linestyles='-',
            )
        cbar_g = fig.colorbar(cf, cax=cax_g, extend='max')
        cbar_g.set_label(f'Rest-of-years JJA (0–{density_vmax:.2f})', fontsize=10)
        cbar_g.ax.yaxis.set_major_formatter(cbar_fmt)
        cbar_g.ax.tick_params(labelsize=9)
    elif sm_r.size > 0:
        ax.scatter(sm_r, LE_r, s=8, alpha=0.25, color='seagreen', label='Rest-of-years JJA')

    # Conditioned years: orange contours
    if sm_c.size > 5:
        kde_c = gaussian_kde(np.vstack([sm_c, LE_c]))
        z_c = kde_c(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        ax.contour(
            xgrid, ygrid, z_c,
            levels=cont_levels, cmap='Oranges', vmin=0, vmax=density_vmax,
            linewidths=1.4,
        )
        colors_c = [mpl.cm.Oranges(lvl / density_vmax) for lvl in cont_levels]
        if len(cont_levels) > 1:
            half = 0.5 * (cont_levels[1] - cont_levels[0])
            boundaries = np.concatenate([
                [cont_levels[0] - half],
                0.5 * (cont_levels[:-1] + cont_levels[1:]),
                [cont_levels[-1] + half],
            ])
        else:
            boundaries = np.array([cont_levels[0] - 0.01, cont_levels[0] + 0.01])
        cmap_c = ListedColormap(colors_c)
        norm_c = BoundaryNorm(boundaries, ncolors=len(colors_c))
        smap = mpl.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
        smap.set_array([])
        cbar_o = fig.colorbar(smap, cax=cax_o)
        cbar_o.set_ticks(cont_levels)
        cbar_o.ax.set_yticklabels([f'{v:.2f}' for v in cont_levels])
        cbar_o.set_label('Conditioned-years JJA', fontsize=10)
        cbar_o.ax.tick_params(labelsize=9)
    elif sm_c.size > 0:
        ax.scatter(sm_c, LE_c, s=10, alpha=0.35, color='darkorange', label='Conditioned JJA')

    if show_critical_vline:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax_le)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'LE (W m$^{-2}$)', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=11)

    info = []
    if n_pixels is not None:
        info.append(f'n pixels = {int(n_pixels)}')
    if n_years_cond is not None:
        info.append(f'n cond years = {int(n_years_cond)}')
    if n_years_rest is not None:
        info.append(f'n rest years = {int(n_years_rest)}')
    info.append(f'n points rest/cond = {sm_r.size}/{sm_c.size}')
    ax.text(
        0.98, 0.02, '\n'.join(info),
        fontsize=9, transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9),
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=10, loc='best')
    fig.savefig(outfile, dpi=500)
    plt.close(fig)
    print(f'Saved {outfile}')


def _lon_band_mask(lons, lon_lo, lon_hi):
    """Inclusive lon band in 0–360°E (both bounds required here)."""
    lons360 = _lon360_arr(lons)
    return (
        (lons360 >= float(lon_lo)) & (lons360 <= float(lon_hi)),
        f'{lon_lo:g} ≤ lon ≤ {lon_hi:g}',
    )


print('Opening LE and SM for all-pixel summer densities...')
ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine='netcdf4')
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_LE.time.values])
lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), '%Y%m%d').date() for t in ds_sm.time.values])

# Common calendar
common_dates = np.intersect1d(dates_LE, dates_sm)
i_LE = np.isin(dates_LE, common_dates)
i_sm = np.isin(dates_sm, common_dates)
dates_common = dates_LE[i_LE]  # same set/order as LE if chronological
# Re-align SM to LE common order
date_to_ism = {d: j for j, d in enumerate(dates_sm)}
sm_order = np.array([date_to_ism[d] for d in dates_common])

jja_mask = np.array([d.month in (6, 7, 8) for d in dates_common])
years_common = np.array([d.year for d in dates_common])

# Load full fields once (time × lat × lon) — may be large; subset spatially per region below
# Use SM grid as master for regimes; select LE by lat/lon values

for region_tag, lon_lo, lon_hi in US_LON_REGIONS:
    lat_mask_sm = (lats_sm >= ANALYSIS_LAT_MIN) & (lats_sm <= ANALYSIS_LAT_MAX)
    lon_mask_sm, lon_desc = _lon_band_mask(lons_sm, lon_lo, lon_hi)
    ilat_sm = np.where(lat_mask_sm)[0]
    ilon_sm = np.where(lon_mask_sm)[0]
    if ilat_sm.size == 0 or ilon_sm.size == 0:
        print(f'Skipping {region_tag}: no SM grid points')
        continue

    lat_mask_LE = (lats_LE >= ANALYSIS_LAT_MIN) & (lats_LE <= ANALYSIS_LAT_MAX)
    lon_mask_LE, _ = _lon_band_mask(lons_LE, lon_lo, lon_hi)
    ilat_LE = np.where(lat_mask_LE)[0]
    ilon_LE = np.where(lon_mask_LE)[0]
    if ilat_LE.size == 0 or ilon_LE.size == 0:
        print(f'Skipping {region_tag}: no LE grid points')
        continue

    print(f'\n=== Region {region_tag} ({lon_desc}, lat [{ANALYSIS_LAT_MIN}, {ANALYSIS_LAT_MAX}]) ===')
    print(f'  SM grid: {ilat_sm.size} × {ilon_sm.size}; LE grid: {ilat_LE.size} × {ilon_LE.size}')

    # Prefer SM spatial grid for regime classification; require matching shape with LE subset
    # If shapes differ, reindex LE onto SM lat/lon by nearest neighbor once
    sm_sub = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values  # (t_sm, ny, nx)
    LE_sub_raw = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values

    sm_aligned = sm_sub[sm_order]  # (t_common, ny_sm, nx_sm)
    # Align LE in time
    date_to_iLE = {d: j for j, d in enumerate(dates_LE)}
    LE_order = np.array([date_to_iLE[d] for d in dates_common])
    LE_time = LE_sub_raw[LE_order]

    # Spatial align LE → SM grid if needed
    if LE_time.shape[1:] != sm_aligned.shape[1:]:
        from scipy.interpolate import RegularGridInterpolator
        lats_LE_sub = lats_LE[ilat_LE]
        lons_LE_sub = _lon360_arr(lons_LE[ilon_LE])
        lats_sm_sub = lats_sm[ilat_sm]
        lons_sm_sub = _lon360_arr(lons_sm[ilon_sm])
        # Ensure increasing for interpolator
        lat_LE_ord = np.argsort(lats_LE_sub)
        lon_LE_ord = np.argsort(lons_LE_sub)
        lats_LE_sorted = lats_LE_sub[lat_LE_ord]
        lons_LE_sorted = lons_LE_sub[lon_LE_ord]
        LE_sorted = LE_time[:, lat_LE_ord, :][:, :, lon_LE_ord]
        Lon_t, Lat_t = np.meshgrid(lons_sm_sub, lats_sm_sub)
        pts = np.column_stack([Lat_t.ravel(), Lon_t.ravel()])
        LE_on_sm = np.empty_like(sm_aligned)
        for it in range(LE_sorted.shape[0]):
            # Skip all-nan slices
            slab = LE_sorted[it]
            if not np.isfinite(slab).any():
                LE_on_sm[it] = np.nan
                continue
            # Fill interpolator; nan-safe via nearest of finite
            interp = RegularGridInterpolator(
                (lats_LE_sorted, lons_LE_sorted),
                np.where(np.isfinite(slab), slab, np.nanmean(slab)),
                bounds_error=False, fill_value=np.nan,
            )
            LE_on_sm[it] = interp(pts).reshape(Lat_t.shape)
        LE_aligned = LE_on_sm
    else:
        LE_aligned = LE_time

    LE_aligned = np.where((LE_aligned >= 0) & np.isfinite(LE_aligned), LE_aligned, np.nan)

    regime_sub = regime_field[np.ix_(ilat_sm, ilon_sm)]
    bp_sub = critical_sm_bp[np.ix_(ilat_sm, ilon_sm)]
    weak_pix = regime_sub == 0
    ml_pix = (regime_sub == 3) | (regime_sub == 4)
    ml_pix = ml_pix & np.isfinite(bp_sub)

    n_weak = int(np.sum(weak_pix))
    n_ml = int(np.sum(ml_pix))
    print(f'  Regime pixels: weak={n_weak}, moistlim(3+4)={n_ml}')

    sm_weak = sm_aligned[:, weak_pix]
    LE_weak = LE_aligned[:, weak_pix]
    sm_ml_abs = sm_aligned[:, ml_pix]
    LE_ml = LE_aligned[:, ml_pix]
    bp_ml = bp_sub[ml_pix]
    sm_ml = sm_ml_abs - bp_ml[np.newaxis, :]

    for eof_num, year_neg in year_neg_by_eof.items():
        year_neg_set = set(int(y) for y in np.asarray(year_neg))
        cond_day = jja_mask & np.array([y in year_neg_set for y in years_common])
        rest_day = jja_mask & (~np.array([y in year_neg_set for y in years_common]))
        n_years_cond = len({y for y in years_common[cond_day]})
        n_years_rest = len({y for y in years_common[rest_day]})
        cond_tag = (
            f'EOF{eof_num}neg_{eof_period_label}_thresh{pc_thresh:g}_{region_tag}'
        )
        print(
            f'  EOF{eof_num}: cond JJA days={cond_day.sum()}, rest JJA days={rest_day.sum()} '
            f'(years {n_years_cond}/{n_years_rest})'
        )

        # Weak coupling
        if n_weak > 0:
            _plot_cond_vs_rest(
                sm_weak[rest_day].ravel(), LE_weak[rest_day].ravel(),
                sm_weak[cond_day].ravel(), LE_weak[cond_day].ravel(),
                xlabel=r'SMs',
                title=(
                    f'SM vs LE — weak coupling (regime 0) | {region_tag}\n'
                    f'{lon_desc} | green: rest JJA, orange: EOF{eof_num}-neg JJA'
                ),
                outfile=(
                    f'{path_figures}density_{variable_LE}_{variable_sm}_absSM_JJAcond_'
                    f'05deg_weak_{cond_tag}_{name_land}.png'
                ),
                show_critical_vline=False,
                density_vmax=0.15,
                n_years_cond=n_years_cond,
                n_years_rest=n_years_rest,
                n_pixels=n_weak,
            )

        # Moisture-limited (3+4)
        if n_ml > 0:
            _plot_cond_vs_rest(
                sm_ml[rest_day].ravel(), LE_ml[rest_day].ravel(),
                sm_ml[cond_day].ravel(), LE_ml[cond_day].ravel(),
                xlabel=r'SMs $-$ critical SM',
                title=(
                    f'SM vs LE — moisture-limited (regime 3+4) | {region_tag}\n'
                    f'{lon_desc} | green: rest JJA, orange: EOF{eof_num}-neg JJA'
                ),
                outfile=(
                    f'{path_figures}density_{variable_LE}_{variable_sm}_relBP_JJAcond_'
                    f'05deg_moistlim_{cond_tag}_{name_land}.png'
                ),
                show_critical_vline=True,
                density_vmax=0.35,
                xmax_sm=0.15,
                n_years_cond=n_years_cond,
                n_years_rest=n_years_rest,
                n_pixels=n_ml,
            )

ds_LE.close()
ds_sm.close()
print('Done.')
