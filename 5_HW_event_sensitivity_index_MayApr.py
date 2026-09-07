# Heatwave-specific soil-moisture → temperature sensitivity:
#   T'_HW(x,y) = α(x,y) + β(x,y) SM'_lateApr(x,y) + ε
# where SM' is the late-April (15–30) soil-moisture anomaly in the event year
# and T'_HW is the mean temperature anomaly over the event footprint at that pixel.
from Functions import *
import argparse
import yaml
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
case = cfg['case']
path_case = cfg['path_case_land']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
topography = cfg['topography']
initial_year = cfg['initial_year']
apr_day_start = int(cfg['apr_day_start'])
apr_day_end = int(cfg['apr_day_end'])
min_n = int(cfg['min_n'])
if not (1 <= apr_day_start <= apr_day_end <= 30):
    raise ValueError(
        f'Need 1 ≤ apr_day_start ≤ apr_day_end ≤ 30 '
        f'(got {apr_day_start}–{apr_day_end})'
    )

variable_sm = 'SMrz'
variable_t = 'TS'
lat_minHW = 25
lat_maxHW = 50
lon_minHW = 235
lon_maxHW = 290
sm_period_tag = f'lateApr{apr_day_start}to{apr_day_end}'

path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

# ---------------------------------------------------------------- HW mask + events
path_file_mask_hw = f'{path_outputs}{case}/HW_pixel_mask_daily_ERA5_Teng.nc'
ds_mask_hw = xr.open_dataset(path_file_mask_hw, decode_times=False)
hw_mask = ds_mask_hw['hw_mask']  # 0=none, 1=onset, 2=later
lats = np.asarray(ds_mask_hw['lat'].values)
lons = np.asarray(ds_mask_hw['lon'].values)
lons_plot = np.where(lons > 180.0, lons - 360.0, lons)

time_hw = pd.to_datetime(np.asarray(ds_mask_hw['time'].values).astype(str), format='%Y%m%d')
hw_mask = hw_mask.assign_coords(time=('time', time_hw))
print(f'HW mask: {time_hw[0].date()}–{time_hw[-1].date()}, '
      f'grid {len(lats)}×{len(lons)}')

hw_csv = f'{path_case}/Heat_waves_events_list.csv'
df_hw_events = pd.read_csv(hw_csv, index_col=0)
df_hw_events['Date 0'] = pd.to_datetime(df_hw_events['Date 0'])
print(f'Loaded {len(df_hw_events)} HW events from {hw_csv}')

# ---------------------------------------------------------------- Daily TS anomalies (ERA5 US)
path_file_TS = f'/scratch/negishi/castanev/Amplification_RW/ERA5/TS_ERA5_daily_anoma_window.nc'
ds_ts = xr.open_dataset(path_file_TS, decode_times=False, engine='netcdf4')
ts = ds_ts[variable_t]
dates_d = np.array([
    dt.datetime(initial_year, 1, 1) + dt.timedelta(days=i)
    for i in range(len(ds_ts['time']))
])
ts = ts.assign_coords(time=('time', dates_d))
lats_ts_full = np.asarray(ds_ts['lat'].values)
lons_ts_full = np.asarray(ds_ts['lon'].values)
pos_lats_ts = np.where((lats_ts_full >= lat_minHW) & (lats_ts_full <= lat_maxHW))[0]
pos_lons_ts = np.where((lons_ts_full >= lon_minHW) & (lons_ts_full <= lon_maxHW))[0]
ts = ts.isel(lat=pos_lats_ts, lon=pos_lons_ts)
print(f'TS daily: {dates_d[0].date()}–{dates_d[-1].date()}, '
      f'US crop {ts.sizes["lat"]}×{ts.sizes["lon"]}')

# ---------------------------------------------------------------- Daily SM anomalies (full year, GLEAM US)
path_file_SM = (
    f'{path_outputs}daily_anoma_window_smoothed15daysw_'
    f'{variable_sm}_{name_land}_US.nc'
)
ds_sm = xr.open_dataset(path_file_SM, decode_times=False, engine='netcdf4')
sm = ds_sm[variable_sm]
sm = sm.assign_coords(
    time=('time', pd.to_datetime(np.asarray(ds_sm['time'].values).astype(str), format='%Y%m%d'))
)
print(f'SM daily: {pd.Timestamp(sm.time.values[0]).date()}–'
      f'{pd.Timestamp(sm.time.values[-1]).date()}, '
      f'grid {sm.sizes["lat"]}×{sm.sizes["lon"]}')

# Put SM and TS on the HW lat/lon grid used for mapping
def _align_to_hw(da):
    """Regrid `da` onto the HW lat/lon grid, fixing 0–360 vs −180–180 lon."""
    hw_lat = np.asarray(hw_mask['lat'].values, dtype=float)
    hw_lon = np.asarray(hw_mask['lon'].values, dtype=float)
    da_lon = np.asarray(da['lon'].values, dtype=float)

    same_grid = (
        len(da['lat']) == len(hw_lat)
        and len(da['lon']) == len(hw_lon)
        and np.allclose(da['lat'].values, hw_lat)
        and np.allclose(da_lon, hw_lon)
    )
    if same_grid:
        return da

    # Match target longitudes to the source convention for interpolation
    if da_lon.min() < 0 and hw_lon.min() >= 0:
        target_lon = np.where(hw_lon > 180.0, hw_lon - 360.0, hw_lon)
    elif da_lon.min() >= 0 and hw_lon.min() < 0:
        target_lon = np.where(hw_lon < 0.0, hw_lon + 360.0, hw_lon)
    else:
        target_lon = hw_lon

    out = da.interp(lat=hw_lat, lon=target_lon, method='nearest')
    # Keep HW 0–360 longitude labels for plotting / netCDF
    out = out.assign_coords(lon=('lon', hw_lon), lat=('lat', hw_lat))
    return out


ts_on_hw = _align_to_hw(ts)
sm_on_hw = _align_to_hw(sm)
print(
    f'Aligned to HW grid: TS finite={np.isfinite(ts_on_hw.values[0]).mean():.2f}, '
    f'SM finite={np.isfinite(sm_on_hw.values[min(100, sm_on_hw.sizes["time"]-1)]).mean():.2f}'
)

time_ts = pd.DatetimeIndex(ts_on_hw['time'].values).normalize()
time_sm = pd.DatetimeIndex(sm_on_hw['time'].values).normalize()
time_hw_full = pd.DatetimeIndex(hw_mask['time'].values).normalize()
date_to_its = {d: i for i, d in enumerate(time_ts)}
date_to_ism = {d: i for i, d in enumerate(time_sm)}
date_to_ihw = {d: i for i, d in enumerate(time_hw_full)}

ts_vals = np.asarray(ts_on_hw.values, dtype=float)
sm_vals = np.asarray(sm_on_hw.values, dtype=float)
hw_vals = np.asarray(hw_mask.values)
nlat, nlon = hw_vals.shape[1], hw_vals.shape[2]

# Sufficient statistics for T = α + β SM at each pixel (equal weight per event)
n_pix = np.zeros((nlat, nlon), dtype=float)
sum_sm = np.zeros((nlat, nlon), dtype=float)
sum_t = np.zeros((nlat, nlon), dtype=float)
sum_sm2 = np.zeros((nlat, nlon), dtype=float)
sum_t2 = np.zeros((nlat, nlon), dtype=float)
sum_smt = np.zeros((nlat, nlon), dtype=float)
n_events_used = 0
n_skip_no_ts = 0
n_skip_no_sm = 0
n_skip_no_footprint = 0
n_skip_allnan = 0
n_skip_onset_before_apr = 0

for _, row in df_hw_events.iterrows():
    onset = pd.Timestamp(row['Date 0']).normalize()
    dur = int(row['Duration'])

    # Event footprint days (onset … onset+duration−1)
    its_e, ihw_e = [], []
    for k in range(dur):
        d = onset + pd.Timedelta(days=k)
        if d in date_to_its and d in date_to_ihw:
            its_e.append(date_to_its[d])
            ihw_e.append(date_to_ihw[d])
    if not its_e:
        n_skip_no_ts += 1
        continue

    # Late April SM in the same calendar year as the event onset
    ism_apr = []
    for day in range(apr_day_start, apr_day_end + 1):
        d = pd.Timestamp(year=int(onset.year), month=4, day=day)
        if d in date_to_ism:
            ism_apr.append(date_to_ism[d])
    if not ism_apr:
        n_skip_no_sm += 1
        continue

    ts_e = ts_vals[np.asarray(its_e)]
    hw_e = hw_vals[np.asarray(ihw_e)] > 0
    n_hw_days = hw_e.sum(axis=0).astype(float)
    affected = n_hw_days > 0
    if not np.any(affected):
        n_skip_no_footprint += 1
        continue

    with np.errstate(invalid='ignore'):
        t_hw = np.nansum(np.where(hw_e, ts_e, np.nan), axis=0) / n_hw_days

    sm_apr = sm_vals[np.asarray(ism_apr)]
    with np.errstate(invalid='ignore'):
        sm_pre = np.nanmean(sm_apr, axis=0)

    ok = affected & np.isfinite(t_hw) & np.isfinite(sm_pre)
    if not np.any(ok):
        n_skip_allnan += 1
        continue

    x = sm_pre[ok]
    y = t_hw[ok]
    n_pix[ok] += 1.0
    sum_sm[ok] += x
    sum_t[ok] += y
    sum_sm2[ok] += x * x
    sum_t2[ok] += y * y
    sum_smt[ok] += x * y
    n_events_used += 1

print(
    f'Event filter: used={n_events_used}, '
    f'skip onset before/during late Apr={n_skip_onset_before_apr}, '
    f'skip no TS/HW days={n_skip_no_ts}, '
    f'skip no late-Apr SM (e.g. pre-1980)={n_skip_no_sm}, '
    f'skip empty footprint={n_skip_no_footprint}, '
    f'skip all-NaN SM/T={n_skip_allnan}'
)
# OLS: y = α + β x
with np.errstate(invalid='ignore', divide='ignore'):
    denom = n_pix * sum_sm2 - sum_sm ** 2
    beta = (n_pix * sum_smt - sum_sm * sum_t) / denom
    alpha = (sum_t - beta * sum_sm) / n_pix
    # Pearson r
    var_x = n_pix * sum_sm2 - sum_sm ** 2
    var_y = n_pix * sum_t2 - sum_t ** 2
    r = (n_pix * sum_smt - sum_sm * sum_t) / np.sqrt(var_x * var_y)
    # two-sided p-value for β (equiv. for r) with n−2 df
    df = n_pix - 2.0
    tstat = r * np.sqrt(df / np.maximum(1.0 - r ** 2, 1e-12))
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df)

too_few = n_pix < min_n
beta = np.where(too_few | ~np.isfinite(beta), np.nan, beta)
alpha = np.where(too_few | ~np.isfinite(alpha), np.nan, alpha)
r = np.where(too_few | ~np.isfinite(r), np.nan, r)
pvalue = np.where(too_few | ~np.isfinite(pvalue), np.nan, pvalue)

print(
    f'Sensitivity: events used={n_events_used}/{len(df_hw_events)}, '
    f'SM={sm_period_tag} (event year), min_n={min_n}, '
    f'pixels with β={int(np.isfinite(beta).sum())}'
)
if np.isfinite(beta).any():
    print(f'β range {np.nanmin(beta):.2f} to {np.nanmax(beta):.2f}')
else:
    raise RuntimeError(
        'No finite β values. Check SM/HW longitude conventions and SM time coverage.'
    )


def plot_clim_map(field, outfile, title, cbar_label, vmax, cmap='YlOrRd',
                  levels=None, extend='max', sig_mask=None):
    """Single-panel US map; optional significance hatch where sig_mask==1."""
    if levels is None:
        levels = np.linspace(0.0, vmax, 12)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.5)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
    im = ax.contourf(
        lons_plot, lats, field,
        levels=levels, cmap=cmap, extend=extend,
        transform=ccrs.PlateCarree(),
    )
    if sig_mask is not None:
        ax.contourf(
            lons_plot, lats, sig_mask,
            levels=[0.5, 1.5], colors='none', hatches=['///'],
            transform=ccrs.PlateCarree(), zorder=5,
        )
    ax.set_extent([lons_plot.min(), lons_plot.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.35, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_title(title, fontsize=12)
    fig.subplots_adjust(bottom=0.12, top=0.90, left=0.08, right=0.95)
    cax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


sig = np.where(np.isfinite(pvalue) & (pvalue <= 0.05), 1.0, np.nan)

vmax_beta = float(np.nanpercentile(np.abs(beta), 98))
if not np.isfinite(vmax_beta) or vmax_beta <= 0:
    vmax_beta = 50.0
plot_clim_map(
    beta,
    f'{path_figures}HW_SMtoT_beta_{sm_period_tag}_{variable_sm}_{name_land}.png',
    title=rf'HW-specific SM→T sensitivity $\beta$ (SM late Apr {apr_day_start}–{apr_day_end})',
    cbar_label=rf'$\beta$ [°C / (m$^3$ m$^{{-3}}$)]',
    vmax=vmax_beta,
    cmap='RdBu_r',
    levels=np.linspace(-vmax_beta, vmax_beta, 13),
    extend='both',
    sig_mask=sig,
)

vmax_r = 1.0
plot_clim_map(
    r,
    f'{path_figures}HW_SMtoT_r_{sm_period_tag}_{variable_sm}_{name_land}.png',
    title=rf'HW SM→T correlation $r$ (SM late Apr {apr_day_start}–{apr_day_end})',
    cbar_label=r'$r$',
    vmax=vmax_r,
    cmap='RdBu_r',
    levels=np.linspace(-vmax_r, vmax_r, 13),
    extend='neither',
    sig_mask=sig,
)

vmax_n = float(np.nanpercentile(n_pix, 99))
if not np.isfinite(vmax_n) or vmax_n <= 0:
    vmax_n = 20.0
plot_clim_map(
    n_pix,
    f'{path_figures}HW_SMtoT_n_{sm_period_tag}_{variable_sm}_{name_land}.png',
    title='Number of HW events per pixel used for β',
    cbar_label='Number of events',
    vmax=vmax_n,
    cmap='YlOrRd',
    extend='max',
)

# Save fields for later analysis
out_nc = (
    f'{path_outputs}{case}/HW_SMtoT_sensitivity_{sm_period_tag}_{variable_sm}_{name_land}.nc'
)
xr.Dataset(
    {
        'beta': (('lat', 'lon'), beta.astype('float32')),
        'alpha': (('lat', 'lon'), alpha.astype('float32')),
        'r': (('lat', 'lon'), r.astype('float32')),
        'pvalue': (('lat', 'lon'), pvalue.astype('float32')),
        'n_events': (('lat', 'lon'), n_pix.astype('float32')),
    },
    coords={'lat': lats, 'lon': lons},
    attrs={
        'description': (
            'Heatwave-specific SM→T sensitivity: T_HW = alpha + beta*SM_lateApr '
            '(SM = mean anomaly Apr day_start–day_end in the event year)'
        ),
        'SM_period': sm_period_tag,
        'apr_day_start': apr_day_start,
        'apr_day_end': apr_day_end,
        'min_n': min_n,
        'SM_file': path_file_SM,
        'TS_file': path_file_TS,
    },
).to_netcdf(out_nc)
print(f'Saved {out_nc}')
