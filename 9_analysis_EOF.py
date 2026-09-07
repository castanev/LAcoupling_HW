from Functions import *
import argparse
import yaml
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
# |PC*| threshold. 0.5 is mild (~1/3 of years each side if near-Gaussian).
# 0.75–1.0 gives clearer composites with fewer years.
pc_thresh = 0.75
vmax=0.04

lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat = 45  # US, 0–360°
path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

# EOF (May PCs) ========================================================================================================
variable = 'SMrz'
path_file_eof = f'{path_outputs}EOF_timeseries_monthly_anoma_SMrz_{name_land}_May_US.nc'

ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
time_ymd = ds_eof['time_ymd']

t_dim = eof_ts.dims[-1]
eof_ts_std = (eof_ts - eof_ts.mean(t_dim)) / eof_ts.std(t_dim)
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')
years_eof = np.array(time_eof_ts.year)

# Daily HW pixel mask (ERA5 US) ========================================================================================
path_file_mask_hw = f'{path_outputs}{case}/HW_pixel_mask_daily_ERA5_Teng.nc'
ds_mask_hw = xr.open_dataset(path_file_mask_hw, decode_times=False)
hw_mask = ds_mask_hw['hw_mask']  # (time, lat, lon): 0=none, 1=onset, 2=later HW day
lats = np.asarray(ds_mask_hw['lat'].values)
lons = np.asarray(ds_mask_hw['lon'].values)
lons_plot = np.where(lons > 180.0, lons - 360.0, lons)

time_hw = pd.to_datetime(np.asarray(ds_mask_hw['time'].values).astype(str), format='%Y%m%d')
print("time_hw", time_hw)
hw_mask = hw_mask.assign_coords(time=('time', time_hw))

# Summer = JJA
months = hw_mask['time'].dt.month
hw_summer = hw_mask.sel(time=(months >= 6) & (months <= 8))

# SMrz monthly anomalies (GLEAM grid)
variable_sm = 'SMrz'
path_file_SMrz = f'{path_outputs}monthly_anoma_{variable_sm}_{name_land}_US.nc'
ds_sm = xr.open_dataset(path_file_SMrz, decode_times=False, engine='netcdf4')
sm = ds_sm[variable_sm]
sm = sm.assign_coords(
    time=('time', pd.to_datetime(np.asarray(ds_sm['time'].values).astype(str), format='%Y%m%d'))
)
lats_sm = np.asarray(ds_sm['lat'].values)
lons_sm = np.asarray(ds_sm['lon'].values)
lons_sm_plot = np.where(lons_sm > 180.0, lons_sm - 360.0, lons_sm)

# TS monthly anomalies (ERA5 US grid)
# variable_t = 'TS'
# path_file_TS = f'{path_outputs}monthly_anoma_{variable_t}_ERA5_US.nc'
# ds_ts = xr.open_dataset(path_file_TS, decode_times=False, engine='netcdf4')
# ts = ds_ts[variable_t]
# ts = ts.assign_coords(
#     time=('time', pd.to_datetime(np.asarray(ds_ts['time'].values).astype(str), format='%Y%m%d'))
# )
# lats_ts = np.asarray(ds_ts['lat'].values)
# lons_ts = np.asarray(ds_ts['lon'].values)
# lons_ts_plot = np.where(lons_ts > 180.0, lons_ts - 360.0, lons_ts)

variable_t = 'TS'
path_file_TS = f'/scratch/negishi/castanev/Amplification_RW/ERA5/TS_ERA5_daily_anoma_window.nc'
ds_ts = xr.open_dataset(path_file_TS, decode_times=False, engine='netcdf4')
ts = ds_ts[variable_t]
dates_d = np.array([
    dt.datetime(initial_year, 1, 1) + dt.timedelta(days=i)
    for i in range(len(ds_ts['time']))
])
ts = ts.assign_coords(time=('time', dates_d))
# Daily anomalies → one map per month (mean of days in that month)
ts = ts.resample(time='MS').mean('time')

# Crop to US using THIS file's lat/lon (0–360), same as heatwave detection
lats_ts_full = np.asarray(ds_ts['lat'].values)
lons_ts_full = np.asarray(ds_ts['lon'].values)
pos_lats_ts = np.where((lats_ts_full >= lat_minHW) & (lats_ts_full <= lat_maxHW))[0]
pos_lons_ts = np.where((lons_ts_full >= lon_minHW) & (lons_ts_full <= lon_maxHW))[0]
ts = ts.isel(lat=pos_lats_ts, lon=pos_lons_ts)
lats_ts = lats_ts_full[pos_lats_ts]
lons_ts = lons_ts_full[pos_lons_ts]
lons_ts_plot = np.where(lons_ts > 180.0, lons_ts - 360.0, lons_ts)
print(f'TS US crop: lat {lats_ts[0]}–{lats_ts[-1]} (n={len(lats_ts)}), '
      f'lon {lons_ts[0]}–{lons_ts[-1]} (n={len(lons_ts)})')



def month_composite(da, years, month):
    """Mean monthly anomaly field for `month` over the given years."""
    years = np.asarray(years, dtype=int)
    mask = (da['time'].dt.month == month) & np.isin(da['time'].dt.year, years)
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    return sel.mean('time').values.astype(float)


def plot_sm_ts_composites(fields, n_pos, n_weak, n_neg, eof_i, outfile,
                          vmax_sm=0.04, vmax_ts=1.5):
    """
    3 rows (pos/weak/neg) × 4 columns:
      col0 = May SMrz anomaly, col1–3 = Jun/Jul/Aug TS anomaly.
    fields: dict phase -> [May_SM, Jun_TS, Jul_TS, Aug_TS]
    """
    row_labels = [
        f'Positive (PC* > {pc_thresh:g}, n={n_pos})',
        f'Weak (|PC*| < {pc_thresh:g}, n={n_weak})',
        f'Negative (PC* < -{pc_thresh:g}, n={n_neg})',
    ]
    phases = ['pos', 'weak', 'neg']
    col_titles = ['May SMrz', 'June TS', 'July TS', 'August TS']
    levels_sm = np.linspace(-vmax_sm, vmax_sm, 21)
    levels_ts = np.linspace(-vmax_ts, vmax_ts, 21)

    fig = plt.figure(figsize=(14, 9))
    im_sm = im_ts = None
    for irow, (phase, row_title) in enumerate(zip(phases, row_labels)):
        for icol in range(4):
            ax = fig.add_subplot(3, 4, irow * 4 + icol + 1, projection=ccrs.PlateCarree())
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)

            if icol == 0:
                im_sm = ax.contourf(
                    lons_sm_plot, lats_sm, fields[phase][0],
                    levels=levels_sm, cmap='BrBG', extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_sm_plot, lats_sm
            else:
                im_ts = ax.contourf(
                    lons_ts_plot, lats_ts, fields[phase][icol],
                    levels=levels_ts, cmap='RdBu_r', extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_ts_plot, lats_ts

            ax.set_extent([lon_p.min(), lon_p.max(), lat_p.min(), lat_p.max()],
                          crs=ccrs.PlateCarree())
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = (irow == 2)
            gl.ylocator = mticker.FixedLocator([30, 40])
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}
            if irow == 0:
                ax.set_title(col_titles[icol], fontsize=11)
            if icol == 0:
                ax.text(
                    -0.22, 0.5, row_title, transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    fig.suptitle(
        f'May SM & summer TS composites | May {variable} EOF{eof_i + 1}',
        fontsize=13, y=0.98,
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.08, bottom=0.12, top=0.92, left=0.10, right=0.95)
    cax_sm = fig.add_axes([0.12, 0.04, 0.28, 0.015])
    cb_sm = fig.colorbar(im_sm, cax=cax_sm, orientation='horizontal')
    cb_sm.set_label(r'May SMrz anomaly [m$^{3}$ m$^{-3}$]', fontsize=9)
    cb_sm.ax.tick_params(labelsize=8)
    cax_ts = fig.add_axes([0.52, 0.04, 0.38, 0.015])
    cb_ts = fig.colorbar(im_ts, cax=cax_ts, orientation='horizontal')
    cb_ts.set_label(r'TS anomaly [°C]', fontsize=9)
    cb_ts.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def summer_hw_probability(years):
    """Mean fraction of JJA days in a tracked HW cluster, for the given years."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_summer.sel(time=np.isin(hw_summer['time'].dt.year, years))
    return (sel > 0).astype(float).mean('time').values, int(years.size)


def summer_hw_year_probability(years, month=None):
    """Probability a year has ≥1 HW day at each pixel (JJA, or one summer month).

    For each year in `years`: mark pixels with any HW day in the season/month,
    then average over years → P(heatwave occurs that summer/month).
    """
    years = np.asarray(years, dtype=int)
    if month is None:
        hw_sel = hw_summer
    else:
        hw_sel = hw_mask.sel(time=hw_mask['time'].dt.month == month)
    years = years[np.isin(years, hw_sel['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0

    annual = []
    for y in years:
        ymask = hw_sel.sel(time=hw_sel['time'].dt.year == y)
        # 1 if any HW day that year at the pixel, else 0
        annual.append((ymask.max('time') > 0).astype(float).values)
    return np.mean(annual, axis=0), int(years.size)


def plot_prob_panels(
    prob_pos, prob_weak, prob_neg, n_pos, n_weak, n_neg, eof_i, outfile,
    vmax=0.03, title=None, cbar_label='Fraction of JJA days in HW cluster',
):
    """Three vertical panels: positive / weak / negative May PC phase."""
    panels = [
        (prob_pos, f'Positive (PC* > {pc_thresh:g}, n={n_pos})'),
        (prob_weak, f'Weak (|PC*| < {pc_thresh:g}, n={n_weak})'),
        (prob_neg, f'Negative (PC* < -{pc_thresh:g}, n={n_neg})'),
    ]
    levels = np.linspace(0.0, vmax, 12)
    if title is None:
        title = f'Summer (JJA) HW-day probability | May {variable} EOF{eof_i + 1}'

    fig = plt.figure(figsize=(6.5, 11))
    last_im = None
    for i, (prob, panel_title) in enumerate(panels):
        ax = fig.add_subplot(3, 1, i + 1, projection=ccrs.PlateCarree())
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.5)
            ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        last_im = ax.contourf(
            lons_plot, lats, prob,
            levels=levels, cmap='YlOrRd', extend='max',
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent([lons_plot.min(), lons_plot.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.35, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        ax.set_title(panel_title, fontsize=11)

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(hspace=0.28, bottom=0.08, top=0.93, left=0.08, right=0.95)
    cax = fig.add_axes([0.18, 0.03, 0.64, 0.015])
    cb = fig.colorbar(last_im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_prob_panels_by_month(
    probs, n_pos, n_weak, n_neg, eof_i, outfile, vmax=0.5,
    title=None, cbar_label='P(≥1 HW day in month)',
):
    """3 rows (pos/weak/neg) × 3 columns (June/July/August)."""
    row_labels = [
        f'Positive (PC* > {pc_thresh:g}, n={n_pos})',
        f'Weak (|PC*| < {pc_thresh:g}, n={n_weak})',
        f'Negative (PC* < -{pc_thresh:g}, n={n_neg})',
    ]
    phases = ['pos', 'weak', 'neg']
    month_names = {6: 'June', 7: 'July', 8: 'August'}
    levels = np.linspace(0.0, vmax, 12)
    if title is None:
        title = f'HW occurrence probability by month | May {variable} EOF{eof_i + 1}'

    fig = plt.figure(figsize=(12, 10))
    last_im = None
    for irow, (phase, row_title) in enumerate(zip(phases, row_labels)):
        for icol, month in enumerate([6, 7, 8]):
            ax = fig.add_subplot(3, 3, irow * 3 + icol + 1, projection=ccrs.PlateCarree())
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)
            last_im = ax.contourf(
                lons_plot, lats, probs[phase][icol],
                levels=levels, cmap='YlOrRd', extend='max',
                transform=ccrs.PlateCarree(),
            )
            ax.set_extent(
                [lons_plot.min(), lons_plot.max(), lats.min(), lats.max()],
                crs=ccrs.PlateCarree(),
            )
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = (irow == 2)
            gl.ylocator = mticker.FixedLocator([30, 40])
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}
            if irow == 0:
                ax.set_title(month_names[month], fontsize=11)
            if icol == 0:
                ax.text(
                    -0.18, 0.5, row_title, transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.subplots_adjust(hspace=0.18, wspace=0.08, bottom=0.08, top=0.92, left=0.12, right=0.95)
    cax = fig.add_axes([0.25, 0.03, 0.5, 0.015])
    cb = fig.colorbar(last_im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


# Day-fraction of JJA (how often a summer day is in a HW) ============================================================
vmax_sm = 0.04
vmax_ts = 2

for EOF_i in range(eof_ts_std.shape[0]):
    pc = np.asarray(eof_ts_std[EOF_i].values)

    year_pos = years_eof[pc > pc_thresh]
    year_weak = years_eof[np.abs(pc) < pc_thresh]
    year_neg = years_eof[pc < -pc_thresh]

    print(
        f'EOF{EOF_i + 1}: n_pos={len(year_pos)}, n_weak={len(year_weak)}, '
        f'n_neg={len(year_neg)} (thresh={pc_thresh})'
    )
    print(f'  pos years: {year_pos.tolist()}')
    print(f'  neg years: {year_neg.tolist()}')

    prob_pos, n_pos = summer_hw_probability(year_pos)
    prob_weak, n_weak = summer_hw_probability(year_weak)
    prob_neg, n_neg = summer_hw_probability(year_neg)

    outfile = (
        f'{path_figures}HW_prob_JJA_MayEOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_prob_panels(
        prob_pos, prob_weak, prob_neg,
        n_pos, n_weak, n_neg,
        EOF_i, outfile,
        vmax=vmax,
    )

    # May SM + Jun/Jul/Aug TS composites for the same phase years
    fields = {}
    for phase, years in [('pos', year_pos), ('weak', year_weak), ('neg', year_neg)]:
        may_sm = month_composite(sm, years, 5)
        jun_ts = month_composite(ts, years, 6)
        jul_ts = month_composite(ts, years, 7)
        aug_ts = month_composite(ts, years, 8)
        fields[phase] = [may_sm, jun_ts, jul_ts, aug_ts]

    outfile_comp = (
        f'{path_figures}composite_MaySMrz_JJAdailyTS_monthlymean_MayEOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_sm_ts_composites(
        fields, n_pos, n_weak, n_neg, EOF_i, outfile_comp,
        vmax_sm=vmax_sm, vmax_ts=vmax_ts,
    )

