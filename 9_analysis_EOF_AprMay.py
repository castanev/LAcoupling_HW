from Functions import *
import argparse
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('--name_land', type=str, required=True)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--path_case', type=str, required=True)
parser.add_argument('--path_outputs', type=str, required=True)
parser.add_argument('--topography', type=bool, required=True)
parser.add_argument('--initial_year', type=int, required=True)
args = parser.parse_args()

name_land = args.name_land
case = args.case
path_case = args.path_case
path_outputs = args.path_outputs
topography = args.topography
initial_year = args.initial_year
# |PC*| threshold. 0.5 is mild (~1/3 of years each side if near-Gaussian).
# 0.75–1.0 gives clearer composites with fewer years.
pc_thresh = 0.75
vmax=0.04

lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat = 45  # US, 0–360°
path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

# EOF (daily Apr–May PCs) ==============================================================================================
variable = 'SMrz'
path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'{variable}_{name_land}_AprMay_US.nc'
)

ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
eof_patterns = ds_eof['n_eof']  # (neval, lat, lon); GLEAM grid
time_ymd = ds_eof['time_ymd']
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')

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
# Onset-only mask (1 in file); bool so .sum counts events
hw_onset = (hw_mask == 1)
hw_onset_summer = hw_onset.sel(time=(months >= 6) & (months <= 8))

# Annual HW event counts: each event contributes +1 on the union of its lifetime pixels
path_file_hw_events = (
    f'{path_outputs}{case}/HW_event_count_annual_JJA_ERA5_Teng.nc'
)
ds_hw_events = xr.open_dataset(path_file_hw_events, decode_times=False)
hw_event_count = ds_hw_events['hw_event_count']
# time coord is YYYY0101 strings from detection
_ev_years = pd.to_datetime(
    np.asarray(ds_hw_events['time'].values).astype(str), format='%Y%m%d'
).year
hw_event_count = hw_event_count.assign_coords(year=('time', _ev_years)).swap_dims({'time': 'year'})
hw_event_count = hw_event_count.sortby('year')
hw_event_years = np.asarray(hw_event_count['year'].values, dtype=int)
print(f'Loaded annual HW event counts: {path_file_hw_events} '
      f'({hw_event_years[0]}–{hw_event_years[-1]})')

# SMrz daily anomalies (smoothed; Apr–May), same as EOF input — for 15-day composites
variable_sm = 'SMrz'
path_file_SMrz = (
    f'{path_outputs}daily_anoma_window_smoothed15daysw_'
    f'{variable_sm}_{name_land}_AprMay_US.nc'
)
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

# Keep daily TS anomalies for HW-day footprint composites; monthly for Jun/Jul/Aug panels
ts_daily = ts
ts = ts_daily.resample(time='MS').mean('time')



def month_composite(da, years, month):
    """Mean monthly anomaly field for `month` over the given years.

    Only years present in `da` are used (intersection); prints a warning if any
    requested years are missing (e.g. TS ends before the EOF record).
    """
    years = np.asarray(years, dtype=int)
    years_avail = np.unique(da['time'].dt.year.values).astype(int)
    years_use = years[np.isin(years, years_avail)]
    missing = years[~np.isin(years, years_avail)]
    if missing.size:
        print(
            f'  month_composite({month:02d}): dropping years missing from field '
            f'→ {missing.tolist()} (using n={len(years_use)}/{len(years)})'
        )
    if years_use.size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    mask = (da['time'].dt.month == month) & da['time'].dt.year.isin(years_use)
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    return sel.mean('time').values.astype(float)


def dayrange_composite(da, years, month, day_lo, day_hi):
    """Mean daily anomaly over [day_lo, day_hi] of `month` for the given years."""
    years = np.asarray(years, dtype=int)
    years_avail = np.unique(da['time'].dt.year.values).astype(int)
    years_use = years[np.isin(years, years_avail)]
    missing = years[~np.isin(years, years_avail)]
    if missing.size:
        print(
            f'  dayrange_composite({month:02d} {day_lo}-{day_hi}): dropping years '
            f'missing from field → {missing.tolist()} '
            f'(using n={len(years_use)}/{len(years)})'
        )
    if years_use.size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    t = da['time']
    mask = (
        (t.dt.month == month)
        & (t.dt.day >= day_lo)
        & (t.dt.day <= day_hi)
        & t.dt.year.isin(years_use)
    )
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    return sel.mean('time').values.astype(float)


def period_mean_pc(pc_daily, dates, month, day_lo, day_hi):
    """One mean PC value per year over the given calendar day window.

    Returns (pc_year, years) with matching length, years sorted ascending.
    """
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


def plot_sm_ts_composites(fields, n_pos, n_weak, n_neg, eof_i, outfile,
                          vmax_sm=0.04, vmax_ts=1.5, period_label='May'):
    """
    3 rows (pos/weak/neg) × 4 columns:
      col0 = period SMrz anomaly, col1–3 = Jun/Jul/Aug TS anomaly.
    fields: dict phase -> [period_SM, Jun_TS, Jul_TS, Aug_TS]
    """
    row_labels = [
        f'Positive (PC* > {pc_thresh:g}, n={n_pos})',
        f'Weak (|PC*| < {pc_thresh:g}, n={n_weak})',
        f'Negative (PC* < -{pc_thresh:g}, n={n_neg})',
    ]
    phases = ['pos', 'weak', 'neg']
    col_titles = [f'{period_label} SMrz', 'June TS', 'July TS', 'August TS']
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
        f'{period_label} SM & summer TS composites | Apr–May {variable} EOF{eof_i + 1}',
        fontsize=13, y=0.98,
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.08, bottom=0.12, top=0.92, left=0.10, right=0.95)
    cax_sm = fig.add_axes([0.12, 0.04, 0.28, 0.015])
    cb_sm = fig.colorbar(im_sm, cax=cax_sm, orientation='horizontal')
    cb_sm.set_label(rf'{period_label} SMrz anomaly [m$^{{3}}$ m$^{{-3}}$]', fontsize=9)
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
    # Mask is 0 / 1(onset) / 2(later); any HW day counts as 1 for the fraction
    return (sel > 0).astype(float).mean('time').values, int(years.size)


def summer_onset_probability(years):
    """Mean fraction of JJA days that are HW onset (mask==1), for the given years."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_onset_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_onset_summer.sel(time=np.isin(hw_onset_summer['time'].dt.year, years))
    return sel.astype(float).mean('time').values, int(years.size)


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


def mean_onset_count(years, month=None):
    """Mean number of HW onsets (mask==1) per year at each pixel.

    Counts onset days in JJA (month=None) or in a single calendar month,
    then averages those annual counts over `years`.
    """
    years = np.asarray(years, dtype=int)
    if month is None:
        onset_sel = hw_onset_summer
    else:
        onset_sel = hw_onset.sel(time=hw_onset['time'].dt.month == month)
    years = years[np.isin(years, onset_sel['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0

    annual = []
    for y in years:
        ymask = onset_sel.sel(time=onset_sel['time'].dt.year == y)
        annual.append(ymask.sum('time').values.astype(float))
    return np.mean(annual, axis=0), int(years.size)


def mean_event_count(years):
    """Mean number of HW events per summer at each pixel (lifetime-union counts)."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_event_years)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_event_count.sel(year=years)
    return sel.mean('year').values.astype(float), int(years.size)


def total_event_count(years):
    """Total number of HW events over the selected years at each pixel."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_event_years)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_event_count.sel(year=years)
    return sel.sum('year').values.astype(float), int(years.size)


def _annual_jja_day_fraction(da_bool):
    """Per-year mean of a boolean JJA daily field → (nyear, lat, lon) and year list."""
    years = np.unique(da_bool['time'].dt.year.values).astype(int)
    stacks = []
    for y in years:
        sel = da_bool.sel(time=da_bool['time'].dt.year == y)
        stacks.append(sel.astype(float).mean('time').values)
    return years, np.stack(stacks, axis=0)


def ttest_ind_pvalue_maps(annual, years_all, years_a, years_b):
    """Welch t-test p-value map: mean(years_a) vs mean(years_b) along years (independent).

    annual: (nyear, lat, lon); years_all aligned with axis 0.
    Only years present in `years_all` are used (intersection with annual record).
    """
    years_all = np.asarray(years_all, dtype=int)
    years_a = np.asarray(years_a, dtype=int)
    years_b = np.asarray(years_b, dtype=int)
    # Restrict to years that exist in the annual array
    years_a = years_a[np.isin(years_a, years_all)]
    years_b = years_b[np.isin(years_b, years_all)]
    ia = np.isin(years_all, years_a)
    ib = np.isin(years_all, years_b)
    a = np.asarray(annual, dtype=float)[ia]
    b = np.asarray(annual, dtype=float)[ib]
    shape = annual.shape[1:]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.full(shape, np.nan, dtype=float)
    # equal_var=False → Welch; independent samples (phase vs rest)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='omit')
    p = np.asarray(res.pvalue, dtype=float)
    # Zero variance in both samples with equal means → nan; leave as nan (not significant)
    return p


def ttest_1samp_pvalue_maps(annual, years_all, years_a, popmean):
    """One-sample t-test p-value map: mean(years_a) vs climatological mean.

    Tests whether the phase-year mean differs from `popmean` (clim) at each
    grid point. Note: phase years are part of the clim sample, so this is not
    a fully independent two-sample test (prefer vs-rest for that).
    """
    years_all = np.asarray(years_all, dtype=int)
    years_a = np.asarray(years_a, dtype=int)
    years_a = years_a[np.isin(years_a, years_all)]
    ia = np.isin(years_all, years_a)
    a = np.asarray(annual, dtype=float)[ia]
    shape = annual.shape[1:]
    if a.shape[0] < 2:
        return np.full(shape, np.nan, dtype=float)
    popmean = np.asarray(popmean, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_1samp(a, popmean=popmean, axis=0, nan_policy='omit')
    return np.asarray(res.pvalue, dtype=float)


def significance_mask(pvalue, alpha=0.05):
    """1 where significant (p ≤ alpha), else NaN (for hatch overlay)."""
    p = np.asarray(pvalue, dtype=float)
    sig = np.isfinite(p) & (p <= alpha)
    return np.where(sig, 1.0, np.nan)


def plot_clim_map(field, outfile, title, cbar_label, vmax, cmap='YlOrRd',
                  levels=None, extend='max'):
    """Single-panel US map (e.g. climatological mean HW events/year)."""
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


def plot_eof_hw_by_eof(eof_maps, hw_fields, ns, outfile, period_label,
                       vmax_eof=None, vmax_hw=0.04, vmin_hw=0.0, n_eof=4,
                       hw_cmap='YlOrRd', hw_center=None, hw_extend='max',
                       cbar_label='Fraction of JJA days in HW cluster',
                       title=None, sig_fields=None, sig_alpha=0.05):
    """
    4 rows × n_eof columns:
      row0 = spatial EOF pattern for that column,
      rows1–3 = HW field for pos / weak / neg (that EOF's years).

    eof_maps: list of n_eof 2D arrays (lat, lon) on the GLEAM grid
    hw_fields: phase -> list of n_eof maps (absolute prob, ratio, or anomaly)
    ns: phase -> list of n_eof year counts
    hw_center: if set (e.g. 1 for ratio, 0 for anomaly), use diverging levels
               symmetric about that center with half-range vmax_hw.
    sig_fields: optional phase -> list of n_eof p-value maps; hatch where p ≤ sig_alpha
    """
    phases = ['pos', 'weak', 'neg']
    row_labels = [
        'EOF pattern',
        f'Positive (PC* > {pc_thresh:g})',
        f'Weak (|PC*| < {pc_thresh:g})',
        f'Negative (PC* < -{pc_thresh:g})',
    ]
    col_titles = [f'EOF{i + 1}' for i in range(n_eof)]
    if vmax_eof is None:
        vmax_eof = float(np.nanpercentile(np.abs(np.asarray(eof_maps)), 98))
        if not np.isfinite(vmax_eof) or vmax_eof == 0:
            vmax_eof = 0.01
    levels_eof = np.linspace(-vmax_eof, vmax_eof, 21)
    if hw_center is None:
        levels_hw = np.linspace(vmin_hw, vmax_hw, 12)
    else:
        half = vmax_hw  # half-range about center
        levels_hw = np.linspace(hw_center - half, hw_center + half, 21)
    nrows = 4
    ncols = n_eof

    fig = plt.figure(figsize=(3.2 * ncols, 2.6 * nrows))
    im_eof = im_hw = None
    for irow in range(nrows):
        for icol in range(ncols):
            ax = fig.add_subplot(nrows, ncols, irow * ncols + icol + 1, projection=ccrs.PlateCarree())
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)

            if irow == 0:
                im_eof = ax.contourf(
                    lons_sm_plot, lats_sm, eof_maps[icol],
                    levels=levels_eof, cmap='BrBG', extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_sm_plot, lats_sm
            else:
                phase = phases[irow - 1]
                im_hw = ax.contourf(
                    lons_plot, lats, hw_fields[phase][icol],
                    levels=levels_hw, cmap=hw_cmap, extend=hw_extend,
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_plot, lats
                if sig_fields is not None:
                    # Hatch = statistically significant (p ≤ sig_alpha)
                    sig01 = significance_mask(sig_fields[phase][icol], alpha=sig_alpha)
                    mpl.rcParams['hatch.linewidth'] = 0.5
                    cs_hatch = ax.contourf(
                        lons_plot, lats, sig01,
                        levels=[0.5, 1.5],
                        colors='none',
                        hatches=['///'],
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                    # Cartopy often needs explicit edge color or hatches are invisible
                    for coll in getattr(cs_hatch, 'collections', []):
                        coll.set_edgecolor('k')
                        coll.set_linewidth(0.45)
                        coll.set_facecolor('none')
                ax.text(
                    0.03, 0.04, f'n={ns[phase][icol]}', transform=ax.transAxes,
                    fontsize=7, color='k',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, lw=0),
                )

            ax.set_extent([lon_p.min(), lon_p.max(), lat_p.min(), lat_p.max()],
                          crs=ccrs.PlateCarree())
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = (irow == nrows - 1)
            gl.ylocator = mticker.FixedLocator([30, 40])
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            if irow == 0:
                ax.set_title(col_titles[icol], fontsize=11)
            if icol == 0:
                ax.text(
                    -0.28, 0.5, row_labels[irow], transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    if title is None:
        title = (
            f'JJA HW-day probability | {period_label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof}'
        )
    if sig_fields is not None:
        title = f'{title} | hatch: significant (p≤{sig_alpha:g})'
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(hspace=0.18, wspace=0.08, bottom=0.10, top=0.92, left=0.08, right=0.95)
    cax_eof = fig.add_axes([0.10, 0.035, 0.28, 0.012])
    cb_eof = fig.colorbar(im_eof, cax=cax_eof, orientation='horizontal')
    cb_eof.set_label(r'EOF loading (SMrz)', fontsize=9)
    cb_eof.ax.tick_params(labelsize=8)
    cax_hw = fig.add_axes([0.48, 0.035, 0.40, 0.012])
    cb_hw = fig.colorbar(im_hw, cax=cax_hw, orientation='horizontal')
    cb_hw.set_label(cbar_label, fontsize=9)
    cb_hw.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def _safe_ratio(cond, clim, clim_min=1e-6):
    """P(cond) / P(clim); NaN where climatology is ~0."""
    out = np.full_like(cond, np.nan, dtype=float)
    ok = np.isfinite(cond) & np.isfinite(clim) & (clim > clim_min)
    out[ok] = cond[ok] / clim[ok]
    return out


def _anomaly(cond, clim):
    """P(cond) − P(clim)."""
    return cond.astype(float) - clim.astype(float)


def _relative_anomaly(cond, baseline, baseline_min=1e-6):
    """(cond − baseline) / baseline; NaN where baseline is ~0."""
    return _safe_ratio(_anomaly(cond, baseline), baseline, clim_min=baseline_min)


# Condition on 15-day mean PC (Apr/May); EOF patterns + HW for EOF1–4; summer TS composites ====
vmax_sm = 0.04
vmax_ts = 2
vmax_hw = vmax
# Ratio: center 1, show ~0.5–1.5 (half-range 0.5). Diff: center 0, ± ~±0.02
vmax_hw_ratio = 4
vmax_hw_diff = 0.03
n_eof_plot = min(4, int(eof_ts.shape[0]))  # EOF1–4 only
eof_maps = [np.asarray(eof_patterns[i].values) for i in range(n_eof_plot)]

# Climatological JJA HW-day fraction (all years); mask may be 0/1/2
prob_clim = (hw_summer > 0).astype(float).mean('time').values

# Per-year JJA fields for independent-sample t-tests (phase vs rest)
years_hw_day, annual_hw_day = _annual_jja_day_fraction(hw_summer > 0)
years_onset_day, annual_onset_day = _annual_jja_day_fraction(hw_onset_summer)
# Event counts already annual
annual_events = np.asarray(hw_event_count.values, dtype=float)
# Climatological means consistent with annual samples (for one-sample t-tests)
clim_hw_day_annual = np.nanmean(annual_hw_day, axis=0)
clim_onset_day_annual = np.nanmean(annual_onset_day, axis=0)
clim_events_annual = np.nanmean(annual_events, axis=0)
print(f'Annual maps ready for t-tests: HW-day {years_hw_day[0]}–{years_hw_day[-1]}, '
      f'events {hw_event_years[0]}–{hw_event_years[-1]}')

# Climatological mean JJA onset counts (per year)
all_years_hw = np.unique(hw_mask['time'].dt.year.values)
onset_clim_jja, _ = mean_onset_count(all_years_hw, month=None)

vmax_onset_diff_jja = 0.5
vmax_onset_ratio = 4  # half-range about 1 → show 0–2
vmax_hw_vs_rest = 0.03
vmax_hw_ratio_vs_rest = 4  # half-range about 1
vmax_onset_vs_rest = 0.008
vmax_events = 0.5
vmax_events_diff = 0.5
vmax_events_ratio = 1.0  # half-range about 1
vmax_events_vs_rest = 0.6
vmax_events_rel_vs_rest = 4.0  # half-range about 0 → show −1…+1

# Mean TS anomaly with equal weight per heatwave event (not per HW day).
# For each event: mean T' over days when that pixel is in the event footprint;
# then average those event means over all events that affect each pixel.
hw_csv = f'{path_case}/Heat_waves_events_list.csv'
df_hw_events = pd.read_csv(hw_csv, index_col=0)
df_hw_events['Date 0'] = pd.to_datetime(df_hw_events['Date 0'])

# Align daily TS to the HW lat/lon grid
ts_on_hw = ts_daily
if (
    len(ts_on_hw['lat']) != len(hw_mask['lat'])
    or len(ts_on_hw['lon']) != len(hw_mask['lon'])
    or not np.allclose(ts_on_hw['lat'].values, hw_mask['lat'].values)
    or not np.allclose(ts_on_hw['lon'].values, hw_mask['lon'].values)
):
    ts_on_hw = ts_on_hw.interp(lat=hw_mask['lat'], lon=hw_mask['lon'], method='nearest')

time_ts = pd.DatetimeIndex(ts_on_hw['time'].values).normalize()
time_hw_full = pd.DatetimeIndex(hw_mask['time'].values).normalize()
date_to_its = {d: i for i, d in enumerate(time_ts)}
date_to_ihw = {d: i for i, d in enumerate(time_hw_full)}

ts_vals = np.asarray(ts_on_hw.values, dtype=float)
hw_vals = np.asarray(hw_mask.values)
nlat_hw, nlon_hw = hw_vals.shape[1], hw_vals.shape[2]
sum_event_means = np.zeros((nlat_hw, nlon_hw), dtype=float)
n_events_pix = np.zeros((nlat_hw, nlon_hw), dtype=float)
n_events_used = 0

for _, row in df_hw_events.iterrows():
    onset = pd.Timestamp(row['Date 0']).normalize()
    dur = int(row['Duration'])
    its_list, ihw_list = [], []
    for k in range(dur):
        d = onset + pd.Timedelta(days=k)
        if d in date_to_its and d in date_to_ihw:
            its_list.append(date_to_its[d])
            ihw_list.append(date_to_ihw[d])
    if not its_list:
        continue

    ts_e = ts_vals[np.asarray(its_list)]
    hw_e = hw_vals[np.asarray(ihw_list)] > 0
    n_days = hw_e.sum(axis=0).astype(float)
    affected = n_days > 0
    if not np.any(affected):
        continue

    # Per-pixel mean over this event's HW days only
    with np.errstate(invalid='ignore'):
        event_mean = np.nansum(np.where(hw_e, ts_e, np.nan), axis=0) / n_days
    sum_event_means[affected] += event_mean[affected]
    n_events_pix[affected] += 1.0
    n_events_used += 1

with np.errstate(invalid='ignore'):
    ts_hw_mean = np.where(n_events_pix > 0, sum_event_means / n_events_pix, np.nan)

vmax_ts_hw = float(np.nanpercentile(np.abs(ts_hw_mean), 99))
if not np.isfinite(vmax_ts_hw) or vmax_ts_hw <= 0:
    vmax_ts_hw = 2.0
outfile_ts_hw_mean = (
    f'{path_figures}TS_mean_on_HWevents_equalweight_{variable}_{name_land}.png'
)
plot_clim_map(
    ts_hw_mean, outfile_ts_hw_mean,
    title='Mean TS anomaly on HW events (equal weight per event)',
    cbar_label='Temperature anomaly (°C)',
    vmax=vmax_ts_hw,
    cmap='RdBu_r',
    levels=np.linspace(-vmax_ts_hw, vmax_ts_hw, 13),
    extend='both',
)
print(
    f'TS equal-event mean: n_events={n_events_used}/{len(df_hw_events)}, '
    f'pixels with ≥1 event={(n_events_pix > 0).sum()}, '
    f'range {np.nanmin(ts_hw_mean):.2f} to {np.nanmax(ts_hw_mean):.2f} °C'
)


aaaaa

# Total HW days over the full record (sum of JJA days with mask>0 at each pixel)
hw_days_total = np.asarray((hw_summer > 0).sum('time').values, dtype=float)
vmax_hw_days_total = float(np.nanpercentile(hw_days_total, 99))
if not np.isfinite(vmax_hw_days_total) or vmax_hw_days_total <= 0:
    vmax_hw_days_total = 10.0
outfile_hw_days_total = (
    f'{path_figures}HW_days_total_JJA_{variable}_{name_land}.png'
)
plot_clim_map(
    hw_days_total, outfile_hw_days_total,
    title='Total HW days over the full record',
    cbar_label='Number of HW days',
    vmax=vmax_hw_days_total,
)



# Climatological mean HW events per summer (lifetime-union counts)
events_clim, _ = mean_event_count(hw_event_years)
outfile_events_clim = (
    f'{path_figures}HW_events_clim_JJA_{variable}_{name_land}.png'
)
plot_clim_map(
    events_clim, outfile_events_clim,
    title='Climatological mean HW events per summer',
    cbar_label='Mean number of heatwaves per year',
    vmax=vmax_events,
)

# Total events over the full record (sum of annual counts at each pixel)
events_total = np.asarray(hw_event_count.sum('year').values, dtype=float)
vmax_events_total = float(np.nanpercentile(events_total, 99))
if not np.isfinite(vmax_events_total) or vmax_events_total <= 0:
    vmax_events_total = 10.0
outfile_events_total = (
    f'{path_figures}HW_events_total_JJA_{variable}_{name_land}.png'
)
plot_clim_map(
    events_total, outfile_events_total,
    title='Total HW events affecting each pixel (full record)',
    cbar_label='Number of heatwave events',
    vmax=vmax_events_total,
)

# (label, month, day_lo, day_hi)
periods = [
    ('earlyApr', 4, 1, 15),
    ('lateApr', 4, 16, 30),
    ('earlyMay', 5, 1, 15),
    ('lateMay', 5, 16, 31),
]
phases = ['pos', 'weak', 'neg']

for label, month, day_lo, day_hi in periods:
    hw_fields = {p: [] for p in phases}
    hw_ratio = {p: [] for p in phases}
    hw_diff = {p: [] for p in phases}
    hw_vs_rest = {p: [] for p in phases}
    hw_ratio_vs_rest = {p: [] for p in phases}
    hw_sig_vs_rest = {p: [] for p in phases}
    hw_sig_vs_clim = {p: [] for p in phases}
    ns = {p: [] for p in phases}
    onset_ratio = {p: [] for p in phases}
    onset_diff = {p: [] for p in phases}
    onset_vs_rest = {p: [] for p in phases}
    onset_sig_vs_rest = {p: [] for p in phases}
    onset_sig_vs_clim = {p: [] for p in phases}
    events_fields = {p: [] for p in phases}
    events_total_fields = {p: [] for p in phases}
    events_anom = {p: [] for p in phases}
    events_ratio = {p: [] for p in phases}
    events_vs_rest = {p: [] for p in phases}
    events_rel_vs_rest = {p: [] for p in phases}
    events_sig_vs_rest = {p: [] for p in phases}
    events_sig_vs_clim = {p: [] for p in phases}

    for EOF_i in range(n_eof_plot):
        pc_daily = np.asarray(eof_ts[EOF_i].values)
        pc_yr, years_yr = period_mean_pc(pc_daily, time_eof_ts, month, day_lo, day_hi)
        pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()

        year_pos = years_yr[pc_std > pc_thresh]
        year_weak = years_yr[np.abs(pc_std) < pc_thresh]
        year_neg = years_yr[pc_std < -pc_thresh]
        years_by_phase = {'pos': year_pos, 'weak': year_weak, 'neg': year_neg}

        print(
            f'EOF{EOF_i + 1} {label}: n_pos={len(year_pos)}, n_weak={len(year_weak)}, '
            f'n_neg={len(year_neg)} (thresh={pc_thresh})'
        )
        print(f'  pos years: {year_pos.tolist()}')
        print(f'  neg years: {year_neg.tolist()}')

        for phase, years in years_by_phase.items():
            rest = years_yr[~np.isin(years_yr, years)]

            prob, n = summer_hw_probability(years)
            prob_rest, _ = summer_hw_probability(rest)
            hw_fields[phase].append(prob)
            hw_ratio[phase].append(_safe_ratio(prob, prob_clim))
            hw_diff[phase].append(_anomaly(prob, prob_clim))
            hw_vs_rest[phase].append(_anomaly(prob, prob_rest))
            hw_ratio_vs_rest[phase].append(_safe_ratio(prob, prob_rest))
            hw_sig_vs_rest[phase].append(
                ttest_ind_pvalue_maps(annual_hw_day, years_hw_day, years, rest)
            )
            hw_sig_vs_clim[phase].append(
                ttest_1samp_pvalue_maps(
                    annual_hw_day, years_hw_day, years, clim_hw_day_annual
                )
            )
            ns[phase].append(n)

            onset_mean, _ = mean_onset_count(years, month=None)
            onset_ratio[phase].append(_safe_ratio(onset_mean, onset_clim_jja))
            onset_diff[phase].append(_anomaly(onset_mean, onset_clim_jja))
            onset_sig_vs_clim[phase].append(
                ttest_1samp_pvalue_maps(
                    annual_onset_day, years_onset_day, years, clim_onset_day_annual
                )
            )

            onset_p, _ = summer_onset_probability(years)
            onset_rest, _ = summer_onset_probability(rest)
            onset_vs_rest[phase].append(_anomaly(onset_p, onset_rest))
            onset_sig_vs_rest[phase].append(
                ttest_ind_pvalue_maps(annual_onset_day, years_onset_day, years, rest)
            )

            ev_mean, _ = mean_event_count(years)
            ev_total, _ = total_event_count(years)
            ev_rest, _ = mean_event_count(rest)
            events_fields[phase].append(ev_mean)
            events_total_fields[phase].append(ev_total)
            events_anom[phase].append(_anomaly(ev_mean, events_clim))
            events_ratio[phase].append(_safe_ratio(ev_mean, events_clim))
            events_vs_rest[phase].append(_anomaly(ev_mean, ev_rest))
            events_rel_vs_rest[phase].append(_relative_anomaly(ev_mean, ev_rest))
            events_sig_vs_rest[phase].append(
                ttest_ind_pvalue_maps(annual_events, hw_event_years, years, rest)
            )
            events_sig_vs_clim[phase].append(
                ttest_1samp_pvalue_maps(
                    annual_events, hw_event_years, years, clim_events_annual
                )
            )

        # Per-EOF period SM + Jun/Jul/Aug TS composites
        # Intersect phase years with both SM and TS availability so both columns
        # use the same year set (TS record ends earlier than GLEAM EOF/SM).
        years_sm = np.unique(sm['time'].dt.year.values).astype(int)
        years_ts = np.unique(ts['time'].dt.year.values).astype(int)
        years_both = np.intersect1d(years_sm, years_ts)

        fields = {}
        for phase, years in years_by_phase.items():
            years_c = np.intersect1d(np.asarray(years, dtype=int), years_both)
            dropped = np.setdiff1d(np.asarray(years, dtype=int), years_c)
            if dropped.size:
                print(
                    f'  composite EOF{EOF_i + 1} {phase}: using n={len(years_c)}/'
                    f'{len(years)} years (dropped {dropped.tolist()} — not in SM∩TS)'
                )
            period_sm = dayrange_composite(sm, years_c, month, day_lo, day_hi)
            jun_ts = month_composite(ts, years_c, 6)
            jul_ts = month_composite(ts, years_c, 7)
            aug_ts = month_composite(ts, years_c, 8)
            fields[phase] = [period_sm, jun_ts, jul_ts, aug_ts]

        outfile_comp = (
            f'{path_figures}composite_{label}SMrz_JJATS_smoothed15daysw_AprMayEOF{EOF_i + 1}_'
            f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
        )
        plot_sm_ts_composites(
            fields, ns['pos'][-1], ns['weak'][-1], ns['neg'][-1], EOF_i, outfile_comp,
            vmax_sm=vmax_sm, vmax_ts=vmax_ts, period_label=label,
        )
        

    # Absolute conditional probability (original)
    outfile = (
        f'{path_figures}HW_prob_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, hw_fields, ns, outfile, label,
        vmax_hw=vmax_hw, n_eof=n_eof_plot,
    )



    # Difference vs climatology: P(HW|phase) − P(HW|clim)
    outfile_diff = (
        f'{path_figures}HW_prob_anom_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, hw_diff, ns, outfile_diff, label,
        vmax_hw=vmax_hw_diff, n_eof=n_eof_plot,
        hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
        cbar_label=r'P(HW|phase) − P(HW|clim) [day fraction]',
        title=(
            f'JJA HW-day probability anomaly | {label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof_plot}'
        ),
        # sig_fields=hw_sig_vs_clim,
    )


    # outfile_on_diff = (
    #     f'{path_figures}HW_onset_anom_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    #     f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    # )
    # plot_eof_hw_by_eof(
    #     eof_maps, onset_diff, ns, outfile_on_diff, label,
    #     vmax_hw=vmax_onset_diff_jja, n_eof=n_eof_plot,
    #     hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
    #     cbar_label=r'N(onset|phase) − N(onset|clim) [per year]',
    #     title=(
    #         f'JJA HW onset count anomaly | {label} mean PC | '
    #         f'Apr–May {variable} EOF1–{n_eof_plot}'
    #     ),
    #     sig_fields=onset_sig_vs_clim,
    # )

    # vs rest of years (complement of phase): P|phase − P|rest
    outfile_hw_rest = (
        f'{path_figures}HW_prob_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, hw_vs_rest, ns, outfile_hw_rest, label,
        vmax_hw=vmax_hw_vs_rest, n_eof=n_eof_plot,
        hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
        cbar_label=r'P(HW day|phase) − P(HW day|rest)',
        title=(
            f'JJA HW-day prob vs rest of years | {label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof_plot}'
        ),
        # sig_fields=hw_sig_vs_rest,
    )


    # outfile_onset_rest = (
    #     f'{path_figures}HW_onset_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    #     f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    # )
    # plot_eof_hw_by_eof(
    #     eof_maps, onset_vs_rest, ns, outfile_onset_rest, label,
    #     vmax_hw=vmax_onset_vs_rest, n_eof=n_eof_plot,
    #     hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
    #     cbar_label=r'P(HW onset|phase) − P(HW onset|rest)',
    #     title=(
    #         f'JJA HW-onset prob vs rest of years | {label} mean PC | '
    #         f'Apr–May {variable} EOF1–{n_eof_plot}'
    #     ),
    #     sig_fields=onset_sig_vs_rest,
    # )

    # Mean HW events/year (lifetime-union counts) under EOF phase vs climatology
    outfile_events = (
        f'{path_figures}HW_events_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, events_fields, ns, outfile_events, label,
        vmax_hw=vmax_events, n_eof=n_eof_plot,
        cbar_label='Mean HW events per summer',
        title=(
            f'JJA HW events/year | {label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof_plot}'
        ),
    )



    outfile_events_anom = (
        f'{path_figures}HW_events_anom_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, events_anom, ns, outfile_events_anom, label,
        vmax_hw=vmax_events_diff, n_eof=n_eof_plot,
        hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
        cbar_label=r'N(events|phase) − N(events|clim) [per year]',
        title=(
            f'JJA HW events/year anomaly | {label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof_plot}'
        ),
        # sig_fields=events_sig_vs_clim,
    )

    # vs rest of years: N|phase − N|rest and relative (N|phase − N|rest) / N|rest
    outfile_events_rest = (
        f'{path_figures}HW_events_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_eof_hw_by_eof(
        eof_maps, events_vs_rest, ns, outfile_events_rest, label,
        vmax_hw=vmax_events_vs_rest, n_eof=n_eof_plot,
        hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
        cbar_label=r'N(events|phase) − N(events|rest) [per year]',
        title=(
            f'JJA HW events/year vs rest of years | {label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof_plot}'
        ),
        # sig_fields=events_sig_vs_rest,
    )

    # outfile_events_rel_rest = (
    #     f'{path_figures}HW_events_rel_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    #     f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    # )
    # plot_eof_hw_by_eof(
    #     eof_maps, events_rel_vs_rest, ns, outfile_events_rel_rest, label,
    #     vmax_hw=vmax_events_rel_vs_rest, n_eof=n_eof_plot,
    #     hw_cmap='RdBu_r', hw_center=0.0, hw_extend='both',
    #     cbar_label=r'(N|phase − N|rest) / N|rest',
    #     title=(
    #         f'JJA HW events/year relative vs rest | {label} mean PC | '
    #         f'Apr–May {variable} EOF1–{n_eof_plot}'
    #     ),
    #     sig_fields=events_sig_vs_rest,
    # )

