from Functions import *
import argparse
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

# |PC*| threshold
pc_thresh = 0.75

# Late-April 15-day window for phase definition (same as 9_analysis_EOF_AprMay.py)
eof_period_label = 'lateApr'
eof_sel_month, eof_day_lo, eof_day_hi = 4, 16, 30

path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

# EOF (Apr–May daily PCs) ==============================================================================================
variable = 'SMrz'
path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'{variable}_{name_land}_AprMay_US.nc'
)

ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
time_ymd = ds_eof['time_ymd']
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')


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


# Regime map + critical SM (same as 4_analysis_regimes_LE_allHWdays.py) ===============================================
miss_bp = -9.99e08
ncfilenm = f'{path_outputs}/map_wet_regime_fraction_{name_land}_US.nc'
ds_reg = xr.open_dataset(ncfilenm)


def _squeeze2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return np.squeeze(arr)


bp_rhs_2seg = np.where(
    np.isclose(_squeeze2d(ds_reg['BPx_2Seg_RHSflat']), miss_bp),
    np.nan,
    _squeeze2d(ds_reg['BPx_2Seg_RHSflat']),
)
bp_rhs_3seg = np.where(
    np.isclose(_squeeze2d(ds_reg['BPx2_3Seg']), miss_bp),
    np.nan,
    _squeeze2d(ds_reg['BPx2_3Seg']),
)
regime_field = _squeeze2d(ds_reg['regime_map']).astype(int)

# Critical SM only defined for moisture-limited regimes 3 and 4
critical_sm_bp = np.full(regime_field.shape, np.nan, dtype=float)
critical_sm_bp[regime_field == 3] = bp_rhs_2seg[regime_field == 3]
critical_sm_bp[regime_field == 4] = bp_rhs_3seg[regime_field == 4]
regime34 = (regime_field == 3) | (regime_field == 4)

lats_sm = np.asarray(ds_reg['lat'].values)
lons_sm = np.asarray(ds_reg['lon'].values)
lons_sm_plot = np.where(lons_sm > 180.0, lons_sm - 360.0, lons_sm)

crit_da = xr.DataArray(
    critical_sm_bp, dims=('lat', 'lon'),
    coords={'lat': lats_sm, 'lon': lons_sm},
)
regime34_da = xr.DataArray(
    regime34, dims=('lat', 'lon'),
    coords={'lat': lats_sm, 'lon': lons_sm},
)

# Daily absolute SMrz (compare to critical SM; NOT anomalies) ==========================================================
path_file_SMrz_abs = f'{path_outputs}SMrz_{name_land}_US.nc'
ds_sm_abs = xr.open_dataset(
    path_file_SMrz_abs, decode_times=False, engine='netcdf4', chunks={'time': 365}
)
sm_abs = ds_sm_abs[variable]
sm_abs = sm_abs.assign_coords(
    time=('time', pd.to_datetime(np.asarray(ds_sm_abs['time'].values).astype(str), format='%Y%m%d'))
)
# Summer days only
sm_jja = sm_abs.sel(time=(sm_abs['time'].dt.month >= 6) & (sm_abs['time'].dt.month <= 8))


def p_sm_below_critical(years):
    """Fraction of JJA days with SM < critical SM (regimes 3/4 only; else NaN)."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, sm_jja['time'].dt.year.values)]
    if years.size == 0:
        return np.full(regime_field.shape, np.nan, dtype=float), 0
    sel = sm_jja.sel(time=np.isin(sm_jja['time'].dt.year, years))
    below = (sel < crit_da).astype('float32')
    below = below.where(regime34_da)
    return below.mean('time').values.astype(float), int(years.size)


def plot_p_below_critical(p_pos, p_weak, p_neg, n_pos, n_weak, n_neg, eof_i, outfile):
    """3 panels: P(SM < critical) in JJA for pos / weak / neg lateApr PC years."""
    panels = [
        (p_pos, f'Positive (PC* > {pc_thresh:g}, n={n_pos})'),
        (p_weak, f'Weak (|PC*| < {pc_thresh:g}, n={n_weak})'),
        (p_neg, f'Negative (PC* < -{pc_thresh:g}, n={n_neg})'),
    ]
    levels = np.linspace(0.0, 1.0, 21)

    fig = plt.figure(figsize=(6.5, 11))
    last_im = None
    for i, (prob, panel_title) in enumerate(panels):
        ax = fig.add_subplot(3, 1, i + 1, projection=ccrs.PlateCarree())
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.5)
            ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        last_im = ax.contourf(
            lons_sm_plot, lats_sm, prob,
            levels=levels, cmap='YlOrBr', extend='neither',
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent(
            [lons_sm_plot.min(), lons_sm_plot.max(), lats_sm.min(), lats_sm.max()],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.35, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        ax.set_title(panel_title, fontsize=11)

    fig.suptitle(
        f'P(SM < critical) in JJA | regimes 3–4 only | '
        f'{eof_period_label} {variable} EOF{eof_i + 1}',
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(hspace=0.28, bottom=0.08, top=0.93, left=0.08, right=0.95)
    cax = fig.add_axes([0.18, 0.03, 0.64, 0.015])
    cb = fig.colorbar(last_im, cax=cax, orientation='horizontal')
    cb.set_label('Fraction of summer days with SM below critical', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_delta_below_critical(p_phase, p_ref, n_phase, n_ref, eof_i, outfile,
                              phase_label='pos', ref_label='rest'):
    """Map of ΔP = P_phase − P_ref (regimes 3/4)."""
    delta = p_phase - p_ref
    vmax = 0.35
    levels = np.linspace(-vmax, vmax, 21)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.5)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
    im = ax.contourf(
        lons_sm_plot, lats_sm, delta,
        levels=levels, cmap='RdBu_r', extend='both',
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent(
        [lons_sm_plot.min(), lons_sm_plot.max(), lats_sm.min(), lats_sm.max()],
        crs=ccrs.PlateCarree(),
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.35, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(
        f'ΔP(SM < critical): {phase_label} − {ref_label} | '
        f'{eof_period_label} EOF{eof_i + 1} '
        f'(n_{phase_label}={n_phase}, n_{ref_label}={n_ref})',
        fontsize=11,
    )
    fig.subplots_adjust(bottom=0.18)
    cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Change in fraction of JJA days below critical SM', fontsize=10)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def domain_mean_pct(prob_map):
    """Area-mean % over regime 3/4 pixels with finite probability."""
    vals = np.asarray(prob_map, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return 100.0 * float(np.mean(vals))


# Main: P(SM < critical) by lateApr EOF phase ==========================================================================
print(f'Regime 3/4 pixels: {int(np.sum(regime34))} / {regime34.size}')
print(
    f'Phase years from {eof_period_label} mean PC '
    f'(month={eof_sel_month}, days {eof_day_lo}–{eof_day_hi}), thresh={pc_thresh:g}'
)

# Climatology: all available JJA years in the absolute SM record
years_clim = np.unique(sm_jja['time'].dt.year.values).astype(int)
p_clim, n_clim = p_sm_below_critical(years_clim)
pct_clim = domain_mean_pct(p_clim)
print(
    f'JJA climatology P(SM<crit) over regimes 3–4: '
    f'{pct_clim:.1f}% (n={n_clim} years, {years_clim[0]}–{years_clim[-1]})'
)

for EOF_i in range(eof_ts.shape[0]):
    pc_daily = np.asarray(eof_ts[EOF_i].values)
    pc_yr, years_yr = period_mean_pc(
        pc_daily, time_eof_ts, eof_sel_month, eof_day_lo, eof_day_hi
    )
    pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()

    year_pos = years_yr[pc_std > pc_thresh]
    year_weak = years_yr[np.abs(pc_std) < pc_thresh]
    year_neg = years_yr[pc_std < -pc_thresh]
    year_rest_pos = years_yr[~np.isin(years_yr, year_pos)]
    year_rest_neg = years_yr[~np.isin(years_yr, year_neg)]

    print(
        f'EOF{EOF_i + 1} {eof_period_label}: n_pos={len(year_pos)}, '
        f'n_weak={len(year_weak)}, n_neg={len(year_neg)} (thresh={pc_thresh})'
    )
    print(f'  pos years: {year_pos.tolist()}')
    print(f'  neg years: {year_neg.tolist()}')

    p_pos, n_pos = p_sm_below_critical(year_pos)
    p_weak, n_weak = p_sm_below_critical(year_weak)
    p_neg, n_neg = p_sm_below_critical(year_neg)
    p_rest_pos, n_rest_pos = p_sm_below_critical(year_rest_pos)
    p_rest_neg, n_rest_neg = p_sm_below_critical(year_rest_neg)

    pct_pos = domain_mean_pct(p_pos)
    pct_weak = domain_mean_pct(p_weak)
    pct_neg = domain_mean_pct(p_neg)
    pct_rest_pos = domain_mean_pct(p_rest_pos)
    pct_rest_neg = domain_mean_pct(p_rest_neg)
    print(
        f'  Domain-mean P(SM<crit) over regimes 3–4: '
        f'pos={pct_pos:.1f}% (vs rest {pct_rest_pos:.1f}%, '
        f'Δ={pct_pos - pct_rest_pos:+.1f} pp; vs clim {pct_clim:.1f}%, '
        f'Δ={pct_pos - pct_clim:+.1f} pp); '
        f'neg={pct_neg:.1f}% (vs rest {pct_rest_neg:.1f}%, '
        f'Δ={pct_neg - pct_rest_neg:+.1f} pp; vs clim {pct_clim:.1f}%, '
        f'Δ={pct_neg - pct_clim:+.1f} pp); weak={pct_weak:.1f}%'
    )

    outfile = (
        f'{path_figures}P_SMbelowCrit_JJA_regimes34_{eof_period_label}EOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_p_below_critical(p_pos, p_weak, p_neg, n_pos, n_weak, n_neg, EOF_i, outfile)

    outfile_pos = (
        f'{path_figures}dP_SMbelowCrit_posMinusRest_JJA_regimes34_'
        f'{eof_period_label}EOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_delta_below_critical(
        p_pos, p_rest_pos, n_pos, n_rest_pos, EOF_i, outfile_pos,
        phase_label='pos', ref_label='rest',
    )

    outfile_neg = (
        f'{path_figures}dP_SMbelowCrit_negMinusRest_JJA_regimes34_'
        f'{eof_period_label}EOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_delta_below_critical(
        p_neg, p_rest_neg, n_neg, n_rest_neg, EOF_i, outfile_neg,
        phase_label='neg', ref_label='rest',
    )

    outfile_pos_clim = (
        f'{path_figures}dP_SMbelowCrit_posMinusClim_JJA_regimes34_'
        f'{eof_period_label}EOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_delta_below_critical(
        p_pos, p_clim, n_pos, n_clim, EOF_i, outfile_pos_clim,
        phase_label='pos', ref_label='clim',
    )

    outfile_neg_clim = (
        f'{path_figures}dP_SMbelowCrit_negMinusClim_JJA_regimes34_'
        f'{eof_period_label}EOF{EOF_i + 1}_'
        f'{variable}_{name_land}_thresh{pc_thresh:g}.png'
    )
    plot_delta_below_critical(
        p_neg, p_clim, n_neg, n_clim, EOF_i, outfile_neg_clim,
        phase_label='neg', ref_label='clim',
    )

ds_sm_abs.close()
ds_reg.close()
print('Done.')
