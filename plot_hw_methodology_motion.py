"""
Methodology figure: object-tracked clusters and T anomalies around heatwave onset.

For each event in selected calendar years, produces a 2x4 panel figure:
  top row    - tracked cluster warm points with daily motion-tracking point
  bottom row - temperature anomalies on the same four days
  columns    - day -1, onset (day 0), day +1, day +2

Matches 1_heatwaves_detection_teng_clusters_object.py.
"""

from __future__ import annotations

import argparse
import yaml
import datetime as dt
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset
from scipy import ndimage
from scipy.signal import butter, filtfilt

VEL = [5, 1]
MIN_DURATION = 5
MIN_AREA = 678000
MIN_CLUSTER_FRACTION = 0.007
MIN_CLUSTER_AREA = 100000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_v2.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    return argparse.Namespace(
        name=cfg['name'],
        case=cfg['case'],
        percentile=cfg['percentile'],
        region=cfg['region'],
        t_file=cfg['t_file'],
        t_file_anoma=cfg['t_file_anoma'],
        var=cfg['var'],
        seasons=cfg['seasons'],
        initial_year=cfg['initial_year'],
        path_case=cfg['path_case_land'],
        global_mean_file=cfg.get('global_mean_file'),
        global_mean_var=cfg.get('global_mean_var'),
        events_csv=cfg.get('events_csv'),
        years=cfg.get('years', [2012, 2015, 2021]),
        topography=cfg.get('topography', True),
        max_events=cfg.get('max_events'),
    )


def conus_map_bounds():
    """Fixed continental US extent for all map panels."""
    return 24.0, 50.0, 235.0, 295.0  # 50°N–24°N, 125°W–65°W


def region_bounds(region):
    if region == 'US':
        return 25, 50, 235, 290
    if region == 'westUS':
        return 25, 50, 235, 260
    if region == 'centerUS':
        return 25, 45, 250, 280
    if region == 'centralUS':
        return 27, 47, 255, 275
    if region == 'westsouthUS':
        return 25, 39, 252, 270
    if region == 'midwestUS':
        return 35, 47, 255, 275
    raise ValueError(f'Region {region} not supported')


def lon_for_plot(lon):
    lon = float(lon)
    return lon - 360.0 if lon > 180.0 else lon


def _cluster_com(mask, lats_arr, lons_arr):
    mask = np.squeeze(mask)
    lats_1d = np.asarray(lats_arr).ravel()
    lons_1d = np.asarray(lons_arr).ravel()
    if mask.ndim != 2 or mask.sum() == 0:
        return np.nan, np.nan
    w = mask.astype(float)
    lat_grid = np.broadcast_to(lats_1d[:, None], w.shape)
    lon_grid = np.broadcast_to(lons_1d[None, :], w.shape)
    return float(np.average(lat_grid, weights=w)), float(np.average(lon_grid, weights=w))


def _grid_cell_area_km2(lats_arr, dlat_deg, dlon_deg, nlon):
    lat_rad = np.deg2rad(np.asarray(lats_arr).ravel())
    meridional_km = 2.0 * np.pi * 6371.0 / 360.0 * dlat_deg
    zonal_km = 2.0 * np.pi * 6371.0 * np.cos(lat_rad) / 360.0 * dlon_deg
    area_lat = meridional_km * zonal_km
    return np.broadcast_to(area_lat[:, None], (len(area_lat), nlon))


def _domain_warm_fraction(cond, grid_cell_area_km2):
    cond = np.squeeze(np.asarray(cond))
    total_area = np.sum(grid_cell_area_km2)
    if total_area == 0:
        return 0.0
    return float(np.sum(cond.astype(float) * grid_cell_area_km2) / total_area)


def get_clusters(cond, lats_arr, lons_arr, min_cluster_area_km2, grid_cell_area_km2):
    cond = np.squeeze(np.asarray(cond))
    labels, nlab = ndimage.label(cond)
    if nlab == 0:
        return []
    sizes_km2 = ndimage.sum(grid_cell_area_km2, labels, range(1, nlab + 1))
    clusters = []
    for lab, size_km2 in zip(range(1, nlab + 1), sizes_km2):
        if size_km2 < min_cluster_area_km2:
            continue
        mask = labels == lab
        lat, lon = _cluster_com(mask, lats_arr, lons_arr)
        clusters.append({'mask': mask, 'lat': lat, 'lon': lon, 'size': int(size_km2)})
    return clusters


def _com_motion_ok(lat1, lon1, lat2, lon2, max_move):
    return abs(lat2 - lat1) < max_move and abs(lon2 - lon1) < max_move


def match_clusters(clusters_prev, clusters_curr, max_move):
    if not clusters_prev or not clusters_curr:
        return [], list(range(len(clusters_prev))), list(range(len(clusters_curr)))
    pairs = []
    still_prev = list(range(len(clusters_prev)))
    still_curr = list(range(len(clusters_curr)))
    while still_prev and still_curr:
        best_pair = None
        best_dist = np.inf
        for i in still_prev:
            cp = clusters_prev[i]
            for j in still_curr:
                cc = clusters_curr[j]
                if not _com_motion_ok(cp['lat'], cp['lon'], cc['lat'], cc['lon'], max_move):
                    continue
                dist = (cp['lat'] - cc['lat']) ** 2 + (cp['lon'] - cc['lon']) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        pairs.append((i, j))
        still_prev.remove(i)
        still_curr.remove(j)
    return pairs, still_prev, still_curr


def _start_track(cluster, day):
    return {
        'obs': [{'day': day, 'lat': cluster['lat'], 'lon': cluster['lon']}],
        'last_cluster': cluster,
        'onset_mask': cluster['mask'].copy(),
        'cluster_masks': [cluster['mask'].copy()],
    }


def _finalize_tracks(tracks, min_duration, finished_events):
    for track in tracks:
        if len(track['obs']) >= min_duration:
            finished_events.append(track)


def track_objects_across_days(
    get_cond, day_indices, lats_arr, lons_arr,
    grid_cell_area_km2, min_cluster_area_km2,
    min_warm_fraction, min_duration, max_move, day_step=1,
):
    finished_events = []
    open_tracks = []
    for i, day in enumerate(day_indices):
        cond = get_cond(day)
        if _domain_warm_fraction(cond, grid_cell_area_km2) < min_warm_fraction:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = []
            continue
        clusters = get_clusters(
            cond, lats_arr, lons_arr, min_cluster_area_km2, grid_cell_area_km2
        )
        gap = (i == 0) or (day - day_indices[i - 1] != day_step)
        if gap:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = [_start_track(c, day) for c in clusters]
            continue
        if not clusters:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = []
            continue
        if not open_tracks:
            open_tracks = [_start_track(c, day) for c in clusters]
            continue
        prev_clusters = [t['last_cluster'] for t in open_tracks]
        pairs, unmatched_prev, unmatched_curr = match_clusters(
            prev_clusters, clusters, max_move
        )
        new_open = []
        for pi, ci in pairs:
            track = open_tracks[pi]
            track['obs'].append(
                {'day': day, 'lat': clusters[ci]['lat'], 'lon': clusters[ci]['lon']}
            )
            track['last_cluster'] = clusters[ci]
            track['cluster_masks'].append(clusters[ci]['mask'].copy())
            new_open.append(track)
        for pi in unmatched_prev:
            _finalize_tracks([open_tracks[pi]], min_duration, finished_events)
        for ci in unmatched_curr:
            new_open.append(_start_track(clusters[ci], day))
        open_tracks = new_open
    _finalize_tracks(open_tracks, min_duration, finished_events)
    return finished_events


def load_temperature(args, lat_min, lat_max, lon_min, lon_max):
    ncfile = Dataset(args.t_file)
    time = np.array(ncfile['time'][:])
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])
    pos_lats = np.where((lats >= lat_min) & (lats <= lat_max))
    pos_lons = np.where((lons >= lon_min) & (lons <= lon_max))
    lats_us = lats[pos_lats]
    lons_us = lons[pos_lons]
    t_k = np.array(ncfile[args.var][:, pos_lats[0], :])[:, :, pos_lons[0]]
    if args.global_mean_file is not None:
        nc_glob = Dataset(args.global_mean_file)
        x = np.array(nc_glob[args.global_mean_var][:])
        fs, fc = 365.0, 1.0 / (9 * 365.0)
        b, a = butter(N=4, Wn=fc / (fs / 2), btype='low')
        xc = filtfilt(b, a, x) - filtfilt(b, a, x).mean()
        var_x = (xc ** 2).mean()
        b_map = (t_k * xc[:, None, None]).mean(axis=0) / var_x
        t_k = t_k - b_map * xc[:, None, None]
    t_us = t_k - 273.15
    if args.name == 'ERA5':
        dates_d = np.array([
            dt.datetime(args.initial_year, 1, 1) + dt.timedelta(days=i)
            for i in range(len(time))
        ])
        date_0 = dt.datetime(args.initial_year, 1, 1)
        pos_date_0 = np.where(dates_d == date_0)[0][0]
        dates_d = dates_d[pos_date_0:]
        t_us = t_us[pos_date_0:]
    else:
        raise ValueError(f'Date handling for name={args.name} not implemented')
    dlat = float(np.abs(lats_us[1] - lats_us[0]))
    dlon = float(np.abs(lons_us[1] - lons_us[0]))
    grid_cell_area = _grid_cell_area_km2(lats_us, dlat, dlon, len(lons_us))
    return dates_d, lats_us, lons_us, t_us, grid_cell_area


def build_summer_threshold(dates_d, t_us, percentile, case):
    days_summer = np.array([
        dt.datetime(2021, 6, 1) + relativedelta(days=int(xx)) for xx in range(92)
    ])
    threshold = np.zeros((len(days_summer), t_us.shape[1], t_us.shape[2]))
    for i, d in enumerate(days_summer):
        t_pos1 = np.where(
            (np.array([dd.month for dd in dates_d]) == d.month)
            & (np.array([dd.day for dd in dates_d]) == d.day)
        )[0]
        if 'nowindow' in case:
            data = t_us[t_pos1, :, :]
        else:
            t_pos = [np.concatenate((t_pos1, t_pos1 + offset)) for offset in range(-7, 8)]
            t_pos = np.unique(np.concatenate(t_pos))
            t_pos = t_pos[t_pos < t_us.shape[0]]
            data = t_us[t_pos, :, :]
        threshold[i] = np.percentile(data, percentile, axis=0)
    df_threshold = pd.DataFrame(
        index=days_summer,
        data=np.reshape(threshold, (len(days_summer), t_us.shape[1] * t_us.shape[2])),
    )
    return threshold, df_threshold


def threshold_for_day(day_idx, dates_d, df_threshold, threshold):
    date = dates_d[day_idx]
    threshold_pos = df_threshold.index.get_indexer_for(
        df_threshold.loc[
            (df_threshold.index.month == date.month) & (df_threshold.index.day == date.day)
        ].index
    )
    return threshold[int(threshold_pos[0]), :, :]


def make_get_cond(dates_d, t_us, df_threshold, threshold):
    def get_cond(day_idx):
        thr = threshold_for_day(day_idx, dates_d, df_threshold, threshold)
        return t_us[day_idx, :, :] > thr
    return get_cond


def load_anomalies(args, lat_min, lat_max, lon_min, lon_max, n_time, return_coords=False):
    anom = xr.open_dataset(args.t_file_anoma)[args.var]
    lats_an = anom.lat.values
    lons_an = anom.lon.values
    pos_lats = np.where((lats_an >= lat_min) & (lats_an <= lat_max))[0]
    pos_lons = np.where((lons_an >= lon_min) & (lons_an <= lon_max))[0]
    lats_out = lats_an[pos_lats]
    lons_out = lons_an[pos_lons]
    anom_us = anom.isel(lat=pos_lats, lon=pos_lons).values
    anom_us = np.asarray(anom_us).reshape(anom_us.shape[0], len(pos_lats), len(pos_lons))
    if anom_us.shape[0] > n_time:
        anom_us = anom_us[:n_time]
    if return_coords:
        return anom_us, lats_out, lons_out
    return anom_us


def parse_events_csv(path_csv, years):
    df = pd.read_csv(path_csv, index_col=0)
    years_set = set(years)
    events = []
    for onset_idx, row in df.iterrows():
        date = pd.to_datetime(row['Date 0'])
        if date.year not in years_set:
            continue
        events.append({
            'onset_idx': int(onset_idx),
            'date': date,
            'duration': int(row['Duration']),
        })
    return events


def find_track_for_onset(tracks, onset_idx):
    for track in tracks:
        if track['obs'][0]['day'] == onset_idx:
            return track
    return None


def cluster_state_for_day(
    track,
    day_idx,
    get_cond,
    lats_us,
    lons_us,
    grid_cell_area,
    min_cluster_area,
    max_move,
):
    """Return tracked cluster mask and daily tracking point (cluster lat/lon)."""
    if track is not None:
        for i, obs in enumerate(track['obs']):
            if obs['day'] == day_idx:
                return track['cluster_masks'][i], obs['lat'], obs['lon']

    if track is None:
        return None, np.nan, np.nan

    onset_day = track['obs'][0]['day']
    if day_idx == onset_day - 1:
        cond = get_cond(day_idx)
        clusters = get_clusters(cond, lats_us, lons_us, min_cluster_area, grid_cell_area)
        if not clusters:
            return None, np.nan, np.nan
        onset_lat, onset_lon = track['obs'][0]['lat'], track['obs'][0]['lon']
        best = None
        best_dist = np.inf
        for cluster in clusters:
            if not _com_motion_ok(
                cluster['lat'], cluster['lon'], onset_lat, onset_lon, max_move
            ):
                continue
            dist = (cluster['lat'] - onset_lat) ** 2 + (cluster['lon'] - onset_lon) ** 2
            if dist < best_dist:
                best_dist = dist
                best = cluster
        if best is None:
            return None, np.nan, np.nan
        return best['mask'], best['lat'], best['lon']

    return None, np.nan, np.nan


def setup_map_ax(
    ax, lat_min, lat_max, lon_min, lon_max, topography,
    show_left_labels=False, show_bottom_labels=False,
):
    lon_min_p = lon_for_plot(lon_min)
    lon_max_p = lon_for_plot(lon_max)
    ax.set_extent([lon_min_p, lon_max_p, lat_min, lat_max], crs=ccrs.PlateCarree())
    if topography:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=11)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='0.4', zorder=10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = show_bottom_labels


def plot_event_figure(
    event,
    track,
    dates_d,
    anom_map,
    lats_map,
    lons_map,
    get_cond,
    lats_us,
    lons_us,
    grid_cell_area,
    min_cluster_area,
    map_lat_min,
    map_lat_max,
    map_lon_min,
    map_lon_max,
    out_path,
    topography=True,
):
    onset = event['onset_idx']
    # day_offsets = [-1, 0, 1, 2]
    # day_offsets = [-2, 0, 2, 4]
    day_offsets = [0, 2, 4, 5]
    # day_labels = ['Day -2', 'Onset (day 0)', 'Day +2', 'Day +4']
    day_labels = ['Onset (day 0)', 'Day +2', 'Day +4', 'Day +5']

    lon_plot_region = np.array([lon_for_plot(l) for l in lons_us])
    lon2d_region, lat2d_region = np.meshgrid(lon_plot_region, lats_us)
    lon_plot_map = np.array([lon_for_plot(l) for l in lons_map])
    lon2d_map, lat2d_map = np.meshgrid(lon_plot_map, lats_map)

    warm_cmap = mcolors.ListedColormap(['#f7f7f7', '#e34a33'])
    anom_vmax = 8.0

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 7.2),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.22, wspace=0.06, hspace=0.06)

    for col, offset in enumerate(day_offsets):
        day_idx = onset + offset
        date_str = dates_d[day_idx].strftime('%Y-%m-%d')
        cluster_mask, lat_track, lon_track = cluster_state_for_day(
            track, day_idx, get_cond, lats_us, lons_us,
            grid_cell_area, min_cluster_area, VEL[0],
        )

        ax_warm = axes[0, col]
        setup_map_ax(
            ax_warm, map_lat_min, map_lat_max, map_lon_min, map_lon_max, topography,
            show_left_labels=(col == 0),
        )
        if cluster_mask is not None:
            warm_plot = np.where(cluster_mask, 1.0, np.nan)
            ax_warm.pcolormesh(
                lon2d_region, lat2d_region, warm_plot,
                cmap=warm_cmap, vmin=0.5, vmax=1.5,
                transform=ccrs.PlateCarree(), zorder=5,
            )
        if np.isfinite(lat_track) and np.isfinite(lon_track):
            ax_warm.scatter(
                lon_for_plot(lon_track), lat_track,
                s=70, c='black', edgecolors='white', linewidths=1.2,
                transform=ccrs.PlateCarree(), zorder=20,
            )
        ax_warm.set_title(f'{day_labels[col]}\n{date_str}', fontsize=10)

        ax_anom = axes[1, col]
        setup_map_ax(
            ax_anom, map_lat_min, map_lat_max, map_lon_min, map_lon_max, topography,
            show_left_labels=(col == 0), show_bottom_labels=True,
        )
        pcm = ax_anom.pcolormesh(
            lon2d_map, lat2d_map, anom_map[day_idx],
            cmap='RdBu_r', vmin=-anom_vmax, vmax=anom_vmax,
            transform=ccrs.PlateCarree(), zorder=5,
        )

    axes[0, 0].set_ylabel(
        'Tracked cluster\n(T > threshold)', fontsize=11, labelpad=30,
    )
    axes[1, 0].set_ylabel('T anomaly [°C]', fontsize=11, labelpad=30)

    fig.suptitle(
        f"Object-based heatwave methodology — onset {event['date'].strftime('%Y-%m-%d')} "
        f"(duration {event['duration']} d)",
        fontsize=13, y=0.96,
    )
    cax = fig.add_axes([0.22, 0.07, 0.56, 0.03])
    cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
    cbar.set_label('Temperature anomaly [°C]', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)




def event_severity_cluster(anom, track, grid_cell_area_km2, dates_d, path_figure):
    """Severity = AUC of daily area-weighted mean positive SAT anomaly over the cluster.

    Each event day:
        daily_mean = ∑ (max(T', 0) · A_cell) / ∑ A_cell   over that day's cluster mask
    Then sum daily_mean over the event duration (Δt = 1 day).

    Units: °C·day. Uses the object footprint instead of a fixed lat–lon box.
    """
    if track is None:
        return np.nan
    area = np.asarray(grid_cell_area_km2, dtype=float)
    if np.asarray(anom[0]).shape != area.shape:
        raise ValueError(
            f'Anomaly grid {np.asarray(anom[0]).shape} does not match cluster mask '
            f'{area.shape}; severity must use the tracking grid.'
        )

    lags = []
    dates_event = []
    daily_means = []
    onset_day = track['obs'][0]['day']
    for obs, mask in zip(track['obs'], track['cluster_masks']):
        day = obs['day']
        m = np.squeeze(mask).astype(bool)
        if not np.any(m):
            continue
        tprime = np.asarray(anom[day], dtype=float)
        pos = np.maximum(tprime, 0.0)
        area_sum = float(np.nansum(area[m]))
        if area_sum <= 0:
            continue
        daily_mean = float(np.nansum(pos[m] * area[m]) / area_sum)
        daily_means.append(daily_mean)
        lags.append(day - onset_day)
        dates_event.append(dates_d[day])

    if not daily_means:
        return np.nan

    daily_means = np.asarray(daily_means, dtype=float)
    lags = np.asarray(lags)
    total = float(np.nansum(daily_means))
    plot_event_severity_cluster(lags, dates_event, daily_means, total, path_figure)
    return total


def plot_event_severity_cluster(lags, dates_event, daily_means, event_severity, path_figure):
    """Time series of cluster-mean positive T' whose area is the event severity."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(lags, 0.0, daily_means, color='#e34a33', alpha=0.35, zorder=1)
    ax.plot(lags, daily_means, color='#b2182b', lw=2.0, marker='o', ms=5, zorder=2)
    ax.axhline(0.0, color='0.4', lw=0.8)
    ax.axvline(0.0, color='0.25', ls='--', lw=1.0)
    ax.set_xlabel('Days since onset', fontsize=13)
    ax.set_ylabel("Area-weighted mean T' [°C]", fontsize=13)
    ax.set_title('Daily cluster anomaly (severity = area under curve)', fontsize=12)
    ax.set_xlim(lags.min() - 0.3, lags.max() + 0.3)
    ymax = float(np.nanmax(daily_means)) if daily_means.size else 3.0
    ylim_top = max(3.0, np.ceil((ymax + 0.8) / 3.0) * 3.0)
    ax.set_ylim(0, ylim_top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
    ax.tick_params(labelsize=12)
    for lag, y, d in zip(lags, daily_means, dates_event):
        ax.annotate(
            pd.Timestamp(d).strftime('%d %b'),
            xy=(lag, y),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, color='0.15',
            clip_on=False,
        )
    ax.text(
        0.03, 0.06,
        f"Severity = {event_severity:.1f} °C·day",
        transform=ax.transAxes, va='bottom', ha='left', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9),
        zorder=4,
    )
    fig.tight_layout()
    fig.savefig(path_figure, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    if args.percentile % 1 == 0:
        args.percentile = int(args.percentile)

    lat_min, lat_max, lon_min, lon_max = region_bounds(args.region)
    map_lat_min, map_lat_max, map_lon_min, map_lon_max = conus_map_bounds()
    path_figures = os.path.join(args.path_case, 'Figures', 'methodology_motion')
    os.makedirs(path_figures, exist_ok=True)

    events_csv = args.events_csv or os.path.join(args.path_case, 'Heat_waves_events_list.csv')
    if not os.path.exists(events_csv):
        raise FileNotFoundError(f'Events CSV not found: {events_csv}')

    dates_d, lats_us, lons_us, t_us, grid_cell_area = load_temperature(
        args, lat_min, lat_max, lon_min, lon_max,
    )
    if not args.seasons:
        raise ValueError('This figure script currently supports seasons=True only.')

    threshold, df_threshold = build_summer_threshold(
        dates_d, t_us, args.percentile, args.case,
    )
    get_cond = make_get_cond(dates_d, t_us, df_threshold, threshold)
    anom_map, lats_map, lons_map = load_anomalies(
        args, map_lat_min, map_lat_max, map_lon_min, map_lon_max, t_us.shape[0],
        return_coords=True,
    )
    # Severity uses cluster masks on the tracking (region) grid, not the wider CONUS map.
    anom_us = load_anomalies(
        args, lat_min, lat_max, lon_min, lon_max, t_us.shape[0],
    )

    total_area = float(np.sum(grid_cell_area))
    min_cluster_area = max(MIN_CLUSTER_AREA, MIN_CLUSTER_FRACTION * total_area)
    min_warm_fraction = MIN_AREA / total_area
    month = np.array([d.month for d in dates_d])
    pos_summer = np.where(np.isin(month, [6, 7, 8]))[0].tolist()

    print('Running object tracking (same as clusters_object.py)...')
    tracks = track_objects_across_days(
        get_cond,
        pos_summer,
        lats_us,
        lons_us,
        grid_cell_area,
        min_cluster_area,
        min_warm_fraction,
        MIN_DURATION,
        VEL[0],
        day_step=VEL[1],
    )
    print(f'Found {len(tracks)} tracked events')

    events = parse_events_csv(events_csv, args.years)
    if args.max_events is not None:
        events = events[:args.max_events]
    years_label = ', '.join(str(y) for y in sorted(set(args.years)))
    print(f'Plotting {len(events)} events from year(s): {years_label}')

    for event in events:
        onset = event['onset_idx']
        if onset - 2 < 0 or onset + 4 >= t_us.shape[0]:
            print(f"Skipping {event['date'].date()}: day window out of bounds")
            continue
        track = find_track_for_onset(tracks, onset)
        if track is None:
            print(f"Warning: no object track found for onset {event['date'].date()} (idx {onset})")

        out_name = f"methodology_motion_{event['date'].strftime('%Y%m%d')}_idx{onset}.png"
        out_path = os.path.join(path_figures, out_name)
        plot_event_figure(
            event, track, dates_d, anom_map, lats_map, lons_map, get_cond,
            lats_us, lons_us, grid_cell_area, min_cluster_area,
            map_lat_min, map_lat_max, map_lon_min, map_lon_max,
            out_path, topography=args.topography,
        )
        print(f'Saved {out_path}')

        out_name = f"severity_{event['date'].strftime('%Y%m%d')}_idx{onset}.png"
        out_path = os.path.join(path_figures, out_name)
        event_severity = event_severity_cluster(
            anom_us, track, grid_cell_area, dates_d, out_path,
        )
        if np.isfinite(event_severity):
            print(f'Saved {out_path}  (severity = {event_severity:.1f} °C·day)')
        else:
            print(f'Skipped severity plot for {event["date"].date()}: no cluster/track')


if __name__ == '__main__':
    main()


