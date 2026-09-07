# This code is for analyzing the RWP variables diagnosed from ERA5 by Yuan-Bing.  
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import scipy as scp
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
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
path_ncr = '/apps/spack/negishi/apps/nco/5.0.1-gcc-12.2.0-f3lr7i3/bin/ncrcat'

def cut_region_lats_lons(lats, lons, min_lat, max_lat, min_lon, max_lon):
    pos_lat = np.where((lats >= min_lat) & (lats <= max_lat))[0]
    if np.any(lons < 0):
        if min_lon > 180:
            min_lon = min_lon - 360
        if max_lon > 180:
            max_lon = max_lon - 360

    pos_lon = np.where((lons >= min_lon) & (lons <= max_lon))[0]
    new_lats = lats[pos_lat]
    new_lons = lons[pos_lon]
    return pos_lat, pos_lon, new_lats, new_lons

def calculate_time_series_standard_deviation(
    pos_HW,
    path_matriz,
    var,
    num_time_laps,
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    std=True
):
    import numpy as np
    import xarray as xr

    # --- Load data ---
    ds = xr.open_dataset(path_matriz)

    # --- Select spatial region safely ---
    lats = ds["lat"].values
    lons = ds["lon"].values

    pos_lat, pos_lon, lats_cut, lons_cut = cut_region_lats_lons(lats, lons, min_lat, max_lat, min_lon, max_lon)

    ds = ds.isel(lat=pos_lat, lon=pos_lon)

    print(ds.lat.values)
    print(ds.lon.values)

    da = ds[var]

    # --- Area-averaged time series ---
    series = da.mean(dim=("lat", "lon"), skipna=True).values
    ntime = series.shape[0]

    # --- Build time-lag array ---
    time_lags = np.arange(-num_time_laps, num_time_laps + 1)

    # --- HW indices ---
    if hasattr(pos_HW, "iloc"):
        hw_indices = np.array(pos_HW.iloc[:, 0], dtype=int)
    else:
        hw_indices = np.array(pos_HW, dtype=int)

    # --- Dictionary of lagged indices ---
    dic_composites = {}
    for lag in time_lags:
        idx = hw_indices + lag
        idx = idx[(idx >= 0) & (idx < ntime)]
        dic_composites[lag] = idx

    # --- Compute lagged composite mean and std ---
    composites_series = np.full(len(time_lags), np.nan)
    

    for ii, lag in enumerate(time_lags):
        idx = dic_composites[lag]
        if len(idx) > 0:
            composites_series[ii] = np.nanmean(series[idx])

    if std:
        standard_deviation_series = np.full(len(time_lags), np.nan)
        for ii, lag in enumerate(time_lags):
            idx = dic_composites[lag]
            if len(idx) > 0:
                standard_deviation_series[ii] = np.nanstd(series[idx])
    else:
        standard_deviation_series = None
    ds.close()


    results = {
        "composites_series": composites_series,
        "standard_deviation_series": standard_deviation_series,
        "time_lags": time_lags,
    }

    return results

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--path_case', type=str, required=True)
parser.add_argument('--path_file_E', type=str, required=True)
parser.add_argument('--path_file_Cp', type=str, required=True)
parser.add_argument('--path_file_t', type=str, required=True)
parser.add_argument('--path_file_v', type=str, required=True)
parser.add_argument('--initial_year', type=int, required=True)
parser.add_argument('--path_outputs', type=str, required=True)
parser.add_argument('--E0', type=int, required=False, default=25)
args = parser.parse_args()

name = args.name
case = args.case
region = args.region
path_case = args.path_case
path_file_E = args.path_file_E
path_file_Cp = args.path_file_Cp
path_file_t = args.path_file_t
path_file_v = args.path_file_v
initial_year = args.initial_year
path_outputs = args.path_outputs
E0 = args.E0


if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #-125 to -70
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat=45 #-125 to -100   WEST
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat=45 #-125 to -100   WEST
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{path_case}/Figures/'
path_outputs_case = f'/scratch/negishi/castanev/GLEAM/{case}/'
path_figures_all = f'{path_case}/../Figures/'

num_time_laps = 20 

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
name_file_posHW = f'resume_positions_HWdays_{name}_Teng.csv'
pos_HWdays = pd.read_csv(f'{path_figures}../{name_file_posHW}', index_col=0)
pos_HWdays = pos_HWdays.iloc[np.where(pos_HWdays<=len(timei))[0]]

name_file_posHW = f'Heat_waves_events_list.csv'
df_heatwaves = pd.read_csv(f'{path_figures}../{name_file_posHW}', index_col=0)
pos_HW_day0 = df_heatwaves.index
pos_HW_day0 = pos_HW_day0[np.where(pos_HW_day0<=len(timei))[0]]
pos_HW_day0 = pd.DataFrame(pos_HW_day0)


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

path_t_daily_anoma_window = f'{path_outputs}TS_ERA5_daily_anoma_window.nc'  
path_t_annual_cycle = f'{path_outputs}annual_cycle_TS_daily_window.nc'  




path_t_daily_anoma_std_window_US = f'{path_outputs}TS_ERA5_daily_anoma_std_window_US.nc'  
path_t_annual_cycle_std_US = f'{path_outputs}annual_cycle_std_TS_daily_window_US.nc'


# ==================================================== Envelope  ==============================================================================
variable_env = 'amplitude'

ncfile_E = Dataset(f'{path_file_E}')
lats_env = np.array(ncfile_E['lat'])
lons_env = np.array(ncfile_E['lon'])  
timei_env = np.array(ncfile_E['time'][:]) #hours since 1900-01-01 00:00:00, gregorian
dates_env_hourly = np.array([dt.datetime(1900,1,1) + dt.timedelta(hours = i) for i in timei_env])
pos_initial_year = np.where([d.year == initial_year for d in dates_env_hourly])[0][0]
print(pos_initial_year)
pos_lats_midlat = np.where((lats_env >= 35) & (lats_env <= 65))[0]
pos_middle_HW = np.where(abs(lons_env - (lon_minHW+lon_maxHW)/2) == np.min(abs(lons_env - (lon_minHW+lon_maxHW)/2)))[0][0]

path_envelope_daily = f'{path_outputs}envelope_daily.nc'
path_envelope_daily_anoma_window = f'{path_outputs}envelope_daily_anoma_window.nc'
path_envelope_annual_cycle = f'{path_outputs}annual_cycle_envelope_daily_window.nc'

time_env_daily = np.array(Dataset(path_envelope_daily)["time"][:])
dates_env = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(time_env_daily))])
dates_env = np.unique([x.date() for x in dates_env])
dates_env_str = np.array([x.strftime("%Y%m%d") for x in dates_env])
del time_env_daily




pos_HWdays = []
for num in range(len(dates_env)): 
    if dates_env[num] in dates_d_HWdays: pos_HWdays.append(num)
pos_HWdays = pd.DataFrame(pos_HWdays)


pos_HW_day0 = []
for num in range(len(dates_env)): 
    if dates_env[num] in dates_d_HW_day0: pos_HW_day0.append(num)  
pos_HW_day0 = pd.DataFrame(pos_HW_day0)

pos_summer = np.where([i.month in [6,7,8] for i in dates_env])[0]
pos_summer = pos_summer[pos_summer<min(len(dates_env),len(dates_d))]
pos_all_non_HW = []
for i in pos_summer:
    if i not in pos_HWdays.values:
        pos_all_non_HW.append(i)
pos_all_non_HW = np.array(pos_all_non_HW)




# ==================================================== cp  ==============================================================================
variable_cp = 'phase_speed'

path_cp_daily = f'{path_outputs}cp_{str(int(E0))}_min3timesteps_daily_smoothed.nc'
path_cp_daily_anoma_window = f'{path_outputs}cp_{str(int(E0))}_min3timesteps_daily_smoothed_anoma_window.nc'
path_cp_annual_cycle = f'{path_outputs}annual_cycle_cp_{str(int(E0))}_min3timesteps_smoothed_daily_window.nc'
path_cp_daily_seasonal_anoma = f'{path_outputs}cp_{str(int(E0))}_min3timesteps_daily_smoothed_seasonal_anoma.nc'

# path_cp_daily = f'{path_outputs}cp_daily.nc'
# path_cp_daily_anoma_window = f'{path_outputs}cp_daily_anoma_window.nc'
# path_cp_annual_cycle = f'{path_outputs}annual_cycle_cp_daily_window.nc'
# path_cp_daily_seasonal_anoma = f'{path_outputs}cp_daily_seasonal_anoma.nc'



lon_min_pacific = 265 - 95
lon_max_pacific = 265




# path_cp_daily_seasonal_anoma_std_pacific = f'{path_outputs}cp_daily_seasonal_anoma_std_pacific.nc'
# path_cp_seasonal_std_pacific =  f'{path_outputs}seasonal_std_cp_daily_pacific.nc'
path_cp_daily_seasonal_anoma_std_pacific = f'{path_outputs}cp_{str(int(E0))}_min3timesteps_daily_smoothed_seasonal_anoma_std_pacific.nc'
path_cp_seasonal_std_pacific =  f'{path_outputs}seasonal_std_cp_{str(int(E0))}_min3timesteps_daily_smoothed_pacific.nc'



lon_min_env = 265 - 65
lon_max_env = 250
pos_lat_env, pos_lon_env, lats_cut, lons_cut = cut_region_lats_lons(lats_env, lons_env, 35, 55, lon_min_env, lon_max_env)
if not os.path.exists(f"{path_outputs_case}time_series_results_cp_{str(int(E0))}_min3timesteps_daily_smoothed_seasonal_anoma_std_yb_{region}_envregion.npy"):
    # time_series_results = calculate_time_series_standard_deviation(pos_HW_day0, path_cp_daily_seasonal_anoma_std_pacific, variable_cp, 30, 35, 55, lon_min_pacific, lon_max_pacific, std=False)
    # np.save(f"{path_outputs_case}time_series_results_cp_daily_seasonal_anoma_std_yb_{region}_pacific.npy", time_series_results)
    time_series_results = calculate_time_series_standard_deviation(pos_HW_day0, path_cp_daily_seasonal_anoma_std_pacific, variable_cp, 30, 35, 55, lon_min_env, lon_max_env, std=False)
    np.save(f"{path_outputs_case}time_series_results_cp_{str(int(E0))}_min3timesteps_daily_smoothed_seasonal_anoma_std_yb_{region}_envregion.npy", time_series_results)





# # if not os.path.exists(f'{path_outputs_case}composites_matriz_cp_daily_anoma_window_yb_{region}.npy'):
#     composites_matriz_cp, composites_matrix_complete_cp, pvalue_composites_matriz_cp = calculate_composites_ttest_optimized(pos_HWdays, pos_all_non_HW, path_cp_daily, variable_cp, num_time_laps, pvalue=True)
#     np.save(f'{path_outputs_case}composites_matriz_cp_yb_{region}.npy', composites_matriz_cp)
#     np.save(f'{path_outputs_case}composites_matriz_cp_pvalues_yb_{region}.npy', pvalue_composites_matriz_cp)
#     composites_matrix_complete_cp = np.nanmean(composites_matrix_complete_cp[:,pos_lats_midlat], axis=1)
#     np.save(f'{path_outputs_case}composites_matrix_complete_cp_yb_{region}.npy', composites_matrix_complete_cp)
#     hovmoller_Cp_mean = np.roll(composites_matrix_complete_cp,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)

# if not os.path.exists(f'{path_outputs_case}composites_matriz_cp{str(int(E0))}_min3timesteps_smoothed_daily_anoma_window_mean3days_yb_{region}.npy'):
#     composites_matriz_cp_daily_anoma_window, composites_matrix_complete_cp_anoma_window, pvalue_composites_matriz_cp_daily_anoma_window = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_anoma_window, variable_cp, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
#     np.save(f'{path_outputs_case}composites_matriz_cp{str(int(E0))}_min3timesteps_smoothed_daily_anoma_window_mean3days_yb_{region}.npy', composites_matriz_cp_daily_anoma_window)
#     np.save(f'{path_outputs_case}composites_matriz_cp{str(int(E0))}_min3timesteps_smoothed_daily_anoma_window_pvalues_welch_mean3days_yb_{region}.npy', pvalue_composites_matriz_cp_daily_anoma_window)
#     # composites_matrix_complete_cp_anoma_window_mean = np.nanmean(composites_matrix_complete_cp_anoma_window[:,pos_lats_midlat], axis=1)
#     # np.save(f'{path_outputs_case}composites_matrix_complete_cp{str(int(E0))}_min3timesteps_smoothed_anoma_window_yb_{region}.npy', composites_matrix_complete_cp_anoma_window_mean)
#     # hovmoller_anoma_cp_window_mean = np.roll(composites_matrix_complete_cp_anoma_window_mean,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)


# # if not os.path.exists(f'{path_outputs_case}composites_matrix_complete_cp_seasonal_anoma_yb_{region}.npy'):
# composites_matriz_cp_synoptic_seasonal_anoma, composites_matrix_complete_cp_anoma, pvalue_composites_matriz_cp_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_seasonal_anoma, variable_cp, num_time_laps, pvalue=True, welch_test=True)
# np.save(f'{path_outputs_case}composites_matriz_cp_seasonal_anoma_yb_{region}.npy', composites_matriz_cp_synoptic_seasonal_anoma)
# np.save(f'{path_outputs_case}composites_matriz_cp_seasonal_anoma_pvalues_welch_yb_{region}.npy', pvalue_composites_matriz_cp_seasonal_anoma)

# pos_season = np.where([i.month in [6, 7, 8] for i in dates_env])[0]
# pos_season = pos_season[pos_season<min(len(dates_env),len(dates_d))]
# ncfile_cp_daily = Dataset(path_cp_daily)
# cp_season = np.nanmean(np.array(ncfile_cp_daily[variable_cp][pos_season]), axis=0)
# np.save(f'{path_outputs}climatology_summer_cp{str(int(E0))}.npy', cp_season)
# value_threshold = np.nanmax(np.nanmax(cp_season, axis=1))
# print("value_threshold:", value_threshold)


# if not os.path.exists(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_mean3days_yb_{region}.npy'):
composites_matriz_cp_smoothed_seasonal_anoma, composites_matrix_complete_cp_smoothed_seasonal_anoma, pvalue_composites_matriz_cp_smoothed_seasonal_anoma = calculate_composites_ind_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_seasonal_anoma, variable_cp, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_mean3days_yb_{region}.npy', composites_matriz_cp_smoothed_seasonal_anoma)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_ind_welch_mean3days_yb_{region}.npy', pvalue_composites_matriz_cp_smoothed_seasonal_anoma)


# if not os.path.exists(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_yb_{region}.npy'):
composites_matriz_cp_smoothed_seasonal_anoma, composites_matrix_complete_cp_smoothed_seasonal_anoma, pvalue_composites_matriz_cp_smoothed_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_seasonal_anoma, variable_cp, num_time_laps, pvalue=True, welch_test=True)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_yb_{region}.npy', composites_matriz_cp_smoothed_seasonal_anoma)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_yb_{region}.npy', pvalue_composites_matriz_cp_smoothed_seasonal_anoma)
aaaa

# if not os.path.exists(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_mean3days_yb_{region}.npy'):
composites_matriz_cp_smoothed_seasonal_anoma, composites_matrix_complete_cp_smoothed_seasonal_anoma, pvalue_composites_matriz_cp_smoothed_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_seasonal_anoma, variable_cp, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_mean3days_yb_{region}.npy', composites_matriz_cp_smoothed_seasonal_anoma)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_mean3days_yb_{region}.npy', pvalue_composites_matriz_cp_smoothed_seasonal_anoma)
aaaa

# if not os.path.exists(f'{path_outputs_case}composites_matrix_complete_cp_seasonal_anoma_yb_{region}.npy'):
composites_matriz_cp_smoothed_seasonal_anoma, composites_matrix_complete_cp_smoothed_seasonal_anoma, pvalue_composites_matriz_cp_smoothed_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_cp_daily_seasonal_anoma, variable_cp, num_time_laps, pvalue=True, welch_test=True)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_yb_{region}.npy', composites_matriz_cp_smoothed_seasonal_anoma)
np.save(f'{path_outputs_case}composites_matriz_cp_{str(int(E0))}_min3timesteps_smoothed_seasonal_anoma_pvalues_welch_yb_{region}.npy', pvalue_composites_matriz_cp_smoothed_seasonal_anoma)
aaaa



composites_matrix_complete_cp_anoma_mean = np.nanmean(composites_matrix_complete_cp_anoma[:,pos_lats_midlat], axis=1)
np.save(f'{path_outputs_case}composites_matrix_complete_cp_seasonal_anoma_yb_{region}.npy', composites_matrix_complete_cp_anoma_mean)
hovmoller_seasonal_anoma_Cp_mean = np.roll(composites_matrix_complete_cp_anoma_mean,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)
aaaa

# if not os.path.exists(f'{path_outputs_case}frequency_cp_seasonal_anoma_yb_{region}.npy'):
#     cp_daily_seasonal_anoma = np.array(Dataset(path_cp_daily_seasonal_anoma)[variable_cp][:])
#     cp_daily_seasonal_anoma_mean = np.nanmean(cp_daily_seasonal_anoma[:,pos_lats_midlat], axis=1)
#     frequency_cp_seasonal_anoma_mean = calculate_frequency_composites(pos_HWdays, cp_daily_seasonal_anoma_mean, 0, test_excedence=False) # interested in when the seasonal anomaly is negative
#     np.save(f'{path_outputs_case}frequency_cp_seasonal_anoma_yb_{region}.npy', frequency_cp_seasonal_anoma_mean)


aaaa





# ==================================================== u ==============================================================================
variable_u = 'u'

path_u_daily = f'{path_outputs}u250_daily.nc'
path_u_annual_cycle = f'{path_outputs}annual_cycle_u250_daily.nc'
path_u_daily_seasonal_anoma = f'{path_outputs}u250_daily_smoothed_seasonal_anoma.nc'
if not os.path.exists(path_u_daily):
    ncfile_u = Dataset(f'{path_file_u}')
    u = np.array(ncfile_u[variable_u][pos_initial_year:])
    u_daily = u.reshape(-1, 4, u.shape[1], u.shape[2]).astype('float32')
    u_daily = np.nanmean(u_daily, axis=1)  # Average over the 4 time steps per day
    pos_midlat = np.where(np.isin(np.array(ncfile_u['lat']), lats_env))[0]
    u_daily = u_daily[:, pos_midlat, :]
    save_nc_3d(path_u_daily, u_daily, lats_env, lons_env, dates_env_str, variable_u)

if not os.path.exists(path_u_annual_cycle):
    ds = xr.open_dataset(path_u_daily, decode_times=False, engine="netcdf4")
    ds['time'] = np.array([pd.to_datetime(iii, format="%Y%m%d") for iii in ds['time'].values])
    print(ds['time'].values[:3])
    u_daily = ds[variable_u]
    print("u_daily shape:", u_daily.shape)

    # compute annual cycle
    annual_cycle_0 = compute_annual_cycle_window(u_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
    print("annual_cycle_0 shape:", annual_cycle_0.shape)
    save_nc_3d(path_u_annual_cycle, annual_cycle_0, lats_env, lons_env, np.arange(1, len(annual_cycle_0.time)+1).astype(str), variable_u)

    del u_daily, annual_cycle_0

if not os.path.exists(path_u_daily_seasonal_anoma):
    ds = xr.open_dataset(path_u_daily, decode_times=False, engine="netcdf4")
    u_seasonal_anoma = seasonal_anomalies_by_year_optimized(dates_env, ds[variable_u].astype('float32')).astype('float32')
    save_nc_3d(path_u_daily_seasonal_anoma, u_seasonal_anoma, lats_env, lons_env, dates_env_str, variable_u)
    del u_seasonal_anoma

if not os.path.exists(f'{path_outputs_case}composites_matriz_u_daily_yb_{region}_mean3days.npy'):
    composites_matriz_u_daily, composites_matrix_complete_u, pvalue_composites_matriz_u_daily = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_u_daily, variable_u, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
    np.save(f'{path_outputs_case}composites_matriz_u_daily_yb_{region}_mean3days.npy', composites_matriz_u_daily)
    np.save(f'{path_outputs_case}composites_matriz_u_daily_pvalues_yb_{region}_mean3days.npy', pvalue_composites_matriz_u_daily)
    composites_matrix_complete_u_mean = np.nanmean(composites_matrix_complete_u[:,pos_lats_midlat], axis=1)
    np.save(f'{path_outputs_case}composites_matrix_complete_u_yb_{region}_mean3days.npy', composites_matrix_complete_u_mean)
    hovmoller_anoma_u_filtered_mean = np.roll(composites_matrix_complete_u_mean,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)

if not os.path.exists(f'{path_outputs_case}composites_matrix_complete_u_seasonal_anoma_yb_{region}.npy'):
    composites_matriz_u_synoptic_seasonal_anoma, composites_matrix_complete_u_anoma, pvalue_composites_matriz_u_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_u_daily_seasonal_anoma, variable_u, num_time_laps, pvalue=True, welch_test=True)
    np.save(f'{path_outputs_case}composites_matriz_u_seasonal_anoma_yb_{region}.npy', composites_matriz_u_synoptic_seasonal_anoma)
    np.save(f'{path_outputs_case}composites_matriz_u_seasonal_anoma_pvalues_welch_yb_{region}.npy', pvalue_composites_matriz_u_seasonal_anoma)

if not os.path.exists(f'{path_outputs_case}composites_matrix_complete_u_seasonal_anoma_yb_{region}_mean3days.npy'):
    composites_matriz_u_synoptic_seasonal_anoma, composites_matrix_complete_u_anoma, pvalue_composites_matriz_u_seasonal_anoma = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_u_daily_seasonal_anoma, variable_u, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
    np.save(f'{path_outputs_case}composites_matriz_u_seasonal_anoma_yb_{region}_mean3days.npy', composites_matriz_u_synoptic_seasonal_anoma)
    np.save(f'{path_outputs_case}composites_matriz_u_seasonal_anoma_pvalues_welch_yb_{region}_mean3days.npy', pvalue_composites_matriz_u_seasonal_anoma)


# pos_season = np.where([i.month in [6, 7, 8] for i in dates_env])[0]
# pos_season = pos_season[pos_season<min(len(dates_env),len(dates_d))]
# ncfile_u_daily = Dataset(path_u_daily)
# u_season = np.nanmean(np.array(ncfile_u_daily[variable_u][pos_season]), axis=0)
# np.save(f'{path_outputs}climatology_summer_u.npy', u_season)




# ==================================================== v  ==============================================================================
variable_v = 'v_wnf'

path_v_daily_anoma_filtered = f'{path_outputs}v250_daily_anoma_filtered.nc'
path_v_annual_cycle = f'{path_outputs}annual_cycle_v250_daily_anoma_filtered.nc'

if not os.path.exists(path_v_daily_anoma_filtered):
    ncfile_v = Dataset(f'{path_file_v}')
    v = np.array(ncfile_v[variable_v][pos_initial_year:])
    v_daily = v.reshape(-1, 4, v.shape[1], v.shape[2]).astype('float32')
    v_daily = np.nanmean(v_daily, axis=1)  # Average over the 4 time steps per day
    save_nc_3d(path_v_daily_anoma_filtered, v_daily, lats_env, lons_env, dates_env_str, variable_v)

    ds = xr.open_dataset(path_v_daily_anoma_filtered, decode_times=False, engine="netcdf4")
    ds['time'] = np.array([pd.to_datetime(iii, format="%Y%m%d") for iii in ds['time'].values])
    print(ds['time'].values[:3])
    v_daily = ds[variable_v]
    print("v_daily shape:", v_daily.shape)

    # compute annual cycle
    annual_cycle_0 = compute_annual_cycle_window(v_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
    print("annual_cycle_0 shape:", annual_cycle_0.shape)
    save_nc_3d(path_v_annual_cycle, annual_cycle_0, lats_env, lons_env, np.arange(1, len(annual_cycle_0.time)+1).astype(str), variable_v)

    del v_daily, annual_cycle_0

if not os.path.exists(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_yb_{region}_mean3days.npy'):

    composites_matriz_v_daily_anoma_filtered, composites_matrix_complete_v_anoma_filtered, pvalue_composites_matriz_v_daily_anoma_filtered = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_v_daily_anoma_filtered, variable_v, num_time_laps, pvalue=True, welch_test=True, mean_3days=True)
    np.save(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_yb_{region}_mean3days.npy', composites_matriz_v_daily_anoma_filtered)
    np.save(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_pvalues_yb_{region}_mean3days.npy', pvalue_composites_matriz_v_daily_anoma_filtered)
    composites_matrix_complete_v_anoma_filtered_mean = np.nanmean(composites_matrix_complete_v_anoma_filtered[:,pos_lats_midlat], axis=1)
    np.save(f'{path_outputs_case}composites_matrix_complete_v_anoma_filtered_yb_{region}_mean3days.npy', composites_matrix_complete_v_anoma_filtered_mean)
    hovmoller_anoma_v_filtered_mean = np.roll(composites_matrix_complete_v_anoma_filtered_mean,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)

if not os.path.exists(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_yb_{region}.npy'):

    composites_matriz_v_daily_anoma_filtered, composites_matrix_complete_v_anoma_filtered, pvalue_composites_matriz_v_daily_anoma_filtered = calculate_composites_ttest_optimized(pos_HW_day0, pos_summer, path_v_daily_anoma_filtered, variable_v, num_time_laps, pvalue=True, welch_test=True)
    np.save(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_yb_{region}.npy', composites_matriz_v_daily_anoma_filtered)
    np.save(f'{path_outputs_case}composites_matriz_v_daily_anoma_filtered_pvalues_yb_{region}.npy', pvalue_composites_matriz_v_daily_anoma_filtered)
    composites_matrix_complete_v_anoma_filtered_mean = np.nanmean(composites_matrix_complete_v_anoma_filtered[:,pos_lats_midlat], axis=1)
    np.save(f'{path_outputs_case}composites_matrix_complete_v_anoma_filtered_yb_{region}.npy', composites_matrix_complete_v_anoma_filtered_mean)
    hovmoller_anoma_v_filtered_mean = np.roll(composites_matrix_complete_v_anoma_filtered_mean,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)

















# ==================================================== plotting  ==============================================================================
pos_middle_HW = np.where(abs(lons_env - (lon_minHW+lon_maxHW)/2) == np.min(abs(lons_env - (lon_minHW+lon_maxHW)/2)))[0][0]


colors = ['#FFFFF0',    # white
          '#FFE5B4',    # Light orange
          '#FFA500',    # Orange
          '#FF4500',    # Red-orange
          '#8C000F',    # Red
          '#FFFF00']    # Yellow
li_env, ls_env, steps, colormap_env = colormap_custom(10,24,15,colors[:])
li_cp, ls_cp, colormap_cp, bounds_cp = colorm(0,8,15,1.5,'RdYlBu_r')


composites_matriz_env = np.load(f'{path_outputs_case}composites_matriz_env_yb_{region}.npy')
pvalue_composites_matriz_env = np.load(f'{path_outputs_case}composites_matriz_env_pvalues_yb_{region}.npy')
composites_matriz_cp = np.load(f'{path_outputs_case}composites_matriz_cp_yb_{region}.npy')
pvalue_composites_matriz_cp = np.load(f'{path_outputs_case}composites_matriz_cp_pvalues_yb_{region}.npy')


composites_matriz_cp_synoptic_anoma_window = np.load(f'{path_outputs_case}composites_matriz_cp_daily_anoma_window_yb_{region}.npy')
pvalue_composites_matriz_cp_anoma_window = np.load(f'{path_outputs_case}composites_matriz_cp_daily_anoma_window_pvalues_yb_{region}.npy')
composites_matriz_cp_synoptic_seasonal_anoma = np.load(f'{path_outputs_case}composites_matriz_cp_seasonal_anoma_yb_{region}.npy')
pvalue_composites_matriz_cp_seasonal_anoma = np.load(f'{path_outputs_case}composites_matriz_cp_seasonal_anoma_pvalues_yb_{region}.npy')

composites_coastlines_2var(lats_env, lons_env, composites_matriz_env, li_env, ls_env, pvalue_composites_matriz_env, composites_matriz_env, [200], lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap_env, path_figures + 'Composites_coastlines_env_synoptic_yb.png', var = '', topography = True, pvalue=True, center_lon = lons_env[pos_middle_HW])
composites_matriz_cp = np.array([gaussian_filter_with_nans(composites_matriz_cp[t], sigma=4) for t in range(composites_matriz_cp.shape[0])])
composites_coastlines_2var(lats_env, lons_env, composites_matriz_cp, li_cp, ls_cp, pvalue_composites_matriz_cp, composites_matriz_cp, [200], lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap_cp, path_figures + 'Composites_coastlines_cp_synoptic_yb.png', var = '', topography = True, pvalue=True, center_lon = lons_env[pos_middle_HW])

li_anom, ls_anom, colormap_anom, bounds_anom = colorm(-4,4,20,0.04,'BrBG')
composites_matriz_cp_synoptic_anoma_window = np.array([gaussian_filter_with_nans(composites_matriz_cp_synoptic_anoma_window[t], sigma=4) for t in range(composites_matriz_cp_synoptic_anoma_window.shape[0])])
composites_coastlines_2var(lats_env, lons_env, composites_matriz_cp_synoptic_anoma_window, li_anom, ls_anom, pvalue_composites_matriz_cp_anoma_window, composites_matriz_cp_synoptic_anoma_window, [200], lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap_anom, path_figures + 'Composites_coastlines_cp_synoptic_daily_anoma_window_yb.png', var = '', topography = True, pvalue=True, center_lon = lons_env[pos_middle_HW])

composites_matriz_cp_synoptic_seasonal_anoma = np.array([gaussian_filter_with_nans(composites_matriz_cp_synoptic_seasonal_anoma[t], sigma=4) for t in range(composites_matriz_cp_synoptic_seasonal_anoma.shape[0])])
composites_coastlines_2var(lats_env, lons_env, composites_matriz_cp_synoptic_seasonal_anoma, li_anom, ls_anom, pvalue_composites_matriz_cp_seasonal_anoma, composites_matriz_cp_synoptic_seasonal_anoma, [200], lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap_anom, path_figures + 'Composites_coastlines_cp_synoptic_seasonal_anoma_yb.png', var = '', topography = True, pvalue=True, center_lon = lons_env[pos_middle_HW])


# Climatological maps
ncfile_E_daily = Dataset(path_envelope_daily)
ncfile_E_daily_anoma_window = Dataset(path_envelope_daily_anoma_window)
ncfile_cp_daily = Dataset(path_cp_daily)
ncfile_cp_daily_anoma_window = Dataset(path_cp_daily_anoma_window)
ncfile_cp_daily_seasonal_anoma = Dataset(path_cp_daily_seasonal_anoma)

for season, months in [('summer', [6, 7, 8]), ('winter', [12, 1, 2]), ('spring', [3, 4, 5]), ('autumn', [9, 10, 11])]:
    pos_season = np.where([i.month in months for i in dates_env])[0]
    pos_season = pos_season[pos_season<min(len(dates_env),len(dates_d))]

    env_season = np.nanmean(np.array(ncfile_E_daily[variable_env][pos_season]), axis=0)

    li_env_season, ls_env_season, steps, colormap_env_season = colormap_custom(5,30,15,colors[:])
    maps_midlat(lons_env, lats_env, li_env_season, ls_env_season, env_season, colormap_env_season, f'{path_figures_all}Climatological_env_seasons_v2_{season}_yb.png', topography = True, units='', center_lon=lons_env[pos_middle_HW])

    cp_season = np.nanmean(np.array(ncfile_cp_daily[variable_cp][pos_season]), axis=0)
    maps_midlat(lons_env, lats_env, li_cp, ls_cp, cp_season, colormap_cp, f'{path_figures_all}Climatological_cp_seasons_{season}_yb.png', topography = True, units='', center_lon=lons_env[pos_middle_HW])




if not os.path.exists(f'{path_figures_all}Climatological_env_summer_yb.png'):
    env_summer = np.nanmean(np.array(ncfile_E_daily[variable_env][pos_summer]), axis=0)
    maps_midlat(lons_env, lats_env, li_env, ls_env, env_summer, colormap_env, f'{path_figures_all}Climatological_env_summer_yb.png', topography = True, units='', center_lon=lons_env[pos_middle_HW])

    cp_summer = np.nanmean(np.array(ncfile_cp_daily[variable_cp][pos_summer]), axis=0)
    maps_midlat(lons_env, lats_env, li_cp, ls_cp, cp_summer, colormap_cp, f'{path_figures_all}Climatological_cp_summer_yb.png', topography = True, units='', center_lon=lons_env[pos_middle_HW])

    cp_summer_seasonal_anoma = np.nanmean(np.array(ncfile_cp_daily_seasonal_anoma[variable_cp][pos_summer]), axis=0)
    maps_midlat(lons_env, lats_env, li_anom, ls_anom, cp_summer_seasonal_anoma, colormap_anom, f'{path_figures_all}Climatological_cp_summer_seasonal_anoma_yb.png', topography = True, units='', center_lon=lons_env[pos_middle_HW])



# # ==================================================== hovmoller climatologies  ==============================================================================
# #BORRAR#BORRAR#BORRAR
# #BORRAR#BORRAR#BORRAR
# #BORRAR#BORRAR#BORRAR
# #BORRAR#BORRAR#BORRAR
# env_anoma_midlat_mean = np.nanmean(np.array(ncfile_E_daily_anoma_window[variable_env][:,pos_lats_midlat,:]), axis=1)
# df_env_anoma_midlat = pd.DataFrame(index=pd.DatetimeIndex(dates_env), data=env_anoma_midlat_mean).astype('float32')
# df_env_anoma_midlat['dayofyear'] = df_env_anoma_midlat.index.dayofyear

# cp_anoma_midlat_mean = np.nanmean(np.array(ncfile_cp_daily_anoma_window[variable_cp][:,pos_lats_midlat,:]), axis=1)
# df_cp_anoma_midlat = pd.DataFrame(index=pd.DatetimeIndex(dates_env), data=cp_anoma_midlat_mean).astype('float32')
# df_cp_anoma_midlat['dayofyear'] = df_cp_anoma_midlat.index.dayofyear


# df_slow_amplified_midlat_boolean = (df_cp_anoma_midlat < 0) & (df_env_anoma_midlat.drop(columns='dayofyear', errors='ignore') > 0)
# df_slow_amplified_midlat_boolean['dayofyear'] = df_slow_amplified_midlat_boolean.index.dayofyear
# true_counts_per_doy = (df_slow_amplified_midlat_boolean.groupby('dayofyear').sum(numeric_only=True))
# np.save(f'{path_outputs}frequency_slow_amplified_daily_anoma_window.npy', true_counts_per_doy)


# # env_midlat_mean = np.nanmean(np.array(ncfile_E_daily[variable_env][:,pos_lats_midlat,:]), axis=1)
# # df_env_midlat = pd.DataFrame(index=pd.DatetimeIndex(dates_env), data=env_midlat_mean).astype('float32')
# # df_env_midlat['dayofyear'] = df_env_midlat.index.dayofyear
# # env_climatology = np.array(df_env_midlat.groupby('dayofyear').mean())
# # np.save(f'{path_outputs}env_synoptic_midlat_climatology.npy', env_climatology)
# # env_climatology = np.load(f'{path_outputs}env_synoptic_midlat_climatology.npy')
# # hovmoller_env_climatology = np.roll(env_climatology,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)
# # time_lags = np.arange(0, 366, 1)
# # hovmoller_onevar_summer(time_lags, np.linspace(-180,180,len(lons_env)), hovmoller_env_climatology, colormap_env,li_env, ls_env,steps, path_figures_all + f'Hovmoller_env_synoptic_climatology.png', var_1 = '')


# # cp_midlat_mean = np.nanmean(np.array(ncfile_cp_daily[variable_cp][:,pos_lats_midlat,:]), axis=1)
# # df_cp_midlat = pd.DataFrame(index=pd.DatetimeIndex(dates_env), data=cp_midlat_mean).astype('float32')
# # df_cp_midlat['dayofyear'] = df_cp_midlat.index.dayofyear
# # cp_climatology = np.array(df_cp_midlat.groupby('dayofyear').mean())
# # np.save(f'{path_outputs}cp_synoptic_midlat_climatology.npy', cp_climatology)
# # cp_climatology = np.load(f'{path_outputs}cp_synoptic_midlat_climatology.npy')
# # hovmoller_cp_climatology = np.roll(cp_climatology,  -(round(abs(pos_middle_HW-len(lons_env)/2))), axis = 1)
# # time_lags = np.arange(0, 366, 1)
# # hovmoller_onevar_summer(time_lags, np.linspace(-180,180,len(lons_env)), hovmoller_cp_climatology, colormap_cp,li_cp, ls_cp,steps, path_figures_all + f'Hovmoller_cp_synoptic_climatology.png', var_1 = '')










# Define heatwave region for phase speed averaging
# lon_mincp = 235 - 20
# lon_maxcp = 260 - 20
lon_mincp = 265 - 90
lon_maxcp = 265
lat_mincp = 25 + 10
lat_maxcp = 50 + 5
pos_lats_hw_cp = np.where((lats_env >= lat_mincp) & (lats_env <= lat_maxcp))[0]
pos_lons_hw_cp = np.where((lons_env >= lon_mincp) & (lons_env <= lon_maxcp))[0]

# Get heatwave and non-heatwave indices
# pos_HW_indices = pos_HWdays.values.flatten()
pos_HW_indices = pos_HWdays.values.flatten()
pos_non_HW_indices = pos_summer

# Extract temperature data directly for heatwave and non-heatwave periods only
# Temperature data from ncfile_t (already loaded)
temp_hw_data = np.array(ncfile_t['TS'][pos_HW_indices, :, :])  # Only heatwave days
temp_non_hw_data = np.array(ncfile_t['TS'][pos_non_HW_indices,:, :])  # Only non-heatwave days

# Define heatwave region for temperature averaging (using temperature file coordinates)
lats_t = np.array(ncfile_t['lat'])
lons_t = np.array(ncfile_t['lon'])
pos_lats_hw_temp = np.where((lats_t >= lat_minHW) & (lats_t <= lat_maxHW))[0]
pos_lons_hw_temp = np.where((lons_t >= lon_minHW) & (lons_t <= lon_maxHW))[0]


# Calculate temperature indices (mean over heatwave region) for each group
temp_hw = np.nanmean(temp_hw_data[:, pos_lats_hw_temp, :], axis=1)  # Average over latitudes
temp_hw = np.nanmean(temp_hw[:, pos_lons_hw_temp], axis=1)  # Average over longitudes

temp_non_hw = np.nanmean(temp_non_hw_data[:, pos_lats_hw_temp, :], axis=1)  # Average over latitudes
temp_non_hw = np.nanmean(temp_non_hw[:, pos_lons_hw_temp], axis=1)  # Average over longitudes

# for variable, ncfile, output_var_name in [(variable_env, ncfile_E_daily_anoma_window, 'E_daily_anoma_window'), (f'{variable_cp}', ncfile_cp_daily_anoma_window, 'Phase_speed_daily_anoma_window'), (variable_cp, ncfile_cp_daily_seasonal_anoma, 'Phase_speed_seasonal_anoma'), (variable_env, ncfile_E_daily, 'E')]:
for variable, ncfile, output_var_name in [(f'{variable_cp}', ncfile_cp_daily, 'Phase_speed')]:

    # ===================================== Phase Speed Analysis: Heatwaves vs Summer ================================================================
    print(f"Starting {variable} analysis...")

    # ==========================================================
    # 1) COMBINE ALL LAGS INTO ONE DISTRIBUTION (HW ONLY)
    # ==========================================================

    lag_range = range(3, 8)   # 3, 4, 5, 6, 7

    cp_hw_all_lags = []

    for L in lag_range:
        print(f"Processing lag {L}...")
        cp_lagged = lag_array(np.array(ncfile[variable]), L)  # [time, lat, lon]

        # extract region & remove NaNs (only heatwave days)
        # step 1: subset in time and latitude
        hw_L = cp_lagged[pos_HW_indices][:, :, :]  # first extract time
        hw_L = hw_L[:, pos_lats_hw_cp, :]          # then lat
        hw_L = hw_L[:, :, pos_lons_hw_cp]          # then lon
        hw_L = np.nanmean(hw_L, axis=(1,2))


        cp_hw_all_lags.extend(hw_L[~np.isnan(hw_L)])

    cp_hw_all_lags = np.array(cp_hw_all_lags)

    # ==========================================================
    # 1b) NON-HW = ALL SUMMER DATA (UNLAGGED)
    # ==========================================================
    # NOTE: here pos_non_HW_indices should represent ALL summer days you want,
    # not only non-HW if you truly intend "all summer".
    cp_summer_3d = np.array(ncfile[variable][pos_non_HW_indices, :, :])   # [time_summer, lat, lon]
    cp_non_hw = cp_summer_3d[:, pos_lats_hw_cp, :]
    cp_non_hw = cp_non_hw[:, :, pos_lons_hw_cp]
    cp_non_hw = np.nanmean(cp_non_hw, axis=(1,2))
    cp_non_hw = cp_non_hw[np.isfinite(cp_non_hw)]

    print("Combined lag data:")
    print(f" HW samples (all lags combined): {len(cp_hw_all_lags)}")
    print(f" Summer samples (all days): {len(cp_non_hw)}")

    # ==========================================================
    # 2) CALCULATE STATS *BEFORE* PLOTTING
    # ==========================================================

    x_vals = np.linspace(
        min(cp_hw_all_lags.min(), cp_non_hw.min()),
        max(cp_hw_all_lags.max(), cp_non_hw.max()),
        300
    )

    kde_hw = gaussian_kde(cp_hw_all_lags)
    kde_non = gaussian_kde(cp_non_hw)

    y_hw = kde_hw(x_vals)
    y_non = kde_non(x_vals)

    mode_hw = x_vals[np.argmax(y_hw)]
    mode_non = x_vals[np.argmax(y_non)]

    std_hw = np.nanstd(cp_hw_all_lags)
    std_non = np.nanstd(cp_non_hw)

    # ==========================================================
    # 3) TEMPERATURE STATS
    # ==========================================================

    valid_hw_temp = ~np.isnan(temp_hw)
    valid_non_temp = ~np.isnan(temp_non_hw)

    temp_hw_clean = temp_hw[valid_hw_temp]
    temp_non_hw_clean = temp_non_hw[valid_non_temp]

    x_temp_vals = np.linspace(
        min(temp_hw_clean.min(), temp_non_hw_clean.min()),
        max(temp_hw_clean.max(), temp_non_hw_clean.max()),
        300
    )

    kde_hw_temp = gaussian_kde(temp_hw_clean)
    kde_non_temp = gaussian_kde(temp_non_hw_clean)

    y_temp_hw = kde_hw_temp(x_temp_vals)
    y_temp_non = kde_non_temp(x_temp_vals)

    mode_hw_temp = x_temp_vals[np.argmax(y_temp_hw)]
    mode_non_temp = x_temp_vals[np.argmax(y_temp_non)]

    std_hw_temp = np.nanstd(temp_hw_clean)
    std_non_temp = np.nanstd(temp_non_hw_clean)

    # ==========================================================
    # 4) PLOT PDFs
    # ==========================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # =======================
    # PHASE SPEED PANEL
    # =======================

    ax1.hist(cp_hw_all_lags, bins=35, density=True, alpha=0.55, color='red')
    ax1.hist(cp_non_hw,      bins=35, density=True, alpha=0.55, color='blue')

    ax1.plot(x_vals, y_hw,  'r-', lw=2)
    ax1.plot(x_vals, y_non, 'b-', lw=2)

    ax1.set_xlabel(f'{variable} (m/s)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'{variable}: Combined lag distribution (HW lags 3–7 vs. all summer)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax1.legend([
        f'HW lags 3–7 (n={len(cp_hw_all_lags)}, mode={mode_hw:.2f}, σ={std_hw:.2f})',
        f'All summer (n={len(cp_non_hw)}, mode={mode_non:.2f}, σ={std_non:.2f})'
    ])

    # =======================
    # TEMPERATURE PANEL
    # =======================

    ax2.hist(temp_hw_clean,      bins=35, density=True, alpha=0.55, color='red')
    ax2.hist(temp_non_hw_clean,  bins=35, density=True, alpha=0.55, color='blue')

    ax2.plot(x_temp_vals, y_temp_hw,  'r-', lw=2)
    ax2.plot(x_temp_vals, y_temp_non, 'b-', lw=2)

    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Temperature KDE Distribution (HW vs. all summer)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax2.legend([
        f'HW (n={len(temp_hw_clean)}, mode={mode_hw_temp:.2f}, σ={std_hw_temp:.2f})',
        f'All summer (n={len(temp_non_hw_clean)}, mode={mode_non_temp:.2f}, σ={std_non_temp:.2f})'
    ])

    plt.tight_layout()
    plt.savefig(path_figures + f'{output_var_name}_PDF_KDE_HW_vs_all_summer_lag3to7.png',
                dpi=500, bbox_inches='tight')
    plt.close()


    # # # ===================================== Spatial correlation analysis ================================================================

    # corr_cp_temp_non_hw_2d = np.zeros((len(pos_lats_hw_temp), len(pos_lons_hw_temp)))
    # for i, pos_lat in enumerate(pos_lats_hw_temp):
    #     for j, pos_lon in enumerate(pos_lons_hw_temp):
    #         temp_i = temp_non_hw_data[:,pos_lat,pos_lon]
    #         valid_i = ~(np.isnan(cp_non_hw) | np.isnan(temp_i))
    #         corr_cp_temp_non_hw, pval_cp_temp_non_hw = stats.pearsonr(cp_non_hw[valid_i], temp_i[valid_i])
    #         print(f"Correlation between phase speed and temperature at {lats_t[pos_lat]}°N, {lons_t[pos_lon]}°E: {corr_cp_temp_non_hw:.3f}")
    #         print(f"P-value between phase speed and temperature at {lats_t[pos_lat]}°N, {lons_t[pos_lon]}°E: {pval_cp_temp_non_hw:.3f}")
    #         corr_cp_temp_non_hw_2d[i, j] = corr_cp_temp_non_hw
    #         np.save(path_figures + f'corr_{output_var_name}_temp_summer.npy', corr_cp_temp_non_hw_2d)

    # li_anom, ls_anom, colormap_anom, bounds_anom = colorm(-0.4,0.4,15,0.04,'RdBu')
    # maps_USA(lons_t[pos_lons_hw_temp], np.ndarray.round(lats_t[pos_lats_hw_temp],2), -0.4, 0.4, corr_cp_temp_non_hw_2d, r'Correlation between phase speed and temperature',
    #     colormap_anom, path_figures + f'corr_{output_var_name}_temp_summer_lag{lag}d.png', True)  

    # valid_i = ~(np.isnan(cp_non_hw) | np.isnan(temp_non_hw))
    # corr_cp_temp_non_hw, pval_cp_temp_non_hw = stats.pearsonr(cp_non_hw[valid_i], temp_non_hw[valid_i])
    # corr_cp_temp_hw, pval_cp_temp_hw = stats.pearsonr(cp_hw[valid_hw], temp_hw[valid_hw])
    # print(f"Correlation between {output_var_name} and temperature: {corr_cp_temp_non_hw:.3f}")
    # print(f"P-value between {output_var_name} and temperature: {pval_cp_temp_non_hw:.3f}")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(cp_non_hw[valid_i], temp_non_hw[valid_i], alpha=0.5, label=f'Summer r={corr_cp_temp_non_hw:.2f}')
    # ax.scatter(cp_hw[valid_hw], temp_hw[valid_hw], alpha=0.5, color='red', label=f'Heatwaves r={corr_cp_temp_hw:.2f}')
    # ax.set_xlabel(f'{variable} (m/s)')
    # ax.set_ylabel('Temperature (°C)')
    # ax.set_title(f'Correlation between {variable} and temperature')
    # ax.grid(True)
    # ax.legend()
    # plt.savefig(path_figures + f'{output_var_name}_temperature_scatter_HW_vs_summer_lag{lag}d.png', dpi=500, bbox_inches='tight')
    # plt.close()



aaaaa
for variable, ncfile, output_var_name in [(variable_env, ncfile_E_daily_anoma_window, 'E_daily_anoma_window'), (f'{variable_cp}', ncfile_cp_daily_anoma_window, 'Phase_speed_daily_anoma_window'), (variable_cp, ncfile_cp_daily_seasonal_anoma, 'Phase_speed_seasonal_anoma'), (variable_env, ncfile_E_daily, 'E')]:

    # ===================================== Phase Speed Analysis: Heatwaves vs Summer ================================================================

    print(f"Starting {variable} analysis...")

    # Extract phase speed data directly for heatwave and non-heatwave periods only
    # Phase speed data from ncfile_cp_daily (already loaded)
    cp_hw_data = np.array(ncfile[variable][pos_HW_indices, :, :])  # Only heatwave days
    cp_non_hw_data = np.array(ncfile[variable][pos_non_HW_indices, :, :])  # Only non-heatwave days

    # Calculate phase speed indices (mean over heatwave region) for each group
    cp_hw = np.nanmean(cp_hw_data[:, pos_lats_hw_cp, :], axis=1)  # Average over latitudes
    cp_hw = np.nanmean(cp_hw[:, pos_lons_hw_cp], axis=1)  # Average over longitudes
    cp_non_hw = np.nanmean(cp_non_hw_data[:, pos_lats_hw_cp, :], axis=1)  # Average over latitudes
    cp_non_hw = np.nanmean(cp_non_hw[:, pos_lons_hw_cp], axis=1)  # Average over longitudes



    # Remove NaN values
    valid_hw = ~(np.isnan(cp_hw) | np.isnan(temp_hw))
    valid_non_hw = ~(np.isnan(cp_non_hw) | np.isnan(temp_non_hw))

    cp_hw_clean = cp_hw[valid_hw]
    cp_non_hw_clean = cp_non_hw[valid_non_hw]
    temp_hw_clean = temp_hw[valid_hw]
    temp_non_hw_clean = temp_non_hw[valid_non_hw]

    print(f"Heatwave days: {len(cp_hw_clean)}")
    print(f"Non-heatwave days: {len(cp_non_hw_clean)}")



    # ===================================== Approach 1: Probability Distribution Function (PDF) ================================================================

    from scipy import stats
    from scipy.stats import norm

    # Create PDF comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Phase speed PDFs
    ax1.hist(cp_hw_clean, bins=30, density=True, alpha=0.7, color='red', label=f'Heatwaves (n={len(cp_hw_clean)})')
    ax1.hist(cp_non_hw_clean, bins=30, density=True, alpha=0.7, color='blue', label=f'Summer (n={len(cp_non_hw_clean)})')

    # Fit normal distributions
    mu_hw_cp, sigma_hw_cp = norm.fit(cp_hw_clean)
    mu_non_hw_cp, sigma_non_hw_cp = norm.fit(cp_non_hw_clean)

    # Plot fitted normal distributions
    x_cp = np.linspace(min(np.min(cp_hw_clean), np.min(cp_non_hw_clean)), 
                    max(np.max(cp_hw_clean), np.max(cp_non_hw_clean)), 100)
    ax1.plot(x_cp, norm.pdf(x_cp, mu_hw_cp, sigma_hw_cp), 'r--', linewidth=2, 
            label=f'HW Normal: μ={mu_hw_cp:.2f}, σ={sigma_hw_cp:.2f}')
    ax1.plot(x_cp, norm.pdf(x_cp, mu_non_hw_cp, sigma_non_hw_cp), 'b--', linewidth=2,
            label=f'Summer Normal: μ={mu_non_hw_cp:.2f}, σ={sigma_non_hw_cp:.2f}')

    ax1.set_xlabel(f'{variable} (m/s)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'{variable} Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Temperature PDFs
    ax2.hist(temp_hw_clean, bins=30, density=True, alpha=0.7, color='red', label=f'Heatwaves (n={len(temp_hw_clean)})')
    ax2.hist(temp_non_hw_clean, bins=30, density=True, alpha=0.7, color='blue', label=f'Summer (n={len(temp_non_hw_clean)})')

    # Fit normal distributions
    mu_hw_temp, sigma_hw_temp = norm.fit(temp_hw_clean)
    mu_non_hw_temp, sigma_non_hw_temp = norm.fit(temp_non_hw_clean)

    # Plot fitted normal distributions
    x_temp = np.linspace(min(np.min(temp_hw_clean), np.min(temp_non_hw_clean)), 
                        max(np.max(temp_hw_clean), np.max(temp_non_hw_clean)), 100)
    ax2.plot(x_temp, norm.pdf(x_temp, mu_hw_temp, sigma_hw_temp), 'r--', linewidth=2,
            label=f'HW Normal: μ={mu_hw_temp:.2f}, σ={sigma_hw_temp:.2f}')
    ax2.plot(x_temp, norm.pdf(x_temp, mu_non_hw_temp, sigma_non_hw_temp), 'b--', linewidth=2,
            label=f'Summer Normal: μ={mu_non_hw_temp:.2f}, σ={sigma_non_hw_temp:.2f}')

    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Temperature Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_figures + f'{output_var_name}_temperature_PDF_HW_vs_summer.png', dpi=500, bbox_inches='tight')
    plt.close()



    # # ===================================== Spatial correlation analysis ================================================================

    corr_cp_temp_non_hw_2d = np.zeros((len(pos_lats_hw_temp), len(pos_lons_hw_temp)))
    for i, pos_lat in enumerate(pos_lats_hw_temp):
        for j, pos_lon in enumerate(pos_lons_hw_temp):
            temp_i = temp_non_hw_data[:,pos_lat,pos_lon]
            valid_i = ~(np.isnan(cp_non_hw) | np.isnan(temp_i))
            corr_cp_temp_non_hw, pval_cp_temp_non_hw = stats.pearsonr(cp_non_hw[valid_i], temp_i[valid_i])
            print(f"Correlation between phase speed and temperature at {lats_t[pos_lat]}°N, {lons_t[pos_lon]}°E: {corr_cp_temp_non_hw:.3f}")
            print(f"P-value between phase speed and temperature at {lats_t[pos_lat]}°N, {lons_t[pos_lon]}°E: {pval_cp_temp_non_hw:.3f}")
            corr_cp_temp_non_hw_2d[i, j] = corr_cp_temp_non_hw
            np.save(path_figures + f'corr_{output_var_name}_temp_summer.npy', corr_cp_temp_non_hw_2d)

    li_anom, ls_anom, colormap_anom, bounds_anom = colorm(-0.4,0.4,15,0.04,'RdBu')
    maps_USA(lons_t[pos_lons_hw_temp], np.ndarray.round(lats_t[pos_lats_hw_temp],2), -0.4, 0.4, corr_cp_temp_non_hw_2d, r'Correlation between phase speed and temperature',
        colormap_anom, path_figures + f'corr_{output_var_name}_temp_summer.png', True)  
















# # ===================================== Statistical Tests ================================================================

# from scipy.stats import ttest_ind, ks_2samp

# # Statistical tests for phase speed
# tstat_cp, pval_cp = ttest_ind(cp_hw_clean, cp_non_hw_clean)
# ksstat_cp, pval_ks_cp = ks_2samp(cp_hw_clean, cp_non_hw_clean)

# # Statistical tests for temperature
# tstat_temp, pval_temp = ttest_ind(temp_hw_clean, temp_non_hw_clean)
# ksstat_temp, pval_ks_temp = ks_2samp(temp_hw_clean, temp_non_hw_clean)

# # Print results
# print("\n" + "="*60)
# print("STATISTICAL ANALYSIS RESULTS")
# print("="*60)
# print(f"Phase Speed:")
# print(f"  Heatwaves:     mean = {np.mean(cp_hw_clean):.3f} m/s, std = {np.std(cp_hw_clean):.3f} m/s")
# print(f"  Summer: mean = {np.mean(cp_non_hw_clean):.3f} m/s, std = {np.std(cp_non_hw_clean):.3f} m/s")
# print(f"  t-test: t = {tstat_cp:.3f}, p = {pval_cp:.3f}")
# print(f"  KS-test: D = {ksstat_cp:.3f}, p = {pval_ks_cp:.3f}")
# print(f"\nTemperature:")
# print(f"  Heatwaves:     mean = {np.mean(temp_hw_clean):.3f} °C, std = {np.std(temp_hw_clean):.3f} °C")
# print(f"  Summer: mean = {np.mean(temp_non_hw_clean):.3f} °C, std = {np.std(temp_non_hw_clean):.3f} °C")
# print(f"  t-test: t = {tstat_temp:.3f}, p = {pval_temp:.3f}")
# print(f"  KS-test: D = {ksstat_temp:.3f}, p = {pval_ks_temp:.3f}")
# print(f"\nCorrelations:")
# print(f"  Heatwaves:     r = {corr_hw:.3f}, p = {pval_hw:.3f}")
# print(f"  Summer: r = {corr_non_hw:.3f}, p = {pval_non_hw:.3f}")
# print("="*60)

# # Save results to file
# results_summary = f"""
# Phase Speed Analysis Results - {region}
# =====================================

# Sample Sizes:
# - Heatwave days: {len(cp_hw_clean)}
# - Non-heatwave days: {len(cp_non_hw_clean)}

# Phase Speed Statistics:
# - Heatwaves:     mean = {np.mean(cp_hw_clean):.3f} m/s, std = {np.std(cp_hw_clean):.3f} m/s
# - Summer: mean = {np.mean(cp_non_hw_clean):.3f} m/s, std = {np.std(cp_non_hw_clean):.3f} m/s
# - t-test: t = {tstat_cp:.3f}, p = {pval_cp:.3f}
# - KS-test: D = {ksstat_cp:.3f}, p = {pval_ks_cp:.3f}

# Temperature Statistics:
# - Heatwaves:     mean = {np.mean(temp_hw_clean):.3f} °C, std = {np.std(temp_hw_clean):.3f} °C
# - Summer: mean = {np.mean(temp_non_hw_clean):.3f} °C, std = {np.std(temp_non_hw_clean):.3f} °C
# - t-test: t = {tstat_temp:.3f}, p = {pval_temp:.3f}
# - KS-test: D = {ksstat_temp:.3f}, p = {pval_ks_temp:.3f}

# Correlations:
# - Heatwaves:     r = {corr_hw:.3f}, p = {pval_hw:.3f}
# - Summer: r = {corr_non_hw:.3f}, p = {pval_non_hw:.3f}
# """

# with open(path_figures + f'Phase_speed_analysis_results_{region}.txt', 'w') as f:
#     f.write(results_summary)

# print(f"\nAnalysis complete! Results saved to:")
# print(f"- Scatter plots: {path_figures}Phase_speed_temperature_scatter_HW_vs_nonHW_{region}.png")
# print(f"- PDF plots: {path_figures}Phase_speed_temperature_PDF_HW_vs_nonHW_{region}.png")
# print(f"- Summary: {path_figures}Phase_speed_analysis_results_{region}.txt")

