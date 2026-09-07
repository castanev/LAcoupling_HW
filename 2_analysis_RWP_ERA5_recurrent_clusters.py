# This code is for analyzing the RWP variables diagnosed from ERA5 by Yuan-Bing.  
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import scipy as srecurrent3d_vanom_cp
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy as srecurrent3d_vanom_cp
import os
import glob
import re
from scipy.ndimage import uniform_filter
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import argparse
import yaml
import xarray as xr

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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
case = cfg['case']
region = cfg['region']
path_case = cfg['path_case_land']
path_file_E = cfg['path_file_E_recurrent_vanom']
path_file_Cp = cfg['path_file_Cp_recurrent_vanom']
path_file_t = cfg['path_file_t']
path_file_v = cfg['path_file_v_recurrent_vanom']
initial_year = cfg['initial_year']
path_outputs = cfg['path_outputs']


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
path_outputs_case = f'{path_outputs}{case}/'
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



# ==================================================== recurrent3d_vanom_envelope  ==============================================================================
variable_env = 'amplitude'

ncfile_E = Dataset(f'{path_file_E}')
lats_env = np.array(ncfile_E['lat'])
lons_env = np.array(ncfile_E['lon'])  
timei_env = np.array(ncfile_E['time'][:]) #hours since 1900-01-01 00:00:00, gregorian
dates_env_recurrent_hourly = np.array([pd.to_datetime(iii, format="%Y-%m-%d-%H") for iii in timei_env])
pos_initial_year = np.where([d.year == initial_year for d in dates_env_recurrent_hourly])[0][0]
print(pos_initial_year)
pos_lats_midlat = np.where((lats_env >= 35) & (lats_env <= 65))[0]
pos_middle_HW = np.where(abs(lons_env - (lon_minHW+lon_maxHW)/2) == np.min(abs(lons_env - (lon_minHW+lon_maxHW)/2)))[0][0]

path_recurrent3d_vanom_envelope_daily = f'{path_outputs}recurrent3d_vanom_envelope_daily.nc'
path_recurrent3d_vanom_envelope_daily_anoma_window = f'{path_outputs}recurrent3d_vanom_envelope_daily_anoma_window.nc'
path_recurrent3d_vanom_envelope_daily_seasonal_anoma = f'{path_outputs}recurrent3d_vanom_envelope_daily_seasonal_anoma.nc'
path_recurrent3d_vanom_envelope_annual_cycle = f'{path_outputs}annual_cycle_recurrent3d_vanom_envelope_daily_window.nc'
if not os.path.exists(path_recurrent3d_vanom_envelope_daily_seasonal_anoma):
    # Convert 6-hourly data to daily data
    recurrent3d_vanom_envelope = np.array(ncfile_E[variable_env][pos_initial_year:])
    # Reshape to group 4 time steps per day (6-hourly to daily)
    # Assuming 4 time steps per day: 00, 06, 12, 18 UTC
    recurrent3d_vanom_envelope_daily = recurrent3d_vanom_envelope.reshape(-1, 4, recurrent3d_vanom_envelope.shape[1], recurrent3d_vanom_envelope.shape[2]).astype('float32')
    recurrent3d_vanom_envelope_daily = np.nanmean(recurrent3d_vanom_envelope_daily, axis=1)  # Average over the 4 time steps per day
    
    dates_env = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(recurrent3d_vanom_envelope_daily))])
    dates_env = np.unique([x.date() for x in dates_env])
    dates_recurrent3d_vanom_env_recurrent_str = np.array([x.strftime("%Y%m%d") for x in dates_env])
    save_nc_3d(path_recurrent3d_vanom_envelope_daily, recurrent3d_vanom_envelope_daily, lats_env, lons_env, dates_recurrent3d_vanom_env_recurrent_str, variable_env)

    # Convert dates to daily
    ds = xr.open_dataset(path_recurrent3d_vanom_envelope_daily, decode_times=False, engine="netcdf4")
    ds['time'] = np.array([pd.to_datetime(iii, format="%Y%m%d") for iii in ds['time'].values])
    print(ds['time'].values[:3])
    recurrent3d_vanom_envelope_daily = ds[variable_env]
    print("recurrent3d_vanom_envelope_daily shape:", recurrent3d_vanom_envelope_daily.shape)

    
    # compute annual cycle
    annual_cycle_0 = compute_annual_cycle_window(recurrent3d_vanom_envelope_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
    print("annual_cycle_0 shape:", annual_cycle_0.shape)
    save_nc_3d(path_recurrent3d_vanom_envelope_annual_cycle, annual_cycle_0, lats_env, lons_env, np.arange(1, len(annual_cycle_0.time)+1).astype(str), variable_env)
    recurrent3d_vanom_envelope_daily_annual_cycle = expand_annual_cycle(recurrent3d_vanom_envelope_daily, annual_cycle_0)
    print("annual_cycle shape:", recurrent3d_vanom_envelope_daily_annual_cycle.shape)
    recurrent3d_vanom_envelope_daily_anoma_windowly = recurrent3d_vanom_envelope_daily - recurrent3d_vanom_envelope_daily_annual_cycle
    print("recurrent3d_vanom_envelope_daily_anoma_windowly shape:", recurrent3d_vanom_envelope_daily_anoma_windowly.shape)
    save_nc_3d(path_recurrent3d_vanom_envelope_daily_anoma_window, recurrent3d_vanom_envelope_daily_anoma_windowly, lats_env, lons_env, dates_recurrent3d_vanom_env_recurrent_str, variable_env)

    # recurrent3d_vanom_envelope_seasonal_anoma = seasonal_anomalies_by_year_optimized(dates_env, recurrent3d_vanom_envelope_daily).astype('float32')
    # save_nc_3d(path_recurrent3d_vanom_envelope_daily_seasonal_anoma, recurrent3d_vanom_envelope_seasonal_anoma, lats_env, lons_env, dates_recurrent3d_vanom_env_recurrent_str, variable_env)

    # del recurrent3d_vanom_envelope_daily, recurrent3d_vanom_envelope_daily_annual_cycle, recurrent3d_vanom_envelope_daily_anoma_windowly

else:
    time_recurrent3d_vanom_env_recurrent_daily = np.array(Dataset(path_recurrent3d_vanom_envelope_daily)["time"][:])
    dates_env = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(time_recurrent3d_vanom_env_recurrent_daily))])
    dates_env = np.unique([x.date() for x in dates_env])
    dates_recurrent3d_vanom_env_recurrent_str = np.array([x.strftime("%Y%m%d") for x in dates_env])
    del time_recurrent3d_vanom_env_recurrent_daily

lon_min_env = 265 - 65
lon_max_env = 250
pos_lat_env, pos_lon_env, lats_cut, lons_cut = cut_region_lats_lons(lats_env, lons_env, 35, 55, lon_min_env, lon_max_env)
path_recurrent3d_vanom_envelope_daily_anoma_std_window_env = f'{path_outputs}recurrent3d_vanom_envelope_daily_anoma_std_window_envregion.nc'  
path_recurrent3d_vanom_envelope_annual_cycle_std_env = f'{path_outputs}annual_cycle_std_recurrent3d_vanom_envelope_daily_window_envregion.nc'



pos_HWdays = []
for num in range(len(dates_env)): 
    if dates_env[num] in dates_d_HWdays: pos_HWdays.append(num)
pos_HWdays = pd.DataFrame(pos_HWdays)

pos_HW_day0 = []
for num in range(len(dates_env)): 
    if dates_env[num] in dates_d_HW_day0: pos_HW_day0.append(num)  
pos_HW_day0 = pd.DataFrame(pos_HW_day0)
print("pos_HW_day0 :", pos_HW_day0)


pos_summer = np.where([i.month in [6,7,8] for i in dates_env])[0]
pos_summer = pos_summer[pos_summer<min(len(dates_env),len(dates_d))]
pos_all_non_HW = []
for i in pos_summer:
    if i not in pos_HWdays.values:
        pos_all_non_HW.append(i)
pos_all_non_HW = np.array(pos_all_non_HW)





# -------------------------------
# 2D scatter with color:
# env and Cp at onset (lag 0),
# temperature 3 days after onset
# -------------------------------

pos_HW_indices = pos_HW_day0.values.flatten()
pos_preHW = pos_HW_indices - 1
pos_postHW = pos_HW_indices + 1

pos_non_HW_indices = pos_summer

ncfile_t_anom = Dataset(f'{path_outputs}TS_ERA5_daily_anoma_window.nc')


L_temp = -3   # because lag_array(..., -3) gives values 3 days after onset

E0 = 10
ncfile_E_daily_anoma = Dataset(path_recurrent3d_vanom_envelope_daily_anoma_window)
ncfile_Cp_daily_anoma = Dataset(f"/scratch/negishi/castanev/Amplification_RW/ERA5/cp_{E0}_min3timesteps_daily_smoothed_seasonal_anoma.nc")
# ncfile_E_daily_anoma = Dataset(path_recurrent3d_vanom_envelope_daily_anoma_std_window_env)
# ncfile_Cp_daily_anoma = Dataset("/scratch/negishi/castanev/Amplification_RW/ERA5/cp_15_min3timesteps_daily_smoothed_seasonal_anoma_std_pacific.nc")

# ncfile_Cp_daily_anoma = Dataset("/scratch/negishi/castanev/Amplification_RW/ERA5/cp_15_min3timesteps_daily_smoothed_anoma_window.nc")

lon_min_pacific = 265 - 95
lon_max_pacific = 265
pos_lat_cp, pos_lon_cp, lats_cut, lons_cut = cut_region_lats_lons(lats_env, lons_env, 35, 55, lon_min_pacific, lon_max_pacific)
# Example names:
variable_cp = 'phase_speed'


# Envelope at onset
recurrent_env_preonset = np.array(ncfile_E_daily_anoma[variable_env][pos_preHW, :, :])
recurrent_env_onset = np.array(ncfile_E_daily_anoma[variable_env][pos_HW_indices, :, :])
recurrent_env_postonset = np.array(ncfile_E_daily_anoma[variable_env][pos_postHW, :, :])
recurrent_env_onset_mean = np.nanmean(
    np.stack([recurrent_env_preonset, recurrent_env_onset, recurrent_env_postonset], axis=0),
    axis=0
)
del recurrent_env_preonset, recurrent_env_onset, recurrent_env_postonset

env_hw = np.nanmean(
    recurrent_env_onset_mean[:, pos_lat_env, :],
    axis=1
)
env_hw = np.nanmean(env_hw[:, pos_lon_env], axis=1)

recurrent_env_summer = np.array(ncfile_E_daily_anoma[variable_env][pos_non_HW_indices, :, :])
env_non_hw = np.nanmean(
    recurrent_env_summer[:, pos_lat_env, :],
    axis=1
)
env_non_hw = np.nanmean(env_non_hw[:, pos_lon_env], axis=1)
del recurrent_env_summer

# Cp at onset
cp_preonset = np.array(ncfile_Cp_daily_anoma[variable_cp][pos_preHW, :, :])
cp_onset = np.array(ncfile_Cp_daily_anoma[variable_cp][pos_HW_indices, :, :])
cp_postonset = np.array(ncfile_Cp_daily_anoma[variable_cp][pos_postHW, :, :])
cp_onset_mean = np.nanmean(
    np.stack([cp_preonset, cp_onset, cp_postonset], axis=0),
    axis=0
)
del cp_preonset, cp_onset, cp_postonset

cp_hw = np.nanmean(
    cp_onset_mean[:, pos_lat_env, :],
    axis=1
)
cp_hw = np.nanmean(cp_hw[:, pos_lon_env], axis=1)

cp_summer = np.array(ncfile_Cp_daily_anoma[variable_cp][pos_non_HW_indices, :, :])
cp_non_hw = np.nanmean(
    cp_summer[:, pos_lat_env, :],
    axis=1
)
cp_non_hw = np.nanmean(cp_non_hw[:, pos_lon_env], axis=1)
del cp_summer

# Temperature 3 days after onset
temp_lagged = lag_array(np.array(ncfile_t_anom['TS']), L_temp)

temp_hw_data_lag = temp_lagged[pos_HW_indices, :, :]
temp_non_hw_data_lag = temp_lagged[pos_non_HW_indices, :, :]

pos_lats_hw_temp = np.where((lats_t >= lat_minHW) & (lats_t <= lat_maxHW))[0]
pos_lons_hw_temp = np.where((lons_t >= lon_minHW) & (lons_t <= lon_maxHW))[0]
temp_hw_lag = np.nanmean(temp_hw_data_lag[:, pos_lats_hw_temp, :], axis=1)
temp_hw_lag = np.nanmean(temp_hw_lag[:, pos_lons_hw_temp], axis=1)

temp_non_hw_lag = np.nanmean(temp_non_hw_data_lag[:, pos_lats_hw_temp, :], axis=1)
temp_non_hw_lag = np.nanmean(temp_non_hw_lag[:, pos_lons_hw_temp], axis=1)
del temp_lagged

# Consistent valid masks
ok_hw = np.isfinite(env_hw) & np.isfinite(cp_hw) & np.isfinite(temp_hw_lag)
ok_non_hw = np.isfinite(env_non_hw) & np.isfinite(cp_non_hw) & np.isfinite(temp_non_hw_lag)

env_hw_ok = env_hw[ok_hw]
cp_hw_ok = cp_hw[ok_hw]
temp_hw_ok = temp_hw_lag[ok_hw]

env_non_hw_ok = env_non_hw[ok_non_hw]
cp_non_hw_ok = cp_non_hw[ok_non_hw]
temp_non_hw_ok = temp_non_hw_lag[ok_non_hw]

# -------------------------------
# 2D scatter with color
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

vmin = 0
vmax = 4

# Calculate percentage of heatwaves with env_hw_ok > 0 and cp_hw_ok < 0
mask_neg = (env_hw_ok > 0) & (cp_hw_ok < 0)
n_neg = np.sum(mask_neg)
total = len(env_hw_ok)
percent_neg = 100.0 * n_neg / total if total > 0 else 0.0

label_str = f"Heatwaves ({percent_neg:.1f}% with R'>0 & Cp'<0)"

sc2 = ax.scatter(
    env_hw_ok,
    cp_hw_ok,
    c=temp_hw_ok,
    cmap='Reds',
    vmin=vmin,
    vmax=vmax,
    alpha=0.8,
    s=48,
    edgecolor='k',
    linewidth=1.4,
    label=label_str
)

ax.set_xlim(-6, 13)
ax.set_ylim(-8, 5)
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.axvline(0, color='k', linestyle='--', linewidth=1)

ax.set_xlabel(f"{variable_env} at onset (m/s)", fontsize=13)
ax.set_ylabel(f"{variable_cp} at onset", fontsize=13)
ax.set_title("Temperature 3 days after onset vs amplitude and phase speed at onset", fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=13)

cbar = plt.colorbar(sc2, ax=ax)
cbar.set_label("Temperature anomaly at day +3 (°C)", fontsize=13)

ax.grid(True, alpha=0.3)
ax.legend(fontsize=13)

plt.tight_layout()

plt.savefig(
    path_figures + f"scatter2D_env_cp{E0}_aroundonset_Tplus3_HWday0.png",
    dpi=500,
    bbox_inches='tight'
)
plt.close()

