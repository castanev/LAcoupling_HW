# This code is for the detection of heatwaves. It's based on the methodology used by Hayan Teng
# Here, a heat wave event is defined as at least 5 consecutive days following:
# i). More than 5% of the US has daily averaged SAT exceeding the threshold value
# ii). Centre of these warm points does not move faster than 5◦ latitude or longitude per day

# Threshold: x percentile for historical t within a 15-day window centred on the day (for each day and each gridpoint)
# To avoid contamination, we use only events that have no heat wave days in the preceding 20 day

# NOTE: 
# For the detection of heat waves in the Dry Core GCM, the threshold is defined as the 97.5 percentile of all the data (the model has no seasons)

from Functions import *
import argparse
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import matplotlib.pyplot as plt
import cartopy
from cartopy import crs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from scipy import interpolate
from scipy.ndimage import label, find_objects
import json
from scipy.signal import butter, filtfilt


# INPUTS TO CHANGE ===========================================================================================================================


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--percentile', type=float, required=True)
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--t_file', type=str, required=True)
parser.add_argument('--t_file_anoma', type=str, required=True)
parser.add_argument('--var', type=str, required=True)
parser.add_argument('--seasons', type=bool, required=True)
parser.add_argument('--topography', type=bool, required=True)
parser.add_argument('--methodology', type=str, required=True)
parser.add_argument('--initial_year', type=int, required=True)
parser.add_argument('--path_case', type=str, required=True)
parser.add_argument('--path_outputs_case', type=str, required=True)
parser.add_argument('--global_mean_file', type=str, required=False)
parser.add_argument('--global_mean_var', type=str, required=False)
args = parser.parse_args()

name = args.name
case = args.case
percentile = args.percentile
if percentile % 1 == 0: percentile = int(percentile)
region = args.region
t_file = args.t_file
t_file_anoma = args.t_file_anoma
var = args.var
seasons = args.seasons
topography = args.topography
methodology = args.methodology
initial_year = args.initial_year
path_case = args.path_case
path_outputs_case = args.path_outputs_case
global_mean_file = args.global_mean_file
global_mean_var = args.global_mean_var

vel = [5, 1]
min_duration = 5
min_area = 535000 # [km2-->5% of US from Teng 2013 region] 
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
elif region == 'midwestUS':
    lat_minHW = 35; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{os.path.abspath(os.getcwd())}/{case}/Figures/'
vel_str = f'vel{str(vel[0])}{str(vel[1])}'


# ============================================================================================================================================

def anomalies_seasons_optimized(df_VAR):
    # Process one month at a time
    ANOMA = df_VAR.copy()* np.nan
    
    for month in range(1, 13):
        # Get data for current month
        month_mask_ = pd.DatetimeIndex(df_VAR.index).month == month
        month_mask = np.where(month_mask_)[0]
        print(f"Processing month {month}, found {len(month_mask)} time steps")
        
        if len(month_mask) == 0:
            continue
            
        month_data = pd.DataFrame(index=pd.DatetimeIndex(df_VAR.index[month_mask]), 
                                data=np.array(df_VAR.iloc[month_mask]))
        
        if month_data.empty: continue
            
        # Group by day within the month
        day_idx = pd.DatetimeIndex(month_data.index).day
        grouped = month_data.groupby(day_idx)
        
        # Calculate means for each day
        means = grouped.transform('mean')
        
        # Calculate anomalies for this month
        ANOMA.iloc[month_mask] = month_data - means

    return ANOMA


# To calculate the duration and the position of the first day of each event
def duration_heat_waves(pos_hw, min_duration):
    count = 1
    duration_hw = []
    pos_day1_hw = []
    for i in range(len(pos_hw[:-1])):
        if pos_hw[i] + 1 == pos_hw[i + 1]:
            count += 1
        else:
            if count >= min_duration and len(pos_day1_hw) == 0:
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            elif count >= min_duration and abs((pos_day1_hw[-1] + duration_hw[-1]) - pos_hw[i - count + 1]):
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            count = 1

    return duration_hw, pos_day1_hw



ncfile = Dataset(f'{t_file}')
time = np.array(ncfile['time'][:])
lats = np.array(ncfile['lat'])
lons = np.array(ncfile['lon'])

# Spatial cut: United States
pos_NH = np.where(lats>0)[0]
pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))
pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))
lats_US = lats[pos_lats]
lons_US = lons[pos_lons]
km_1degree = np.pi*2*6371*np.cos(np.radians(38))/360 # [km]
total_area =(lat_maxHW-lat_minHW)*km_1degree*(lon_maxHW-lon_minHW)*km_1degree
percentage_min_area = min_area/total_area
min_grid_5 = round(lats_US.shape[0] * lons_US.shape[0] * percentage_min_area)

t_k_US = np.array(ncfile[var][:,pos_lats[0],:])  #(t, lat, lon) °K
t_k_US = t_k_US[:, :, pos_lons[0]]

if global_mean_file is not None:
    print("Detrending the data")
    # 1) Center the regressor so you preserve local means
    ncfile_global = Dataset(f'{global_mean_file}')
    X = np.array(ncfile_global[global_mean_var][:])  #(t) K

    # --- 8)Apply 9-year low-pass filter to global mean ---
    # Sampling freq fs = 365 day⁻¹; cutoff = 1/(9*365) ≈ 0.000304 yr⁻¹ = 1/(9 years)
    # Normalized frequency for butter: Wn = fc / (fs/2) = (1/3285) / (182.5) ≈ 0.00055
    fs = 365.0
    fc = 1.0 / (9 * 365.0)
    Wn = fc / (fs / 2)

    b, a = butter(N=4, Wn=Wn, btype='low')
    X_lp_vals = filtfilt(b, a, X)  # X has no NaNs, so it's safe
    X_lp = X_lp_vals

    Xc = X_lp - X_lp.mean()            # (time,)
    varX = (Xc**2).mean()              # scalar
    # 2) Slope map b(lat,lon) = Cov(T, Xc) / Var(Xc)
    covY_X = (t_k_US * Xc[:, np.newaxis, np.newaxis]).mean(axis=0)        # (lat, lon)
    b_map   = covY_X / varX                  # (lat, lon)
    # 3) Detrend: remove only the time-varying b*Xc
    t_k_US_detr = t_k_US - b_map * Xc[:, np.newaxis, np.newaxis]            # (time, lat, lon)

    t_US = pd.DataFrame(data=np.reshape(t_k_US_detr, [t_k_US_detr.shape[0], t_k_US_detr.shape[1] * t_k_US_detr.shape[2]]))
    t_US = t_US - 273.15 # °C
    t_US = t_US.values.reshape(t_k_US_detr.shape[0], t_k_US_detr.shape[1], t_k_US_detr.shape[2])
    print(t_US.shape)
    print(Xc.shape)

    plt.figure(figsize=(10,5))
    plt.plot(Xc, label='Trend', color='k')
    plt.xlabel('Time')
    plt.ylabel('Trend (°C)')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(path_figures + f'trend_temperature.png', dpi=500)
    plt.close()

    series_detrended= np.nanmean(t_US, axis=1)
    series_detrended= np.nanmean(series_detrended, axis=1)

    series = np.nanmean(t_k_US, axis=1)
    series= np.nanmean(series, axis=1) - 273.15


else: 
    t_US = pd.DataFrame(data=np.reshape(t_k_US, [t_k_US.shape[0], t_k_US.shape[1] * t_k_US.shape[2]]))
    t_US = t_US - 273.15 # °C
    t_US = t_US.values.reshape(t_k_US.shape[0], t_k_US.shape[1], t_k_US.shape[2])


print(f'% of the total grids corresponds to = {percentage_min_area}')  

if seasons == True:
    if   name == 'NCEP': dates_d = np.array([dt.datetime(1800,1,1) + dt.timedelta(hours = int(time[i])) for i in range(len(time))])
    elif   name == 'MERRA2': 
        dates_d_summer = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in time.astype(str)])
        timei_summer = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d_summer])

        dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(365*(2023-initial_year))])
        timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
        pos = np.where([i in timei_summer for i in timei])[0]
        t_US_complete = np.empty([len(dates_d), len(lats_US), len(lons_US)]) *np.nan
        t_US_complete[pos] = t_US
        t_US = t_US_complete
    elif   name == 'ERA5': 
        dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(time))])
        timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
        
        date_0 = dt.datetime(initial_year,1,1)
        pos_date_0 = np.where(dates_d == date_0)
        dates_d = dates_d[pos_date_0[0][0]:]; timei = timei[pos_date_0[0][0]:]; t_US = t_US[pos_date_0[0][0]:]
    elif 'dryc' in name:
        dates_d = np.array([pd.to_datetime(iii, format="%Y-%m-%d") for iii in time])
        date_0 = dt.datetime(initial_year,1,1)
        pos_date0 = np.where(dates_d == date_0)[0]
        print(pos_date0)
        dates_d = dates_d[pos_date0[0]:]; t_US = t_US[pos_date0[0]:]


    Month = np.array([ii.month for ii in dates_d])
    df_t = pd.DataFrame(index=dates_d, data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
    days_summer = np.array([dt.datetime(2021, 6, 1) + relativedelta(days=int(xx)) for xx in range(92)])  # random year
    threshold = np.zeros([len(days_summer), len(lats_US), len(lons_US)])
    for i, d in enumerate(days_summer):
        
        t_pos1 = np.where((np.array([ii.month for ii in dates_d])==d.month) & (np.array([ii.day for ii in dates_d])==d.day))[0]
        #Positions of the 15-day window centred on the day of the year of the potential heat wave day:
        t_pos = [np.concatenate((t_pos1, t_pos1 + iii)) for iii in range(-7, 8)]
        t_pos = np.unique(t_pos)
        t_pos = t_pos[t_pos < t_US.shape[0]] # For the last days of the dataset
        if "nowindow" in case:
            print("Using no window")
            data = t_US[t_pos1, :, :]
        else:
            print("Using window")
            data = t_US[t_pos, :, :]
        threshold_i = np.percentile(data, percentile, axis=0)
        threshold[i, :, :] = threshold_i


    df_threshold = pd.DataFrame(index=days_summer,
                                data=np.reshape(threshold, [days_summer.shape[0], t_US.shape[1] * t_US.shape[2]]))

    pos_summer = np.where([ii in [6, 7, 8] for ii in Month])[0]
    dates_summer = dates_d[pos_summer]
    t_summer = t_US[pos_summer, :, :]

    pos_heat_wavesi = []
    for i, date in enumerate(dates_summer):

        threshold_pos = df_threshold.index.get_indexer_for(
            (df_threshold.loc[(df_threshold.index.month == date.month) & (df_threshold.index.day == date.day)].index))
        
        if i+1 == dates_summer.shape[0]: break
        date_2 = dates_summer[i+1]
        threshold_pos_2 = df_threshold.index.get_indexer_for(
            (df_threshold.loc[(df_threshold.index.month == date_2.month) & (df_threshold.index.day == date_2.day)].index))
        
        # Condition i). More than 5% of the domain (US) has daily averaged SAT exceeding the threshold value
        cond_1 = t_US[pos_summer[i], :, :] > threshold[threshold_pos, :, :]
        cond_1_2 = t_US[pos_summer[i] + vel[1], :, :] > threshold[threshold_pos_2, :, :]
        
        # labels, nlab = ndimage.label(cond_1)
        # labels_2, nlab_2 = ndimage.label(cond_1_2)
        # if nlab == 0 or nlab_2 == 0:
        #     continue

        # Condition i): >5% of domain exceeds threshold (all warm cells, not largest cluster only)
        grid_cont = np.count_nonzero(cond_1)
        if grid_cont <= min_grid_5:
            continue

        # sizes = ndimage.sum(cond_1, labels, range(1, nlab + 1))
        # largest = int(np.argmax(sizes)) + 1
        # cond_main = labels == largest

        # # Condition ii): track the same blob — day-2 cluster with max overlap vs day-1 main cluster
        # overlaps = ndimage.sum(cond_main, labels_2, index=range(1, nlab_2 + 1))
        # largest_2 = int(np.argmax(overlaps)) + 1
        # if overlaps[largest_2 - 1] > 0:
        #     cond_main_2 = labels_2 == largest_2
        # else:
        #     # No spatial continuity: fall back to all warm cells on day 2 (original Teng COM)
        #     cond_main_2 = cond_1_2

        # Condition ii): Center of these warm points does not move faster than 5◦ latitude or longitude per day
        if pos_summer[i] + vel[1] == t_US.shape[0]: break

        # pos_max = ndimage.measurements.center_of_mass(cond_main)
        # pos_max2 = ndimage.measurements.center_of_mass(cond_main_2)
        pos_max = ndimage.measurements.center_of_mass(cond_1)
        pos_max2 = ndimage.measurements.center_of_mass(cond_1_2)

        if len(cond_1.shape)==2:
            dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
            dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])
        elif len(cond_1.shape)==3:
            dif_lats = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lats_US[1]-lats_US[0])
            dif_lons = np.abs(pos_max2[2] - pos_max[2]) * np.abs(lons_US[1]-lons_US[0])

        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos_summer[i])  # dates_d[pos_summer[i]] is the date that meets both conditions


elif seasons == False:
    dates_d = time
    t_US = t_US[:len(time),:,:]
    df_t = pd.DataFrame(data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
    threshold = np.percentile(t_US, percentile, axis=0)


    pos_heat_wavesi = []
    for pos in range(t_US.shape[0]):
        if pos + vel[1] == t_US.shape[0]: break
        # Condition i). More than 5% of the domain (US) has daily averaged SAT exceeding the threshold value
        cond_1 = t_US[pos, :, :] > threshold
        cond_1_2 = t_US[pos + vel[1], :, :] > threshold
        
        grid_cont = np.count_nonzero(cond_1)


        # Condition ii): Centre of these warm points does not move faster than 5◦ latitude or longitude per day
        # Center defined as the point with the max temperature in the domain     
        pos_max = ndimage.measurements.center_of_mass(cond_1)
        pos_max2 = ndimage.measurements.center_of_mass(cond_1_2)

        if len(cond_1.shape)==2:
            dif_lats = np.abs(pos_max2[0] - pos_max[0]) * np.abs(lats_US[1]-lats_US[0])
            dif_lons = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lons_US[1]-lons_US[0])
        elif len(cond_1.shape)==3:
            dif_lats = np.abs(pos_max2[1] - pos_max[1]) * np.abs(lats_US[1]-lats_US[0])
            dif_lons = np.abs(pos_max2[2] - pos_max[2]) * np.abs(lons_US[1]-lons_US[0])

        if (grid_cont > min_grid_5) and (dif_lats < vel[0]) and (dif_lons < vel[0]):
            pos_heat_wavesi.append(pos) 
        

# # Calculating anomalies
# if seasons == True: 
#     anom_t_US = anomalies_seasons_optimized(df_t)
# elif seasons == False: 
#     anom_t_US = anomalies_noseasons(df_t)


# Calculating anomalies
if seasons == True: 
    if os.path.exists(t_file_anoma):
        anom = xr.open_dataset(t_file_anoma)[var]
        lats_anoma = anom.lat.values
        pos_lats_anoma = np.where((lats_anoma >= lat_minHW) & (lats_anoma <= lat_maxHW))[0]
        lons_anoma = anom.lon.values
        pos_lons_anoma = np.where((lons_anoma >= lon_minHW) & (lons_anoma <= lon_maxHW))[0]
        anom_t_US = anom.isel(lat=pos_lats_anoma, lon=pos_lons_anoma)
        del anom
    
    else:
        aaaaa
        path_t_annual_cycle = f'{path_outputs}annual_cycle_TS_daily_window.nc'  
        path_t_daily_anoma_window = f'{path_outputs}TS_ERA5_daily_anoma_window.nc'  
        # Convert dates to daily
        ds = xr.open_dataset(t_file, decode_times=False, engine="netcdf4")
        ds['time'] = dates_d
        print(ds['time'].values[:3])
        t_daily = ds[var]
        print("t_daily shape:", t_daily.shape)

        # compute annual cycle
        annual_cycle_0 = compute_annual_cycle_window(t_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        print("annual_cycle_0 shape:", annual_cycle_0.shape)
        save_nc_3d(path_t_annual_cycle, annual_cycle_0, lats_t, lons_t, np.arange(1, len(annual_cycle_0.time)+1).astype(str), var)

        ds_clim = xr.open_dataset(path_t_annual_cycle, decode_times=False, engine="netcdf4")
        clim = ds_clim[var]  
        cycle_times = pd.date_range(f"{2000}-01-01", f"{2000}-12-31", freq="D")
        clim = clim.assign_coords(time=("time", cycle_times))

        ds = xr.open_dataset(path_t_daily, decode_times=False, engine="netcdf4")
        expanded_clim = expand_annual_cycle_optimized(dates_d, len(lats_t), len(lons_t), clim)

        anom = ds[var].astype("float32") - expanded_clim
        del expanded_clim
        print("anom shape:", anom.shape)
        save_nc_3d(path_t_daily_anoma_window, anom, lats_env, lons_env, dates_env_str, variable_cp)
        
        anom_t_US = anom.isel(lat=pos_lats, lon=pos_lons)
        del anom

elif seasons == False: 
    if os.path.exists(path_file_t_anoma):
        anom_t_US = xr.open_dataset(path_file_t_anoma)['t_anoma'].values
    else:
        anom_t_US = anomalies_noseasons(df_t)

print('good')
duration_hw, pos_day1_hw = duration_heat_waves(pos_heat_wavesi, min_duration)

# for heatwaves that are too close to each other (<20 days), keep the longest one
# Remove overlapping events by keeping only the most intense one
i = 0
while i < len(pos_day1_hw) - 1:
    # Check if events overlap (within 20 days)
    if abs((pos_day1_hw[i] + duration_hw[i]) - pos_day1_hw[i + 1]) < 20:
        # Calculate intensity as the maximum anomaly over the entire event period
        intensity_i = np.nanmax(np.nanmean(anom_t_US[pos_day1_hw[i]:pos_day1_hw[i]+duration_hw[i]-1], axis=0))
        intensity_i_2 = np.nanmax(np.nanmean(anom_t_US[pos_day1_hw[i + 1]:pos_day1_hw[i + 1]+duration_hw[i + 1]-1], axis=0))
        
        if intensity_i < intensity_i_2:
            # Remove the less intense event (current one)
            duration_hw.pop(i)
            pos_day1_hw.pop(i)
            # Don't increment i since we removed an element
        else:
            # Remove the less intense event (next one)
            duration_hw.pop(i + 1)
            pos_day1_hw.pop(i + 1)
            # Don't increment i since we removed an element
    else:
        # No overlap, move to next pair
        i += 1

print(f'Number of heat waves events: {len(duration_hw)}')
# print(pos_day1_hw)


# Probability distribution function for duration
bins = np.arange(np.unique(duration_hw)[0], np.unique(duration_hw)[-1] + 1, 1)
Hist, bins1 = np.histogram(duration_hw, len(bins))
PDF_temp = Hist / len(duration_hw)


fig = plt.figure(figsize=[4, 4])
df = pd.DataFrame({'x': bins, 'PDF': PDF_temp})
df.plot.bar(x='x', y='PDF', rot=0, color='dimgray', width=.4)
plt.ylabel('PDF', fontsize=13)
plt.xlabel('Duration (d)', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.legend(fontsize=12, labelcolor='linecolor')
#plt.show()
plt.savefig(path_figures + f'SAT_PDF_{methodology}_{vel_str}.png', dpi=500)
plt.close()


pos_heat_waves = []
for day1, dur in zip(pos_day1_hw, duration_hw): 
    pos_heat_waves.append([day1 + i for i in range(dur)])
pos_heat_waves = [item for sublist in pos_heat_waves for item in sublist]
print(f'Number heat waves days: {len(pos_heat_waves)}')


# Saving PDF of duration in a .csv
pos_heat_waves_serie = pd.Series(index = bins[:10], data = PDF_temp[:10])
pos_heat_waves_serie.to_csv(f'{path_outputs_case}/PDF_duration_{name}_{methodology}.csv')

t_US_hw = t_US[pos_heat_waves, :, :]

print('good')

anom_t_US = anom_t_US.values.reshape(anom_t_US.shape[0], lats_US.shape[0], lons_US.shape[0])
anom_t_US_hw = anom_t_US[pos_heat_waves, :, :]



coord_hw = []
freq_hw = np.zeros([len(lats_US), len(lons_US)])
for pos in pos_heat_waves:
    cond_1 = np.zeros([len(lats_US), len(lons_US)])
    cond_1 = anom_t_US[pos, :, :] > 0
    freq_hw += cond_1.astype(int) 
    if pos in pos_day1_hw:
        pos_max = ndimage.measurements.center_of_mass(t_US[pos])
        lat_max = lats_US[int(pos_max[0])]
        lon_max = lons_US[int(pos_max[1])]
        coord_hw.append([lat_max, lon_max])
coord_hw = coord_hw
freq_hw_NH = np.zeros([len(lats), len(lons)])
for num, l in enumerate(pos_lats[0]):
    freq_hw_NH[l, pos_lons] = freq_hw[num]
freq_hw_NH = freq_hw_NH[pos_NH]
np.save(str(path_outputs_case)+"HW_frequency.npy",freq_hw_NH)

coord_hw_onset = []
coord_hw_max = []

freq_hw = np.zeros([len(pos_heat_waves), len(lats_US), len(lons_US)]) * np.nan
for num, pos in enumerate(pos_heat_waves):
    cond_1 = anom_t_US[pos, :, :] > 0
    freq_hw[num] = cond_1.astype(int) * anom_t_US[pos]

    # --- Coordinates at onset: COM of anom_pos on largest warm cluster
    if pos in pos_day1_hw:
        cond_mask = anom_t_US[pos, :, :] > 0
        labels, nlab = ndimage.label(cond_mask)
        if nlab == 0:
            lat_max_onset = np.nan
            lon_max_onset = np.nan
        else:
            sizes = ndimage.sum(cond_mask, labels, range(1, nlab + 1))
            largest = np.argmax(sizes) + 1
            anom_pos = np.where(labels == largest, anom_t_US[pos, :, :], 0)
            anom_pos = np.where(anom_pos > 0, anom_pos, 0)
            if np.sum(anom_pos) > 0:
                pos_max = ndimage.measurements.center_of_mass(anom_pos)
                lat_max_onset = lats_US[int(pos_max[0])]
                lon_max_onset = lons_US[int(pos_max[1])]
            else:
                lat_max_onset = np.nan
                lon_max_onset = np.nan
        coord_hw_onset.append([lat_max_onset, lon_max_onset])
        
# # --- Now: For each event, find coordinates of max anomaly over whole event duration
# for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
#     event_slice = anom_t_US[pos1_i:pos1_i+dur_i-1, :, :]
#     if np.all(np.isnan(event_slice)):
#         coord_hw_max.append([np.nan, np.nan])
#         continue
#     maxval = np.nanmax(event_slice)
#     # Find where the max is (may be multiple, pick the first)
#     maxpos = np.where(event_slice == maxval)
#     if maxpos[0].size > 0:
#         # Get time, lat, lon indices
#         max_time_idx = maxpos[0][0]
#         max_lat_idx = maxpos[1][0]
#         max_lon_idx = maxpos[2][0]
#         lat_max_event = lats_US[max_lat_idx]
#         lon_max_event = lons_US[max_lon_idx]
#     else:
#         lat_max_event = np.nan
#         lon_max_event = np.nan
#     coord_hw_max.append([lat_max_event, lon_max_event])


# # --- For each event: COM of max anomaly over time (spatial peak pattern center)
# for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
#     event_slice = anom_t_US[pos1_i:pos1_i+dur_i-1, :, :]
#     if np.all(np.isnan(event_slice)):
#         coord_hw_max.append([np.nan, np.nan])
#         continue
#     spatial_max = np.nanmax(event_slice, axis=0)
#     anom_pos = np.where(spatial_max > 0, spatial_max, 0)
#     if np.nansum(anom_pos) > 0:
#         pos_max = ndimage.measurements.center_of_mass(anom_pos)
#         lat_max_event = lats_US[int(pos_max[0])]
#         lon_max_event = lons_US[int(pos_max[1])]
#     else:
#         lat_max_event = np.nan
#         lon_max_event = np.nan
#     coord_hw_max.append([lat_max_event, lon_max_event])

# # --- For each event: COM on the day of maximum intensity (max domain-mean anomaly)
# for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
#     event_slice = anom_t_US[pos1_i:pos1_i+dur_i-1, :, :]
#     event_slice_positive = np.where(event_slice > 0, event_slice, np.nan)
#     if np.all(np.isnan(event_slice)):
#         coord_hw_max.append([np.nan, np.nan])
#         continue
#     daily_intensity = np.nanmean(event_slice_positive, axis=(1, 2))
#     t_peak = int(np.nanargmax(daily_intensity))
#     cond_mask = event_slice[t_peak, :, :] > 0
#     labels, nlab = ndimage.label(cond_mask)
#     if nlab == 0:
#         lat_max_event = np.nan
#         lon_max_event = np.nan
#     else:
#         sizes = ndimage.sum(cond_mask, labels, range(1, nlab + 1))
#         largest = np.argmax(sizes) + 1
#         anom_pos = np.where(labels == largest, event_slice[t_peak, :, :], 0)
#         anom_pos = np.where(anom_pos > 0, anom_pos, 0)
#         if np.sum(anom_pos) > 0:
#             pos_max = ndimage.measurements.center_of_mass(anom_pos)
#             lat_max_event = lats_US[int(pos_max[0])]
#             lon_max_event = lons_US[int(pos_max[1])]
#         else:
#             lat_max_event = np.nan
#             lon_max_event = np.nan
#     coord_hw_max.append([lat_max_event, lon_max_event])


# # --- For each event: COM on the distribution of event intensity
# for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
#     event_slice = anom_t_US[pos1_i:pos1_i+dur_i-1, :, :]
#     event_slice_positive = np.where(event_slice > 0, event_slice, np.nan)
#     if np.all(np.isnan(event_slice)):
#         coord_hw_max.append([np.nan, np.nan])
#         continue
#     event_intensity = np.nanmean(event_slice_positive, axis=0)

#     cond_mask = event_intensity > 0

#     labels, nlab = ndimage.label(cond_mask)
#     if nlab == 0:
#         lat_max_event = np.nan
#         lon_max_event = np.nan
#     else:
#         sizes = ndimage.sum(cond_mask, labels, range(1, nlab + 1))
#         largest = np.argmax(sizes) + 1
#         anom_pos = np.where(labels == largest, event_intensity, 0)
#         anom_pos = np.where(anom_pos > 0, anom_pos, 0)
#         if np.sum(anom_pos) > 0:
#             pos_max = ndimage.measurements.center_of_mass(anom_pos)
#             lat_max_event = lats_US[int(pos_max[0])]
#             lon_max_event = lons_US[int(pos_max[1])]
#         else:
#             lat_max_event = np.nan
#             lon_max_event = np.nan
#     coord_hw_max.append([lat_max_event, lon_max_event])

# --- For each event: COM on the distribution of event intensity
for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
    event_slice = anom_t_US[pos1_i:pos1_i+dur_i-1, :, :]
    event_slice_positive = np.where(event_slice > 0, event_slice, np.nan)
    if np.all(np.isnan(event_slice)):
        coord_hw_max.append([np.nan, np.nan])
        continue
    event_intensity = np.nanmean(event_slice_positive, axis=0)

    cond_mask = event_intensity > 0

    labels, nlab = ndimage.label(cond_mask)
    if nlab == 0:
        lat_max_event = np.nan
        lon_max_event = np.nan
    else:
        sizes = ndimage.sum(cond_mask, labels, range(1, nlab + 1))
        largest = np.argmax(sizes) + 1
        anom_pos = np.where(labels == largest, event_intensity, 0)
        anom_pos = np.where(anom_pos > 0, anom_pos, 0)
        if np.sum(anom_pos) > 0:
            pos_max = ndimage.measurements.center_of_mass(anom_pos)
            lat_max_event = lats_US[int(pos_max[0])]
            lon_max_event = lons_US[int(pos_max[1])]
        else:
            lat_max_event = np.nan
            lon_max_event = np.nan
    coord_hw_max.append([lat_max_event, lon_max_event])

freq_hw = np.nanmean(freq_hw, axis=0)
freq_hw_NH = np.zeros([len(lats), len(lons)])
for num, l in enumerate(pos_lats[0]):
    freq_hw_NH[l, pos_lons] = freq_hw[num] 
freq_hw_NH = freq_hw_NH[pos_NH]
np.save(str(path_outputs_case)+"HW_intensity.npy",freq_hw_NH)


# # Creating list with all heat waves events 
intensities = []
maxTs = []
pos_maxTs = []
for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
    event_slice = anom_t_US[pos1_i:pos1_i + dur_i - 1, :, :]
    # Compute mean anomaly over the spatial dimensions, then take the max as intensity
    intensity_i = round(np.max(np.nanmean(event_slice, axis=0)), 2)
    intensities.append(intensity_i)
    maxTs_i = round(np.nanmax(event_slice), 2)
    maxTs.append(maxTs_i)
    # Day index (in full time series) when the event maximum anomaly occurs
    t_rel = np.unravel_index(np.nanargmax(event_slice), event_slice.shape)[0]
    pos_maxTs.append(pos1_i + t_rel)

# Save both onset (center-of-mass) and max anomaly coordinates for each heatwave event
if seasons == True:
    pos_heat_waves_serie = pd.DataFrame(
        data={
            'Date 0': dates_d[pos_day1_hw],
            'Duration': duration_hw,
            'Intensity': intensities,
            'Max Ts': maxTs,
            'Onset Position': coord_hw_onset,
            'Max Anomaly Position': coord_hw_max,
            'Pos max Ts': pos_maxTs
        },
        index=pos_day1_hw
    )
else:
    pos_heat_waves_serie = pd.DataFrame(
        data={
            'Date 0': dates_d[pos_day1_hw],
            'Duration': duration_hw,
            'Intensity': intensities,
            'Max Ts': maxTs,
            'Onset Position': coord_hw_onset,
            'Max Anomaly Position': coord_hw_max,
            'Pos max Ts': pos_maxTs
        },
        index=pos_day1_hw
    )
pos_heat_waves_serie.to_csv(f'{path_case}/Heat_waves_events_list.csv')



# Ploting intensity
RdYlBu_v2_list = ['rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
my_cmap = matplotlib.colors.ListedColormap(RdYlBu_v2_list, name='RdYlBu')


# Saving the resume of statistics in a .csv
try:
    resume = pd.read_csv(f'{path_case}../resume_heatWaves_statistics.csv', index_col = 0)
    resume.loc[case, f'HW days'] = len(pos_heat_waves)
    resume.loc[case, f'HW events'] = len(duration_hw)
    resume.loc[case, f'Intensity'] = round(np.nanmax(np.nanmean(anom_t_US_hw, axis=0)),2)
    if seasons == True: resume.loc[case, f'Analized days'] = len(pos_summer)
    else: resume.loc[case, f'Analized days'] = t_US.shape[0]
    resume.to_csv(f'{path_case}/../resume_heatWaves_statistics.csv')
    print(len(pos_heat_waves))
except:
    pass
# Saving the position of HW days in a .csv
pos_heat_waves_serie = pd.Series(pos_heat_waves)
pos_heat_waves_serie.to_csv(f'{path_case}/resume_positions_HWdays_{name}_{methodology}.csv')




li, ls, intervalos, limite, color = 0.5, 4.5, 15, 1, 'RdYlBu_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps_USA(lons_US, np.ndarray.round(lats_US,2), li, ls + np.abs(bounds[0] - bounds[1]), np.nanmean(anom_t_US_hw, axis=0), r'SAT anomalies [°C]',
    colormap, path_figures + f'SAT_mean_anomt_hw_{methodology}_{vel_str}.png', topography)



