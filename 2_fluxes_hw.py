# This code is for analyzing 
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
import yaml
import xarray as xr
from scipy.stats import gaussian_kde
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
path_case = cfg['path_case']
path_file_SMs = cfg['path_file_SMs']
path_file_SMrz = cfg['path_file_SMrz']
path_file_E = cfg['path_file_E']
path_file_H = cfg['path_file_H']
path_file_t = cfg['path_file_t']
path_file_EF = cfg['path_file_EF']
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

def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)


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









# ======================================================= DENSITY PLOT ====================================================================

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers
# =======================================================
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

# =======================================================
# Containers for all events
# =======================================================
all_sm_summer = []
all_LE_summer = []

all_sm_hw = []
all_LE_hw = []

all_sm_hw5 = []
all_LE_hw5 = []

# =======================================================
# Loop over events
# =======================================================
for lat_center_hw, lon_center_hw, onset_day in zip(lats_events, lons_events, dates_d_HW_day0):

    if onset_day < dates_sm[0] or onset_day < dates_LE[0]:
        print(f"Skipping event {onset_day}: onset date is before the EF/SM data.")
        continue

    # --- local box ±0.5°
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=1)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=1)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0:
        continue

    # --- regional mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values

    if LE_region.ndim != 3 or sm_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align dates
    common_dates = np.intersect1d(dates_LE, dates_sm)

    LE_mask = np.isin(dates_LE, common_dates)
    sm_mask = np.isin(dates_sm, common_dates)

    LE_aligned = LE_point[LE_mask]
    sm_aligned = sm_point[sm_mask]

    # --- summer points from this event-centered box
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    valid = np.isfinite(LE_summer) & np.isfinite(sm_summer)
    if np.any(valid):
        all_LE_summer.append(LE_summer[valid])
        all_sm_summer.append(sm_summer[valid])

    # --- onset
    onset_idx = np.where(common_dates == onset_day)[0]
    if len(onset_idx) == 1:
        LE0 = LE_aligned[onset_idx[0]]
        SM0 = sm_aligned[onset_idx[0]]
        if np.isfinite(LE0) and np.isfinite(SM0):
            all_LE_hw.append(LE0)
            all_sm_hw.append(SM0)

    # --- onset +3
    onset5_day = onset_day + dt.timedelta(days=3)
    onset5_idx = np.where(common_dates == onset5_day)[0]
    if len(onset5_idx) == 1:
        LE5 = LE_aligned[onset5_idx[0]]
        SM5 = sm_aligned[onset5_idx[0]]
        if np.isfinite(LE5) and np.isfinite(SM5):
            all_LE_hw5.append(LE5)
            all_sm_hw5.append(SM5)

# =======================================================
# Concatenate all summer points
# =======================================================
all_sm_summer = np.concatenate(all_sm_summer) if len(all_sm_summer) > 0 else np.array([])
all_LE_summer = np.concatenate(all_LE_summer) if len(all_LE_summer) > 0 else np.array([])

all_sm_hw = np.array(all_sm_hw)
all_LE_hw = np.array(all_LE_hw)

all_sm_hw5 = np.array(all_sm_hw5)
all_LE_hw5 = np.array(all_LE_hw5)

print("all_sm_summer shape:", all_sm_summer.shape)
print("all_LE_summer shape:", all_LE_summer.shape)
print("all_sm_hw:", all_sm_hw)
print("all_LE_hw:", all_LE_hw)

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(8, 5))

mask = np.isfinite(all_sm_summer) & np.isfinite(all_LE_summer)
sm_summer_valid = all_sm_summer[mask] 
LE_summer_valid = all_LE_summer[mask] 

if len(sm_summer_valid) > 5:
    xy = np.vstack([sm_summer_valid, LE_summer_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_summer_valid), np.nanmax(sm_summer_valid)
    ymin, ymax = np.nanmin(LE_summer_valid), np.nanmax(LE_summer_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45)
    )

    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    levels = np.linspace(0, 45, 21)

    cf = plt.contourf(
        xgrid, ygrid, z,
        levels=levels,
        cmap="Blues",
        vmin=0,
        vmax=45,
        alpha=0.8,
        extend="max")
    plt.colorbar(cf, label="Density", extend="max")
else:
    plt.scatter(sm_summer_valid, LE_summer_valid, s=10, alpha=0.3, color="cornflowerblue", label="Summer")

plt.scatter(all_sm_hw, all_LE_hw, marker="x", s=22, label="Onset", color="red")
# plt.scatter(all_sm_hw5, all_LE_hw5, marker="x", s=22, label="Onset +3 days", color="orange")

plt.xlabel("SMs", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of Soil Moisture vs Evaporative Fraction\n"
    "Summer points from ±0.5° around each event center",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HWonset_eachEvent_05deg_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()
AAAA




# -------------------- latent heat / EF --------------------
variable_LE = "EF"
ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
timei_LE = ds_LE.time.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in timei_LE])

ilat_LE = np.where((lats_LE >= lat_minHW) & (lats_LE <= lat_maxHW))[0]

if (lons_LE < 0).any():
    lon_minHW_adj = lon_minHW - 360 if lon_minHW > 180 else lon_minHW
    lon_maxHW_adj = lon_maxHW - 360 if lon_maxHW > 180 else lon_maxHW
    ilon_LE = np.where((lons_LE >= lon_minHW_adj) & (lons_LE <= lon_maxHW_adj))[0]
else:
    ilon_LE = np.where((lons_LE >= lon_minHW) & (lons_LE <= lon_maxHW))[0]

LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
LE_point = np.nanmean(LE_region, axis=(1, 2))
LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

# -------------------- Soil moisture --------------------
variable_sm = "SMs"
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
timei_sm = ds_sm.time.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in timei_sm])

ilat_sm = np.where((lats_sm >= lat_minHW) & (lats_sm <= lat_maxHW))[0]

if (lons_sm < 0).any():
    lon_minHW_adj = lon_minHW - 360 if lon_minHW > 180 else lon_minHW
    lon_maxHW_adj = lon_maxHW - 360 if lon_maxHW > 180 else lon_maxHW
    ilon_sm = np.where((lons_sm >= lon_minHW_adj) & (lons_sm <= lon_maxHW_adj))[0]
else:
    ilon_sm = np.where((lons_sm >= lon_minHW) & (lons_sm <= lon_maxHW))[0]

sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values
sm_point = np.nanmean(sm_region, axis=(1, 2))

# -------------------- Align by common dates --------------------
common_dates = np.intersect1d(dates_LE, dates_sm)
print("common_dates:", common_dates)

LE_mask = np.isin(dates_LE, common_dates)
sm_mask = np.isin(dates_sm, common_dates)

LE_aligned = LE_point[LE_mask]
sm_aligned = sm_point[sm_mask]

print("aligned shapes:", LE_aligned.shape, sm_aligned.shape)

# -------------------- Summer days --------------------
summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

LE_summer = LE_aligned[summer_mask]
sm_summer = sm_aligned[summer_mask]

print("LE_summer shape:", LE_summer.shape)
print("sm_summer shape:", sm_summer.shape)

# -------------------- Heatwave onset days --------------------
hw_mask = np.isin(common_dates, dates_d_HW_day0)

LE_hw = LE_aligned[hw_mask]
sm_hw = sm_aligned[hw_mask]

print("LE_hw shape:", LE_hw.shape)
print("sm_hw shape:", sm_hw.shape)

# -------------------- Plot --------------------
mask = np.isfinite(sm_summer) & np.isfinite(LE_summer)
sm_valid = sm_summer[mask]
LE_valid = LE_summer[mask]


xy = np.vstack([sm_valid, LE_valid])
kde = gaussian_kde(xy)

xmin, xmax = np.nanmin(sm_valid), np.nanmax(sm_valid)
ymin, ymax = np.nanmin(LE_valid), np.nanmax(LE_valid)

xgrid, ygrid = np.meshgrid(
    np.linspace(xmin, xmax, 40),
    np.linspace(ymin, ymax, 40)
)
z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xgrid, ygrid, z, levels=20, cmap="Blues", alpha=0.8)
plt.colorbar(label="Density")
plt.scatter(sm_hw, LE_hw, s=35, color="Red", label="Heatwave onset")

plt.xlabel("Soil moisture", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title("Scatter plot of Evaporative Fraction vs Soil Moisture\n(Regional mean over summer and heatwave onset days)", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_HW_regionmean_{name_land}_{region}.png")
plt.close()

# -------------------- Plot --------------------
plt.figure(figsize=(8, 5))
plt.contourf(xgrid, ygrid, z, levels=20, cmap="Blues", alpha=0.8)
plt.colorbar(label="Density")
plt.xlabel("Soil moisture", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title("Scatter plot of Evaporative Fraction vs Soil Moisture\n(Regional mean over summer and heatwave onset days)", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summer_regionmean_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()



# ======================================================= DENSITY PLOT ====================================================================

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================

def get_lon_idx(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# =======================================================
# Event centers
# =======================================================
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# =======================================================
# Open datasets once
# =======================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

# Common timeline between EF and SM
common_dates = np.intersect1d(dates_LE, dates_sm)

# Summer-only mask on common timeline
summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])
summer_dates = common_dates[summer_mask]

# =======================================================
# Containers: one summer time series per event
# =======================================================
LE_summer_all_events = []
sm_summer_all_events = []

# =======================================================
# Build one summer time series per event-centered box
# =======================================================
for lat_center_hw, lon_center_hw in zip(lats_events, lons_events):

    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_idx(lons_LE, lon_center_hw, half_width=0.5)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_idx(lons_sm, lon_center_hw, half_width=0.5)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0:
        continue

    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values

    if LE_region.ndim != 3 or sm_region.ndim != 3:
        continue

    if not np.isfinite(LE_region).any() or not np.isfinite(sm_region).any():
        continue

    LE_point = np.nanmean(LE_region, axis=(1, 2))
    sm_point = np.nanmean(sm_region, axis=(1, 2))

    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # Align to common dates
    LE_mask = np.isin(dates_LE, common_dates)
    sm_mask = np.isin(dates_sm, common_dates)

    LE_aligned = LE_point[LE_mask]
    sm_aligned = sm_point[sm_mask]

    # Keep only summer
    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    LE_summer_all_events.append(LE_summer)
    sm_summer_all_events.append(sm_summer)

# =======================================================
# Stack and average across events
# =======================================================
LE_summer_all_events = np.array(LE_summer_all_events)   # (nevents, nsummerdays)
sm_summer_all_events = np.array(sm_summer_all_events)   # (nevents, nsummerdays)

LE_summer_mean = np.nanmean(LE_summer_all_events, axis=0)
sm_summer_mean = np.nanmean(sm_summer_all_events, axis=0)

# =======================================================
# Onset and onset+3 from the averaged time series
# =======================================================
all_LE_hw = []
all_sm_hw = []

all_LE_hw3 = []
all_sm_hw3 = []

for onset_day in dates_d_HW_day0:
    if onset_day in summer_dates:
        idx0 = np.where(summer_dates == onset_day)[0]
        if len(idx0) == 1:
            i0 = idx0[0]
            if np.isfinite(LE_summer_mean[i0]) and np.isfinite(sm_summer_mean[i0]):
                all_LE_hw.append(LE_summer_mean[i0])
                all_sm_hw.append(sm_summer_mean[i0])

    onset3_day = onset_day + dt.timedelta(days=3)
    if onset3_day in summer_dates:
        idx3 = np.where(summer_dates == onset3_day)[0]
        if len(idx3) == 1:
            i3 = idx3[0]
            if np.isfinite(LE_summer_mean[i3]) and np.isfinite(sm_summer_mean[i3]):
                all_LE_hw3.append(LE_summer_mean[i3])
                all_sm_hw3.append(sm_summer_mean[i3])

all_LE_hw = np.array(all_LE_hw)
all_sm_hw = np.array(all_sm_hw)
all_LE_hw3 = np.array(all_LE_hw3)
all_sm_hw3 = np.array(all_sm_hw3)

# =======================================================
# Density plot from averaged summer time series
# =======================================================
mask = np.isfinite(sm_summer_mean) & np.isfinite(LE_summer_mean)
sm_valid = sm_summer_mean[mask]
LE_valid = LE_summer_mean[mask]

plt.figure(figsize=(8, 5))

if len(sm_valid) > 5:
    xy = np.vstack([sm_valid, LE_valid])
    kde = gaussian_kde(xy)

    xmin, xmax = np.nanmin(sm_valid), np.nanmax(sm_valid)
    ymin, ymax = np.nanmin(LE_valid), np.nanmax(LE_valid)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 40),
        np.linspace(ymin, ymax, 40)
    )
    z = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)

    plt.contourf(xgrid, ygrid, z, levels=20, cmap="Blues", alpha=0.8)
    plt.colorbar(label="Density")
else:
    plt.scatter(sm_valid, LE_valid, s=12, alpha=0.5, color="cornflowerblue", label="Summer mean")

plt.scatter(all_sm_hw, all_LE_hw, s=24, color="red", label="Onset")
plt.scatter(all_sm_hw3, all_LE_hw3, s=24, color="orange", label="Onset +3 days")

plt.xlabel("SMs", fontsize=12)
plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
plt.title(
    "Density plot of averaged summer SMs vs EF\n"
    "using event-centered ±0.5° boxes",
    fontsize=12
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{path_figures}density_{variable_LE}_{variable_sm}_summerMean_HW_eachEvent_05deg_{name_land}_{region}.png")
plt.close()

ds_LE.close()
ds_sm.close()




aaaaa


# ==================================================== T - SM  ======================================================================
variable_sm = 'SMs'
ds_sm = xr.open_dataset(f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc', decode_times=False, engine="netcdf4")
lats_sm = np.array(ds_sm.lat.values)
lons_sm = np.array(ds_sm.lon.values)  
timei_sm = np.array(ds_sm.time.values) 
dates_sm = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei_sm])

pos_HW_day0_sm = []
for num in range(len(dates_sm)): 
    if dates_sm[num] in dates_d_HW_day0: pos_HW_day0_sm.append(num)
pos_HW_day0_sm = pd.DataFrame(pos_HW_day0_sm)
# --- indices: HEATWAVE DAYS ONLY
sm_idx = pos_HW_day0_sm.values.flatten()   # <-- HW days (not pos_summer)



pos_summer_sm = np.where([i.month in [6,7,8] for i in dates_sm])[0]
composites_matriz_sm, composites_matrix_complete_sm, pvalue_composites_matriz_sm = calculate_composites_ttest_optimized(pos_HW_day0_sm, pos_summer_sm, f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc', variable_sm, num_time_laps=20, pvalue=True, welch_test=True)
np.save(f'{path_outputs_case}composites_matriz_{variable_sm}_daily_anoma_window_{region}.npy', composites_matriz_sm)
np.save(f'{path_outputs_case}composites_matriz_{variable_sm}_daily_anoma_window_pvalues_welch_{region}.npy', pvalue_composites_matriz_sm)







# ======================================================= time series ====================================================================
# for var, path_file in zip(['SMrz', 'SMs', 'E', 'H'], [path_file_SMrz, path_file_SMs, path_file_E, path_file_H]):
for var, path_file in zip(['SMs', 'E', 'H'], [path_file_SMs, path_file_E, path_file_H]):
    variable_sm = var

    ncfile_sm = Dataset(f'{path_file}')
    lats_sm = np.array(ncfile_sm['lat'])
    lons_sm = np.array(ncfile_sm['lon'])  
    timei_sm = np.array(ncfile_sm['time'][:]) 
    dates_sm = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei_sm])
    print(dates_d_HWdays[:4])
    print(dates_sm[:4])
    # pos_initial_year = np.where([d.year == initial_year for d in dates_sm])[0][0]

    # pos_HWdays = []
    # for num in range(len(dates_sm)): 
    #     if dates_sm[num] in dates_d_HWdays: pos_HWdays.append(num)
    # pos_HWdays = pd.DataFrame(pos_HWdays)
    # print(pos_HWdays[:4])

    pos_HW_day0_sm = []
    for num in range(len(dates_sm)): 
        if dates_sm[num] in dates_d_HW_day0: pos_HW_day0_sm.append(num)
    pos_HW_day0_sm = pd.DataFrame(pos_HW_day0_sm)
    # --- indices: HEATWAVE DAYS ONLY
    sm_idx = pos_HW_day0_sm.values.flatten()   # <-- HW days (not pos_summer)



    path_sm_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable_sm}_{name_land}_US.nc'  
    path_sm_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc'


    if not os.path.exists(path_sm_annual_cycle_window):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d_sm = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in time.values])
        ds['time'] = dates_d_sm
        sm_daily = ds[variable_sm]

        annual_cycle_window_0 = compute_annual_cycle_window(sm_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        save_nc_3d(path_sm_annual_cycle_window, annual_cycle_window_0, lats, lons, np.arange(1, len(annual_cycle_window_0.time)+1).astype(str), variable_sm)


    if not os.path.exists(path_sm_daily_anoma_window):

        # 1) climatology (dayofyear, lat, lon)
        ds_clim = xr.open_dataset(path_sm_annual_cycle_window, decode_times=False, engine="netcdf4")
        clim = ds_clim[variable_sm]  # ideally has coord dayofyear=1..366
        cycle_times = pd.date_range(f"{2000}-01-01", f"{2000}-12-31", freq="D")
        clim = clim.assign_coords(time=("time", cycle_times))

        # 2) daily data
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4")  # keep time unchunked if possible
        # ds = ds.isel(time=slice(0,400))
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d_lwa = np.array([dt.datetime.strptime(iii, "%Y%m%d") for iii in time.values])
        ds = ds.assign_coords(time=("time", dates_d_lwa))
        
        expanded_clim = expand_annual_cycle_optimized(dates_d_lwa, len(lats), len(lons), clim)

        # 3) anomaly WITHOUT expanding
        anom = ds[variable_sm].astype("float32") - expanded_clim
        del expanded_clim
        # 4) save (prefer to_netcdf over save_nc_3d if save_nc_3d materializes .values)
        save_nc_3d(path_sm_daily_anoma_window, anom.values, lats, lons, time.values, variable_sm)
        del anom, ds, ds_clim

    path_sm_daily_anoma_std_window_US = f'{path_outputs}{variable_sm}_ERA5_daily_anoma_std_window_US.nc'  
    path_sm_annual_cycle_std_US = f'{path_outputs}annual_cycle_std_{variable_sm}_daily_window_US.nc'
    if not os.path.exists(path_sm_daily_anoma_std_window_US):
        # Convert dates to daily
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4")
        pos_lat, pos_lon, lats_cut, lons_cut = cut_region_lats_lons(lats_sm, lons_sm, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
        ds = ds.isel(lat=pos_lat, lon=pos_lon)
        ds['time'] = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in ds['time'].values])
        print(ds['time'].values[:3])
        sm_daily = ds[variable_sm]
        print("sm_daily shape:", sm_daily.shape)
        # compute annual cycle
        std = compute_std_window(sm_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        print("std shape:", std.shape)
        save_nc_3d(path_sm_annual_cycle_std_US, std, lats_cut, lons_cut, np.arange(1, len(std.time)+1).astype(str), variable_sm)
        expanded_std = expand_annual_cycle_optimized(dates_sm, len(lats_cut), len(lons_cut), std).astype("float32")
        ds = xr.open_dataset(path_sm_daily_anoma_window, decode_times=False, engine="netcdf4")
        ds = ds.isel(lat=pos_lat, lon=pos_lon)
        print("good")
        del std
        anom = ds[variable_sm].astype("float32")
        std_anom = anom / expanded_std
        del expanded_std, anom
        save_nc_3d(path_sm_daily_anoma_std_window_US, std_anom, lats_cut, lons_cut, dates_d_str, variable_sm)
        del std_anom

    time_series_results = calculate_time_series_standard_deviation(pos_HW_day0_sm, path_sm_daily_anoma_std_window_US, variable_sm, 30, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, std=False)
    np.save(f"{path_outputs_case}time_series_results_{variable_sm}_daily_anoma_std_window_yb_{region}_US.npy", time_series_results)

    # time_series_results = calculate_time_series_standard_deviation(pos_HWdays, path_sm_daily_anoma_window, variable_sm, 30, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
    # np.save(f"{path_outputs_case}time_series_results_{variable_sm}_anoma_window_{name_land}_{region}.npy", time_series_results)

    # time_series_results = calculate_time_series_standard_deviation(pos_HWdays, path_file, variable_sm, 30, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
    # np.save(f"{path_outputs_case}time_series_results_{variable_sm}_{name_land}_{region}.npy", time_series_results)



aaaaa


aaaaaa

# ============================================================
# Helper: longitude-aware box selection
# ============================================================
def get_lon_indices(lons, lon_center, half_width=2):
    """
    Returns longitude indices for a +/- half_width box around lon_center.
    Works for longitude arrays in either [0,360] or [-180,180].
    """
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# ============================================================
# Open datasets only once
# ============================================================
variable_t = "TS"
variable_sm = "SMs"

ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_t = ds_t.lat.values
lons_t = ds_t.lon.values
dates_t = np.array([d.date() for d in dates_d])


lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])


# event centers from dataframe
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# output directory
os.makedirs(path_figures + "individual", exist_ok=True)

# ============================================================
# Loop over each event
# ============================================================
for iev, (lat_center_hw, lon_center_hw, onset_day) in enumerate(zip(lats_events, lons_events, dates_d_HW_day0)):
    print(f"Event {iev+1}: onset={onset_day}, lat={lat_center_hw:.2f}, lon={lon_center_hw:.2f}")

    if onset_day < dates_t[0] or onset_day < dates_sm[0]:
        print(f"Skipping event {iev+1}: onset date is before the EF/SM data.")
        continue

    # --- region bounds
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_t = np.where((lats_t >= lat_min) & (lats_t <= lat_max))[0]
    ilon_t = get_lon_indices(lons_t, lon_center_hw, half_width=2)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_indices(lons_sm, lon_center_hw, half_width=2)

    if len(ilat_t) == 0 or len(ilon_t) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0:
        print(f"Skipping event {iev+1}: empty spatial selection.")
        continue

    # --- spatial mean time series for this event-centered box
    t_region = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values

    t_point = np.nanmean(t_region, axis=(1, 2)) if t_region.ndim == 3 else np.nanmean(t_region)
    sm_point = np.nanmean(sm_region, axis=(1, 2)) if sm_region.ndim == 3 else np.nanmean(sm_region)


    # --- align by common dates
    common_dates = np.intersect1d(dates_t, dates_sm)


    t_mask = np.isin(dates_t, common_dates)
    sm_mask = np.isin(dates_sm, common_dates)

    t_aligned = t_point[t_mask]
    sm_aligned = sm_point[sm_mask]

    # --- summer cloud
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])
    t_summer = t_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    # --- helper to extract one day
    def get_point_for_day(target_day):
        idx = np.where(common_dates == target_day)[0]
        if len(idx) == 1:
            return sm_aligned[idx[0]], t_aligned[idx[0]]
        return np.nan, np.nan

    # day 0
    sm_hw, t_hw = get_point_for_day(onset_day)

    # days -3, -2, -1
    pre_days = [onset_day - dt.timedelta(days=d) for d in [3, 2, 1]]
    pre_points = np.array([get_point_for_day(d) for d in pre_days], dtype=float)

    # days +1, +2, +3, +4
    post_days = [onset_day + dt.timedelta(days=d) for d in [1, 2, 3, 4]]
    post_points = np.array([get_point_for_day(d) for d in post_days], dtype=float)

    # keep only valid rows
    pre_points = pre_points[~np.isnan(pre_points).any(axis=1)] if len(pre_points) > 0 else np.empty((0, 2))
    post_points = post_points[~np.isnan(post_points).any(axis=1)] if len(post_points) > 0 else np.empty((0, 2))

    # ========================================================
    # Plot for this event
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.scatter(
        sm_summer, t_summer,
        s=22, alpha=0.4, color="cornflowerblue",
        edgecolors="none", label="Summer"
    )

    if len(pre_points) > 0:
        plt.scatter(
            pre_points[:, 0], pre_points[:, 1],
            s=36, color="saddlebrown", label="Days -3 to -1", zorder=5
        )

    if not (np.isnan(sm_hw) or np.isnan(t_hw)):
        plt.scatter(
            sm_hw, t_hw,
            s=42, color="red", label="Onset", zorder=6
        )

    if len(post_points) > 0:
        plt.scatter(
            post_points[:, 0], post_points[:, 1],
            s=36, color="orange", label="Days +1 to +4", zorder=5
        )

    plt.xlabel("SMs", fontsize=12)
    plt.ylabel(r"T", fontsize=12)
    plt.title(
        f"Event {iev+1}: SMs vs T around onset\n"
        f"Center: ({lat_center_hw:.1f}°, {lon_center_hw:.1f}°), onset: {onset_day}",
        fontsize=12
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(
        f"{path_figures}/individual/scatter_{variable_t}_{variable_sm}_event{iev+1:03d}_{name_land}_{region}.png",
        dpi=200
    )
    plt.close()



print("Done.")

aaaaaaa







aaaaaaa


# ==================================================== T - SM  ======================================================================
variable_sm = 'SMs'
# ds_sm = xr.open_dataset(f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc', decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
lats_sm = np.array(ds_sm.lat.values)
lons_sm = np.array(ds_sm.lon.values)  
timei_sm = np.array(ds_sm.time.values) 
dates_sm = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei_sm])

pos_HWdays_sm = []
for num in range(len(dates_sm)): 
    if dates_sm[num] in dates_d_HWdays: pos_HWdays_sm.append(num)
pos_HWdays_sm = pd.DataFrame(pos_HWdays_sm)
# --- indices: HEATWAVE DAYS ONLY
sm_idx = pos_HWdays_sm.values.flatten()   # <-- HW days (not pos_summer)



# load temperature 
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")
# ds_t = xr.open_dataset(f'{path_outputs}TS_ERA5_daily_anoma_window.nc', decode_times=False, engine="netcdf4")
ds_t['time'] = dates_d
pos_initial_t = np.where([d.year == dates_sm[0].year for d in dates_d])[0][0]
pos_lats_hw_t, pos_lons_hw_t, new_lats_t, new_lons_t = cut_region_lats_lons(lats_t, lons_t, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
print("new_lons_t :", new_lons_t[:4])
print("new_lats_t :", new_lats_t[:4])
ds_t = ds_t.isel(lat=pos_lats_hw_t, lon=pos_lons_hw_t)
t = ds_t[variable_t][pos_initial_t:].values
t_hw = t[sm_idx, :, :]
print(t_hw.shape)
del t


pos_lats_hw_sm, pos_lons_hw_sm, new_lats_sm, new_lons_sm = cut_region_lats_lons(lats_sm, lons_sm, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
new_lons_sm_360 = np.where(new_lons_sm < 0, new_lons_sm + 360, new_lons_sm)
print("new_lons_sm_360 :", new_lons_sm_360[:4])
print("new_lats_sm :", new_lats_sm[:4])
ds_sm = ds_sm.isel(lat=pos_lats_hw_sm, lon=pos_lons_hw_sm)
sm_full = np.array(ds_sm[variable_sm].values)  # (time_hw, lat, lon)
sm_full_interpolated = interpolate_matrix_with_nan(sm_full, new_lats_sm, new_lons_sm_360, new_lats_t, new_lons_t)


# --- lagged correlation maps (lags you want)
lags_list = [20, 15, 10, 5, 0, -5]
corr_map_lags = np.full((len(lags_list), len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)
pval_map_lags = np.full((len(lags_list), len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)


for num, lag in enumerate(lags_list):

    print(f"Computing lag {lag}", flush=True)

    # Lag SM in time
    sm_lagged = lag_array(sm_full_interpolated, lag)   # (time_all, lat, lon)

    # Keep only HW days
    X = sm_lagged[sm_idx, :, :]           # lagged SM on HW days
    Y = t_hw.copy()                       # local T on HW days

    # Match lengths
    nT = min(X.shape[0], Y.shape[0])
    X = X[:nT, :, :]
    Y = Y[:nT, :, :]

    corr_map = np.full((len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)
    p_map = np.full((len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)

    for i in range(corr_map.shape[0]):
        for j in range(corr_map.shape[1]):
            x = X[:, i, j]
            y = Y[:, i, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if np.sum(ok) > 10:
                r, p = stats.spearmanr(x[ok], y[ok])
                corr_map[i, j] = r
                p_map[i, j] = p

    corr_map_lags[num] = corr_map
    pval_map_lags[num] = p_map

# -------------------------------
# 5) SAVE
# -------------------------------
np.save(f"{path_outputs_case}/corr_spearman_SMs_vs_localT_lags_map_HWonly.npy", corr_map_lags)
np.save(f"{path_outputs_case}/pval_spearman_SMs_vs_localT_lags_map_HWonly.npy", pval_map_lags)
# np.save(f"{path_outputs_case}/corr_anoma_SMs_vs_anoma_localT_lags_map_HWonly.npy", corr_map_lags)
# np.save(f"{path_outputs_case}/pval_anoma_SMs_vs_anoma_localT_lags_map_HWonly.npy", pval_map_lags)

print("Saved correlation and p-value maps.", flush=True)
del corr_map_lags, pval_map_lags



















# ==================================================== T - SM  ======================================================================
variable_sm = 'SMs'
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
# ds_sm = xr.open_dataset(f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc', decode_times=False, engine="netcdf4")
lats_sm = np.array(ds_sm.lat.values)
lons_sm = np.array(ds_sm.lon.values)  
timei_sm = np.array(ds_sm.time.values) 
dates_sm = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei_sm])

pos_HWdays_sm = []
for num in range(len(dates_sm)): 
    if dates_sm[num] in dates_d_HWdays: pos_HWdays_sm.append(num)
pos_HWdays_sm = pd.DataFrame(pos_HWdays_sm)
# --- indices: HEATWAVE DAYS ONLY
sm_idx = pos_HWdays_sm.values.flatten()   # <-- HW days (not pos_summer)


pos_summer_sm = np.where([i.month in [6,7,8] for i in dates_sm])[0]
composites_matriz_sm, composites_matrix_complete_sm, pvalue_composites_matriz_sm = calculate_composites_ttest_optimized(pos_HWdays_sm, pos_summer_sm, path_file_SMs, variable_sm, num_time_laps=20, pvalue=True)
np.save(f'{path_outputs_case}composites_matriz_{variable_sm}_{region}.npy', composites_matriz_sm)
np.save(f'{path_outputs_case}composites_matriz_{variable_sm}_pvalues_{region}.npy', pvalue_composites_matriz_sm)

aaaaa

# load temperature 
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")
# ds_t = xr.open_dataset(f'{path_outputs}TS_ERA5_daily_anoma_window.nc', decode_times=False, engine="netcdf4")
ds_t['time'] = dates_d
pos_initial_t = np.where([d.year == dates_sm[0].year for d in dates_d])[0][0]
pos_lats_hw_t, pos_lons_hw_t, new_lats_t, new_lons_t = cut_region_lats_lons(lats_t, lons_t, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
print("new_lons_t :", new_lons_t[:4])
print("new_lats_t :", new_lats_t[:4])
ds_t = ds_t.isel(lat=pos_lats_hw_t, lon=pos_lons_hw_t)
t = ds_t[variable_t][pos_initial_t:].values
t_hw = t[sm_idx, :, :]
print(t_hw.shape)
del t


pos_lats_hw_sm, pos_lons_hw_sm, new_lats_sm, new_lons_sm = cut_region_lats_lons(lats_sm, lons_sm, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW)
new_lons_sm_360 = np.where(new_lons_sm < 0, new_lons_sm + 360, new_lons_sm)
print("new_lons_sm_360 :", new_lons_sm_360[:4])
print("new_lats_sm :", new_lats_sm[:4])
ds_sm = ds_sm.isel(lat=pos_lats_hw_sm, lon=pos_lons_hw_sm)
sm_full = np.array(ds_sm[variable_sm].values)  # (time_hw, lat, lon)
sm_full_interpolated = interpolate_matrix_with_nan(sm_full, new_lats_sm, new_lons_sm_360, new_lats_t, new_lons_t)


# --- lagged correlation maps (lags you want)
lags_list = [20, 15, 10, 5, 0, -5]
corr_map_lags = np.full((len(lags_list), len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)
pval_map_lags = np.full((len(lags_list), len(new_lats_t), len(new_lons_t)), np.nan, dtype=float)


for num, lag in enumerate(lags_list):

    print(f"Computing lag {lag}", flush=True)

    # Lag SM in time
    sm_lagged = lag_array(sm_full_interpolated, lag)   # (time_all, lat, lon)

    # Keep only HW days
    X = sm_lagged[sm_idx, :, :]           # lagged SM on HW days
    Y = t_hw.copy()                       # local T on HW days

    # Match lengths
    nT = min(X.shape[0], Y.shape[0])
    X = X[:nT, :, :]
    Y = Y[:nT, :, :]

    # -------------------------------
    # Vectorized Pearson correlation
    # -------------------------------
    Xmean = np.nanmean(X, axis=0)
    Ymean = np.nanmean(Y, axis=0)

    X0 = X - Xmean
    Y0 = Y - Ymean

    cov = np.nanmean(X0 * Y0, axis=0)
    stdX = np.nanstd(X, axis=0)
    stdY = np.nanstd(Y, axis=0)

    corr_map = cov / (stdX * stdY)
    corr_map = np.where(np.isfinite(corr_map), corr_map, np.nan)

    # -------------------------------
    # p-values
    # -------------------------------
    p_map = np.full(corr_map.shape, np.nan, dtype=np.float32)

    for i in range(corr_map.shape[0]):
        for j in range(corr_map.shape[1]):
            x = X[:, i, j]
            y = Y[:, i, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if np.sum(ok) > 10:
                r, p = stats.pearsonr(x[ok], y[ok])
                p_map[i, j] = p

    corr_map_lags[num] = corr_map
    pval_map_lags[num] = p_map

# -------------------------------
# 5) SAVE
# -------------------------------
np.save(f"{path_outputs_case}/corr_SMs_vs_localT_lags_map_HWonly.npy", corr_map_lags)
np.save(f"{path_outputs_case}/pval_SMs_vs_localT_lags_map_HWonly.npy", pval_map_lags)
# np.save(f"{path_outputs_case}/corr_anoma_SMs_vs_anoma_localT_lags_map_HWonly.npy", corr_map_lags)
# np.save(f"{path_outputs_case}/pval_anoma_SMs_vs_anoma_localT_lags_map_HWonly.npy", pval_map_lags)

print("Saved correlation and p-value maps.", flush=True)
aaaaaa















# ============================================================
# Helper: longitude-aware box selection
# ============================================================
def get_lon_indices(lons, lon_center, half_width=2):
    """
    Returns longitude indices for a +/- half_width box around lon_center.
    Works for longitude arrays in either [0,360] or [-180,180].
    """
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]


# ============================================================
# Open datasets only once
# ============================================================
variable_LE = "EF"
variable_sm = "SMs"

ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_LE.time.values])

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in ds_sm.time.values])

# event centers from dataframe
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)

# output directory
os.makedirs(path_figures + "individual", exist_ok=True)

# ============================================================
# Loop over each event
# ============================================================
for iev, (lat_center_hw, lon_center_hw, onset_day) in enumerate(zip(lats_events, lons_events, dates_d_HW_day0)):
    print(f"Event {iev+1}: onset={onset_day}, lat={lat_center_hw:.2f}, lon={lon_center_hw:.2f}")

    if onset_day < dates_LE[0] or onset_day < dates_sm[0]:
        print(f"Skipping event {iev+1}: onset date is before the EF/SM data.")
        continue

    # --- region bounds
    lat_min = lat_center_hw - 0.5
    lat_max = lat_center_hw + 0.5

    ilat_LE = np.where((lats_LE >= lat_min) & (lats_LE <= lat_max))[0]
    ilon_LE = get_lon_indices(lons_LE, lon_center_hw, half_width=2)

    ilat_sm = np.where((lats_sm >= lat_min) & (lats_sm <= lat_max))[0]
    ilon_sm = get_lon_indices(lons_sm, lon_center_hw, half_width=2)

    if len(ilat_LE) == 0 or len(ilon_LE) == 0 or len(ilat_sm) == 0 or len(ilon_sm) == 0:
        print(f"Skipping event {iev+1}: empty spatial selection.")
        continue

    # --- spatial mean time series for this event-centered box
    LE_region = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
    sm_region = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values

    LE_point = np.nanmean(LE_region, axis=(1, 2)) if LE_region.ndim == 3 else np.nanmean(LE_region)
    sm_point = np.nanmean(sm_region, axis=(1, 2)) if sm_region.ndim == 3 else np.nanmean(sm_region)

    # optional quality control for EF
    LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)

    # --- align by common dates
    common_dates = np.intersect1d(dates_LE, dates_sm)

    LE_mask = np.isin(dates_LE, common_dates)
    sm_mask = np.isin(dates_sm, common_dates)

    LE_aligned = LE_point[LE_mask]
    sm_aligned = sm_point[sm_mask]

    # --- summer cloud
    summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])
    LE_summer = LE_aligned[summer_mask]
    sm_summer = sm_aligned[summer_mask]

    # --- helper to extract one day
    def get_point_for_day(target_day):
        idx = np.where(common_dates == target_day)[0]
        if len(idx) == 1:
            return sm_aligned[idx[0]], LE_aligned[idx[0]]
        return np.nan, np.nan

    # day 0
    sm_hw, LE_hw = get_point_for_day(onset_day)

    # days -3, -2, -1
    pre_days = [onset_day - dt.timedelta(days=d) for d in [3, 2, 1]]
    pre_points = np.array([get_point_for_day(d) for d in pre_days], dtype=float)

    # days +1, +2, +3, +4
    post_days = [onset_day + dt.timedelta(days=d) for d in [1, 2, 3, 4]]
    post_points = np.array([get_point_for_day(d) for d in post_days], dtype=float)

    # keep only valid rows
    pre_points = pre_points[~np.isnan(pre_points).any(axis=1)] if len(pre_points) > 0 else np.empty((0, 2))
    post_points = post_points[~np.isnan(post_points).any(axis=1)] if len(post_points) > 0 else np.empty((0, 2))

    # ========================================================
    # Plot for this event
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.scatter(
        sm_summer, LE_summer,
        s=22, alpha=0.4, color="cornflowerblue",
        edgecolors="none", label="Summer"
    )

    if len(pre_points) > 0:
        plt.scatter(
            pre_points[:, 0], pre_points[:, 1],
            s=36, color="saddlebrown", label="Days -3 to -1", zorder=5
        )

    if not (np.isnan(sm_hw) or np.isnan(LE_hw)):
        plt.scatter(
            sm_hw, LE_hw,
            s=42, color="red", label="Onset", zorder=6
        )

    if len(post_points) > 0:
        plt.scatter(
            post_points[:, 0], post_points[:, 1],
            s=36, color="orange", label="Days +1 to +4", zorder=5
        )

    plt.xlabel("SMs", fontsize=12)
    plt.ylabel(r"EF = $\dfrac{LE}{LE + H}$", fontsize=12)
    plt.title(
        f"Event {iev+1}: SMs vs EF around onset\n"
        f"Center: ({lat_center_hw:.1f}°, {lon_center_hw:.1f}°), onset: {onset_day}",
        fontsize=12
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(
        f"{path_figures}/individual/scatter_{variable_LE}_{variable_sm}_event{iev+1:03d}_{name_land}_{region}.png",
        dpi=200
    )
    plt.close()

print("Done.")

aaaaaaa












# ======================================================= soil moisture regimes ====================================================================
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)
lat_center_hw = np.mean(lats_events)
lon_center_hw = np.mean(lons_events)

print("lat_center_hw:", lat_center_hw)
print("lon_center_hw:", lon_center_hw)


# -------------------- latent heat --------------------
variable_LE = "EF"
ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine="netcdf4")

lats_LE = ds_LE.lat.values
lons_LE = ds_LE.lon.values
timei_LE = ds_LE.time.values
dates_LE = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in timei_LE])

ilat_LE = np.argmin(np.abs(lats_LE - lat_center_hw))
if (lons_LE < 0).any():
    lon_center_hw_180 = lon_center_hw - 360
    ilon_LE = np.argmin(np.abs(lons_LE - lon_center_hw_180))
else:
    ilon_LE = np.argmin(np.abs(lons_LE - lon_center_hw))
LE_point = ds_LE[variable_LE].isel(lat=ilat_LE, lon=ilon_LE).values
# LE_point = evaporation_to_latent_heat(LE_point)
LE_point = np.where((LE_point >= 0) & (LE_point <= 1), LE_point, np.nan)


# -------------------- Soil moisture --------------------
variable_sm = "SMs"
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
timei_sm = ds_sm.time.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in timei_sm])

ilat_sm = np.argmin(np.abs(lats_sm - lat_center_hw))
if (lons_sm < 0).any():
    lon_center_hw_180 = lon_center_hw - 360
    ilon_sm = np.argmin(np.abs(lons_sm - lon_center_hw_180))
else:
    ilon_sm = np.argmin(np.abs(lons_sm - lon_center_hw))
sm_point = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values


# -------------------- Align by common dates --------------------
common_dates = np.intersect1d(dates_LE, dates_sm)
print("common_dates :", common_dates)

LE_mask = np.isin(dates_LE, common_dates)
sm_mask = np.isin(dates_sm, common_dates)

LE_aligned = LE_point[LE_mask]
sm_aligned = sm_point[sm_mask]

print("aligned shapes:", LE_aligned.shape, sm_aligned.shape)

# -------------------- Summer days --------------------

summer_mask = np.array([d.month in [6, 7, 8] for d in common_dates])

LE_summer = LE_aligned[summer_mask]
sm_summer = sm_aligned[summer_mask]

print("LE_summer shape:", LE_summer.shape)
print("sm_summer shape:", sm_summer.shape)

# -------------------- Heatwave days --------------------
hw_mask = np.isin(common_dates, dates_d_HW_day0)

LE_hw = LE_aligned[hw_mask]
sm_hw = sm_aligned[hw_mask]

print("LE_hw shape:", LE_hw.shape)
print("sm_hw shape:", sm_hw.shape)

# -------------------- Plot --------------------
plt.figure(figsize=(10, 5))
plt.scatter(sm_summer, LE_summer, marker="x", label="Summer", color="blue")
plt.scatter(sm_hw, LE_hw, marker="x", label="Heatwave", color="orange")
plt.xlabel("Soil moisture")
plt.ylabel("Latent heat")
plt.title("Scatter plot of Latent Heat vs Soil Moisture\n(Summer & Heatwave Days)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{path_figures}scatter_{variable_LE}_{variable_sm}_summer_HW_{name_land}_{region}.png")

AAAA


# ======================================================= scatter plots ====================================================================
# T vs SMs: to check if the events are under hypersensitive regime
coord_events = df_heatwaves.iloc[:, 4]
lats_events, lons_events = extract_latlons_from_coord_events(coord_events)
lat_center_hw = np.mean(lats_events)
lon_center_hw = np.mean(lons_events)

print("lat_center_hw:", lat_center_hw)
print("lon_center_hw:", lon_center_hw)

# -------------------- Soil moisture --------------------
variable_sm = "SMs"
ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")

lats_sm = ds_sm.lat.values
lons_sm = ds_sm.lon.values
timei_sm = ds_sm.time.values
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in timei_sm])

ilat_sm = np.argmin(np.abs(lats_sm - lat_center_hw))
if (lons_sm < 0).any():
    lon_center_hw_180 = lon_center_hw - 360
    ilon_sm = np.argmin(np.abs(lons_sm - lon_center_hw_180))
else:
    ilon_sm = np.argmin(np.abs(lons_sm - lon_center_hw))
sm_point = ds_sm[variable_sm].isel(lat=ilat_sm, lon=ilon_sm).values


# -------------------- Temperature --------------------
ds_t = xr.open_dataset(path_file_t, decode_times=False, engine="netcdf4")
dates_d = np.array([d.date() for d in dates_d])
ds_t = ds_t.assign_coords(time=dates_d)

ilat_t = np.argmin(np.abs(lats_t - lat_center_hw))
ilon_t = np.argmin(np.abs(lons_t - lon_center_hw))


t_point = ds_t[variable_t].isel(lat=ilat_t, lon=ilon_t).values

# -------------------- Align by common dates --------------------
common_dates = np.intersect1d(dates_sm, dates_d)
print("common_dates :", common_dates)

# restrict to analysis period if needed
common_dates = np.array([d for d in common_dates if d.year <= dates_d[-1].year])

sm_mask = np.isin(dates_sm, common_dates)
t_mask = np.isin(dates_d, common_dates)

sm_aligned = sm_point[sm_mask]
t_aligned = t_point[t_mask]

print("aligned shapes:", sm_aligned.shape, t_aligned.shape)

# -------------------- Summer days --------------------
for month in [6, 7, 8]:
    summer_mask = np.array([d.month == month for d in common_dates])

    sm_summer = sm_aligned[summer_mask]
    t_summer = t_aligned[summer_mask]

    print("sm_summer shape:", sm_summer.shape)
    print("t_summer shape:", t_summer.shape)

    # -------------------- Heatwave days --------------------
    dates_HW_day0_month = np.array([d for d in dates_d_HW_day0 if d.month == month])
    hw_mask = np.isin(common_dates, dates_HW_day0_month)

    sm_hw = sm_aligned[hw_mask]
    t_hw = t_aligned[hw_mask]

    print("sm_hw shape:", sm_hw.shape)
    print("t_hw shape:", t_hw.shape)

    # -------------------- Plot --------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(sm_summer, t_summer, marker="x", label="Summer", color="blue")
    plt.scatter(sm_hw, t_hw, marker="x", label="Heatwave", color="orange")
    plt.xlabel("Soil Moisture")
    plt.ylabel("Temperature")
    plt.title("Scatter plot of Soil Moisture vs Temperature\n(Summer & Heatwave Days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_figures}scatter_SM_T_month{str(month)}_HW_{name_land}_{region}.png")

aaaaaa













