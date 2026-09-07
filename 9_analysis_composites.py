from Functions import *
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
import random 


case = 'ERA5_p95_centerUS'
topography = True
# lat_minHW = 20; lat_maxHW = 55; lon_minHW = 235 - 360; lon_maxHW = 290 - 360; midlat=45 #235 to 290
lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235 - 360; lon_maxHW = 290 - 360; midlat=45 #235 to 290
path_figures = f'/home/castanev/Soil_moisture_HW/{case}/Figures/'
# path_figures = f'/home/castanev/Soil_moisture_HW/Figures/'
create_directory(path_figures)

# GLEAM DATA ==========================================================================================================================================
source_SM = 'GLEAM'
variable = 'SMrz'
path_data = f'/depot/wanglei/data/{source_SM}/SM/'
path_outputs = f'{path_data}outputs/'
create_directory(path_outputs)
file_name = f'{variable}_US_1980_2022_GLEAM_v4.1a'
# name_file_posHW = "/home/castanev/Amplification_RW/ERA5_NOFILTER_95_relative_47/Heat_waves_events_list_complete.csv"
# name_file_posHW = "/home/castanev/Heat-waves-dynamics/ERA5_NOFILTER_97.5_relative_47/Heat_waves_events_list.csv"
name_file_posHW = f"/home/castanev/Soil_moisture_HW/{case}/Heat_waves_events_list.csv"



ncfile = Dataset(f'{path_outputs}{file_name}.nc')
# data_us = np.array(ncfile[variable][:])
time = np.array(ncfile['time'][:])
dates = pd.DatetimeIndex(time)
lats_US = np.array(ncfile['lat'])
lons_US = np.array(ncfile['lon'])
# df_data_us = pd.DataFrame(index=time, data=np.reshape(data_us, [data_us.shape[0], data_us.shape[1] * data_us.shape[2]])).astype('float32')

# climatology_month = np.zeros([12, data_us.shape[1], data_us.shape[2]])
# del data_us
# for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
#     month = df_data_us.loc[pd.DatetimeIndex(df_data_us.index).month == i]
#     clima_month = np.nanmean(np.array(month), axis=0)
#     climatology_month[i-1] = clima_month.reshape(lats_US.shape[0], lons_US.shape[0])
#     vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable, colormap, f'{path_figures}{source_SM}_{variable}_{mo}', topography = True)

# data_US_seasonal_mean = np.empty([4,len(lats_US),len(lons_US)]) * np.nan
# for num, j, season in zip([0,1,2,3],[[11,0,1], [2,3,4], [5,6,7], [8,9,10]],['DJF','MAM','JJA','SON']):
#     data_US_seasonal_mean[num,:,:] = np.nanmean(climatology_month[j,:,:], axis=0)
# save_nc_3d(f'{path_outputs}{file_name}_seasonal_mean.nc', data_US_seasonal_mean, lats_US, lons_US, np.arange(0,4,1).astype(str), variable)
# data_US_seasonal_mean = np.array(Dataset(f'{path_outputs}{file_name}_seasonal_mean.nc')[variable])
# for num, season in zip([0,1,2,3],['DJF','MAM','JJA','SON']):
#     vmin, vmax, colormap, bounds = colorm(0, 0.3, 6, 0.05, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, data_US_seasonal_mean[num],  variable, colormap, f'{path_figures}{source_SM}_{variable}_{season}', topography = True)


list_HW = pd.read_csv(f'{name_file_posHW}', index_col=0)
dates_HW = pd.DatetimeIndex(list_HW.iloc[:,0].values)

pos_HW = dates.get_indexer(dates_HW)
pos_HW = pd.DataFrame(pos_HW[pos_HW >= 0])


# anom_data_US = anomalies_seasons(df_data_us)
# save_nc_3d(f'{path_outputs}anom_{file_name}.nc', anom_data_US, lats_US, lons_US, time, variable)
anom_data_US = np.array(Dataset(f'{path_outputs}anom_{file_name}.nc')[variable][:])

# anom_data_US_AMJJ = anom_data_US[np.where(dates.month.isin([4,5,6,7]))[0]]
# time_AMJJ = time[np.where(dates.month.isin([4,5,6,7]))[0]]
# save_nc_3d(f'{path_outputs}anom_{file_name}_AMJJ.nc', anom_data_US_AMJJ, lats_US, lons_US, time_AMJJ, variable)

# anom_data_US_Jul = anom_data_US[np.where(dates.month == 4)[0]]
# time_Jul = time[np.where(dates.month == 4)[0]]
# save_nc_3d(f'{path_outputs}anom_{file_name}_Jul.nc', anom_data_US_Jul, lats_US, lons_US, time_Jul, variable)


# df_anom_data_US = pd.DataFrame(index=pd.to_datetime(time[:], format='%Y%m%d'), data=np.reshape(anom_data_US, [anom_data_US.shape[0], anom_data_US.shape[1] * anom_data_US.shape[2]])).astype('float32')
# monthly_anom_data_US = df_anom_data_US.resample('M').mean()
# monthly_index = pd.to_datetime(monthly_anom_data_US.index)
# monthly_anom_data_US = np.array(monthly_anom_data_US).reshape(monthly_anom_data_US.shape[0], lats_US.shape[0], lons_US.shape[0])
# monthly_index_str = [ii.strftime("%Y%m%d") for ii in monthly_index]
# save_nc_3d(f'{path_outputs}monthly_anom_{file_name}.nc', monthly_anom_data_US, lats_US, lons_US, np.array(monthly_index_str, dtype='str'), variable)
monthly_anom_data_US = np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')[variable][:])
monthly_index = pd.DatetimeIndex(np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')['time'][:]))

vmin, vmax, colormap, bounds = colorm(-0.015, 0.015, 6, 0, 'RdBu_r') 

# composites_matrix_1month = calculate_composites_1month(pos_HW, anom_data_US)
# composites_coastlines_1month(lats_US, lons_US, composites_matrix_1month, vmin, vmax, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap, f'{path_figures}composites_1month_{variable}.jpg', var = 'SM anomaly', topography=topography)
# del composites_matrix_1month

# composites_matrix_3month = calculate_composites_3month(pos_HW, anom_data_US)
# composites_coastlines_3month(lats_US, lons_US, composites_matrix_3month, vmin, vmax, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap, f'{path_figures}composites_3month_{variable}.jpg', var = 'SM anomaly', topography=topography)
# del composites_matrix_3month

# JUNE - JULY EVENTS
dates_HW_Jul = dates_HW[dates_HW.month == 7]
print(dates_HW_Jul.shape)
dates_HW_jun_jul = dates_HW[dates_HW.month.isin([6, 7])]


years_HW = dates_HW.year.unique()
composites_matrix_monthly = calculate_composites_monthly(years_HW, monthly_anom_data_US, monthly_index)
composites_coastlines_monthly(lats_US, lons_US, composites_matrix_monthly, vmin, vmax, colormap, f'{path_figures}compositesHW_monthly_{variable}.jpg', var = 'SM anomaly', topography=topography)
del composites_matrix_monthly

years_HW_jun_jul = dates_HW_jun_jul.year.unique()
composites_matrix_monthly_jun_jul = calculate_composites_monthly(years_HW_jun_jul, monthly_anom_data_US, monthly_index)
composites_coastlines_monthly(lats_US, lons_US, composites_matrix_monthly_jun_jul, vmin, vmax, colormap, f'{path_figures}composites_Jun_JulHW_monthly_{variable}.jpg', var = 'SM anomaly', topography=topography)
del composites_matrix_monthly_jun_jul

years_HW_jul = dates_HW_Jul.year.unique()
composites_matrix_monthly_jul = calculate_composites_monthly(years_HW_jul, monthly_anom_data_US, monthly_index)
composites_coastlines_monthly(lats_US, lons_US, composites_matrix_monthly_jul, vmin, vmax, colormap, f'{path_figures}composites_JulHW_monthly_{variable}.jpg', var = 'SM anomaly', topography=topography)
del composites_matrix_monthly_jul

years = dates.year.unique()
years_nonHW = []
for i in range(len(years_HW)*5):
    year_rand = years[random.randint(0, len(years)-1)]
    if (year_rand in years_HW): continue
    elif len(years_nonHW) == len(years_HW): break
    else: years_nonHW.append(int(year_rand))
years_nonHW=np.array(years_nonHW)

composites_matrix_monthly_nonHW = calculate_composites_monthly(years_nonHW, monthly_anom_data_US, monthly_index)
composites_coastlines_monthly(lats_US, lons_US, composites_matrix_monthly_nonHW, vmin, vmax, colormap, f'{path_figures}composites_nonHW_monthly_{variable}.jpg', var = 'SM anomaly', topography=topography)
del composites_matrix_monthly_nonHW





Month = np.array([ii.month for ii in dates]) 
pos_summer = np.where([ii in [6, 7, 8] for ii in Month])[0]
pos_HW_day0_pre_20 = pd.Series([num - i for num in pos_HW for i in range(20)])
pos_nonHW_jul = []
for i in range(len(dates_HW_Jul)*5):
    pos_rand = pos_summer[random.randint(0, len(pos_summer)-1)]
    if (dates[pos_rand] in pos_HW) or (pos_rand in pos_HW_day0_pre_20): continue
    elif len(pos_nonHW_jul) == len(pos_HW): break
    else: pos_nonHW_jul.append(int(pos_rand))
dates_nonHW_Jul = dates[pos_nonHW_jul]



for months, dates_HW_month_i in zip(['Jul_nonHW', 'Jul', 'Jun_Jul'], [dates_nonHW_Jul, dates_HW_Jul, dates_HW_jun_jul]):
    pos_HW_month = dates.get_indexer(dates_HW_month_i); pos_HW_month = pd.DataFrame(pos_HW_month[pos_HW_month >= 0])

    composites_matrix_3month = calculate_composites_3month(pos_HW_month, anom_data_US)
    composites_coastlines_3month(lats_US, lons_US, composites_matrix_3month, vmin, vmax, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, colormap, f'{path_figures}composites_{months}HW_3month_{variable}.jpg', var = 'SM anomaly', topography=topography)

    dates_HW_month = pd.to_datetime(dates_HW_month_i)

    # Extract years with heatwaves in July
    print(months)
    years_HW_Jul = dates_HW_month.year.unique()

    # Calculate the mean of all Aprils during those years
    pos = np.where((monthly_index.month == 4) & (monthly_index.year.isin(years_HW_Jul)))[0]

    mean_value = monthly_anom_data_US[pos]
    mean_value=np.nanmean(mean_value,axis=0)

    vmin, vmax, colormap, bounds = colorm(-0.02, 0.02, 6, 0, 'RdBu_r') 
    maps_USA(lons_US, lats_US, vmin, vmax, mean_value,  variable, colormap, f'{path_figures}{source_SM}_{variable}_anom_Apr_{months}HW', topography = True)





# ==========================================================================================
# SURFACE TEMPERATURE ======================================================================

source_TS = 'ERA5'
variable = 'TS'
file_name = f'{variable}_{source_TS}_US.nc'
path_data = f'/depot/wanglei/data/Reanalysis/{source_TS}/TS/TS_US/'
path_outputs = f'/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/'

ncfile = Dataset(f'{path_data}{file_name}')
time = np.array(ncfile['time'][:])
lats_US_t = np.array(ncfile['lat'])
lons_US_t = np.array(ncfile['lon'])
# t_k = np.array(ncfile[variable][:,:,:]) #(t, lat, lon) °K
# t = pd.DataFrame(data=np.reshape(t_k, [t_k.shape[0], t_k.shape[1] * t_k.shape[2]]))
# t = t - 273.15 # °C
# t_US = t.values.reshape(t_k.shape[0], t_k.shape[1], t_k.shape[2])

dates_d = np.array([dt.datetime(1950,1,1) + dt.timedelta(days = i) for i in range(len(time))])
timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])

date_0 = dates[0]
pos_date_0 = np.where(dates_d == date_0)
dates_d = dates_d[pos_date_0[0][0]:]; timei = timei[pos_date_0[0][0]:]
# t_US = t_US[pos_date_0[0][0]:]

# df_t = pd.DataFrame(index=dates_d, data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
# anom_t_US = anomalies_seasons(df_t)

# monthly_anom_t_US = anom_t_US.resample('M').mean()
# monthly_index_t = pd.to_datetime(monthly_anom_t_US.index)
# monthly_anom_t_US = np.array(monthly_anom_t_US).reshape(monthly_anom_t_US.shape[0], lats_US_t.shape[0], lons_US_t.shape[0])
# monthly_index_str_t = [ii.strftime("%Y%m%d") for ii in monthly_index_t]
# save_nc_3d(f'{path_outputs}monthly_anom_{nc_name}.nc', monthly_anom_t_US, lats_US_t, lons_US_t, np.array(monthly_index_str_t, dtype='str'), variable)
monthly_anom_t_US = np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')[variable][:])
monthly_index_t = pd.DatetimeIndex(np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')['time'][:]))







# DETECTION OF DIPOLE STRUCTURE IN APRIL 
# Positive dipole defined as wet conditions (anomaly of SMrz > 0) in the Northwestern US and dry conditions
# (anomaly of SMrz < 0) in the Great Plains 

for dipole in ['dipole1', 'dipole2']:
    print(dipole)
    if dipole == 'dipole1':
        lat_minHW_nw = 38; lat_maxHW_nw = 45; lon_minHW_nw = -120; lon_maxHW_nw = -95
        lat_minHW_gp = 32; lat_maxHW_gp = 38; lon_minHW_gp = -93; lon_maxHW_gp = -80 

    elif dipole == 'dipole2':
        lat_minHW_nw = 40; lat_maxHW_nw = 48; lon_minHW_nw = -120; lon_maxHW_nw = -110
        lat_minHW_gp = 31; lat_maxHW_gp = 41; lon_minHW_gp = -100; lon_maxHW_gp = -94


    nw_US = monthly_anom_data_US[:, np.where((lats_US >= lat_minHW_nw) & (lats_US <= lat_maxHW_nw))[0], :]
    nw_US = nw_US[:,:, np.where((lons_US >= lon_minHW_nw) & (lons_US <= lon_maxHW_nw))[0]]

    great_plains = monthly_anom_data_US[:, np.where((lats_US >= lat_minHW_gp) & (lats_US <= lat_maxHW_gp))[0], :]
    great_plains = great_plains[:,:, np.where((lons_US >= lon_minHW_gp) & (lons_US <= lon_maxHW_gp))[0]]

    min_perc = 0.3
    pos_dipole = []
    for i in range(len(monthly_index)):
        cond_1 = nw_US[i] > np.zeros_like(nw_US[i])+0.01
        grid_cont1 = np.count_nonzero(cond_1)/nw_US[i].size

        cond_2 = great_plains[i] < np.zeros_like(great_plains[i])-0.01
        grid_cont2 = np.count_nonzero(cond_2)/great_plains[i].size

        if (grid_cont1 > min_perc) and (grid_cont2 > min_perc) and (monthly_index[i].month == 4): pos_dipole.append(i) 

    dipole_map = np.nanmean(monthly_anom_data_US[pos_dipole], axis = 0)
    vmin, vmax, colormap, bounds = colorm(-0.02, 0.02, 6, 0, 'RdBu_r') 
    maps_USA_dipole(lons_US, lon_minHW_nw, lon_maxHW_nw, lon_minHW_gp, lon_maxHW_gp, lats_US, lat_minHW_nw, lat_maxHW_nw, lat_minHW_gp, lat_maxHW_gp, vmin, vmax, dipole_map,  variable, colormap, f'{path_figures}{source_SM}_SMrz_{dipole}Apr_P30', topography = True)


    pos_dipole = np.array(pos_dipole)[np.array(pos_dipole)<len(monthly_index_t)]

    print(monthly_index[pos_dipole])

    for mon_summ, name_mon_summ in zip([2,3,4], ['Jun', 'Jul', 'Aug']):
        pos_dipole_summer = pos_dipole + mon_summ

        li, ls, intervalos, limite, color = 0.3, 1.1, 15, 0.4, 'RdYlBu_r'
        bounds = np.round(np.linspace(li, ls, intervalos), 3)
        colormap = center_white_anom(color, intervalos, bounds, limite)
        maps_USA(lons_US_t, np.ndarray.round(lats_US_t,2), 0.3, 1.1, np.nanmean(monthly_anom_t_US[pos_dipole_summer], axis=0), r'SAT anomalies [°C]',
            colormap, path_figures + f'TSanom_{name_mon_summ}_{dipole}Apr.png', topography)


# # FIND HOW IS THE TEMPERATURE ANOMALY IN OTHER SUMMER MONTHS 
# # FIND HOW IS THE ANOMALY IN NON DIPOLE YEARS 
# # CHANGE TO CENTER HEATWAVES
aaaaa




# ==========================================================================================
# STREAMFUNCTION ===========================================================================
source = 'ERA5'
variable_sf = 'SF'

ncfile = Dataset(f'/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/v300_ERA5.nc')
time = np.array(ncfile['time'][:])
lats_sf = np.array(ncfile['lat'])
lons_sf = np.array(ncfile['lon'])  
pos_lats_NH_sf = np.where((lats_sf > 0))[0]
lats_NH_sf = lats_sf[pos_lats_NH_sf]

ncfile = Dataset(f'/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/sf_vp_300_ERA5.nc')
sf = np.array(ncfile[variable_sf])[:,pos_lats_NH_sf,:]
sf = sf[:].astype('float32')

dates_d = np.array([dt.datetime(1950,1,1) + dt.timedelta(days = i) for i in range(len(time))])
timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])


df_sf = pd.DataFrame(index=dates_d, data=np.reshape(sf, [sf.shape[0], sf.shape[1] * sf.shape[2]])).astype('float32')
climatology = CLIMA(df_sf)
del df_sf
climatology = climatology.values.reshape(climatology.shape[0], lats_NH_sf.shape[0], lons_sf.shape[0])
sf_anom = np.subtract(sf, climatology)
del climatology
del sf

nc_name = f'sf_anom_NOFILTER_{source}_NH.nc'
ncfile = Dataset(f'{path_outputs}{nc_name}', 'w')

ncfile.createDimension('lat', len(lats_NH_sf))
ncfile.createDimension('lon', len(lons_sf))
ncfile.createDimension('time', len(time))

var_lats = ncfile.createVariable('lat', 'f', ('lat'))
var_lons = ncfile.createVariable('lon', 'f', ('lon'))
var_time = ncfile.createVariable('time', 'f', ('time'))

var_lats[:] = lats_NH_sf
var_lons[:] = lons_sf
var_time[:] = timei

varr = ncfile.createVariable(variable_sf, 'f', ('time', 'lat', 'lon'))
varr[:, :, :] = sf_anom[:,:,:]
ncfile.close()


nc_name = f'sf_anom_NOFILTER_{source}_NH.nc'
ncfile = Dataset(f'{path_outputs}{nc_name}') 
sf_anom = np.array(ncfile[variable_sf]) 
aaaaa


file_name = f'{variable}_{source}_US.nc'
path_data = f'/depot/wanglei/data/Reanalysis/{source}/TS/TS_US/'
path_outputs = f'/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/'

ncfile = Dataset(f'{path_data}{file_name}')
time = np.array(ncfile['time'][:])
lats_US = np.array(ncfile['lat'])
lons_US = np.array(ncfile['lon'])
# t_k = np.array(ncfile[variable][:,:,:]) #(t, lat, lon) °K
# t = pd.DataFrame(data=np.reshape(t_k, [t_k.shape[0], t_k.shape[1] * t_k.shape[2]]))
# t = t - 273.15 # °C
# t_US = t.values.reshape(t_k.shape[0], t_k.shape[1], t_k.shape[2])

dates_d = np.array([dt.datetime(1950,1,1) + dt.timedelta(days = i) for i in range(len(time))])
timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])

date_0 = dates[0]
pos_date_0 = np.where(dates_d == date_0)
dates_d = dates_d[pos_date_0[0][0]:]; timei = timei[pos_date_0[0][0]:]
# t_US = t_US[pos_date_0[0][0]:]

# df_t = pd.DataFrame(index=dates_d, data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
# anom_t_US = anomalies_seasons(df_t)

# monthly_anom_t_US = anom_t_US.resample('M').mean()
# monthly_index_t = pd.to_datetime(monthly_anom_t_US.index)
# monthly_anom_t_US = np.array(monthly_anom_t_US).reshape(monthly_anom_t_US.shape[0], lats_US.shape[0], lons_US.shape[0])
# monthly_index_str_t = [ii.strftime("%Y%m%d") for ii in monthly_index_t]
# save_nc_3d(f'{path_outputs}monthly_anom_{nc_name}.nc', monthly_anom_t_US, lats_US, lons_US, np.array(monthly_index_str_t, dtype='str'), variable)
monthly_anom_t_US = np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')[variable][:])
monthly_index_t = pd.DatetimeIndex(np.array(Dataset(f'{path_outputs}monthly_anom_{file_name}.nc')['time'][:]))


pos_dipole = np.array(pos_dipole)[np.array(pos_dipole)<len(monthly_index_t)]
pos_dipole_jul = pos_dipole + 3
print(monthly_index[pos_dipole])
print(monthly_index[pos_dipole_jul])



li, ls, intervalos, limite, color = 0.3, 1.5, 15, 0.4, 'RdYlBu_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps_USA(lons_US, np.ndarray.round(lats_US,2), 0.3, 1,5 + np.abs(bounds[0] - bounds[1]), np.nanmean(monthly_anom_t_US[pos_dipole_jul], axis=0), r'SAT anomalies [°C]',
    colormap, path_figures + f'TSanom_Jul_dipole2Apr.png', topography)

print(np.nanmax(np.nanmean(monthly_anom_t_US[pos_dipole_jul], axis=0)))













