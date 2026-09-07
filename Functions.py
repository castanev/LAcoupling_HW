
import pandas as pd
import numpy as np 
from multiprocessing import Pool
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.cm 
import os
from scipy.signal import windows
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy
import numpy as np
import matplotlib.ticker as mticker
from cartopy import crs
import matplotlib.ticker as ticker
import xarray as xr
from scipy import stats
from scipy import ndimage
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import RectBivariateSpline
from scipy.signal import hilbert


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



def seasonal_anomalies_by_year_optimized(dates_df_VAR, VAR):
    
    if VAR.ndim == 3:
        df_VAR = pd.DataFrame(index=pd.DatetimeIndex(dates_df_VAR), data=np.reshape(VAR.values, [VAR.shape[0], VAR.shape[1]*VAR.shape[2]])).astype('float32')
    else:
        df_VAR = pd.DataFrame(index=pd.DatetimeIndex(dates_df_VAR), data=np.array(VAR.values).astype('float32'))
    # Create a copy to avoid modifying original
    ANOMA = df_VAR.copy()
    
    # Add year and season columns
    df_with_seasons = df_VAR.copy()
    df_with_seasons['year'] = df_VAR.index.year
    df_with_seasons['season'] = pd.cut(df_with_seasons.index.month, 
                                      bins=[0, 3, 6, 9, 12], 
                                      labels=['winter', 'spring', 'summer', 'fall'],
                                      include_lowest=True)
    
    # Calculate seasonal means for each year using groupby
    seasonal_means_by_year = df_with_seasons.groupby(['year', 'season']).mean()
    
    # Calculate anomalies for each year and season
    for year in df_with_seasons['year'].unique():
        for season in ['spring', 'summer', 'fall']:
            # Get data for this year and season
            mask = (df_with_seasons['year'] == year) & (df_with_seasons['season'] == season)
            season_data = df_with_seasons[mask]
            
            if len(season_data) > 0:
                # Get seasonal mean for this year
                seasonal_mean = seasonal_means_by_year.loc[(year, season)].iloc[:-2]  # Exclude year and season columns
                # Calculate anomaly: data - seasonal_mean
                anomaly = season_data.iloc[:, :-2] - seasonal_mean  # Exclude year and season columns
                # Assign anomalies back to original dataframe
                ANOMA.loc[season_data.index] = anomaly

    if VAR.ndim == 3:
        ANOMA = ANOMA.values.reshape(ANOMA.shape[0], VAR.shape[1], VAR.shape[2])
    else:
        ANOMA = ANOMA.values

    return ANOMA



def monthly_anomalies_by_year(dates_df_VAR, VAR):
    """
    Remove the monthly mean (for each year) from df_VAR.
    df_VAR must be a DataFrame with datetime index and shape [time, n_points].
    Returns a NumPy array of anomalies with the same shape.
    """
    df_VAR = pd.DataFrame(index=pd.DatetimeIndex(dates_df_VAR), data=np.reshape(VAR, [VAR.shape[0], VAR.shape[1]*VAR.shape[2]])).astype('float32')
    # Ensure the index is datetime
    if not isinstance(df_VAR.index, pd.DatetimeIndex):
        raise ValueError("df_VAR must have a DatetimeIndex.")

    # Add year and month for grouping
    df_grouped = df_VAR.copy()
    df_grouped['year'] = df_grouped.index.year
    df_grouped['month'] = df_grouped.index.month

    # Calculate monthly means for each year
    monthly_means = df_grouped.groupby(['year', 'month']).transform('mean')

    # Subtract to get anomalies
    anomalies = df_VAR - monthly_means[df_VAR.columns]  # Only subtract data columns
    anomalies = anomalies.values.reshape(anomalies.shape[0], VAR.shape[1], VAR.shape[2])
    return anomalies


def temporal_smooth_3timesteps_nans(arr):
    smoothed = np.empty_like(arr)
    for i in range(len(arr)):
        window = arr[max(0, i-1):min(len(arr), i+2)]  # 3-point window
        smoothed[i] = np.nanmean(window, axis=0)
    return smoothed

def gaussian_filter_with_nans(data, sigma):
    mask = ~np.isnan(data)
    result = np.full_like(data, np.nan)
    result[mask] = gaussian_filter(data[mask], sigma=sigma)
    return result



def calculate_composites_simple(pos_HW, pos_non_HW, matriz, time, lats, lons, var, num_time_laps, pvalue=False):
    import numpy as np
    from scipy import stats
    dic_composites = {}


    # time_lags = np.arange(-20, 21, 1)
    time_lags = np.arange(-num_time_laps, num_time_laps+1, 1)
    for pos in pos_HW.index:
        if pos == 0: 
            dic_composites[0] = []
            dic_composites[0].append(int(pos_HW.iloc[0][0]))
            continue
        #elif pos_HW.iloc[pos][0] >= 27270: break  # len(other variables)
        elif pos_HW.iloc[pos][0] == time.shape[0]-20: break
        elif pos == pos_HW.index[-1]:
            break
        elif pos_HW.iloc[pos][0]-1 != pos_HW.iloc[pos-1][0]:
            for num in time_lags:  
                if num in dic_composites.keys(): dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))  
                else: 
                    dic_composites[num] = []
                    dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))

    print(dic_composites)
    composites_matrix_complete =  np.zeros((len(time_lags), lats.shape[0], lons.shape[0]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.nanmean(matriz[dic_composites[lag]], axis = 0)

    day_0 = np.nanmean(matriz[dic_composites[0]], axis=0)
    day_5 = np.nanmean(matriz[dic_composites[-5]], axis=0)
    day_10 = np.nanmean(matriz[dic_composites[-10]], axis=0)
    day_15 = np.nanmean(matriz[dic_composites[-15]], axis=0)
    day_20 = np.nanmean(matriz[dic_composites[-20]], axis=0)
    day_5_post = np.nanmean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((6, day_0.shape[0], day_0.shape[1]))
    for i, matriz_i in enumerate([day_20, day_15, day_10, day_5, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    # Significance test
    pvalue_composites_matrix = np.full_like(composites_matrix_complete, np.nan)

    

    if pvalue:
        for ii, lag in enumerate(np.sort(list(dic_composites))):
            # HW composites for this lag: shape (n_events, lat, lon)
            hw_data = np.array(ncfile[var][dic_composites[lag], :, :])

            # Reshape to (n_events, lat*lon) and (n_non_hw, lat*lon)
            hw_data_reshaped = hw_data.reshape(hw_data.shape[0], -1)
            hw_sample_size = np.sum(~np.isnan(hw_data_reshaped), axis=0)

            # Non-HW composites: shape (n_non_hw, lat, lon)
            random_non_hw_pos = np.random.choice(pos_non_HW, size=len(dic_composites[lag]), replace=False)
            non_hw_data = np.array(ncfile[var][random_non_hw_pos, :, :])
            non_hw_reshaped = non_hw_data.reshape(non_hw_data.shape[0], -1)

            # Gridpoint-wise t-test (axis=0 = across events/non-events)
            day_lag_p = stats.ttest_ind(hw_data_reshaped, non_hw_reshaped,
                                        axis=0, nan_policy='omit')[1]
            
            # minimum sample size to perform t-test
            day_lag_p[hw_sample_size < 30] = np.nan
            # Store back in (lat, lon)
            
            pvalue_composites_matrix[ii, :, :] = day_lag_p.reshape(hw_data.shape[1], hw_data.shape[2])
    else:
        pvalue_composites_matrix = None

    return composites_matrix, composites_matrix_complete, pvalue_composites_matrix


def calculate_composites_ttest(pos_HW, pos_non_HW, path_matriz, var, num_time_laps, pvalue=False):
    import numpy as np
    dic_composites = {}

    ncfile = Dataset(path_matriz)
    time = np.array(ncfile["time"][:])
    lats = np.array(ncfile["lat"][:])
    lons = np.array(ncfile["lon"][:])


    # time_lags = np.arange(-20, 21, 1)
    time_lags = np.arange(-num_time_laps, num_time_laps+1, 1)
    for pos in pos_HW.index:
        if pos == 0: 
            dic_composites[0] = []
            dic_composites[0].append(int(pos_HW.iloc[0][0]))
            continue
        #elif pos_HW.iloc[pos][0] >= 27270: break  # len(other variables)
        elif pos_HW.iloc[pos][0] == time.shape[0]-20: break
        elif pos == pos_HW.index[-1]:
            break
        elif pos_HW.iloc[pos][0]-1 != pos_HW.iloc[pos-1][0]:
            for num in time_lags:  
                if num in dic_composites.keys(): dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))  
                else: 
                    dic_composites[num] = []
                    dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))

    print(dic_composites)
    composites_matrix_complete =  np.zeros((len(time_lags), lats.shape[0], lons.shape[0]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.nanmean(np.array(ncfile[var][dic_composites[lag]]), axis = 0)

    day_0 = np.nanmean(np.array(ncfile[var][dic_composites[0]]), axis=0)
    day_5 = np.nanmean(np.array(ncfile[var][dic_composites[-5]]), axis=0)
    day_10 = np.nanmean(np.array(ncfile[var][dic_composites[-10]]), axis=0)
    day_15 = np.nanmean(np.array(ncfile[var][dic_composites[-15]]), axis=0)
    day_20 = np.nanmean(np.array(ncfile[var][dic_composites[-20]]), axis=0)
    day_5_post = np.nanmean(np.array(ncfile[var][dic_composites[5]]), axis=0)

    composites_matrix = np.zeros((6, day_0.shape[0], day_0.shape[1]))
    for i, matriz_i in enumerate([day_20, day_15, day_10, day_5, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    # Significance test
    pvalue_composites_matrix = np.full_like(composites_matrix_complete, np.nan)

    

    if pvalue:
        for ii, lag in enumerate(np.sort(list(dic_composites))):
            # HW composites for this lag: shape (n_events, lat, lon)
            hw_data = np.array(ncfile[var][dic_composites[lag], :, :])

            # Reshape to (n_events, lat*lon) and (n_non_hw, lat*lon)
            hw_data_reshaped = hw_data.reshape(hw_data.shape[0], -1)
            hw_sample_size = np.sum(~np.isnan(hw_data_reshaped), axis=0)

            # Non-HW composites: shape (n_non_hw, lat, lon)
            random_non_hw_pos = np.random.choice(pos_non_HW, size=len(dic_composites[lag]), replace=False)
            non_hw_data = np.array(ncfile[var][random_non_hw_pos, :, :])
            non_hw_reshaped = non_hw_data.reshape(non_hw_data.shape[0], -1)

            # Gridpoint-wise t-test (axis=0 = across events/non-events)
            day_lag_p = stats.ttest_ind(hw_data_reshaped, non_hw_reshaped,
                                        axis=0, nan_policy='omit')[1]
            
            # minimum sample size to perform t-test
            day_lag_p[hw_sample_size < 30] = np.nan
            # Store back in (lat, lon)
            
            pvalue_composites_matrix[ii, :, :] = day_lag_p.reshape(hw_data.shape[1], hw_data.shape[2])
    else:
        pvalue_composites_matrix = None

    return composites_matrix, composites_matrix_complete, pvalue_composites_matrix


def calculate_composites_ind_ttest_optimized(pos_HW, pos_non_HW, path_matriz, var, num_time_laps, pvalue=False, welch_test=False, mean_3days=False):
    import numpy as np
    from scipy import stats
    from netCDF4 import Dataset

    # --- Load data ---
    nc = Dataset(path_matriz)
    data = np.array(nc[var][:])     # may be 2D or 3D
    nc.close()

    # Detect dimensionality
    if data.ndim == 3:
        ntime, nlat, nlon = data.shape
        is_3d = True
    elif data.ndim == 2:
        ntime, nspace = data.shape     # only time x lon or time x lat
        is_3d = False
    else:
        raise ValueError("Input data must be 2D or 3D.")

    # Build time-lag array
    time_lags = np.arange(-num_time_laps, num_time_laps + 1)

    # HW indices
    hw_indices = np.array(pos_HW.iloc[:, 0], dtype=int)

    # Dictionary of composites
    dic_composites = {}
    if mean_3days == True:
        for lag in time_lags:
            idx0 = hw_indices + lag
            idx_before = hw_indices + lag -1
            idx_after = hw_indices + lag +1
            idx = np.concatenate([idx0, idx_before, idx_after])
            idx = idx[(idx >= 0) & (idx < ntime)]
            dic_composites[lag] = idx
    else:
        for lag in time_lags:
            idx = hw_indices + lag
            idx = idx[(idx >= 0) & (idx < ntime)]
            dic_composites[lag] = idx

    # --- Compute full lag composites ---
    if is_3d:
        composites_matrix_complete = np.stack([
            np.nanmean(data[idx, :, :], axis=0) for lag, idx in dic_composites.items()
        ])
    else:
        # 2D: mean over time only
        composites_matrix_complete = np.stack([
            np.nanmean(data[idx, :], axis=0) for lag, idx in dic_composites.items()
        ])

    # Helper for specific lags
    def safe_mean(lag):
        idx = dic_composites.get(lag, [])
        if len(idx) == 0:
            return np.full_like(composites_matrix_complete[0], np.nan)
        return np.nanmean(data[idx], axis=0)

    # Composites at selected lags
    composites_matrix = np.stack([
        safe_mean(-20),
        safe_mean(-15),
        safe_mean(-10),
        safe_mean(-5),
        safe_mean(0),
        safe_mean(5)
    ])

    # --- Optional t-test ---
    pvalue_composites_matrix = None
    if pvalue:
        pos_non_HW = np.array(pos_non_HW, dtype=int)

        if is_3d:
            pvalue_composites_matrix = np.full((len(time_lags), nlat, nlon), np.nan, dtype=np.float32)
        else:
            pvalue_composites_matrix = np.full((len(time_lags), nspace), np.nan, dtype=np.float32)

        count_lags=0
        for i, lag in enumerate(time_lags):
            idx_hw = np.array(dic_composites[lag], dtype=int)
            n_hw = len(idx_hw)

            if n_hw < 2:
                continue

            # remove lagged HW indices from summer sample
            pos_non_HW_clean = np.setdiff1d(pos_non_HW, idx_hw)

            if len(pos_non_HW_clean) < 2:
                continue

            # Flatten spatial dims
            hw_data = data[idx_hw].reshape(n_hw, -1)
            
            if welch_test:
                non_data = data[pos_non_HW_clean].reshape(len(pos_non_HW_clean), -1)
                _, pvals = stats.ttest_ind(
                    hw_data, non_data,
                    equal_var=False, axis=0, nan_policy='omit'
                )
            else:
                n_ref = min(n_hw, len(pos_non_HW_clean))
                idx_non = np.random.choice(pos_non_HW_clean, size=n_ref, replace=False)
                hw_data_sub = hw_data[:n_ref]
                non_data = data[idx_non].reshape(n_ref, -1)

                _, pvals = stats.ttest_ind(
                    hw_data_sub, non_data,
                    axis=0, nan_policy='omit'
                )

            # minimum valid samples
            hw_count = np.sum(~np.isnan(hw_data), axis=0)
            min_frac = 0.3   
            min_count = int(np.ceil(min_frac * n_hw))
            # print(np.max(hw_count))
            pvals[hw_count < min_count] = np.nan

            if is_3d:
                pvalue_composites_matrix[i] = pvals.reshape(nlat, nlon)
            else:
                pvalue_composites_matrix[i] = pvals
            
            
            if lag in [-20, -15, -10, -5, 0, 5]:

                composites_matrix_i = composites_matrix[count_lags]
                # Fix shape mismatch by reshaping hw_count < min_count to match composites_matrix_i shape
                # composites_matrix_i is shape (nlat, nlon), hw_count is flat
                # We reshape the mask and apply only for 3D
                if is_3d:
                    mask = (hw_count < min_count).reshape(nlat, nlon)
                    composites_matrix_i[mask] = np.nan
                else:
                    mask = (hw_count < min_count)
                    composites_matrix_i[mask] = np.nan
                composites_matrix[count_lags] = composites_matrix_i
                count_lags+=1

    return composites_matrix, composites_matrix_complete, pvalue_composites_matrix


def calculate_composites_ttest_optimized(pos_HW, pos_non_HW, path_matriz, var, num_time_laps, pvalue=False, welch_test=False, mean_3days=False):
    import numpy as np
    from scipy import stats
    from netCDF4 import Dataset

    # --- Load data ---
    nc = Dataset(path_matriz)
    data = np.array(nc[var][:])     # may be 2D or 3D
    nc.close()

    # Detect dimensionality
    if data.ndim == 3:
        ntime, nlat, nlon = data.shape
        is_3d = True
    elif data.ndim == 2:
        ntime, nspace = data.shape     # only time x lon or time x lat
        is_3d = False
    else:
        raise ValueError("Input data must be 2D or 3D.")

    # Build time-lag array
    time_lags = np.arange(-num_time_laps, num_time_laps + 1)

    # HW indices
    hw_indices = np.array(pos_HW.iloc[:, 0], dtype=int)

    # Dictionary of composites
    dic_composites = {}
    if mean_3days == True:
        for lag in time_lags:
            idx0 = hw_indices + lag
            idx_before = hw_indices + lag -1
            idx_after = hw_indices + lag +1
            idx = np.concatenate([idx0, idx_before, idx_after])
            idx = idx[(idx >= 0) & (idx < ntime)]
            dic_composites[lag] = idx
    else:
        for lag in time_lags:
            idx = hw_indices + lag
            idx = idx[(idx >= 0) & (idx < ntime)]
            dic_composites[lag] = idx

    # --- Compute full lag composites ---
    if is_3d:
        composites_matrix_complete = np.stack([
            np.nanmean(data[idx, :, :], axis=0) for lag, idx in dic_composites.items()
        ])
    else:
        # 2D: mean over time only
        composites_matrix_complete = np.stack([
            np.nanmean(data[idx, :], axis=0) for lag, idx in dic_composites.items()
        ])

    # Helper for specific lags
    def safe_mean(lag):
        idx = dic_composites.get(lag, [])
        if len(idx) == 0:
            return np.full_like(composites_matrix_complete[0], np.nan)
        return np.nanmean(data[idx], axis=0)

    # Composites at selected lags
    composites_matrix = np.stack([
        safe_mean(-20),
        safe_mean(-15),
        safe_mean(-10),
        safe_mean(-5),
        safe_mean(0),
        safe_mean(5)
    ])

    # --- Optional t-test ---
    pvalue_composites_matrix = None
    if pvalue:
        pos_non_HW = np.array(pos_non_HW, dtype=int)

        if is_3d:
            pvalue_composites_matrix = np.full((len(time_lags), nlat, nlon), np.nan, dtype=np.float32)
        else:
            pvalue_composites_matrix = np.full((len(time_lags), nspace), np.nan, dtype=np.float32)

        count_lags=0
        for i, lag in enumerate(time_lags):
            idx_hw = dic_composites[lag]
            n_hw = len(idx_hw)

            if n_hw < 2:
                continue

            idx_non = np.random.choice(pos_non_HW, size=n_hw, replace=False)

            # Flatten spatial dims
            hw_data = data[idx_hw].reshape(n_hw, -1)
            

            if welch_test == True:
                non_data = data[pos_non_HW].reshape(len(pos_non_HW), -1)
                _, pvals = stats.ttest_ind(hw_data, non_data, equal_var=False, axis=0, nan_policy='omit')
            else:
                non_data = data[idx_non].reshape(n_hw, -1)
                _, pvals = stats.ttest_ind(hw_data, non_data, axis=0, nan_policy='omit')

            # minimum valid samples
            hw_count = np.sum(~np.isnan(hw_data), axis=0)
            min_frac = 0.3   
            min_count = int(np.ceil(min_frac * n_hw))
            # print(np.max(hw_count))
            pvals[hw_count < min_count] = np.nan

            if is_3d:
                pvalue_composites_matrix[i] = pvals.reshape(nlat, nlon)
            else:
                pvalue_composites_matrix[i] = pvals
            
            
            if lag in [-20, -15, -10, -5, 0, 5]:

                composites_matrix_i = composites_matrix[count_lags]
                # Fix shape mismatch by reshaping hw_count < min_count to match composites_matrix_i shape
                # composites_matrix_i is shape (nlat, nlon), hw_count is flat
                # We reshape the mask and apply only for 3D
                if is_3d:
                    mask = (hw_count < min_count).reshape(nlat, nlon)
                    composites_matrix_i[mask] = np.nan
                else:
                    mask = (hw_count < min_count)
                    composites_matrix_i[mask] = np.nan
                composites_matrix[count_lags] = composites_matrix_i
                count_lags+=1

    return composites_matrix, composites_matrix_complete, pvalue_composites_matrix


import numpy as np



def calculate_frequency_composites_3d(
    pos_HW,
    matriz,
    var,
    matrix_threshold,
    test_excedence=True,
    num_lags=20,
    selected_lags=(-20, -15, -10, -5, 0, 5),
):
    """
    Frequency composites for a 3D field (time, lat, lon).

    For each lag L, compute:
        freq(L,lat,lon) = (# of HW-events where condition is true at time = event_start + L)
                          / (# of HW-events with non-NaN at that gridpoint at that time)

    Condition:
        if test_excedence: matriz >= threshold
        else:              matriz <= threshold

    Parameters
    ----------
    pos_HW : pandas DataFrame or Series-like
        Must contain indices in its first column (pos_HW.iloc[:,0]) as integer time indices.
        Typically your HW days list (can include consecutive days).
    matriz : np.ndarray
        Shape (time, lat, lon). Can contain NaNs.
    matrix_threshold : float
        Threshold for exceedance (>=) or below-threshold (<=).
    test_excedence : bool
        True -> exceedances (>= threshold). False -> below (<= threshold).
    num_lags : int
        Compute lags from -num_lags..+num_lags
    selected_lags : iterable of int
        Lags to return in the compact composites_matrix.

    Returns
    -------
    composites_matrix : np.ndarray
        Shape (len(selected_lags), lat, lon) with frequency at selected lags.
    composites_matrix_complete : np.ndarray
        Shape (2*num_lags+1, lat, lon) with frequency for all lags.
    time_lags : np.ndarray
        The lags used (length 2*num_lags+1).
    """

    # --- Load data ---
    nc = Dataset(matriz)
    matriz = np.array(nc[var][:])     # may be 2D or 3D
    nc.close()

    if matriz.ndim != 3:
        raise ValueError(f"matriz must be 3D (time,lat,lon). Got shape {matriz.shape}")
    ntime, nlat, nlon = matriz.shape
    time_lags = np.arange(-num_lags, num_lags + 1, dtype=int)

    # --- 1) Convert HW day list -> event start indices (avoid counting consecutive HW days)
    hw_days = np.asarray(pos_HW.iloc[:, 0], dtype=int)
    hw_days = np.sort(hw_days)

    if hw_days.size == 0:
        # no events -> return NaNs
        composites_matrix_complete = np.full((len(time_lags), nlat, nlon), np.nan, dtype=np.float32)
        composites_matrix = np.full((len(selected_lags), nlat, nlon), np.nan, dtype=np.float32)
        return composites_matrix, composites_matrix_complete, time_lags

    is_start = np.r_[True, np.diff(hw_days) != 1]
    event_starts = hw_days[is_start]

    # Keep only events that have full lag window inside bounds
    event_starts = event_starts[(event_starts + time_lags.min() >= 0) & (event_starts + time_lags.max() < ntime)]

    if event_starts.size == 0:
        composites_matrix_complete = np.full((len(time_lags), nlat, nlon), np.nan, dtype=np.float32)
        composites_matrix = np.full((len(selected_lags), nlat, nlon), np.nan, dtype=np.float32)
        return composites_matrix, composites_matrix_complete, time_lags

    # --- 2) Build boolean "condition met" safely with NaNs preserved
    valid = np.isfinite(matriz)  # True where not NaN/inf
    if test_excedence:
        cond = (matriz >= matrix_threshold) & valid
    else:
        cond = (matriz <= matrix_threshold) & valid

    # --- 3) Compute frequency for each lag
    composites_matrix_complete = np.empty((len(time_lags), nlat, nlon), dtype=np.float32)

    for ii, lag in enumerate(time_lags):
        idx = event_starts + lag  # shape (n_events,)
        # numerator = count of True among events
        num = cond[idx, :, :].sum(axis=0)                      # (lat, lon)
        den = valid[idx, :, :].sum(axis=0).astype(np.float32)  # (lat, lon)

        # avoid division by 0
        out = np.full((nlat, nlon), np.nan, dtype=np.float32)
        ok = den > 0
        out[ok] = (num[ok] / den[ok]).astype(np.float32)
        composites_matrix_complete[ii] = out

    # --- 4) Selected lags (like your safe_mean stack)
    lag_to_pos = {lag: i for i, lag in enumerate(time_lags)}
    selected_lags = list(selected_lags)
    composites_matrix = np.stack(
        [composites_matrix_complete[lag_to_pos[L]] if L in lag_to_pos
         else np.full((nlat, nlon), np.nan, dtype=np.float32)
         for L in selected_lags],
        axis=0
    ).astype(np.float32)

    return composites_matrix


def colormap_custom(min_v, max_v, steps, colors):
    norm_values = np.linspace(0, 1, len(colors))
    bar_vals = list(zip(norm_values, colors))
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", bar_vals)
    return min_v, max_v, steps,custom_cmap



def colorm(vmin, vmax, interv, lim, color):
    import matplotlib.colors as mcolors
    import numpy as np
    li, ls, intervalos, limite, color = vmin, vmax, interv, lim, color
    bounds = np.round(np.linspace(li, ls, intervalos), 8)
    colormap = center_white_anom(color, intervalos, bounds, limite)
    return vmin, vmax, colormap, bounds

def format_latitude(value, pos):
    # Format latitude ticks with degree symbol and 'N'
    if value >= 0:
        return f'{value:.0f}°N'
    else:
        return f'{-value:.0f}°S'
    

def composites_coastlines_2var(lats, lons, Matriz_1, min_1, max_1, Pvalue_matriz, Matriz_2, levels_2, lat_minHW, lat_maxHW, lon_minHW, lon_maxHW, cmap, path, var = '', topography = '', pvalue='', center_lon = 180, levels_cmap = 15):
    import matplotlib.pyplot as plt
    import cartopy
    import numpy as np
    import matplotlib.ticker as mticker
    from cartopy import crs
    
    if pvalue == True:
        Pvalue_matriz = Pvalue_matriz.copy()
        Pvalue_matriz[Pvalue_matriz<=0.05] = True
        Pvalue_matriz[Pvalue_matriz>0.05] = False

    fig = plt.figure(figsize=[17, 18])
    #fig = plt.figure(figsize=[19/2.54, 23/2.54])
    for i, tit in enumerate(['Day -20', 'Day -15', 'Day -10', 'Day -5', 'Day 0', 'Day 5']):
        ax = fig.add_subplot(6, 1, i + 1, projection=crs.PlateCarree(central_longitude=center_lon))
        if topography == True:
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.6, zorder=11)

        ax.set_title(tit, fontsize = 15)
        im = ax.contourf(lons, lats, Matriz_1[i, :, :], cmap=cmap, extend='both', levels=np.linspace(min_1, max_1, levels_cmap),transform=crs.PlateCarree())
        im2 = ax.contour(lons, lats, Matriz_2[i, :, :], extend='both', levels=levels_2, colors='k', linewidths=2,transform=crs.PlateCarree())
        ax.clabel(im2, inline=True, fontsize=10, fmt='%1.1f')
        if pvalue == True:
            true_coords = np.where((Pvalue_matriz[i]==1))
            im5 = ax.scatter(lons[true_coords[1]], lats[true_coords[0]], c= 'k', s = 0.5, transform=crs.PlateCarree())

        # if tit == 'Day 0':
        ax.plot([lon_minHW, lon_maxHW], [lat_minHW, lat_minHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_maxHW], [lat_maxHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_minHW, lon_minHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        ax.plot([lon_maxHW, lon_maxHW], [lat_minHW, lat_maxHW], transform=crs.PlateCarree(), color='b', lw=1.5)
        # ax.set_yticks([30, 50])
        ax.tick_params(axis='both', labelsize=14)
        plt.ylim(20,58)

        gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.ylocator = mticker.FixedLocator([30, 50])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_latitude))
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

    cbaxes = fig.add_axes([0.3, 0.06, 0.4, 0.015])
    cb = plt.colorbar(im, orientation="horizontal", pad=0.2, cax=cbaxes, format='%.1f')
    cb.set_label(var, fontsize=15)
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelsize=14)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.92,
                    wspace=0.1,
                    hspace=0.2)

    plt.savefig(path, dpi=500)
    # plt.show()
    plt.close()

def save_nc_3d(path_name, var, lats, lons, dates_str, var_str):
    ncfile = Dataset(path_name, 'w')
    ncfile.createDimension('lat', len(lats))
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', None)
    var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'str', ('time'))
    var_lats[:] = lats
    var_lons[:] = lons
    var_time[:] = dates_str
    varr = ncfile.createVariable(var_str, 'f', ('time', 'lat', 'lon'), fill_value=None)
    varr[:, :, :] = var
    ncfile.close()

def save_nc_2d(path_name, var, lons, dates_str, var_str):
    ncfile = Dataset(path_name, 'w')
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', None)
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'str', ('time'))
    var_lons[:] = lons
    var_time[:] = dates_str
    varr = ncfile.createVariable(var_str, 'f', ('time', 'lon'), fill_value=None)
    varr[:, :] = var
    ncfile.close()


def maps_USA(Lons, Lats, minn, maxx, matriz, var, cmap, path, topography=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import numpy as np

    # --- FIGURE ---
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # --- FEATURES ---
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.5)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)

    # --- PLOT ---
    levels = np.linspace(minn, maxx, 15)
    im = ax.contourf(
        Lons, Lats, matriz,
        cmap=cmap,
        levels=levels,
        extend='both',
        transform=ccrs.PlateCarree()
    )

    # --- GRIDLINES ---
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.4,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- ENSURE COLORBAR DOES NOT OVERLAP ---
    fig.subplots_adjust(bottom=0.18)   # leaves extra space at the bottom

    # adaptive colorbar position
    cax = fig.add_axes([0.13, 0.08, 0.76, 0.03])
    cb = fig.colorbar(im, orientation='horizontal', cax=cax, format='%.1f')
    cb.set_label(var, fontsize=10, color='dimgrey')
    cb.ax.tick_params(labelsize=9)

    # --- SAVE ---
    fig.savefig(path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def evolution_events(pos_event_pre_after, dates_d, t_US, lons_US, lats_US, path_figures, methodology, date0_str):
        fig = plt.figure(figsize=[10, 7])
        for i, pos in enumerate(pos_event_pre_after):
            date = dates_d[pos]
            ax = fig.add_subplot(3, 3, 1+i, projection=crs.PlateCarree(central_longitude=180))
            ax.add_feature(cartopy.feature.BORDERS, lw=0.5)

            ax.outline_patch.set_edgecolor('None')
            ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

            im2 = ax.contourf(lons_US, lats_US, t_US[pos,:,:], cmap='Reds', extend='both', \
                            levels=np.arange(17, 35, 0.2), transform=crs.PlateCarree())
            gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                            linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            gl.ylocator = mticker.FixedLocator([30, 45])
            gl.xlocator = mticker.FixedLocator([-120, -100, -80])
            gl.xlabel_style = {'size': 9, 'color': 'k'}
            gl.ylabel_style = {'size': 9, 'color': 'k'}
            ax.set_title(f'SAT {date.strftime("%Y-%m-%d")}', fontsize=9, color='k')
        cbaxes = fig.add_axes([0.31, 0.07, 0.38, 0.018])
        cb = plt.colorbar(im2, orientation="horizontal", pad=0.13, cax=cbaxes, format='%.0f')
        tick_locator = ticker.MaxNLocator(nbins=7)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.set_label('SAT [°C]', fontsize=10, color='k')
        cb.outline.set_edgecolor(None)
        cb.ax.tick_params(labelcolor='k', color='k', labelsize=9)
        #plt.show()
        plt.savefig(f'{path_figures}/Methodology_SAT_{methodology}_{date0_str}.png', dpi=200)
        plt.close()



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
    # # for heatwaves that are too close to each other (<20 days), keep the longest one
    # i = 0
    # while i < len(pos_day1_hw) - 1:
    #     if abs((pos_day1_hw[i] + duration_hw[i]) - pos_day1_hw[i + 1]) < 20:
    #         if duration_hw[i] < duration_hw[i + 1]:
    #             # Remove the shorter event (current one)
    #             duration_hw.pop(i)
    #             pos_day1_hw.pop(i)
    #             # Don't increment i since we removed an element
    #         else:
    #             # Remove the shorter event (next one)
    #             duration_hw.pop(i + 1)
    #             pos_day1_hw.pop(i + 1)
    #             # Don't increment i since we removed an element
    #     else:
    #         # No overlap, move to next pair
    #         i += 1
    return duration_hw, pos_day1_hw

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else: print(f'{directory_path} exists')

    
def anomalies_noseasons(df_VAR):
    import numpy as np
    ANOMA = df_VAR * np.nan
    media = df_VAR.mean()
    for i in range(df_VAR.shape[0]):
        dia = df_VAR.iloc[i]
        anoma = dia - media
        ANOMA.iloc[i] = anoma
    return ANOMA


# Find heatwave events in specific years for methodology visualization
def find_heatwave_positions_for_years(pos_day1_hw, duration_hw, dates_d, target_years=[2016, 2021]):
    """
    Find heatwave event positions for specific years to use in methodology visualization
    """
    year_events = {}
    
    for year in target_years:
        year_events[year] = []
        
        # Find events that start in the target year
        for i, (day1, dur) in enumerate(zip(pos_day1_hw, duration_hw)):
            event_date = dates_d[day1]
            if event_date.year == year:
                # Create a sequence of positions for this event
                event_positions = [day1 + j for j in range(dur)]
                year_events[year].append({
                    'event_id': i,
                    'start_pos': day1,
                    'duration': dur,
                    'positions': event_positions,
                    'start_date': event_date
                })
    
    return year_events



def center_white_anom(cmap, num, bounds, limite):
    barcmap = matplotlib.cm.get_cmap(cmap, num)
    barcmap.set_bad(color='white', alpha=0.5)
    bar_vals = barcmap(np.arange(num))  # extract those values as an array
    pos = np.arange(num)
    centro = pos[(bounds >= -limite) & (bounds <= limite)]
    for i in centro:
        bar_vals[i] = [1, 1, 1, 1]  # change the middle value
    newcmap = mpl.colors.LinearSegmentedColormap.from_list("new" + cmap, bar_vals)
    return newcmap



def maps_midlat(x, y, minn, maxx, matriz,  cmap, path, units='', topography = '', center_lon=180):

    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=[6, 3.5])
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=center_lon))
    
    if topography == True:
        # ax.add_feature(cartopy.feature.BORDERS, lw=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

    im = ax.contourf(x, y[:], matriz[:,:], levels=np.linspace(minn, maxx, 15), cmap = cmap, extend='both', transform=crs.PlateCarree())
    if np.any(y>50): 
        ax.set_yticks([25,50,75])
        ax.set_yticklabels(['25°N', '50°N', '75°N'])
    else: 
        ax.set_yticks([30,40,50])
        ax.set_yticklabels(['30°N', '40°N', '50°N'])
    ax.tick_params(axis='both', labelsize=8)
    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                      linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.ylocator = mticker.FixedLocator([25,50,75])
    # gl.yformatter = mticker.FuncFormatter(lambda x, p: f'{abs(x):.0f}°N' if x >= 0 else f'{abs(x):.0f}°S')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.08, format='%.2f', shrink=0.9, aspect=40)
    cb.set_label(units, fontsize=8, color='k')
    cb.ax.tick_params(labelsize=8)
    plt.ylim(20,70)
    plt.subplots_adjust(
                bottom=0.2,
                top=0.995,
                wspace=0.1,
                hspace=0.3)
    plt.savefig(path, dpi=500)
    plt.close()


def maps(x, y, minn, maxx, matriz,  cmap, path, units='', topography = '', center_lon=180):

    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=center_lon))
    
    if topography == True:
        # ax.add_feature(cartopy.feature.BORDERS, lw=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

    im = ax.contourf(x, y[:], matriz[:,:], levels=np.linspace(minn, maxx, 15), cmap = cmap, extend='both', transform=crs.PlateCarree())
    if np.any(y>50): 
        ax.set_yticks([25,50,75])
        ax.set_yticklabels(['25°N', '50°N', '75°N'])
    else: 
        ax.set_yticks([30,40,50])
        ax.set_yticklabels(['30°N', '40°N', '50°N'])
    ax.tick_params(axis='both', labelsize=8)
    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                      linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.ylocator = mticker.FixedLocator([25,50,75])
    # gl.yformatter = mticker.FuncFormatter(lambda x, p: f'{abs(x):.0f}°N' if x >= 0 else f'{abs(x):.0f}°S')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.08, format='%.2f', shrink=0.9, aspect=40)
    cb.set_label(units, fontsize=8, color='k')
    cb.ax.tick_params(labelsize=8)
    plt.subplots_adjust(
                bottom=0.2,
                top=0.995,
                wspace=0.1,
                hspace=0.3)
    plt.savefig(path, dpi=500)
    plt.close()


def hovmoller_onevar_summer(time_lags, lons, Matriz_t, cmap_t, min_t, max_t, steps, path, var_1 = ''):
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    RdYlBu_list = ['rgb(165,0,38)','rgb(215,48,39)','rgb(244,109,67)','rgb(253,174,97)','rgb(254,224,144)','rgb(255,255,191)','rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
    my_cmap = matplotlib.colors.ListedColormap(RdYlBu_list, name='RdYlBu')

    fig = plt.figure(figsize=[6,8.5])
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.74])
    # ax = fig.add_subplot(1, 1, 1)
    im = ax.contourf(lons, time_lags, Matriz_t[:,:], cmap=cmap_t, extend='both', levels=np.linspace(min_t, max_t, steps))
    plt.xlabel(r'Relative longitude', fontsize=13)
    plt.ylim(152,213+31)
    months = ['Jun', 'Jul', 'Aug']
    month_days = [152, 182, 213]  # Approximate middle of each month
    plt.yticks(month_days, months)
    ax.set_xticks([-180,-90,0,90,180])
    ax.tick_params(labelsize=13)
    ax.axhline(0, ls = '--', color='dimgray', lw = 0.3)
    ax.axvline(0, ls = '--', color='dimgray', lw = 0.3)
    cbaxes = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    # cbaxes = fig.add_axes([0.29, 0.12, 0.5, 0.03])
    #cb = plt.colorbar(cm.ScalarMappable(norm=norm_t, cmap=cmap_t), orientation="horizontal", pad=0.2, cax=cbaxes, format='%.2f')
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, cax=cbaxes, format='%.0f')
    cb.set_label(var_1, fontsize=13, color='k')
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelcolor='k', color='k', labelsize=13)

    plt.savefig(path, dpi=500)
    plt.close()


def hovmoller_onevar_event_20days(time_lags, lons, Matriz_t, cmap_t, min_t, max_t, steps, path, var_1 = ''):
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    RdYlBu_list = ['rgb(165,0,38)','rgb(215,48,39)','rgb(244,109,67)','rgb(253,174,97)','rgb(254,224,144)','rgb(255,255,191)','rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
    my_cmap = matplotlib.colors.ListedColormap(RdYlBu_list, name='RdYlBu')

    fig = plt.figure(figsize=[6,8.5])
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.74])
    # ax = fig.add_subplot(1, 1, 1)
    im = ax.contourf(lons, time_lags, Matriz_t[:,:], cmap=cmap_t, extend='both', levels=np.linspace(min_t, max_t, steps))
    plt.xlabel(r'Relative longitude', fontsize=12)
    ax.set_yticks([-20,-15,-10,-5,0,5,10,15,20])
    ax.set_xticks([-180,-90,0,90,180])
    ax.tick_params(labelsize=13)
    ax.axhline(0, ls = '--', color='dimgray', lw = 0.3)
    ax.axvline(0, ls = '--', color='dimgray', lw = 0.3)
    cbaxes = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    # cbaxes = fig.add_axes([0.29, 0.12, 0.5, 0.03])
    #cb = plt.colorbar(cm.ScalarMappable(norm=norm_t, cmap=cmap_t), orientation="horizontal", pad=0.2, cax=cbaxes, format='%.2f')
    cb = plt.colorbar(im, orientation="horizontal", pad=0.1, cax=cbaxes, format='%.0f')
    cb.set_label(var_1, fontsize=13, color='k')
    cb.outline.set_edgecolor('k')
    cb.ax.tick_params(labelcolor='k', color='k', labelsize=13)

    plt.savefig(path, dpi=500)
    plt.close()


def compute_annual_cycle(data, is_leap, frequency):
    # Group by dayofyear and hour to preserve 6-hourly resolution
    grouped = data.groupby(['time.month', 'time.day'])
    # Compute mean over all years for each day and hour combination
    # annual_cycle = grouped.map(lambda x: x.groupby('time.hour').mean(dim='time'))
    annual_cycle = grouped.mean(dim='time')

    # Stack dayofyear and hour into a single dimension
    annual_cycle = annual_cycle.stack(time=('month', 'day'))
    annual_cycle = annual_cycle.drop_vars(['month', 'day'])  # Dimensions without coordinates: time
    annual_cycle = annual_cycle.dropna(dim='time', how='all')

    # Create a new time coordinate for the annual cycle
    # Use pd.date_range with 6-hour frequency for the entire year
    # (1 for leap year, 0 for non-leap year)
    base_year = 2000 if is_leap == 1 else 2001
    if frequency == 4:
        cycle_times = pd.date_range(start=f'{base_year}-01-01 00:00', end=f'{base_year}-12-31 18:00',freq='6h')
    elif frequency == 1:
        cycle_times = pd.date_range(start=f'{base_year}-01-01 00:00', end=f'{base_year}-12-31',freq='D')

    # Assign the timestamp-based time coordinate to annual_cycle
    # Assuming annual_cycle is an xarray object with an existing time dimension
    annual_cycle = annual_cycle.assign_coords(time=cycle_times)
    
    return annual_cycle.transpose('time', 'lat', 'lon')


def compute_annual_cycle_window(data, is_leap, frequency=1, window_size=31):
    """
    Compute the mean annual cycle using a ±(window_size/2)-day moving window
    across all years, preserving daily or 6-hourly resolution.

    Parameters
    ----------
    data : xarray.DataArray
        Time series (time, lat, lon)
    is_leap : bool or int
        1 for leap-year climatology, 0 otherwise.
    frequency : int
        1 for daily data, 4 for 6-hourly data.
    window_size : int
        Length of moving window in days (default 31).
    """

    # 1️⃣ Compute mean for each calendar day (averaged across all years)
    annual_cycle = data.groupby("time.dayofyear").mean("time")

    # 2️⃣ Apply 31-day running mean along dayofyear (wrap around year edges)
    # pad first and last days for circular smoothing
    if window_size is not None:
        padded = xr.concat([annual_cycle[-window_size//2:], annual_cycle, annual_cycle[:window_size//2]], dim="dayofyear")
        annual_cycle_smooth = padded.rolling(dayofyear=window_size, center=True, min_periods=1).mean().isel(dayofyear=slice(window_size//2, -window_size//2))
    else:
        annual_cycle_smooth = annual_cycle

    # 3️⃣ Reconstruct proper time coordinate
    base_year = 2000 if is_leap else 2001
    if frequency == 4:
        cycle_times = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31 18:00", freq="6h")
    elif frequency == 1:
        cycle_times = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    else:
        raise ValueError("frequency must be 1 (daily) or 4 (6-hourly)")

    # 4️⃣ Assign new time coordinate
    annual_cycle_smooth = annual_cycle_smooth.assign_coords(dayofyear=np.arange(1, len(cycle_times)+1))
    annual_cycle_smooth = annual_cycle_smooth.rename(dayofyear="time")
    annual_cycle_smooth["time"] = cycle_times

    if annual_cycle_smooth.ndim == 3:
        return annual_cycle_smooth.transpose("time", "lat", "lon")
    elif annual_cycle_smooth.ndim == 2:
        return annual_cycle_smooth.transpose("time", "lon")


def compute_monthly_anomalies(data):
    """
    From daily data, compute monthly anomalies:
    mean of each calendar month in each year, minus the long-term
    mean for that calendar month across all years.

    Parameters
    ----------
    data : xarray.DataArray
        Daily time series with a ``time`` coordinate (e.g. time, lat, lon).
    is_leap, frequency :
        Unused; kept for API consistency with neighboring cycle helpers.

    Returns
    -------
    xarray.DataArray
        Monthly anomalies with one value per year-month (same spatial dims).
    """
    # One mean per year-month (e.g. 1980-01, 1980-02, ...)
    monthly = data.resample(time="MS").mean("time")
    # Long-term mean for each calendar month (Jan, Feb, ..., Dec)
    monthly_clim = monthly.groupby("time.month").mean("time")
    # Anomaly = that month's mean minus historical mean for that month
    anomalies = monthly.groupby("time.month") - monthly_clim
    return anomalies





def compute_annual_cycle_freq_nonzero(data, is_leap, frequency):
    """
    For each calendar day (month,day), compute:
      (# of timesteps with data != 0) / (total # of timesteps considered)
    across all years in `data`.

    Returns an annual cycle with a synthetic time coordinate (base_year 2000/2001).
    """

    # Boolean mask: 1 where non-zero, 0 where zero (NaNs stay NaN so they don't count)
    nz = xr.where(data != 0, 1.0, 0.0)

    # Group by calendar day
    grouped_nz = nz.groupby(["time.month", "time.day"])

    # Count non-zero and total (excluding NaNs)
    nz_count = grouped_nz.sum(dim="time", skipna=True)
    total_count = grouped_nz.count(dim="time")  # counts non-NaN samples

    # Fraction of non-zero values
    annual_cycle = nz_count / total_count

    # Stack month/day into a single time dimension
    annual_cycle = annual_cycle.stack(time=("month", "day"))
    annual_cycle = annual_cycle.drop_vars(["month", "day"])
    annual_cycle = annual_cycle.dropna(dim="time", how="all")

    # Build synthetic time coordinate for one year
    base_year = 2000 if is_leap == 1 else 2001
    if frequency == 4:  # 6-hourly data
        cycle_times = pd.date_range(
            start=f"{base_year}-01-01 00:00",
            end=f"{base_year}-12-31 18:00",
            freq="6h",
        )
    elif frequency == 1:  # daily data
        cycle_times = pd.date_range(
            start=f"{base_year}-01-01",
            end=f"{base_year}-12-31",
            freq="D",
        )
    else:
        raise ValueError("frequency must be 4 (6-hourly) or 1 (daily)")

    annual_cycle = annual_cycle.assign_coords(time=cycle_times)

    return annual_cycle.transpose("time", "lat", "lon")




def filter_annual_cycle(data, freq=4, n_freq=4):
    """
    Compute the annual cycle from 6-hourly data using FFT, retaining frequencies 0-4 yr^-1.
    
    Parameters:
        data (xarray.DataArray): Input data with dimensions (time, latitude, longitude).
                                Time coordinate should be in 6-hourly intervals.
        freq: data frequency
        n_freq (int): Maximum frequency to retain (default: 4 yr^-1).
    
    Returns:
        xarray.DataArray: Annual cycle with same dimensions as input.
    """
    n_points = len(data['time'])
    n_days = n_points / freq
    
    # Compute FFT and normalize
    fourier = fft(data.values, axis=0) / n_points
    
    # Frequency in cycles thoughout all years (convert from cycles/day)
    freq_daily = fftfreq(n_points, d=1/freq)  # d=1/4 day (6-hourly sampling)
    freq_yearly = freq_daily * n_days  # Convert to cycles/all years
    
    # Filter frequencies (0-4 yr^-1, including negative frequencies for symmetry)
    mask = np.abs(freq_yearly) <= n_freq
    fourier[~mask] = 0
    
    # Inverse FFT and retain real part
    annual_cycle = ifft(fourier * n_points, axis=0).real
    
    # Wrap output in xarray with original coordinates
    return xr.DataArray(
        annual_cycle,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    )



def expand_annual_cycle(v, annual_cycle):
    """
    Extend annual cycle to match original data based on month, day, and hour.
    For ERA5 data 
    Parameters:
        v (xarray.DataArray): Input data with dims (time, lat, lon)
        annual_cycle (xarray.DataArray): Annual cycle with dims (cycle_time, lat, lon),
                                        where cycle_time is a 1464-step leap-year cycle.
    
    Returns:
        xarray.DataArray: v1 with the same dims as v, containing annual cycle values.
    """
    # Get v's time coordinate
    v_times = pd.to_datetime(v['time'].values)
    
    # Create a month-day-hour string for v
    v_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(v['time'].values)]
    
    # Create month-day-hour string for annual_cycle
    cycle_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(annual_cycle['time'].values)]
    
    # Map v's month-day-hour to annual_cycle's cycle_time
    annual_cycle_expanded = xr.full_like(v, np.nan)
    for i, mdh in enumerate(v_month_day_hour):
            idx = cycle_month_day_hour.index(mdh)
            annual_cycle_expanded[i] = annual_cycle.isel(time=idx)
    
    return annual_cycle_expanded



def expand_annual_cycle_optimized(time_series, n_lats, n_lons, annual_cycle):
    """
    Extend annual cycle to match original data based on month, day, and hour.
    For ERA5 data 
    Parameters:
        v (xarray.DataArray): Input data with dims (time, lat, lon)
        annual_cycle (xarray.DataArray): Annual cycle with dims (cycle_time, lat, lon),
                                        where cycle_time is a 1464-step leap-year cycle.
    
    Returns:
        xarray.DataArray: v1 with the same dims as v, containing annual cycle values.
    """
    # Get v's time coordinate
    v_times = pd.to_datetime(time_series)
    
    # Create a month-day-hour string for v
    v_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(time_series)]
    
    # Create month-day-hour string for annual_cycle
    cycle_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(annual_cycle['time'].values)]
    
    # Map v's month-day-hour to annual_cycle's cycle_time
    annual_cycle_expanded = xr.DataArray(np.zeros((len(v_month_day_hour), n_lats, n_lons)), dims=("time", "lat", "lon"))
    for i, mdh in enumerate(v_month_day_hour):
            idx = cycle_month_day_hour.index(mdh)
            annual_cycle_expanded[i, :, :] = annual_cycle.isel(time=idx)
    
    return annual_cycle_expanded


def lag_array(arr, lag):
    out = np.empty_like(arr)
    out[:] = np.nan
    if lag > 0:
        out[lag:] = arr[:-lag]
    elif lag < 0:
        out[:lag] = arr[-lag:]
    else:
        out = arr.copy()
    return out



def rolling_zonalization(pv_da, window_deg=60):
    # Compute grid spacing
    lon = pv_da['lon'].values
    dlon = np.abs(lon[1] - lon[0])   # degrees per grid cell

    # Convert window size from degrees → number of grid cells
    window_cells = int(np.round(window_deg / dlon))
    if window_cells < 1:
        raise ValueError("Window too small for grid resolution.")

    half = window_cells // 2

    # --- Cyclic padding in longitude ---
    pv_ext = xr.concat(
        [pv_da.isel(lon=slice(-half, None)),
         pv_da,
         pv_da.isel(lon=slice(0, half))],
        dim="lon"
    )

    # --- Rolling mean in grid cells ---
    q_b_ext = pv_ext.rolling(lon=window_cells, center=True, min_periods=1).mean()

    # --- Remove padding ---
    q_b = q_b_ext.isel(lon=slice(half, -half))

    return q_b


def meridional_gradient_qgpv(qgpv_zonalized):
    """
    Compute the meridional gradient d(q_b)/dy of zonalized QGPV.
    This avoids log(PV) because QGPV can be negative or small.

    Parameters
    ----------
    qgpv_zonalized : xarray.DataArray
        Zonalized QGPV with dimensions (time, lat, lon).

    Returns
    -------
    dq_dy : xarray.DataArray
        Meridional gradient of zonalized QGPV in units of s^-1 m^-1.
    """
    R = 6371000.0  # Earth radius (m)

    # 1. Latitude in radians
    lat_rad = np.deg2rad(qgpv_zonalized['lat'])

    # 2. Compute Δφ (spacing in radians)
    dphi = np.gradient(lat_rad)

    # 3. dy = R * dφ  (grid spacing in meters)
    dy = xr.DataArray(
        R * dphi,
        dims=['lat'],
        coords={'lat': qgpv_zonalized['lat']}
    )

    # 4. Derivative with respect to latitude coordinate (radians)
    dq_dphi = qgpv_zonalized.differentiate('lat')

    # 5. Convert derivative to per meter
    dq_dy = dq_dphi / dy

    return dq_dy



def truncate_zonal_wavenumbers(data, kmax=4):
    """
    Truncate zonal wavenumbers beyond kmax (keep only k = 0..kmax and their negative counterparts).
    
    Parameters
    ----------
    data : xarray.DataArray
        3D array with dims (time, lat, lon)
    kmax : int
        Largest wavenumber to retain (default: 4)
    
    Returns
    -------
    xarray.DataArray
        DataArray with only planetary-scale wavenumbers (|k| <= kmax)
    """
    # Number of longitude points
    n_lons = data.sizes["lon"]
    
    # 1) FFT along longitude
    fourier = fft(data.values, axis=-1)

    # 2) Build wavenumber array: 0,1,2,...,-2,-1
    k = np.fft.fftfreq(n_lons) * n_lons   # integer wavenumbers
    
    # 3) Mask: keep |k| <= kmax
    mask = (np.abs(k) <= kmax).astype(float)   # shape (n_lons,)
    
    # Reshape mask for broadcasting onto (time,lat,lon)
    mask = mask.reshape(1, 1, n_lons)

    # 4) Apply filter
    fourier_filtered = fourier * mask

    # 5) Inverse FFT → real part only
    filtered = np.fft.ifft(fourier_filtered, axis=-1).real

    # 6) Return as xarray
    return xr.DataArray(
        filtered,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs,
    ).rename(f"{data.name}_kle{kmax}")






def calculate_frequency_composites(pos_HW, matriz, matrix_threshold, test_excedence):
    import numpy as np
    from scipy import stats
    dic_composites = {}

    matriz_extremes = np.copy(matriz)
    for i in range(matriz_extremes.shape[0]):
        
        if test_excedence==True:
            matriz_extremes_i = np.nan_to_num(matriz_extremes[i], nan=-100)
            matriz_extremes_i = np.where(matriz_extremes_i >= matrix_threshold, 1, 0)
        else:
            print('test_excedence is False')
            matriz_extremes_i = np.nan_to_num(matriz_extremes[i], nan=100)
            matriz_extremes_i = np.where(matriz_extremes_i <= matrix_threshold, 1, 0)
        matriz_extremes[i] = matriz_extremes_i

    time_lags = np.arange(-20, 21, 1)
    for pos in pos_HW.index:
        if pos == 0: 
            dic_composites[0] = []
            dic_composites[0].append(int(pos_HW.iloc[0][0]))
            continue
        #elif pos_HW.iloc[pos][0] >= 27270: break  # len(other variables)
        elif pos_HW.iloc[pos][0] == matriz.shape[0]-20: break
        elif pos == pos_HW.index[-1]:
            break
        elif pos_HW.iloc[pos][0]-1 != pos_HW.iloc[pos-1][0]:
            # print(pos_HW.iloc[pos][0])
            for num in time_lags:  
                if num in dic_composites.keys(): dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))  
                else: 
                    dic_composites[num] = []
                    dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.sum(matriz_extremes[dic_composites[lag]] > 0, axis=0) / np.sum(~np.isnan(matriz[dic_composites[lag]]), axis=0) 

    return composites_matrix_complete 


from matplotlib.patches import Ellipse
def add_covariance_ellipse(ax, x, y, color, label, confidence=0.95):

    # === Remove NaNs ===
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # === Mean and covariance ===
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # === Eigen-decomposition ===
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Avoid degenerate cases
    eigvals = np.maximum(eigvals, 1e-12)

    # Sort by eigenvalue size
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # === Confidence scaling ===
    if confidence == 0.95:
        chi2_val = 5.99
    elif confidence == 0.68:
        chi2_val = 2.30
    elif confidence == 0.99:
        chi2_val = 9.21
    else:
        raise ValueError("Use confidence = 0.68, 0.95, or 0.99")

    scale = np.sqrt(chi2_val)

    width  = 2 * scale * np.sqrt(eigvals[0])
    height = 2 * scale * np.sqrt(eigvals[1])

    # Orientation in degrees
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # === Draw ellipse ===
    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=2,
        linestyle='--',
        label=f"{label} ({int(confidence*100)}%)"
    )

    ax.add_patch(ellipse)

def expand_event_dates(start_dates, durations, flat, flon, labels):
    rows = []

    for start, dur, flat_i, flon_i, label_i in zip(
        start_dates, durations, flat, flon, labels
    ):
        mean_lat = np.nanmean(flat_i)
        mean_lon = np.nanmean(flon_i)

        for d in range(int(dur)):
            rows.append(
                (
                    start + np.timedelta64(d, "D"),
                    label_i,
                    mean_lat,
                    mean_lon,
                )
            )

    return pd.DataFrame(
        rows,
        columns=["date", "event_id", "lat", "lon"]
    )


def expand_event_dates_v2(start_dates, durations, flat, flon, labels):
    """
    Expand blocking events into daily blocking coordinates.

    Returns a dataframe with one row per blocking day:
    date | event_id | lat | lon
    """

    rows = []

    for start, dur, flat_i, flon_i, label_i in zip(
        start_dates, durations, flat, flon, labels
    ):

        for d in range(int(dur)):

            lat_d = flat_i[d]
            lon_d = flon_i[d]

            # skip missing days
            if np.isnan(lat_d) or np.isnan(lon_d):
                continue

            rows.append(
                (
                    start + np.timedelta64(d, "D"),
                    label_i,
                    lat_d,
                    lon_d,
                )
            )

    return pd.DataFrame(
        rows,
        columns=["date", "event_id", "lat", "lon"]
    )



import ast
import re
def normalize_coords(x):
    """
    Convert strings like:
      '[[ 65.625 229.5 ]]'
      '[[65.6 229.5]\n [66.1 231.2]]'
    into array shape (N, 2)
    """

    # extract all floats from the string
    nums = np.array(
        re.findall(r"[-+]?\d*\.\d+|\d+", x),
        dtype=float
    )

    # reshape into pairs (lat, lon)
    coords = nums.reshape(-1, 2)

    return coords








#########################################################
#########################################################
def compute_annual_cycle(data, is_leap, frequency=1):
    # Group by dayofyear and hour to preserve 6-hourly resolution
    grouped = data.groupby(['time.month', 'time.day'])
    # Compute mean over all years for each day and hour combination
    # annual_cycle = grouped.map(lambda x: x.groupby('time.hour').mean(dim='time'))
    annual_cycle = grouped.mean(dim='time')

    # Stack dayofyear and hour into a single dimension
    annual_cycle = annual_cycle.stack(time=('month', 'day'))
    annual_cycle = annual_cycle.drop_vars(['month', 'day'])  # Dimensions without coordinates: time
    annual_cycle = annual_cycle.dropna(dim='time', how='all')

    # Create a new time coordinate for the annual cycle
    # Use pd.date_range with 6-hour frequency for the entire year
    # (1 for leap year, 0 for non-leap year)
    base_year = 2000 if is_leap == 1 else 2001

    cycle_times = pd.date_range(start=f'{base_year}-01-01 00:00', end=f'{base_year}-12-31',freq='D')

    # Assign the timestamp-based time coordinate to annual_cycle
    # Assuming annual_cycle is an xarray object with an existing time dimension
    annual_cycle = annual_cycle.assign_coords(time=cycle_times)
    
    return annual_cycle.transpose('time', 'lat', 'lon')


def compute_annual_cycle_6h(data, is_leap):
    # Group by dayofyear and hour to preserve 6-hourly resolution
    grouped = data.groupby('time.dayofyear')
    # Compute mean over all years for each day and hour combination
    annual_cycle = grouped.map(lambda x: x.groupby('time.hour').mean(dim='time'))

    # Stack dayofyear and hour into a single dimension
    annual_cycle = annual_cycle.stack(time=('dayofyear', 'hour'))
    annual_cycle = annual_cycle.drop_vars(['dayofyear', 'hour'])  # Dimensions without coordinates: time
   
    # Create a new time coordinate for the annual cycle
    # Use pd.date_range with 6-hour frequency for the entire year
    # (1 for leap year, 0 for non-leap year)
    base_year = 2000 if is_leap == 1 else 2001
    cycle_times = pd.date_range(start=f'{base_year}-01-01 00:00', end=f'{base_year}-12-31 18:00',freq='6h')

    # Assign the timestamp-based time coordinate to annual_cycle
    # Assuming annual_cycle is an xarray object with an existing time dimension
    annual_cycle = annual_cycle.assign_coords(time=cycle_times)
    
    return annual_cycle.transpose('time', 'lat', 'lon')

def filter_annual_cycle(data, freq=4, n_freq=4):
    """
    Compute the annual cycle from 6-hourly data using FFT, retaining frequencies 0-4 yr^-1.
    
    Parameters:
        data (xarray.DataArray): Input data with dimensions (time, latitude, longitude).
                                Time coordinate should be in 6-hourly intervals.
        freq: data frequency
        n_freq (int): Maximum frequency to retain (default: 4 yr^-1).
    
    Returns:
        xarray.DataArray: Annual cycle with same dimensions as input.
    """
    n_points = len(data['time'])
    n_days = n_points / freq
    
    # Compute FFT and normalize
    fourier = fft(data.values, axis=0) / n_points
    
    # Frequency in cycles thoughout all years (convert from cycles/day)
    freq_daily = fftfreq(n_points, d=1/freq)  # d=1/4 day (6-hourly sampling)
    freq_yearly = freq_daily * n_days  # Convert to cycles/all years
    
    # Filter frequencies (0-4 yr^-1, including negative frequencies for symmetry)
    mask = np.abs(freq_yearly) <= n_freq
    fourier[~mask] = 0
    
    # Inverse FFT and retain real part
    annual_cycle = ifft(fourier * n_points, axis=0).real
    
    # Wrap output in xarray with original coordinates
    return xr.DataArray(
        annual_cycle,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    )



def expand_annual_cycle(v, annual_cycle):
    """
    Extend annual cycle to match original data based on month, day, and hour.
    For ERA5 data 
    Parameters:
        v (xarray.DataArray): Input data with dims (time, lat, lon)
        annual_cycle (xarray.DataArray): Annual cycle with dims (cycle_time, lat, lon),
                                        where cycle_time is a 1464-step leap-year cycle.
    
    Returns:
        xarray.DataArray: v1 with the same dims as v, containing annual cycle values.
    """
    # Get v's time coordinate
    v_times = pd.to_datetime(v['time'].values)
    
    # Create a month-day-hour string for v
    v_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(v['time'].values)]
    
    # Create month-day-hour string for annual_cycle
    cycle_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(annual_cycle['time'].values)]
    
    # Map v's month-day-hour to annual_cycle's cycle_time
    annual_cycle_expanded = xr.full_like(v, np.nan)
    for i, mdh in enumerate(v_month_day_hour):
            idx = cycle_month_day_hour.index(mdh)
            annual_cycle_expanded[i] = annual_cycle.isel(time=idx)
    
    return annual_cycle_expanded

def filter_zonal_wavenumbers_3D(data, k1=4, k2=15):
    """
    Vectorized zonal Fourier filter for 3D data (time, lat, lon) to isolate wavenumbers k1–k2.
    
    Parameters:
    -----------
    data : xarray.DataArray
        3D array with dimensions (time, lat, lon)
    k1 : int
        Smallest wavenumber to keep (default: 4)
    k2 : int
        Largest wavenumber to keep (default: 15)
    
    Returns:
    --------
    xarray.DataArray
        Filtered data with only wavenumbers k1–k2
    """
    # Get longitude spacing (assumes uniform grid)
    lon_spacing = np.abs(data.lon.diff('lon')[0].item())  
    n_lons = len(data.lon)
    
    # Compute wavenumber mask (vectorized)
    freq = fftfreq(n_lons, d=lon_spacing)
    wavenumbers = np.abs(freq) * n_lons * lon_spacing  # 360 =  n_lons * lon_spacing. Convert to integer wavenumbers
    wavenumbers = np.round(wavenumbers).astype(int)
    if k1 is not None and k2 is not None:
        mask = (wavenumbers >= k1) & (wavenumbers <= k2)  # Keep only target wavenumbers
    elif k1 is not None and k2 is None:
        mask = wavenumbers >= k1
    elif k1 is None and k2 is not None:
        mask = wavenumbers <= k2


    # Vectorized FFT along longitude
    fourier = fft(data.values, axis=-1) / n_lons
    
    # Apply filter to all dimensions simultaneously
    filtered_fourier = fourier * mask[np.newaxis, np.newaxis, :]
    
    # Inverse FFT (real part only)
    filtered_data = ifft(filtered_fourier * n_lons, axis=-1).real
    
    # Return as xarray with preserved metadata
    return xr.DataArray(
        filtered_data,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    ).rename('v_wnf')

def filter_zonal_wavenumbers_3D_optimized(data, lons, k1=4, k2=15):
    """
    Vectorized zonal Fourier filter for 3D data (time, lat, lon) to isolate wavenumbers k1–k2.
    
    Parameters:
    -----------
    data : xarray.DataArray
        3D array with dimensions (time, lat, lon)
    k1 : int
        Smallest wavenumber to keep (default: 4)
    k2 : int
        Largest wavenumber to keep (default: 15)
    
    Returns:
    --------
    xarray.DataArray
        Filtered data with only wavenumbers k1–k2
    """
    # Get longitude spacing (assumes uniform grid)
    lon_spacing = np.abs(lons.diff('lon')[0].item())  
    n_lons = len(lons)
    
    # Compute wavenumber mask (vectorized)
    freq = fftfreq(n_lons, d=lon_spacing)
    wavenumbers = np.abs(freq) * n_lons * lon_spacing  # 360 =  n_lons * lon_spacing. Convert to integer wavenumbers
    wavenumbers = np.round(wavenumbers).astype(int)
    if k1 is not None and k2 is not None:
        mask = (wavenumbers >= k1) & (wavenumbers <= k2)  # Keep only target wavenumbers
    elif k1 is not None and k2 is None:
        mask = wavenumbers >= k1
    elif k1 is None and k2 is not None:
        mask = wavenumbers <= k2


    # Vectorized FFT along longitude
    fourier = np.fft.fft(data, axis=-1) / n_lons
    
    # Apply filter to all dimensions simultaneously
    filtered_fourier = fourier * mask[np.newaxis, np.newaxis, :]
    
    # Inverse FFT (real part only)
    filtered_data = np.fft.ifft(filtered_fourier * n_lons, axis=-1).real
    
    # Return as xarray with preserved metadata
    return filtered_data





def synoptic_highpass_filter(data, dt_hours=6, low_cut_days=2, high_cut_days=8):
    """
    Apply a synoptic high-pass filter (e.g., 2-10 day) along the time axis using FFT.
    
    Parameters:
        data (xarray.DataArray): Input 3D data (time, lat, lon)
        dt_hours (float): Time step in hours (default 6h)
        low_cut_days (float): Lower period bound in days (default 2)
        high_cut_days (float): Upper period bound in days (default 10)
        
    Returns:
        xarray.DataArray: Filtered data
    """
    n_time = data.sizes['time']
    dt = dt_hours * 3600  # convert hours to seconds (not strictly needed for FFT, just for clarity)
    
    # FFT along time
    fft_vals = fft(data.values, axis=0)
    
    # Frequencies in cycles per time series
    freq = fftfreq(n_time, d=dt_hours / 24)  # in cycles/day
    
    # Convert period bounds (days) to frequency
    low_cut = 1 / high_cut_days   # high-pass: remove periods > high_cut_days
    high_cut = 1 / low_cut_days   # low-pass: remove periods < low_cut_days
    
    # Build mask: keep only 1/10–1/2 day^-1 (cycles/day)
    mask = (np.abs(freq) >= low_cut) & (np.abs(freq) <= high_cut)
    
    # Apply mask
    fft_vals[~mask, :, :] = 0
    
    # Inverse FFT
    filtered = ifft(fft_vals, axis=0).real
    
    return xr.DataArray(
        filtered,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    ).rename(f'{data.name}_synoptic_highpass')


def compute_analytic_signal(data):
    """
    Compute the analytic signal, local amplitude, and local phase of meridional wind anomaly.
    Following Fragkoulidis and Wirth (2020)

    Parameters:
        data (xarray.DataArray): Input data with dims (time, lat, lon)
        
    Returns:
        tuple: (analytic_signal, amplitude, phase)
            - analytic_signal: Complex analytic signal
            - amplitude: Local amplitude (envelope) E_ell
            - phase: Local phase Phi_v_ell
    """
    # Number of longitude points (L must be even)
    L = data['lon'].size
    if L % 2 != 0:
        raise ValueError("Number of longitude points (L) must be even.")
    
    # Perform discrete Fourier transform along longitude dimension
    # Following Fragkoulidis and Wirth (2020) Eq. (4)
    # Note: The $ 1/L $ factor is applied automatically, as per the code’s structure, though Equation (3) includes it in the DFT definition. This is a minor inconsistency but doesn’t affect the result due to normalization.
    v_tilde = xr.apply_ufunc(
        np.fft.fft, data, input_core_dims=[['lon']], output_core_dims=[['freq']],
        kwargs={'n': L, 'axis': -1}
    )

    # Compute coefficients A_m based on frequency spectrum
    m = np.arange(L)  # Spatial frequencies (wavenumbers) 0 to L-1

    # Following Fragkoulidis and Wirth (2020) Eq. (3)
    # Initialize A_m with v_tilde
    A_m = v_tilde.copy()

    # Apply scaling directly based on m
    A_m = A_m.where((m == 0) | (L // 2), A_m)  # m = 0 or m = L // 2: keep v_tilde
    A_m = A_m.where((m >= 1) & (m <= L // 2 - 1), 2 * A_m)  # 1 to L/2 - 1: double v_tilde
    A_m = A_m.where((m > L // 2) | (m == L), 0)  # L/2 + 1 to L-1: set to 0
    
    '''
    A_m = v_tilde * xr.where(
        (m == 0) | (m == L // 2),  # m = 0 or m = L/2
        1,                         # Keep v_tilde unchanged
        xr.where(
            (1 <= m) & (m <= L // 2 - 1),  # 1 to L/2 - 1
            2,                             # Double v_tilde
            0                              # Zero for other m
        )
    )
    '''

    # Compute analytic signal using inverse DFT
    # Following Fragkoulidis and Wirth (2020) Eq. (2)
    analytic_signal = xr.apply_ufunc(
        np.fft.ifft, A_m, input_core_dims=[['freq']], output_core_dims=[['lon']],
        kwargs={'n': L, 'axis': -1}
    ).astype(complex)


    # Convert to DataArray with original coordinates
    analytic_signal = xr.DataArray(
        analytic_signal,
        dims=data.dims,
        coords=data.coords
    )

    # Extract real and imaginary parts
    real_part = analytic_signal.real
    imag_part = analytic_signal.imag

    # Compute local amplitude (envelope) E_ell
    amplitude = xr.DataArray(
        np.sqrt(real_part**2 + imag_part**2),
        dims=data.dims,
        coords=data.coords
    )

    # Compute local phase Phi_v_ell
    phase = xr.DataArray(
        -1 * np.arctan2(imag_part, real_part),  # why needs -1?
        dims=data.dims,
        coords=data.coords
    )

    return analytic_signal, amplitude, phase


def compute_hilbert_transform(data):
    """
    Compute the Hilbert transform of the data (v') to obtain the analytic signal.
    
    Parameters:
    - data (xarray.DataArray): Input data with dimensions (time, lat, lon).
    
    Returns:
    - analytic_signal (xarray.DataArray): Complex analytic signal of v'.
    - phase (xarray.DataArray): Instantaneous phase of v'.
    - amplitude (xarray.DataArray): Instantaneous amplitude of v'.
    """
    # Apply Hilbert transform along the time dimension
    analytic_signal = xr.apply_ufunc(
        hilbert,  # Function to apply
        data,  # Input data
        input_core_dims=[['lon']],  # Dimension to transform
        output_core_dims=[['lon']],  # Output has same dims
        vectorize=True,  # Vectorize over other dims
        dask='allowed'  # Allow dask if v_prime is a dask array
    )
   
    # Compute instantaneous amplitude (magnitude of analytic signal)
    amplitude = xr.apply_ufunc(
        np.abs,  # Magnitude function
        analytic_signal,
        input_core_dims=[['lon']],
        output_core_dims=[['lon']],
        vectorize=True
    )

    # Compute instantaneous phase
    phase = xr.apply_ufunc(
        np.angle,
        analytic_signal,
        input_core_dims=[['lon']],
        output_core_dims=[['lon']],
        vectorize=True
    )
    
    return analytic_signal, amplitude, phase






#########################################################
#########################################################
def compute_phase_speed(phase_da, amplitude_da, dt, dx, earth_radius=6371000.0, E0_threshold=15.0):
    """
    Compute the Rossby Wave Packet (RWP) phase speed based on local phase derivatives,
    considering cyclic longitude and phase wrapping within (-pi, pi], with boundary handling.
    Following Fragkoulidis and Wirth (2020)
    
    Parameters:
        phase_da (xarray.DataArray): Local phase (Phi_v_ell) with dims (time, lat, lon) in radians
        amplitude_da (xarray.DataArray): Local amplitude (E_ell) with dims (time, lat, lon) in m/s
        earth_radius (float, optional): Earth's radius in meters. Default: 6371000.0 m
        E0_threshold (float, optional): Amplitude threshold in m/s. Default: 15.0 m/s for reanalysis data
    
    Returns:
        xarray.DataArray: Phase speed (c_p) in m/s, with nan where amplitude threshold is not met
    """
    # Convert latitude to radians
    lat_rad = np.deg2rad(phase_da['lat'].values)
    print(lat_rad)
    # Define delta values
    delta_t = dt / 24.0 * 86400 # hours --> seconds
    delta_x = np.deg2rad(dx)  # degrees
    
    print(delta_t)
    print(delta_x)

    # Step 1: Initialize phase_dt array with NaN
    phase_dt = xr.full_like(phase_da, np.nan)

    # Step 1: Extract data array and initialize phase_dt with NaN
    phase_data = phase_da.values  # Shape: (time, lat, lon)
    n_time, n_lat, n_lon = phase_data.shape
    phase_dt = np.full(phase_data.shape, np.nan)

    print(phase_data[100,:,:])

    # Step 2: Fill boundary values first
    # Forward difference at first time step (t_0)
    phase_t0 = phase_data[0, :, :]  # First time step
    phase_t1 = phase_data[1, :, :]  # Second time step
    diff_t0 = phase_t1 - phase_t0
    phase_dt[0, :, :] = np.where(diff_t0 > np.pi,
                                 (phase_t0 + 2 * np.pi - phase_t1) / delta_t,
                                 np.where(diff_t0 < -np.pi,
                                          (phase_t0 - 2 * np.pi - phase_t1) / delta_t,
                                          (phase_t0 - phase_t1) / delta_t))

    # Backward difference at last time step (t_{n-1})
    phase_tn_1 = phase_data[-1, :, :]  # Last time step
    phase_tn_2 = phase_data[-2, :, :]  # Second-to-last time step
    diff_tn = phase_tn_2 - phase_tn_1
    phase_dt[-1, :, :] = np.where(diff_tn > np.pi,
                                  (phase_tn_2 + 2 * np.pi - phase_tn_1) / delta_t,
                                  np.where(diff_tn < -np.pi,
                                           (phase_tn_2 - 2 * np.pi - phase_tn_1) / delta_t,
                                           (phase_tn_2 - phase_tn_1) / delta_t))

    # Step 3: Fill interior points second (centered difference)
    phase_t_minus = phase_data[0:n_time-2, :, :]  # t - delta_t (all but last 2)
    phase_t_plus = phase_data[2:n_time, :, :]     # t + delta_t (all but first 2)
    diff_t = phase_t_plus - phase_t_minus
    phase_dt[1:n_time-1, :, :] = np.where(diff_t > np.pi,
                                          (phase_t_minus + 2 * np.pi - phase_t_plus) / (2 * delta_t),
                                          np.where(diff_t < -np.pi,
                                                   (phase_t_minus - 2 * np.pi - phase_t_plus) / (2 * delta_t),
                                                   (phase_t_minus - phase_t_plus) / (2 * delta_t)))

    # Convert to xarray.DataArray
    phase_dt = xr.DataArray(phase_dt, dims=phase_da.dims, coords=phase_da.coords)
    print('phase_dt:', phase_dt)

    # Longitude derivative with phase wrapping and cyclic boundary
    # Create extended phase_data with last lon before first and first lon after last
    extended_phase_data = np.zeros((n_time, n_lat, n_lon + 2))
    extended_phase_data[:, :, 1:n_lon + 1] = phase_data  # Copy original data to middle
    extended_phase_data[:, :, 0] = phase_data[:, :, -1]  # Add last lon at start
    extended_phase_data[:, :, -1] = phase_data[:, :, 0]  # Add first lon at end

    # Centered difference for interior points (indices 1 to n_lon)
    phase_x_minus = extended_phase_data[:, :, :-2]  # Previous lon
    phase_x_plus = extended_phase_data[:, :, 2:]   # Next lon
    diff_x = phase_x_minus - phase_x_plus
    phase_dlambda = np.where(diff_x > np.pi,
                             (phase_x_plus + 2 * np.pi - phase_x_minus) / (2 * delta_x),
                             np.where(diff_x < -np.pi,
                                      (phase_x_plus - 2 * np.pi - phase_x_minus) / (2 * delta_x),
                                      (phase_x_plus - phase_x_minus) / (2 * delta_x)))

    # Convert to xarray.DataArray with original lon dimension
    phase_dlambda = xr.DataArray(phase_dlambda, dims=phase_da.dims, coords=phase_da.coords)
    print('phase_dlambda:', phase_dlambda)

    
    # Adjust wavenumber with Earth's radius and cosine of latitude
    cos_lat = xr.DataArray(
        np.cos(lat_rad),
        dims=['lat'],
        coords={'lat': phase_da['lat']}
    ).broadcast_like(phase_da)
    k_vell = (1 / (earth_radius * cos_lat)) * phase_dlambda  # Wavenumber (m^-1)

    # Compute phase speed
    phase_speed = phase_dt / k_vell  # Phase speed (m/s)

    # Apply amplitude threshold with cyclic longitude using direct indexing
    amplitude_data = amplitude_da.values  # Shape: (time, lat, lon)

    # Create extended amplitude_data for cyclic longitude
    extended_amplitude_data = np.zeros((n_time, n_lat, n_lon + 2))
    extended_amplitude_data[:, :, 1:n_lon + 1] = amplitude_data  # Copy original data
    extended_amplitude_data[:, :, 0] = amplitude_data[:, :, -1]  # Last lon at start
    extended_amplitude_data[:, :, -1] = amplitude_data[:, :, 0]  # First lon at end

    # Create extended amplitude_data for time (pad with first and last values)
    extended_amplitude_data_time = np.zeros((n_time + 2, n_lat, n_lon))
    extended_amplitude_data_time[1:n_time + 1, :, :] = amplitude_data  # Copy original data
    extended_amplitude_data_time[0, :, :] = amplitude_data[0, :, :]  # Pad with first time step
    extended_amplitude_data_time[-1, :, :] = amplitude_data[-1, :, :]  # Pad with last time step

    # Create mask using direct indexing with extended arrays
    mask = (amplitude_data >= E0_threshold) & \
           (extended_amplitude_data_time[2:n_time + 2, :, :] >= E0_threshold) & \
           (extended_amplitude_data_time[0:n_time, :, :] >= E0_threshold) & \
           (extended_amplitude_data[:, :, 0:n_lon] >= E0_threshold) & \
           (extended_amplitude_data[:, :, 2:n_lon + 2] >= E0_threshold)

    phase_speed = phase_speed.where(mask, np.nan)  # Apply mask

    # Additional condition: Set |phase_speed| > 20 m/s to NaN
    #phase_speed = phase_speed.where(abs(phase_speed) <= 20, np.nan)

    # Return as xarray.DataArray with original coordinates
    return xr.DataArray(
        phase_speed,
        dims=phase_da.dims,
        coords=phase_da.coords
    )



from scipy import ndimage
def interpolate_matrix_with_nan(matrix, lats_matrix, lons_matrix, lats_new, lons_new):
    
    def fill_nans_nearest(mat):
        nan_mask = np.isnan(mat)
        # nearest-neighbour fill
        idx = ndimage.distance_transform_edt(
            nan_mask, return_distances=False, return_indices=True
        )
        filled = mat[tuple(idx)]
        return filled, nan_mask

    # --- 3D case: (time, lat, lon) ---
    if matrix.ndim == 3:
        nt, ny, nx = matrix.shape
        out = np.empty((nt, len(lats_new), len(lons_new))) * np.nan
        
        for t in range(nt):
            mat_t = matrix[t, :, :]

            # fill nans
            filled, nan_mask = fill_nans_nearest(mat_t)

            # spline
            r = RectBivariateSpline(lats_matrix, lons_matrix, filled)
            inter = r(lats_new, lons_new)

            # re-mask: nearest-neighbour mask interpolation
            r_mask = RectBivariateSpline(lats_matrix, lons_matrix, nan_mask.astype(float))
            mask_new = r_mask(lats_new, lons_new) > 0.5
            inter[mask_new] = np.nan

            out[t, :, :] = inter

        return out

    # --- 2D case: (lat, lon) ---
    elif matrix.ndim == 2:
        filled, nan_mask = fill_nans_nearest(matrix)

        r = RectBivariateSpline(lats_matrix, lons_matrix, filled)
        inter = r(lats_new, lons_new)

        r_mask = RectBivariateSpline(lats_matrix, lons_matrix, nan_mask.astype(float))
        mask_new = r_mask(lats_new, lons_new) > 0.5
        inter[mask_new] = np.nan

        return inter

    else:
        raise ValueError("matrix must be 2D or 3D")



# def interpolate_matrix_with_nan_optimized(ds, variable_matrix, nt, lats_matrix, lons_matrix, lats_new, lons_new):
    
#     def fill_nans_nearest(mat):
#         nan_mask = np.isnan(mat)
#         # nearest-neighbour fill
#         idx = ndimage.distance_transform_edt(
#             nan_mask, return_distances=False, return_indices=True
#         )
#         filled = mat[tuple(idx)]
#         return filled, nan_mask

#     # --- 3D case: (time, lat, lon) ---
#     if nt >= 2:
#         ny, nx = lats_matrix.shape, lons_matrix.shape
#         out = np.empty((nt, len(lats_new), len(lons_new))) * np.nan

#         for t in range(nt):
            
#             mat_t = ds[variable_matrix][t, :, :]

#             # fill nans
#             filled, nan_mask = fill_nans_nearest(mat_t.values)

#             # spline
#             r = RectBivariateSpline(lats_matrix, lons_matrix, filled)
#             inter = r(lats_new, lons_new)

#             # re-mask: nearest-neighbour mask interpolation
#             r_mask = RectBivariateSpline(lats_matrix, lons_matrix, nan_mask.astype(float))
#             mask_new = r_mask(lats_new, lons_new) > 0.5
#             inter[mask_new] = np.nan

#             out[t, :, :] = inter

#         return out

#     # --- 2D case: (lat, lon) ---
#     elif nt == 1:
#         mat_t = ds[variable_matrix]
#         filled, nan_mask = fill_nans_nearest(mat_t.values)

#         r = RectBivariateSpline(lats_matrix, lons_matrix, filled)
#         inter = r(lats_new, lons_new)

#         r_mask = RectBivariateSpline(lats_matrix, lons_matrix, nan_mask.astype(float))
#         mask_new = r_mask(lats_new, lons_new) > 0.5
#         inter[mask_new] = np.nan

#         return inter

#     else:
#         raise ValueError("nt must be 2 or 3")


def interpolate_matrix_with_nan_fast(
    ds,
    variable_matrix,
    nt,
    lats_matrix,
    lons_matrix,
    lats_new,
    lons_new,
    order=1,                 # 1=bilinear (fast), 3=cubic (slower)
    assume_static_nanmask=True
):
    """
    Fast interpolation (time,lat,lon) -> (time,lat_new,lon_new)
    with NaN nearest-fill + re-mask.
    Uses ndimage.map_coordinates (much faster than RectBivariateSpline per time step).

    assume_static_nanmask:
      True  -> compute NaN mask + nearest-fill indices ONCE from first timestep
      False -> recompute mask+indices each time (slow, but correct if mask changes)
    """

    # ---- Ensure 1D lat/lon arrays
    lats = np.asarray(lats_matrix)
    lons = np.asarray(lons_matrix)
    lats_new = np.asarray(lats_new)
    lons_new = np.asarray(lons_new)

    # ---- Ensure increasing latitude for index mapping
    # (map_coordinates assumes regular index space; we just need monotonic coord mapping)
    flip_lat = False
    if lats[0] > lats[-1]:
        flip_lat = True
        lats = lats[::-1]

    ny = lats.size
    nx = lons.size

    # ---- Build target sampling coordinates in source INDEX space
    # y_idx: float indices for each target latitude in [0..ny-1]
    # x_idx: float indices for each target longitude in [0..nx-1]
    y_idx_1d = np.interp(lats_new, lats, np.arange(ny, dtype=np.float64))
    x_idx_1d = np.interp(lons_new, lons, np.arange(nx, dtype=np.float64))
    Xg, Yg = np.meshgrid(x_idx_1d, y_idx_1d)   # shapes (ny_new, nx_new)

    coords = np.array([Yg, Xg])  # (2, ny_new, nx_new)

    # ---- output
    out = np.full((nt, len(lats_new), len(lons_new)), np.nan, dtype=np.float32)

    # ---- helper to build nearest-fill indices
    def _nearest_fill_indices(nan_mask_2d):
        # returns indices arrays that map each point to nearest non-NaN
        idx = ndimage.distance_transform_edt(
            nan_mask_2d, return_distances=False, return_indices=True
        )
        return idx

    # ---- Precompute NaN mask handling ONCE (fast path)
    if assume_static_nanmask:
        mat0 = np.asarray(ds[variable_matrix].isel(time=0).values, dtype=np.float32)
        if flip_lat:
            mat0 = mat0[::-1, :]

        nan_mask0 = ~np.isfinite(mat0)
        idx0 = _nearest_fill_indices(nan_mask0)

        # Precompute re-mask on target grid ONCE
        mask_interp0 = ndimage.map_coordinates(
            nan_mask0.astype(np.float32),
            coords,
            order=1,
            mode="nearest"
        ) > 0.5

    # ---- main loop
    for t in range(nt):
        mat = np.asarray(ds[variable_matrix].isel(time=t).values, dtype=np.float32)
        if flip_lat:
            mat = mat[::-1, :]

        if assume_static_nanmask:
            nan_mask = nan_mask0
            idx = idx0
            mask_interp = mask_interp0
        else:
            nan_mask = ~np.isfinite(mat)
            idx = _nearest_fill_indices(nan_mask)
            mask_interp = ndimage.map_coordinates(
                nan_mask.astype(np.float32),
                coords,
                order=1,
                mode="nearest"
            ) > 0.5

        # Fill NaNs using nearest valid neighbor indices
        filled = mat[tuple(idx)]

        # Interpolate filled field to new grid
        inter = ndimage.map_coordinates(
            filled,
            coords,
            order=order,
            mode="nearest"
        ).astype(np.float32)

        # Re-apply NaN mask on target grid
        inter[mask_interp] = np.nan
        out[t] = inter

    return out


def extract_latlons_from_coord_events(coord_events):
    """
    Given a pandas Series of position strings as in df_heatwaves.iloc[:,4],
    returns a tuple of arrays: (lats, lons)
    """
    import numpy as np

    def parse_position(pos_str):
        # Remove possible quotes and white spaces
        s = pos_str.strip().strip('"').strip("'").strip('[]')
        items = s.split('),')
        lat = float(items[0].split('(')[-1])
        lon = float(items[1].split('(')[-1].replace(')', ''))
        return lat, lon

    lats, lons = [], []
    for pos_str in coord_events:
        lat, lon = parse_position(pos_str)
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)



