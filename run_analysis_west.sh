#!/bin/bash
#SBATCH -A wanglei
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G        # leave ~8G for system
#SBATCH --time=0-12:00:00
#SBATCH --output=output_run_west.txt
#SBATCH --error=error_run_west.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set echo
  module load anaconda
  environment=/home/castanev/.conda/envs/2024.02-py311/MyEnv
  conda activate $environment
  export PATH=$environment/bin:$PATH

  name='ERA5'
  percentile=95
  region="westUS"
  case="${name}_P${percentile}_${region}_detrended_object_byonset"
  path_data="/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/"
  t_file="/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/TS_ERA5.nc"
  t_file_anoma="/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/outputs_storm_swamp/TS_ERA5_daily_anoma_window.nc"
  var_TS="TS"
  global_mean_file="/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/series_globalMean_TS_daily.nc"
  global_mean_var="t"
  seasons=True
  topography=True
  methodology="Teng"
  initial_year="1950"
  initial_date="1950-01-01-00"
  path_case="/home/castanev/RW_amplification_heatwaves/${case}/"
  path_outputs="/scratch/negishi/${USER}/Amplification_RW/${name}/"
  path_outputs_case="$path_outputs/${case}/"
  
  

  name_land='GLEAM'
  path_data_land="/depot/wanglei/data/${name_land}/v4.2/"
  path_outputs_land="/scratch/negishi/${USER}/${name_land}/"
  path_outputs_case_land="$path_outputs_land/${case}/"
  path_case_land="/home/castanev/land_atmosphere/${case}/"

  path_figures_land="$path_case_land/Figures/"
  path_figures_all="$path_case_land/../Figures/"
  path_file_SMrz="$path_outputs_land/SMrz_${name_land}_US.nc"
  path_file_SMs="$path_outputs_land/SMs_${name_land}_US.nc"
  path_file_EF="$path_outputs_land/EF_${name_land}_US.nc"
  path_file_E="$path_outputs_land/E_${name_land}_US.nc"
  path_file_H="$path_outputs_land/H_${name_land}_US.nc"

  if [ ! -d "$path_case_land" ]; then
    mkdir -p "$path_case_land"
  fi

  if [ ! -d "$path_figures_land" ]; then
    mkdir -p "$path_figures_land"
  fi

  if [ ! -d "$path_outputs_case_land" ]; then
    mkdir -p "$path_outputs_case_land"
  fi

  if [ ! -d "$path_figures_all" ]; then
    mkdir -p "$path_figures_all"
  fi


  # python3 0_data.py --name_land $name_land --path_data $path_data_land --path_outputs $path_outputs_land --path_temp $path_outputs_land
  # python3 /home/castanev/RW_amplification_heatwaves/1_heatwaves_detection_teng.py --name $name --case $case --percentile $percentile --region $region --t_file $t_file --var $var_TS --seasons $seasons --topography $topography --methodology $methodology --initial_year $initial_year --path_case $path_case_land --path_outputs_case $path_outputs_case_land --global_mean_file $global_mean_file --global_mean_var $global_mean_var
  # python3 1_heatwaves_detection_teng_clusters.py --name $name --case $case --percentile $percentile --region $region --t_file $t_file --t_file_anoma $t_file_anoma --var $var_TS --seasons $seasons --topography $topography --methodology $methodology --initial_year $initial_year --path_case $path_case_land --path_outputs_case $path_outputs_case_land --global_mean_file $global_mean_file --global_mean_var $global_mean_var
  # python3 1_coupling.py --name_land $name_land --case $case --region $region --path_case_land $path_case_land --path_file_SMrz $path_file_SMrz --path_file_SMs $path_file_SMs --path_file_E $path_file_E --path_file_H $path_file_H --path_outputs $path_outputs_land
  # python3 2_fluxes_hw.py --name $name --name_land $name_land --case $case --region $region --path_case $path_case --path_file_SMs $path_file_SMs --path_file_SMrz $path_file_SMrz --path_file_E $path_file_E --path_file_H $path_file_H --path_file_EF $path_file_EF --path_file_t $t_file --initial_year $initial_year --path_outputs $path_outputs
  # python3 3_breakpoints_optimized_LE.py --name_land $name_land --path_file_SMs $path_file_SMs --path_file_E $path_file_E --path_outputs $path_outputs_land
  python3 4_analysis_regimes_LE.py --name $name --name_land $name_land --case $case --region $region --path_case $path_case_land --path_case_land $path_case_land --path_file_SMs $path_file_SMs --path_file_t $t_file --path_file_t_anom $t_file_anoma --initial_year $initial_year --path_outputs $path_outputs_land

  # path_file_E_recurrent_vanom="/scratch/negishi/castanev/Amplification_RW/ERA5/v250_recurrent_vanom_analytic_signal_6h_NH.nc"
  # path_file_Cp_recurrent_vanom="/scratch/negishi/castanev/Amplification_RW/ERA5/v250_Cp_recurrent_vanom_6h_NH.nc"
  # path_file_v_recurrent_vanom="/scratch/negishi/castanev/Amplification_RW/ERA5/v250_recurrent_vanom_filtered_6h_NH.nc"
  # path_file_v="/scratch/negishi/zhao1550/output_ERA5/RWP/v250_anom_filtered_1940_2023_NH.nc"
  # # python3 2_analysis_RWP_ERA5_recurrent_clusters.py --name $name --case $case --region $region --path_case $path_case_land --path_file_E $path_file_E_recurrent_vanom --path_file_Cp $path_file_Cp_recurrent_vanom --path_file_t $t_file --path_file_v $path_file_v_recurrent_vanom --initial_year $initial_year --path_outputs $path_outputs
  # python3 2_analysis_RWP_ERA5_clusters.py --name $name --case $case --region $region --path_case $path_case_land --path_file_E "/scratch/negishi/zhao1550/output_ERA5/RWP/v250_anom_analytic_signal_1940_2023_NH.nc" --path_file_Cp ${path_outputs}/v250_Cp_synoptic_E010_6h_NH.nc --path_file_t $t_file --path_file_v $path_file_v --initial_year $initial_year --path_outputs $path_outputs --E0 10