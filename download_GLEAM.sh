#!/bin/bash

#SBATCH -A wanglei
#SBATCH -p cpu
#SBATCH --nodes=1 --ntasks=5
#SBATCH --time=10:00:00
#SBATCH --error=error_vc_GLEAM.txt
#SBATCH --output=output_vc_GLEAM.txt

# Open a single SFTP session for all downloads, forcing password authentication
sftp -o PreferredAuthentications=password -o PubkeyAuthentication=no -P 2225 gleamuser@hydras.ugent.be <<EOF
lcd /depot/wanglei/data/GLEAM/v4.2/

# Loop through the range of years and download each file
$(for i in $(seq 2015 2025); do
  # echo "get /data/v4.2a/daily/$i/E_${i}_GLEAM_v4.2a.nc"
  # echo "get /data/v4.2a/daily/$i/SMrz_${i}_GLEAM_v4.2a.nc"
  # echo "get /data/v4.2a/daily/$i/SMs_${i}_GLEAM_v4.2a.nc"
  echo "get /data/v4.2a/daily/$i/H_${i}_GLEAM_v4.2a.nc"
done)

EOF