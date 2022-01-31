python3 GEE_ops.py \
                --type Sentinel-1 \
                --input_csv GEE_Download_2021.csv \
                --padding 0.04 \
                --year_percentile 75 \
                --destination Drive \
                --dst_dir Sentinel-1_export_75pt \
                --path_prefix /Volumes/ForLyy/Temp/ReferenceData