from shafts.utils.GEE_ops import sentinel2_download_by_extent


# ---specify the spatial extent and year for Sentinel-2's images
lon_min = -87.740
lat_min = 41.733
lon_max = -87.545
lat_max = 41.996
year = 2018

# ---define the output settings
dst = "Drive"
dst_dir = "Sentinel-2_export"
file_name = "Chicago_2018_sentinel_2.tif"

# ---start data downloading
sentinel2_download_by_extent(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max,
                                year=year, dst_dir=dst_dir, file_name=file_name, dst=dst)