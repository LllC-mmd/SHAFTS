# Example usages of SHAFTS and its global mapping results

Here we offer some example usages of `SHAFTS` and its associated global mapping results, namely the `GLAMOUR` dataset, which offers the average building footprint and height at the resolution of 100 m in global urban centers.

## Overview of examples

- `case_run.py` uses the deep-learning-based models in SHAFTS to predict the average building footprint and height in multiple cities with multiple resolutions.

- `data_download.py` downloads the Sentinel-2 images with a given geospatial extent and year.

- `glamour.py` offers functions such as `get_glamour_by_extent`, `vis_glamour_by_extent_2d`, `vis_glamour_by_extent_3d` and `ana_glamour_joint_distribution` to retrieve, visualize and analysis the average building footprint and height from the the `GLAMOUR` dataset.
The `GLAMOUR` dataset can be requested from this [link](https://zenodo.org/records/10396451) and will be released publicly in January, 2024.
Please also note that additional Python packages including [contextily](https://contextily.readthedocs.io/) and [pydeck](https://pydeck.gl/) should be installed to run this script.

- `minmum_case_run.py` uses the deep-learning-based models in SHAFTS to predict the average building footprint and height in a single city with a single resolution.