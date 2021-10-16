import os
import argparse
import datetime

import pandas as pd
import gdal
import ogr
import osr
import ee


def srtm_download(sample_csv, dst_dir, path_prefix=None, padding=None, target_epsg=4326, dst="Drive"):
    df = pd.read_csv(sample_csv)
    target_spatialRef = osr.SpatialReference()
    target_spatialRef.ImportFromEPSG(target_epsg)

    for row_id in df.index:
        # ------read basic information of Shapefile
        dta_path = os.path.join(path_prefix, df.loc[row_id]["Path"])
        dta_base = os.path.splitext(os.path.basename(dta_path))[0]

        # ------get GeoTiff extent under the target projection (Default: WGS84, EPSG:4326)
        # ---------if the data file is ESRI Shapefile
        if dta_path.endswith(".shp"):
            shp_ds = ogr.Open(dta_path, 0)
            shp_layer = shp_ds.GetLayer()
            shp_spatialRef = shp_layer.GetSpatialRef()
            coordTrans = osr.CoordinateTransformation(shp_spatialRef, target_spatialRef)
            x_min, x_max, y_min, y_max = shp_layer.GetExtent()
        # ---------otherwise, it must be GeoTiff file
        else:
            tiff_ds = gdal.Open(dta_path, 0)
            tiff_spatialRef = osr.SpatialReference(wkt=tiff_ds.GetProjection())
            coordTrans = osr.CoordinateTransformation(tiff_spatialRef, target_spatialRef)
            tiff_geoTransform = tiff_ds.GetGeoTransform()
            x_min = tiff_geoTransform[0]
            y_max = tiff_geoTransform[3]
            x_max = x_min + tiff_geoTransform[1] * tiff_ds.RasterXSize
            y_min = y_max + tiff_geoTransform[5] * tiff_ds.RasterYSize

        point_left_top = coordTrans.TransformPoint(x_min, y_max)[0:2]
        point_right_top = coordTrans.TransformPoint(x_max, y_max)[0:2]
        point_right_bottom = coordTrans.TransformPoint(x_max, y_min)[0:2]
        point_left_bottom = coordTrans.TransformPoint(x_min, y_min)[0:2]

        # ---------select data in our city boundary
        if padding is not None:
            if isinstance(padding, list):
                dx = padding[0]
                dy = padding[1]
            else:
                dx = padding
                dy = padding
            point_left_top = (point_left_top[0] - dx, point_left_top[1] + dy)
            point_right_top = (point_right_top[0] + dx, point_right_top[1] + dy)
            point_right_bottom = (point_right_bottom[0] + dx, point_right_bottom[1] - dy)
            point_left_bottom = (point_left_bottom[0] - dx, point_left_bottom[1] - dy)

        # ------export SRTM data to remote Google Drive directly from Google Earth Engine
        ee.Initialize()
        image = ee.Image("USGS/SRTMGL1_003")
        city_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])
        image = image.clip(city_bd)

        if dst == "Drive":
            task_config = {"image": image,
                           "description": '{0}_strm'.format(dta_base),
                           "folder": dst_dir,
                           "scale": 30,
                           "maxPixels": 1e12,
                           "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toDrive(**task_config)
        elif dst == "CloudStorage":
            task_config = {"image": image,
                           "description": '{0}_strm'.format(dta_base),
                           "bucket": dst_dir,
                           "scale": 30,
                           "maxPixels": 1e12,
                           "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toCloudStorage(**task_config)
        else:
            raise NotImplementedError("Unknown Destination")

        task.start()
        print(df.loc[row_id]["City"], "Start.")


def sentinel1_download(sample_csv, dst_dir, path_prefix=None, padding=None, target_epsg=4326, dst="Drive"):
    df = pd.read_csv(sample_csv)
    target_spatialRef = osr.SpatialReference()
    target_spatialRef.ImportFromEPSG(target_epsg)
    # ------For GDAL 3.0, we must add the folllowing line, see the discussion:
    # ---------https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
    # target_spatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    for row_id in df.index:
        # ------read basic information of Shapefile
        year = df.loc[row_id]["Year"]
        dta_path = os.path.join(path_prefix, df.loc[row_id]["Path"])
        print(dta_path)
        dta_base = os.path.splitext(os.path.basename(dta_path))[0]

        # ------get GeoTiff extent under the target projection (Default: WGS84, EPSG:4326)
        # ---------if the data file is ESRI Shapefile
        if dta_path.endswith(".shp"):
            shp_ds = ogr.Open(dta_path, 0)
            shp_layer = shp_ds.GetLayer()
            shp_spatialRef = shp_layer.GetSpatialRef()
            coordTrans = osr.CoordinateTransformation(shp_spatialRef, target_spatialRef)
            x_min, x_max, y_min, y_max = shp_layer.GetExtent()
        # ---------otherwise, it must be GeoTiff file
        else:
            tiff_ds = gdal.Open(dta_path, 0)
            tiff_spatialRef = osr.SpatialReference(wkt=tiff_ds.GetProjection())
            coordTrans = osr.CoordinateTransformation(tiff_spatialRef, target_spatialRef)
            tiff_geoTransform = tiff_ds.GetGeoTransform()
            x_min = tiff_geoTransform[0]
            y_max = tiff_geoTransform[3]
            x_max = x_min + tiff_geoTransform[1] * tiff_ds.RasterXSize
            y_min = y_max + tiff_geoTransform[5] * tiff_ds.RasterYSize

        point_left_top = coordTrans.TransformPoint(x_min, y_max)[0:2]
        point_right_top = coordTrans.TransformPoint(x_max, y_max)[0:2]
        point_right_bottom = coordTrans.TransformPoint(x_max, y_min)[0:2]
        point_left_bottom = coordTrans.TransformPoint(x_min, y_min)[0:2]

        # ------export Sentinel-1 data to remote Google Drive directly from Google Earth Engine
        ee.Initialize()
        s1_dataset = ee.ImageCollection("COPERNICUS/S1_GRD")
        s1_dataset = s1_dataset.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        s1_dataset = s1_dataset.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        s1_dataset = s1_dataset.filter(ee.Filter.eq('instrumentMode', 'IW'))
        s1_dataset = s1_dataset.select(["VV", "VH"])
        # ---------only keep winter data
        year = min(2020, max(year, 2015))   # Sentinel-1 availability: from 2014-10 to 2020-10
        time_start = datetime.datetime(year=year, month=1, day=1).strftime("%Y-%m-%d")
        time_end = datetime.datetime(year=year, month=12, day=31).strftime("%Y-%m-%d")
        s1_dataset = s1_dataset.filter(ee.Filter.date(time_start, time_end))
        # ---------select data in our city boundary
        if padding is not None:
            if isinstance(padding, list):
                dx = padding[0]
                dy = padding[1]
            else:
                dx = padding
                dy = padding
            point_left_top = (point_left_top[0]-dx, point_left_top[1]+dy)
            point_right_top = (point_right_top[0]+dx, point_right_top[1]+dy)
            point_right_bottom = (point_right_bottom[0]+dx, point_right_bottom[1]-dy)
            point_left_bottom = (point_left_bottom[0]-dx, point_left_bottom[1]-dy)

        city_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])
        s1_dataset = s1_dataset.filterBounds(city_bd)

        # ---------aggregate images in the dataset into its mean value
        # s1_image = s1_dataset.mean()
        # s1_image = s1_dataset.reduce(ee.Reducer.percentile([75]))
        # s1_image = s1_dataset.reduce(ee.Reducer.percentile([25]))
        s1_image = s1_dataset.reduce(ee.Reducer.percentile([50]))

        s1_image = s1_image.clip(city_bd)

        # ---------default DataType for Sentinel-1 data is float64
        # ------------I think it's a little bit large :(
        file_prefix = df.loc[row_id]["City"] + "_" + str(year)
        if dst == "Drive":
            task_config = {"image": s1_image.toFloat(),
                            "description": '{0}_sentinel_1_50pt'.format(file_prefix),
                            "folder": dst_dir,
                            "scale": 10,
                            "maxPixels": 1e13,
                            "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toDrive(**task_config)
        elif dst == "CloudStorage":
            task_config = {"image": s1_image.toFloat(),
                           # "description": '{0}_sentinel_1_avg'.format(file_prefix),
                           # "description": '{0}_sentinel_1_75pt'.format(file_prefix),
                           # "description": '{0}_sentinel_1_25pt'.format(file_prefix),
                           "description": '{0}_sentinel_1_50pt'.format(file_prefix),
                           "bucket": dst_dir,
                           "scale": 10,
                           "maxPixels": 1e13,
                           "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toCloudStorage(**task_config)
        else:
            raise NotImplementedError("Unknown Destination")

        task.start()
        print(df.loc[row_id]["City"], "Start.")

    '''
    # ------check tasks' status
    for i in range(10):
        tasks = ee.batch.Task.list()
        print(tasks)
        time.sleep(5)
    '''


def sentinel2_download(sample_csv, dst_dir, path_prefix=None, padding=None, target_epsg=4326, dst="Drive"):
    df = pd.read_csv(sample_csv)
    target_spatialRef = osr.SpatialReference()
    target_spatialRef.ImportFromEPSG(target_epsg)
    # ------For GDAL 3.0, we must add the folllowing line, see the discussion:
    # ---------https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
    # target_spatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    for row_id in df.index:
        # ------read basic information of Shapefile
        year = df.loc[row_id]["Year"]
        dta_path = os.path.join(path_prefix, df.loc[row_id]["Path"])
        dta_base = os.path.splitext(os.path.basename(dta_path))[0]

        # ------get GeoTiff extent under the target projection (Default: WGS84, EPSG:4326)
        # ---------if the data file is ESRI Shapefile
        if dta_path.endswith(".shp"):
            shp_ds = ogr.Open(dta_path, 0)
            shp_layer = shp_ds.GetLayer()
            shp_spatialRef = shp_layer.GetSpatialRef()
            coordTrans = osr.CoordinateTransformation(shp_spatialRef, target_spatialRef)
            x_min, x_max, y_min, y_max = shp_layer.GetExtent()
        # ---------otherwise, it must be GeoTiff file
        else:
            tiff_ds = gdal.Open(dta_path, 0)
            tiff_spatialRef = osr.SpatialReference(wkt=tiff_ds.GetProjection())
            coordTrans = osr.CoordinateTransformation(tiff_spatialRef, target_spatialRef)
            tiff_geoTransform = tiff_ds.GetGeoTransform()
            x_min = tiff_geoTransform[0]
            y_max = tiff_geoTransform[3]
            x_max = x_min + tiff_geoTransform[1] * tiff_ds.RasterXSize
            y_min = y_max + tiff_geoTransform[5] * tiff_ds.RasterYSize

        point_left_top = coordTrans.TransformPoint(x_min, y_max)[0:2]
        point_right_top = coordTrans.TransformPoint(x_max, y_max)[0:2]
        point_right_bottom = coordTrans.TransformPoint(x_max, y_min)[0:2]
        point_left_bottom = coordTrans.TransformPoint(x_min, y_min)[0:2]

        # ------export Sentinel-1 data to remote Google Drive directly from Google Earth Engine
        ee.Initialize()
        s2_dataset = ee.ImageCollection("COPERNICUS/S2_SR")
        s2_dataset = s2_dataset.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        s2_dataset = s2_dataset.select(["B4", "B3", "B2", "B8"])
        # ---------only keep winter data
        year = min(2020, max(year, 2017))   # Sentinel-2 availability: from 2017-03 to 2020-10
        time_start = datetime.datetime(year=year, month=1, day=1).strftime("%Y-%m-%d")
        time_end = datetime.datetime(year=year, month=12, day=31).strftime("%Y-%m-%d")
        s2_dataset = s2_dataset.filter(ee.Filter.date(time_start, time_end))
        # ---------select data in our city boundary
        if padding is not None:
            if isinstance(padding, list):
                dx = padding[0]
                dy = padding[1]
            else:
                dx = padding
                dy = padding
            point_left_top = (point_left_top[0]-dx, point_left_top[1]+dy)
            point_right_top = (point_right_top[0]+dx, point_right_top[1]+dy)
            point_right_bottom = (point_right_bottom[0]+dx, point_right_bottom[1]-dy)
            point_left_bottom = (point_left_bottom[0]-dx, point_left_bottom[1]-dy)

        city_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])
        s2_dataset = s2_dataset.filterBounds(city_bd)

        # ---------aggregate images in the dataset into its mean value
        # s2_image = s2_dataset.mean()
        # s2_image = s2_dataset.reduce(ee.Reducer.percentile([75]))
        # s2_image = s2_dataset.reduce(ee.Reducer.percentile([25]))
        s2_image = s2_dataset.reduce(ee.Reducer.percentile([50]))

        s2_image = s2_image.clip(city_bd)

        # ---------default DataType for Sentinel-1 data is float64
        # ------------I think it's a little bit large :(
        file_prefix = df.loc[row_id]["City"] + "_" + str(year)
        if dst == "Drive":
            task_config = {"image": s2_image.toFloat(),
                            "description": '{0}_sentinel_2_50pt'.format(file_prefix),
                            "folder": dst_dir,
                            "scale": 10,
                            "maxPixels": 1e13,
                            "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toDrive(**task_config)
        elif dst == "CloudStorage":
            task_config = {"image": s2_image.toFloat(),
                           # "description": '{0}_sentinel_2_avg'.format(file_prefix),
                           # "description": '{0}_sentinel_2_75pt'.format(file_prefix),
                           # "description": '{0}_sentinel_2_25pt'.format(file_prefix),
                           "description": '{0}_sentinel_2_50pt'.format(file_prefix),
                           "bucket": dst_dir,
                           "scale": 10,
                           "maxPixels": 1e13,
                           "crs": 'EPSG:{0}'.format(target_epsg)}
            task = ee.batch.Export.image.toCloudStorage(**task_config)
        else:
            raise NotImplementedError("Unknown Destination")

        task.start()
        print(df.loc[row_id]["City"], "Start.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Earth Engine data downloading")
    parser.add_argument('--type', type=str, choices=["Sentinel-1", "Sentinel-2", "SRTM"])
    parser.add_argument('--input_csv', type=str, help="path of input csv file for city boundary and year specification")
    parser.add_argument("--padding", type=float, default=0.02, help="padding size outside the target region")
    parser.add_argument("--destination", type=str, default="Drive", choices=["Drive", "CloudStorage"],
                        help="destination device where data are export")
    parser.add_argument("--dst_dir", type=str, help="directory on the destination device for data storage")
    parser.add_argument('--path_prefix', type=str, default=".", help="common path prefix for input data")

    args = parser.parse_args()

    if args.type == "SRTM":
        srtm_download(sample_csv=args.input_csv, dst_dir=args.dst_dir, path_prefix=args.path_prefix,
                      padding=args.padding, dst=args.destination)
    elif args.type == "Sentinel-1":
        sentinel1_download(sample_csv=args.input_csv, dst_dir=args.dst_dir, path_prefix=args.path_prefix,
                           padding=args.padding, dst=args.destination)
    elif args.type == "Sentinel-2":
        sentinel2_download(sample_csv=args.input_csv, dst_dir=args.dst_dir, path_prefix=args.path_prefix,
                           padding=args.padding, dst=args.destination)



