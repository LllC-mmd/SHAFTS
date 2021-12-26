import os
import glob
import math
import shutil
import subprocess
import multiprocessing
import numpy as np
import pandas as pd

from osgeo import gdal, ogr, osr
import geopandas as gpd
from geopandas import tools as gtools
from shapely import ops as gops
from pyproj import Transformer

'''
Before execute these functions,
please ensure that GDAL and its command line tool are installed correctly 
'''

    
# ************************* ShapeFile Batch Processing *************************
def mergeShapefile(shp_dir, merged_shp):
    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    for f in shp_files:
        if os.path.exists(merged_shp):
            # os.system("ogr2ogr -f 'ESRI Shapefile' -update -append {0} {1}".format(merged_shp, os.path.join(shp_dir, f)))
            subprocess.run(["ogr2ogr", "-f", "'ESRI Shapefile'", "-update", "-append", merged_shp, os.path.join(shp_dir, f)])
        else:
            prefix = f.split(".")[0]
            suffix = [".dbf", ".prj", ".shp", ".shx"]
            dest_prefix = merged_shp.split(".")[0]
            for s in suffix:
                shutil.copyfile(os.path.join(shp_dir, prefix + s), dest_prefix + s)


def BatchReprojectShapefile(shp_dir, target_epsg=4326):
    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    for f in shp_files:
        shp_path = os.path.join(shp_dir, f)
        shp_base = os.path.splitext(f)[0]
        new_path = os.path.join(shp_dir, shp_base + "_" + str(target_epsg))
        # os.system("ogr2ogr -f 'ESRI Shapefile' -t_srs 'EPSG:{0}' {1} {2}".format(target_epsg, new_path, shp_path))
        subprocess.run(["ogr2ogr", "-f", "'ESRI Shapefile'", "-t_srs", "'EPSG:{0}'".format(target_epsg), new_path, shp_path])


def BatchConvertShapefile(sample_csv, path_prefix=None, target_ext="gpkg"):
    df = pd.read_csv(sample_csv)

    for row_id in df.index:
        shp_path = os.path.join(path_prefix, df.loc[row_id]["SHP_Path"])
        shp_dir = os.path.dirname(shp_path)
        shp_base = os.path.splitext(os.path.basename(shp_path))[0]
        
        if target_ext == "shp":
            target_driver = "ESRI Shapefile"
        elif target_ext == "gpkg":
            target_driver = "GPKG"
        else:
            raise NotImplementedError("Unknown vector format")

        target_path = os.path.join(shp_dir, shp_base + "." + target_ext)
        
        # os.system("ogr2ogr -f '{0}' {1} {2}".format(target_driver, target_path, shp_path))
        subprocess.run(["ogr2ogr", "-f", target_driver, target_path, shp_path])


# ************************* Functional Scripts *************************
def createFishNet(input_shp: str, output_grid: str, height: float, width: float, driver_name="ESRI Shapefile", extent=None, identifier="NID"):
    """Create fishnet layer for a specific region (with projection).

    Parameters
    ----------

    input_shp : str
        Path to a vector layer which specifies the projection (and extent) of the fishnet layer.
    output_grid : str
        Output Path for the fishnet layer.
    height : float
        Height of each grid in the fishnet layer (along the latitude direction).
    width : float
        Width of each grid in the fishnet layer (along the longitude direction).
    driver_name : str
        OGR vector driver for reading `input_shp` and writing `output_grid`
    extent : list
        A list in the format `[x_min, y_min, x_max, y_max]` which specifies the extent of the fishnet layer.
        The default is `None` and then the extent of the fishnet layer would be derived from `input_shp`.
    identifier : str
        Name of the field which specifies the ID of each grid in the fishnet layer.
        The default is `NID`.

    """
    # ------open and read Shapefile
    driver = ogr.GetDriverByName(driver_name)
    shp_ds = driver.Open(input_shp, 1)
    shp_layer = shp_ds.GetLayer()

    # ------get features' extent (or use specified extent) and projection information
    if extent is None:
        x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    else:
        x_min, y_min, x_max, y_max = extent

    input_proj = shp_layer.GetSpatialRef()

    # ------define x,y coordinates of output FishNet
    num_row = int(np.ceil(round(round(y_max - y_min, 6) / height, 6)))
    num_col = int(np.ceil(round(round(x_max - x_min, 6) / width, 6)))

    fishnet_X_left = np.linspace(x_min, x_min+(num_col-1)*width, num_col)
    fishnet_X_right = fishnet_X_left + width
    fishnet_Y_bottom = np.linspace(y_min, y_min+(num_row-1)*width, num_row)
    fishnet_Y_bottom = np.ascontiguousarray(fishnet_Y_bottom[::-1])
    # ------reverse a numpy array to go top from down
    # fishnet_Y_top = np.linspace(y_max-(num_row-1)*width, y_max, num_row)
    #fishnet_Y_top = np.ascontiguousarray(fishnet_Y_top[::-1])
    fishnet_Y_top = fishnet_Y_bottom + height

    # ------create output file
    out_driver = ogr.GetDriverByName(driver_name)
    if os.path.exists(output_grid):
        os.remove(output_grid)
    out_ds = out_driver.CreateDataSource(output_grid)

    if driver_name == "ESRI Shapefile":
        layer_name = output_grid
    else:
        layer_name = "l"

    out_layer = out_ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon)
    idField = ogr.FieldDefn(identifier, ogr.OFTInteger)
    out_layer.CreateField(idField)
    feature_def = out_layer.GetLayerDefn()
    # ---------set output projection same as the input projection by default
    #output_proj = input_proj
    #coord_trans = osr.CoordinateTransformation(input_proj, output_proj)

    # ------create features of grid cell
    for i in range(0, num_row):
        y_top = fishnet_Y_top[i]
        y_bottom = fishnet_Y_bottom[i]
        for j in range(0, num_col):
            fid = i * num_col + j
            x_left = fishnet_X_left[j]
            x_right = fishnet_X_right[j]
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x_left, y_top)
            ring.AddPoint(x_right, y_top)
            ring.AddPoint(x_right, y_bottom)
            ring.AddPoint(x_left, y_bottom)
            ring.AddPoint(x_left, y_top)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            #poly.Transform(coord_trans)
            # ---------add new geometry to layer
            out_feature = ogr.Feature(feature_def)
            out_feature.SetGeometry(poly)
            out_feature.SetField(identifier, int(fid))
            out_layer.CreateFeature(out_feature)
            out_feature.Destroy()

    # ------close data sources
    out_ds.Destroy()

    # ------write ESRI.prj file
    input_proj.MorphToESRI()
    file = open(os.path.splitext(output_grid)[0]+".prj", 'w')
    file.write(input_proj.ExportToWkt())
    file.close()


# ************************* [1] Building Footprint *************************
def getFootprintFromShape(building_shp: str, fishnet_shp: str, output_grid: str, resolution: float, scale=1.0, reserved=False, suffix=None, extent=None, shp_ext="shp", shp_driver="ESRI Shapefile", identifier="NID") -> str:
    """Calculate building footprint density raster layer from a building vector layer using fishnet analysis.

    Parameters
    ----------

    building_shp : str
        Path to the input building vector layer.
    fishnet_shp : str
        Path to the input fishnet layer.
    output_grid : str
        Output Path for the building footprint density raster layer.
    resolution : float
        Spatial resolution of output raster layer (in degrees).
    scale : float
        A factor for unit conversion.
        The default is `1.0`.
    reserved : boolean
        A flag which controls whether temporary files should be reserved after calculation
        The default is `False`.
    suffix : str
        Suffix of the intersection file name.
        The default is `None`.
    extent : list
        A list in the format `[x_min, y_min, x_max, y_max]` which specifies the extent of the fishnet layer.
        The default is `None` and then the extent of the fishnet layer would be derived from `building_shp`.
    shp_ext : str
        File extensions of the input building vector layer and temporary intersection layer.
        The default is `shp`.
    shp_driver : str
        OGR vector driver for reading `input_shp` and writing `output_grid`
        The default is `ESRI Shapefile`.
    identifier : str
        Name of the field which specifies the ID of each grid in the fishnet layer.
        The default is `NID`.
    
    Returns
    -------
    res: str
        Path to the output building footprint density raster layer.
        If `building_shp` has no spatial coverage, `"EMPTY"` is returned.

    """
    building_dir = os.path.dirname(building_shp)
    # fishnet_name = os.path.join(buiding_dir, "_fishnet_{0}.shp".format(suffix))
    pid = output_grid[-6:-4]

    shp_ds = ogr.Open(building_shp)
    shp_layer = shp_ds.GetLayer()
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    input_proj = shp_layer.GetSpatialRef()

    num_row = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
    num_col = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))

    if num_col * num_row != 0:
        # ------create FishNet layer for building Shapefile
        # createFishNet(input_shp=building_shp, output_grid=fishnet_name, height=resolution, width=resolution, extent=extent)

        # ------get the intersection part of building layer and FishNet layer
        # ------the output layer contains segmented buildings using a field named "FID" marking which cell each part belongs to
        fishnet_ds = ogr.Open(fishnet_shp)
        fishnet_layer = fishnet_ds.GetLayer()

        intersect_driver = ogr.GetDriverByName(shp_driver)
        intersect_path = os.path.join(building_dir, "_intersect_{0}.{1}".format(suffix, shp_ext))
        if os.path.exists(intersect_path):
            os.remove(intersect_path)
        intersect_ds = intersect_driver.CreateDataSource(intersect_path)
        intersect_layer = intersect_ds.CreateLayer(pid, geom_type=ogr.wkbMultiPolygon)

        shp_layer.Intersection(fishnet_layer, intersect_layer, ["METHOD_PREFIX=FN_"])

        # ------calculate the average height of each cell in FishNet layer
        # ---------height value is weighted by the area of each part
        val_list = [[feature.GetField("FN_{0}".format(identifier)), feature.GetGeometryRef().GetArea()] for feature in intersect_layer]
        
        if extent is not None:
            x_min, y_min, x_max, y_max = extent
            num_row = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
            num_col = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))

        footprint_arr = np.zeros(num_row * num_col)
        for v in val_list:
            footprint_arr[v[0]] += v[1] * scale

        footprint_arr = footprint_arr.reshape((num_row, num_col))
        footprint_arr = footprint_arr / (resolution * resolution)

        driver = gdal.GetDriverByName("GTiff")
        footprint_ds = driver.Create(output_grid, num_col, num_row, 1, gdal.GDT_Float64)
        footprint_ds.GetRasterBand(1).WriteArray(footprint_arr)
        footprint_ds.SetGeoTransform([x_min, resolution, 0, y_max, 0, -resolution])

        if input_proj is not None:
            footprint_ds.SetProjection(input_proj.ExportToWkt())
        else:
            default_proj = osr.SpatialReference()
            default_proj.ImportFromEPSG(4326)
            footprint_ds.SetProjection(default_proj.ExportToWkt())

        footprint_ds.FlushCache()
        footprint_ds = None

        # ---------if reserve_flag is set to be True, we store intermediate results for further algorithm validation
        if not reserved:
            # os.system("rm {0}".format(os.path.join(buiding_dir, "_*")))
            tmp_list = glob.glob(os.path.join(building_dir, "_*"))
            for f in tmp_list:
                os.remove(f)
        res = output_grid
    else:
        res = "EMPTY"

    return res


def GetFootprintFromCSV(sample_csv: str, path_prefix=None, resolution=0.0009, reserved=False, num_cpu=1, shrink_deg=1e-6, identifier="NID"):
    """Calculate building footprint density raster layers according to a .csv file.

    Parameters
    ----------

    sample_csv : str
        Path to the input .csv file which uses `City`, `SHP_Path`, `SRTM_Path`, `Year`, `BuildingHeightField`, `Unit` as columns.
    path_prefix : str
        Naming prefix to the path of files specified by `SHP_Path` and `SRTM_Path`.
        The default is `None`.
    resolution : float
        Spatial resolution of output raster layer (in degrees).
        The default is `0.0009`.
    reserved : boolean
        A flag which controls whether temporary files should be reserved after calculation
        The default is `False`.
    num_cpu : int
        Number of expected processes to be run in parallel for the rasterization of building layer.
        The default is `1`.
    shrink_deg : float
        A (negative) buffering distance for fixing invalid polygons.
        The default is `1e-6`.
    identifier : str
        Name of the field which specifies the ID of each grid in the fishnet layer.
        The default is `NID`.

    """
    # ------get basic settings for parallel computing
    num_cpu_available = multiprocessing.cpu_count()
    if num_cpu_available < num_cpu:
        print("Use %d CPUs which is available now" % num_cpu_available)
        num_cpu = num_cpu_available

    n_sub = math.floor(math.sqrt(num_cpu))

    df = pd.read_csv(sample_csv)

    for row_id in df.index:
        # ------read basic information of Shapefile and SRTM file
        city = df.loc[row_id]["City"]
        shp_path = os.path.join(path_prefix, df.loc[row_id]["SHP_Path"])
        srtm_path = os.path.join(path_prefix, df.loc[row_id]["SRTM_Path"])
        height_field = df.loc[row_id]["BuildingHeightField"]
        height_unit = df.loc[row_id]["Unit"]

        # ------set the target output
        shp_base = os.path.splitext(os.path.basename(shp_path))[0]
        shp_ext = os.path.splitext(os.path.basename(shp_path))[1][1:]
        if shp_ext == "shp":
            shp_driver = "ESRI Shapefile"
        elif shp_ext == "gpkg":
            shp_driver = "GPKG"
        else:
            raise NotImplementedError("Unknown vector format")

        shp_dir = os.path.dirname(shp_path)
        tiff_path = os.path.join(shp_dir, shp_base + "_building_footprint.tif")

        # ------we hope the projection of output can be the same as SRTM data for the convenience of further comparison
        # ---------read the projection information of SRTM file
        if os.path.exists(srtm_path):
            srtm_data = gdal.Open(srtm_path)
            srtm_proj = osr.SpatialReference(wkt=srtm_data.GetProjection())
            srtm_epsg = srtm_proj.GetAttrValue('AUTHORITY', 1)
        else:
            srtm_epsg = 4326
        # ---------reproject the Shapefile if not exists
        shp_projected_path = os.path.join(shp_dir, shp_base + "_projected_{0}.{1}".format(srtm_epsg, shp_ext))
        if not os.path.exists(shp_projected_path):
            # ------------simplify building layers (some buildings would be stacked)
            # ---------------create the union of buildings (split all the self-intersected polygons for cleaning)
            shp_df = gpd.read_file(shp_path)
            source_crs = shp_df.crs

            ids_list = [shp_df.index[ids] for ids in range(0, len(shp_df.geometry)) if shp_df.at[shp_df.index[ids], "geometry"] is None]
            shp_df.drop(ids_list, inplace=True)

            shp_union_geom = [g for g in gops.unary_union(shp_df.geometry)]
            shp_union_ids = [ids for ids in range(0, len(shp_union_geom))]
            shp_union_df = gpd.GeoDataFrame({"ids": shp_union_ids, "geometry": shp_union_geom})
            if shp_union_df.crs is None:
                shp_union_df.set_crs(crs=source_crs, inplace=True)
            else:
                shp_union_df.to_crs(crs=source_crs, inplace=True)

            # ---------------use the spatial join to attach original attributes to cleaned polygons
            shp_df_copy = shp_df.copy()
            # ------------------shrink the original polygons for strict 'within' identification
            if shp_df_copy.crs.to_epsg() == 4326:
                shp_df_copy.geometry = shp_df_copy.geometry.buffer(distance=-shrink_deg)
            else:
                distance_trans = Transformer.from_crs(source_crs, "epsg:4326")
                x_ini, y_ini = shp_df_copy["geometry"][0].centroid.coords[0]
                x1, y1 = distance_trans.transform(x_ini, y_ini)
                distance_trans_back = Transformer.from_crs("epsg:4326", source_crs)
                x1_p, y1_p = distance_trans_back.transform(x1 + shrink_deg, y1)
                shrink_dis = np.sqrt((x_ini - x1_p)**2 + (y_ini - y1_p)**2)
                shp_df_copy.geometry = shp_df_copy.geometry.buffer(distance=-shrink_dis)

            shp_df_copy.drop(columns=[s for s in shp_df_copy.columns if s not in [height_field, "geometry"]], inplace=True)
            shp_df_copy["area"] = shp_df_copy["geometry"].area

            if height_unit == "m":
                scale = 1.0
            elif height_unit == "cm":
                scale = 0.01
            elif height_unit == "FLOOR":
                scale = 3.0
            elif height_unit == "feet":
                scale = 0.3048
            else:
                raise NotImplementedError("Unknown unit for {0} Shapefile".format(city))
            
            # ------------------in case that height records are strings or None
            for i in range(0, len(shp_df_copy["geometry"])):
                ids = shp_df_copy.index[i]
                v = shp_df_copy.at[ids, height_field]
                if v is not None:
                    v = max(float(v), 2.0 / scale)
                else:
                    v = 3.0 / scale
                shp_df_copy.at[ids, height_field] = v

            res_df = gtools.sjoin(left_df=shp_union_df, right_df=shp_df_copy, how="inner", op="contains")

            # ------------------in case that multiple parts are merged into one shapes 
            res_df["h_tmp"] = -1000000.0
            for i in range(0, len(res_df["geometry"])):
                ids = res_df.index[i]
                v = res_df.at[ids, height_field]
                if not isinstance(v, (int, float, np.integer)):
                    area = res_df.at[ids, "area"]
                    h_tmp = np.average([x for x in v], weights=[a for a in area])
                else:
                    h_tmp = v
                res_df.at[ids, "h_tmp"] = h_tmp
            
            res_df.drop(columns=[height_field], inplace=True)
            res_df.rename(columns={"h_tmp": height_field}, inplace=True)

            # ---------------aggregate the attributes on the same polygon by averaging
            res_df = res_df.dissolve(by="ids", aggfunc="mean")

            tmp_path = os.path.join(shp_dir, shp_base + "_temp.{0}".format(shp_ext))
            res_df.to_file(tmp_path, driver=shp_driver)

            # os.system("ogr2ogr {0} -t_srs 'EPSG:{1}' {2}".format(shp_projected_path, srtm_epsg, tmp_path))
            subprocess.run(["ogr2ogr", shp_projected_path, "-t_srs", "'EPSG:{0}'".format(srtm_epsg), tmp_path])
            
        # ------Divide the whole region into sub-regions
        shp_ds = ogr.Open(shp_projected_path)
        shp_layer = shp_ds.GetLayer()
        x_min, x_max, y_min, y_max = shp_layer.GetExtent()

        num_row = math.ceil((y_max - y_min) / resolution)
        num_col = math.ceil((x_max - x_min) / resolution)

        row_id_ref = np.array_split(np.arange(num_row, dtype=int), n_sub)
        col_id_ref = np.array_split(np.arange(num_col, dtype=int), n_sub)
        x_min_list = np.array([x_min + col_id_ref[i][0]*resolution for i in range(0, n_sub)])
        x_max_list = np.array([x_min + (col_id_ref[i][-1]+1)*resolution for i in range(0, n_sub)])
        y_min_list = np.array([y_min + row_id_ref[i][0]*resolution for i in range(0, n_sub)])
        y_max_list = np.array([y_min + (row_id_ref[i][-1]+1)*resolution for i in range(0, n_sub)])

        subRegion_list = [os.path.join(shp_dir, shp_base + "_projected_{0}_temp_{1}.{2}".format(srtm_epsg, str(i)+str(j), shp_ext)) for i in range(0, n_sub) for j in range(0, n_sub)]
        output_list = [os.path.join(shp_dir, shp_base + "_building_footprint_temp_{0}.tif".format(str(i)+str(j))) for i in range(0, n_sub) for j in range(0, n_sub)]
        suffix_list = [str(i)+str(j) for i in range(0, n_sub) for j in range(0, n_sub)]

        arg_list = []
        for i in range(0, n_sub):
            for j in range(0, n_sub):
                extent = [x_min_list[j], y_min_list[i], x_max_list[j], y_max_list[i]]
                subRegion_name = subRegion_list[i*n_sub+j]
                suffix = suffix_list[i*n_sub+j]
                # ------create FishNet layer for building Shapefile
                fishnet_name = os.path.join(shp_dir, "_fishnet_{0}.{1}".format(suffix, shp_ext))
                createFishNet(input_shp=shp_projected_path, output_grid=fishnet_name, height=resolution, width=resolution, driver_name=shp_driver, extent=extent, identifier=identifier)
                # -----------some sub-region might be empty because no building is present in those grids
                # -----------also, the bounding box would be narrowed automatically until it contains all of buildings
                # -----------considering, we must specify the extent of our fishnet layer manually
                # os.system("ogr2ogr -f '{0}' -clipsrc {1} {2} {3}".format(shp_driver, " ".join([str(x) for x in extent]), subRegion_name, shp_projected_path))
                subprocess.run(["ogr2ogr", "-f", "'{0}'".format(shp_driver), "-clipsrc", " ".join([str(x) for x in extent]), subRegion_name, shp_projected_path])
                arg_list.append((subRegion_name, fishnet_name, output_list[i*n_sub+j], resolution, 1.0, True, suffix, extent, shp_ext, shp_driver))

        # ------call the conversion function for each sub-regions
        pool = multiprocessing.Pool(processes=num_cpu)
        res_list = pool.starmap(getFootprintFromShape, arg_list)
        pool.close()
        pool.join()

        # ------merge GeoTiff from sub-regions
        res_tiff_list = [i for i in res_list if i != "EMPTY"]
        # os.system("gdalwarp {0} {1}".format(" ".join(res_tiff_list), tiff_path))
        gdal.Warp(destNameOrDestDS=tiff_path, srcDSOrSrcDSTab=res_tiff_list)

        if not reserved:
            # os.system("rm {0}".format(os.path.join(shp_dir, "*_temp*")))
            tmp_list = glob.glob(os.path.join(shp_dir, "*_temp*"))
            for f in tmp_list:
                os.remove(f)
            # os.system("rm {0}".format(os.path.join(shp_dir, "_*")))
            tmp_list = glob.glob(os.path.join(shp_dir, "_*"))
            for f in tmp_list:
                os.remove(f)

        print("The building footprint map of {0} has been created at: {1}".format(city, tiff_path))


# ************************* [2] Building Height *************************
def getHeightFromShape(building_shp, fishnet_shp, output_grid, height_field, resolution=0.0009, scale=1.0, noData=-1000000.0, reserved=False, suffix=None, extent=None, shp_ext="shp", shp_driver="ESRI Shapefile", identifier="NID"):
    """Calculate building height raster layer from a building vector layer using fishnet analysis.

    Parameters
    ----------

    building_shp : str
        Path to the input building vector layer.
    fishnet_shp : str
        Path to the input fishnet layer.
    output_grid : str
        Output Path for the building footprint density raster layer.
    height_field : str
        Field in the attribute table of building vector layer which stores the building height information.
    resolution : float
        Spatial resolution of output raster layer (in degrees).
    scale : float
        A factor for unit conversion.
        The default is `1.0`.
    noData : float
        NoData value for grids where no building is present.
        The default is `-1000000.0`.
    reserved : boolean
        A flag which controls whether temporary files should be reserved after calculation
        The default is `False`.
    suffix : str
        Suffix of the intersection file name.
        The default is `None`.
    extent : list
        A list in the format `[x_min, y_min, x_max, y_max]` which specifies the extent of the fishnet layer.
        The default is `None` and then the extent of the fishnet layer would be derived from `building_shp`.
    shp_ext : str
        File extensions of the input building vector layer and temporary intersection layer.
        The default is `shp`.
    shp_driver : str
        OGR vector driver for reading `input_shp` and writing `output_grid`
        The default is `ESRI Shapefile`.
    identifier : str
        Name of the field which specifies the ID of each grid in the fishnet layer.
        The default is `NID`.
    
    Returns
    -------
    res: str
        Path to the output building footprint density raster layer.
        If `building_shp` has no spatial coverage, `"EMPTY"` is returned.

    """
    building_dir = os.path.dirname(building_shp)
    # fishnet_name = os.path.join(buiding_dir, "_fishnet_{0}.shp".format(suffix))
    pid = output_grid[-6:-4]

    shp_ds = ogr.Open(building_shp)
    shp_layer = shp_ds.GetLayer()
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    input_proj = shp_layer.GetSpatialRef()

    num_row = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
    num_col = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))

    if num_col * num_row != 0 and shp_layer is not None:
        # ------create FishNet layer for building Shapefile
        # createFishNet(input_shp=building_shp, output_grid=fishnet_name, height=resolution, width=resolution, extent=extent)

        # ------get the intersection part of building layer and FishNet layer
        # ------the output layer contains segmented buildings using a field named "FID" marking which cell each part belongs to
        fishnet_ds = ogr.Open(fishnet_shp)
        fishnet_layer = fishnet_ds.GetLayer()

        intersect_driver = ogr.GetDriverByName(shp_driver)
        intersect_path = os.path.join(building_dir, "_intersect_{0}.{1}".format(suffix, shp_ext))
        if os.path.exists(intersect_path):
            os.remove(intersect_path)
        intersect_ds = intersect_driver.CreateDataSource(intersect_path)
        intersect_layer = intersect_ds.CreateLayer(pid, geom_type=ogr.wkbMultiPolygon)

        shp_layer.Intersection(fishnet_layer, intersect_layer, ["METHOD_PREFIX=FN_"])

        # ------calculate the average height of each cell in FishNet layer
        # ---------height value is weighted by the area of each part
        val_list = [[feature.GetField("FN_{0}".format(identifier)), feature.GetField(height_field), feature.GetGeometryRef().GetArea()] for feature in intersect_layer]

        if extent is not None:
            x_min, y_min, x_max, y_max = extent
            num_row = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
            num_col = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))

        height_arr = np.zeros(num_row*num_col)
        footprint_arr = np.zeros_like(height_arr)
        for v in val_list:
            # ---------sometimes the type of the specified field could be String
            if v[1] is not None:
                v[1] = max(float(v[1]) * scale, 2.0)
                h_tmp = v[1] * v[2]
            else:
                #v[1] = 0.0
                h_tmp = 3.0 * v[2]
            height_arr[v[0]] += h_tmp
            footprint_arr[v[0]] += v[2]
        height_arr = height_arr / footprint_arr
        # ------------if there is no building in the cell, set its value to noData value
        height_arr[np.isnan(height_arr)] = noData
        height_arr[np.isinf(height_arr)] = noData
        height_arr = height_arr.reshape((num_row, num_col))

        driver = gdal.GetDriverByName("GTiff")
        height_ds = driver.Create(output_grid, num_col, num_row, 1, gdal.GDT_Float64)
        height_ds.GetRasterBand(1).WriteArray(height_arr)
        height_ds.SetGeoTransform([x_min, resolution, 0, y_max, 0, -resolution])

        if input_proj is not None:
            height_ds.SetProjection(input_proj.ExportToWkt())
        else:
            default_proj = osr.SpatialReference()
            default_proj.ImportFromEPSG(4326)
            height_ds.SetProjection(default_proj.ExportToWkt())

        height_ds.GetRasterBand(1).SetNoDataValue(noData)

        height_ds.FlushCache()
        height_ds = None

        # ---------if reserve_flag is set to be True, we store intermediate results for further algorithm validation/debug
        if not reserved:
            # os.system("rm {0}".format(os.path.join(building_dir, "_*")))
            tmp_list = glob.glob(os.path.join(building_dir, "_*"))
            for f in tmp_list:
                os.remove(f)
        res = output_grid
    else:
        res = "EMPTY"

    return res


def GetHeightFromCSV(sample_csv, path_prefix=None, resolution=0.0009, noData=-1000000.0, reserved=False, num_cpu=1, shrink_deg=1e-6, identifier="NID"):
    """Calculate building height raster layers according to a .csv file.

    Parameters
    ----------

    sample_csv : str
        Path to the input .csv file which uses `City`, `SHP_Path`, `SRTM_Path`, `Year`, `BuildingHeightField`, `Unit` as columns.
    path_prefix : str
        Naming prefix to the path of files specified by `SHP_Path` and `SRTM_Path`.
        The default is `None`.
    resolution : float
        Spatial resolution of output raster layer (in degrees).
        The default is `0.0009`.
    noData : float
        NoData value for grids where no building is present.
        The default is `-1000000.0`.
    reserved : boolean
        A flag which controls whether temporary files should be reserved after calculation
        The default is `False`.
    num_cpu : int
        Number of expected processes to be run in parallel for the rasterization of building layer.
        The default is `1`.
    shrink_deg : float
        A (negative) buffering distance for fixing invalid polygons.
        The default is `1e-6`.
    identifier : str
        Name of the field which specifies the ID of each grid in the fishnet layer.
        The default is `NID`.

    """
    
    # ------get basic settings for parallel computing
    num_cpu_available = multiprocessing.cpu_count()
    if num_cpu_available < num_cpu:
        print("Use %d CPUs which is available now" % num_cpu_available)
        num_cpu = num_cpu_available

    n_sub = math.floor(math.sqrt(num_cpu))

    df = pd.read_csv(sample_csv)

    for row_id in df.index:
        # ------read basic information of Shapefile and SRTM file
        city = df.loc[row_id]["City"]
        shp_path = os.path.join(path_prefix, df.loc[row_id]["SHP_Path"])
        srtm_path = os.path.join(path_prefix, df.loc[row_id]["SRTM_Path"])
        height_field = df.loc[row_id]["BuildingHeightField"]
        height_unit = df.loc[row_id]["Unit"]

        # ------set the target output
        shp_base = os.path.splitext(os.path.basename(shp_path))[0]
        shp_ext = os.path.splitext(os.path.basename(shp_path))[1][1:]
        if shp_ext == "shp":
            shp_driver = "ESRI Shapefile"
        elif shp_ext == "gpkg":
            shp_driver = "GPKG"
        else:
            raise NotImplementedError("Unknown vector format")

        shp_dir = os.path.dirname(shp_path)
        tiff_path = os.path.join(shp_dir, shp_base + "_building_height.tif")

        if height_unit == "m":
            scale = 1.0
        elif height_unit == "cm":
            scale = 0.01
        elif height_unit == "FLOOR":
            scale = 3.0
        elif height_unit == "feet":
            scale = 0.3048
        else:
            raise NotImplementedError("Unknown unit for {0} Shapefile".format(city))

        # ------we hope the projection of output can be the same as SRTM data for the convenience of further comparison
        # ---------read the projection information of SRTM file
        if os.path.exists(srtm_path):
            srtm_data = gdal.Open(srtm_path)
            srtm_proj = osr.SpatialReference(wkt=srtm_data.GetProjection())
            srtm_epsg = srtm_proj.GetAttrValue('AUTHORITY', 1)
        else:
            srtm_epsg = 4326
        # ---------reproject the Shapefile if not exists
        shp_projected_path = os.path.join(shp_dir, shp_base + "_projected_{0}.{1}".format(srtm_epsg, shp_ext))
        if not os.path.exists(shp_projected_path):
           # ------------simplify building layers (some buildings would be stacked)
            # ---------------create the union of buildings (split all the self-intersected polygons for cleaning)
            shp_df = gpd.read_file(shp_path)
            source_crs = shp_df.crs

            ids_list = [shp_df.index[ids] for ids in range(0, len(shp_df.geometry)) if shp_df.at[shp_df.index[ids], "geometry"] is None]
            shp_df.drop(ids_list, inplace=True)

            shp_union_geom = [g for g in gops.unary_union(shp_df.geometry)]
            shp_union_ids = [ids for ids in range(0, len(shp_union_geom))]
            shp_union_df = gpd.GeoDataFrame({"ids": shp_union_ids, "geometry": shp_union_geom})
            if shp_union_df.crs is None:
                shp_union_df.set_crs(crs=source_crs, inplace=True)
            else:
                shp_union_df.to_crs(crs=source_crs, inplace=True)

            # ---------------use the spatial join to attach original attributes to cleaned polygons
            shp_df_copy = shp_df.copy()
            # ------------------shrink the original polygons for strict 'within' identification
            # ------------------with the distance specified in degrees
            if shp_df_copy.crs.to_epsg() == 4326:
                shp_df_copy.geometry = shp_df_copy.geometry.buffer(distance=-shrink_deg)
            else:
                distance_trans = Transformer.from_crs(source_crs, "epsg:4326")
                x_ini, y_ini = shp_df_copy["geometry"][0].centroid.coords[0]
                x1, y1 = distance_trans.transform(x_ini, y_ini)
                distance_trans_back = Transformer.from_crs("epsg:4326", source_crs)
                x1_p, y1_p = distance_trans_back.transform(x1 + shrink_deg, y1)
                shrink_dis = np.sqrt((x_ini - x1_p)**2 + (y_ini - y1_p)**2)
                shp_df_copy.geometry = shp_df_copy.geometry.buffer(distance=-shrink_dis)

            shp_df_copy.drop(columns=[s for s in shp_df_copy.columns if s not in [height_field, "geometry"]], inplace=True)
            shp_df_copy["area"] = shp_df_copy["geometry"].area

            # ------------------in case that height records are strings or None
            for i in range(0, len(shp_df_copy["geometry"])):
                ids = shp_df_copy.index[i]
                v = shp_df_copy.at[ids, height_field]
                if v is not None:
                    v = max(float(v), 2.0 / scale)
                else:
                    v = 3.0 / scale
                shp_df_copy.at[ids, height_field] = v

            res_df = gtools.sjoin(left_df=shp_union_df, right_df=shp_df_copy, how="inner", op="contains")

            # ------------------in case that multiple parts are merged into one shapes 
            res_df["h_tmp"] = -1000000.0
            for i in range(0, len(res_df["geometry"])):
                ids = res_df.index[i]
                v = res_df.at[ids, height_field]
                if not isinstance(v, (int, float, np.integer)):
                    area = res_df.at[ids, "area"]
                    h_tmp = np.average([x for x in v], weights=[a for a in area])
                else:
                    h_tmp = v
                res_df.at[ids, "h_tmp"] = h_tmp
            
            res_df.drop(columns=[height_field], inplace=True)
            res_df.rename(columns={"h_tmp": height_field}, inplace=True)

            # ---------------aggregate the attributes on the same polygon by averaging
            res_df = res_df.dissolve(by="ids", aggfunc="mean")

            tmp_path = os.path.join(shp_dir, shp_base + "_temp.{0}".format(shp_ext))
            res_df.to_file(tmp_path, driver=shp_driver)

            # os.system("ogr2ogr {0} -t_srs 'EPSG:{1}' -select '{2}' {3}".format(shp_projected_path, srtm_epsg, height_field, tmp_path))
            subprocess.run(["ogr2ogr", shp_projected_path, "-t_srs", "'EPSG:{0}'".format(srtm_epsg), "-select", height_field, tmp_path])

        # ------Divide the whole region into sub-regions
        shp_ds = ogr.Open(shp_projected_path)
        shp_layer = shp_ds.GetLayer()
        x_min, x_max, y_min, y_max = shp_layer.GetExtent()

        num_row = math.ceil((y_max - y_min) / resolution)
        num_col = math.ceil((x_max - x_min) / resolution)

        row_id_ref = np.array_split(np.arange(num_row, dtype=int), n_sub)
        col_id_ref = np.array_split(np.arange(num_col, dtype=int), n_sub)
        x_min_list = np.array([x_min + col_id_ref[i][0]*resolution for i in range(0, n_sub)])
        x_max_list = np.array([x_min + (col_id_ref[i][-1]+1)*resolution for i in range(0, n_sub)])
        y_min_list = np.array([y_min + row_id_ref[i][0]*resolution for i in range(0, n_sub)])
        y_max_list = np.array([y_min + (row_id_ref[i][-1]+1)*resolution for i in range(0, n_sub)])

        subRegion_list = [os.path.join(shp_dir, shp_base + "_projected_{0}_temp_{1}.{2}".format(srtm_epsg, str(i)+str(j), shp_ext)) for i in range(0, n_sub) for j in range(0, n_sub)]
        output_list = [os.path.join(shp_dir, shp_base + "_building_height_temp_{0}.tif".format(str(i)+str(j))) for i in range(0, n_sub) for j in range(0, n_sub)]
        suffix_list = [str(i)+str(j) for i in range(0, n_sub) for j in range(0, n_sub)]

        arg_list = []
        for i in range(0, n_sub):
            for j in range(0, n_sub):
                extent = [x_min_list[j], y_min_list[i], x_max_list[j], y_max_list[i]]
                subRegion_name = subRegion_list[i*n_sub+j]
                suffix = suffix_list[i*n_sub+j]
                # ------create FishNet layer for building Shapefile
                fishnet_name = os.path.join(shp_dir, "_fishnet_{0}.{1}".format(suffix, shp_ext))
                createFishNet(input_shp=shp_projected_path, output_grid=fishnet_name, height=resolution, width=resolution, driver_name=shp_driver, extent=extent, identifier=identifier)
                # -----------some sub-region might be empty because no building is present in those grids
                # -----------also, the bounding box would be narrowed automatically until it contains all of buildings
                # -----------considering, we must specify the extent of our fishnet layer manually
                # os.system("ogr2ogr -f '{0}' -clipsrc {1} {2} {3}".format(shp_driver, " ".join([str(x) for x in extent]), subRegion_name, shp_projected_path))
                subprocess.run(["ogr2ogr", "-f", "'{0}'".format(shp_driver), "-clipsrc", " ".join([str(x) for x in extent]), subRegion_name, shp_projected_path])
                arg_list.append((subRegion_name, fishnet_name, output_list[i*n_sub+j], height_field, resolution, scale, noData, True, suffix_list[i*n_sub+j], extent, shp_ext, shp_driver))

        # ------call the conversion function for each sub-regions
        pool = multiprocessing.Pool(processes=num_cpu)
        
        res_list = pool.starmap(getHeightFromShape, arg_list)
        
        pool.close()
        pool.join()

        # ------merge GeoTiff from sub-regions
        res_tiff_list = [i for i in res_list if i != "EMPTY"]
        # os.system("gdalwarp {0} {1}".format(" ".join(res_tiff_list), tiff_path))
        gdal.Warp(destNameOrDestDS=tiff_path, srcDSOrSrcDSTab=res_tiff_list)

        if not reserved:
            # os.system("rm {0}".format(os.path.join(shp_dir, "*_temp*")))
            tmp_list = glob.glob(os.path.join(shp_dir, "*_temp*"))
            for f in tmp_list:
                os.remove(f)
            # os.system("rm {0}".format(os.path.join(shp_dir, "_*")))
            tmp_list = glob.glob(os.path.join(shp_dir, "_*"))
            for f in tmp_list:
                os.remove(f)

        print("The building height map of {0} has been created at: {1}".format(city, tiff_path))


if __name__ == "__main__":
    lyy_prefix = "/data/lyy/BuildingProject/ReferenceData"
    # lyy_prefix = "/Users/liruidong/Downloads/tmp_back"
    csv_name = "HeightGen.csv"

    #BatchConvertShapefile(csv_name, path_prefix=lyy_prefix, target_ext="gpkg")
    # ---height, 30m
    # GetHeightFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.00027, noData=-1000000.0, reserved=False, num_cpu=25)
    # ---height, 100m
    GetHeightFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.0009, noData=-1000000.0, reserved=False, num_cpu=36)
    # ---height, 250m
    #GetHeightFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.00225, noData=-1000000.0, reserved=False, num_cpu=36)
    # ---height, 500m
    #GetHeightFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.0045, noData=-1000000.0, reserved=False, num_cpu=36)
    # ---height, 1000m
    # GetHeightFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.009, noData=-1000000.0, reserved=False, num_cpu=25)

    # ---footprint, 30m
    # GetFootprintFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.00027, reserved=False, num_cpu=25)
    # ---footprint, 100m
    GetFootprintFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.0009, reserved=False, num_cpu=36)
    # ---footprint, 250m
    #GetFootprintFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.00225, reserved=False, num_cpu=36)
    # ---footprint, 500m
    #GetFootprintFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.0045, reserved=False, num_cpu=36)
    # ---footprint, 1000m
    # GetFootprintFromCSV(csv_name, path_prefix=lyy_prefix, resolution=0.009, reserved=False, num_cpu=25)