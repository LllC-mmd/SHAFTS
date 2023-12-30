from typing import Union
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from osgeo import gdal
import geopandas as gpd
import rasterio
import rasterio.plot as rplt
import rasterio.mask as rmask
from rasterio.features import geometry_mask
from shapely.geometry import box

import contextily as ctx
import pydeck as pdk
from pydeck.types import String


# ******************** utility functions ********************
def geotiff_to_deckFeat(file_path, map_extent=None, mask_shp=None):
    if map_extent is not None:
        lon_min, lat_min, lon_max, lat_max = map_extent
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        input_file_path = file_path.replace(".tif", "_tmp.tif")
        with rasterio.open(file_path) as src:
            out_image, out_transform = rmask.mask(src, [bbox], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
            with rasterio.open(input_file_path, "w", **out_meta) as dest:
                dest.write(out_image)
        tmp_file = True
    else:
        input_file_path = file_path
        tmp_file = False

    with rasterio.open(input_file_path) as src:
        transform = src.transform
        building_heights = src.read(1)

        if mask_shp is not None:
            gdf_tmp = gpd.read_file(mask_shp)
            mask_nodata = geometry_mask(gdf_tmp.geometry, src.shape, transform=transform, invert=False)
            building_heights[mask_nodata] = np.nan

        rows, cols = np.where(np.logical_and(building_heights > 0, building_heights != src.nodata))
        lon_list, lat_list = rasterio.transform.xy(transform, rows, cols)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)

        val_list = building_heights[rows, cols]

        if tmp_file:
            os.remove(input_file_path)

        return lon_list, lat_list, val_list
# ******************** utility functions ********************
    
# ---[1] data retrieval
def get_glamour_by_extent(output_path: str, glamour_base_dir: str, var: str, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float]):
    """
    Retrieve a subset of building morphology files within a specific geospatial extent from the GLAMOUR dataset.

    Args:
        output_path (str): The path to save the output GLAMOUR file.
        glamour_base_dir (str): The base directory of the GLAMOUR dataset.
        var (str): The variable to retrieve, either "footprint" or "height".
        lon_min (Union[int, float]): The minimum longitude of the geospatial extent.
        lat_min (Union[int, float]): The minimum latitude of the geospatial extent.
        lon_max (Union[int, float]): The maximum longitude of the geospatial extent.
        lat_max (Union[int, float]): The maximum latitude of the geospatial extent.
    """
    lon_spacing = 9.0
    lat_spacing = 9.0

    if var == "footprint":
        gwords = "BF"
        gwords_dir = "BF_100m"
    elif var == "height":
        gwords = "BH"
        gwords_dir = "BH_100m"
    else:
        raise ValueError("var must be either 'footprint' or 'height'")

    lon_min_loc = int(np.floor(lon_min / lon_spacing)) * lon_spacing
    lat_min_loc = int(np.floor(lat_min / lat_spacing)) * lat_spacing
    lon_max_loc = int(np.ceil(lon_max / lon_spacing)) * lon_spacing
    lat_max_loc = int(np.ceil(lat_max / lat_spacing)) * lat_spacing

    glamour_tif_dir = os.path.join(glamour_base_dir, gwords_dir)

    target_glamour_file = []
    for lon in np.arange(lon_min_loc, lon_max_loc, lon_spacing):
        lon = int(lon)
        for lat in np.arange(lat_min_loc, lat_max_loc, lat_spacing):
            lat = int(lat)
            target_file = os.path.join(glamour_tif_dir, f"{gwords}_{lon}_{lon+int(lon_spacing)}_{lat}_{lat+int(lat_spacing)}.tif")
            if os.path.exists(target_file):
                target_glamour_file.append(target_file)
    
    if len(target_glamour_file) == 0:
        print("No GLAMOUR file found in the specified extent: [{0}, {1}, {2}, {3}]".format(lon_min, lat_min, lon_max, lat_max))
        crop_flag = False
    elif len(target_glamour_file) == 1:
        print("Found 1 GLAMOUR file in the specified extent: {0}".format(target_glamour_file[0]))
        crop_flag = True
    else:
        print("Found {0} GLAMOUR files in the specified extent: {1}".format(len(target_glamour_file), target_glamour_file))
        crop_flag = True

    if crop_flag:
        gdal.Warp(output_path, target_glamour_file, format="GTiff", outputBounds=[lon_min, lat_min, lon_max, lat_max], dstNodata=-255)
        print("Output GLAMOUR file saved to: {0}".format(output_path))
    else:
        print("No GLAMOUR file is generated")
    
    return crop_flag


# ---[2] data visualization
def vis_glamour_by_extent_2d(output_path: str, glamour_base_dir: str, var: str, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], vmin=3.0, vmax=30.0, w_h_ratio=1.0, base_size=4.0, tmp_dir=".", mask_path=None):
    """
    Visualize a subset of building morphology files within a specific geospatial extent from the GLAMOUR dataset using 2D figures.

    Args:
        output_path (str): The path to save the output figure.
        glamour_base_dir (str): The base directory of the GLAMOUR dataset.
        var (str): The variable to visualize, either "footprint" or "height".
        lon_min (Union[int, float]): The minimum longitude of the geospatial extent.
        lat_min (Union[int, float]): The minimum latitude of the geospatial extent.
        lon_max (Union[int, float]): The maximum longitude of the geospatial extent.
        lat_max (Union[int, float]): The maximum latitude of the geospatial extent.
        vmin (float, optional): The minimum value for the colorbar range. Defaults to 3.0.
        vmax (float, optional): The maximum value for the colorbar range. Defaults to 30.0.
        w_h_ratio (float, optional): The width-to-height ratio of the figure. Defaults to 1.0.
        base_size (float, optional): The base size of the figure. Defaults to 4.0.
        tmp_dir (str, optional): The temporary directory to save intermediate files. Defaults to ".".
        mask_path (str, optional): The path to the mask file where only pixels inside the mask will be included in the analysis. Defaults to None.
    """
    
    if var == "footprint":
        tmp_path = os.path.join(tmp_dir, "_tmp_BF.tif")
    elif var == "height":
        tmp_path = os.path.join(tmp_dir, "_tmp_BH.tif")
    else:
        raise ValueError("var must be either 'footprint' or 'height'")

    flag = get_glamour_by_extent(tmp_path, glamour_base_dir, var, lon_min, lat_min, lon_max, lat_max)

    if flag:
        # ------define the plotting settings
        cmap = "jet"
        alpha = 0.5
        if var == "footprint":
            title = "Building Footprint in [{0}, {1}, {2}, {3}]".format(lon_min, lat_min, lon_max, lat_max)
        elif var == "height":
            title = "Building Height in [{0}, {1}, {2}, {3}]".format(lon_min, lat_min, lon_max, lat_max)
        else:
            title = ""

        # ------create the figure
        fig, ax = plt.subplots(figsize=(base_size, base_size * w_h_ratio))
        
        with rasterio.open(tmp_path) as src_glamour:
            glamour_dta = src_glamour.read(1)
            if mask_path is not None:
                gdf_tmp = gpd.read_file(mask_path)
                glamour_dta, geotransform_plot = rmask.mask(src_glamour, gdf_tmp.geometry, crop=True, nodata=src_glamour.nodata)
                gdf_tmp.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2, zorder=3)
            else:
                geotransform_plot = src_glamour.transform
            glamour_dta[glamour_dta == src_glamour.nodata] = np.nan
            glamour_dta[glamour_dta <= 0.0] = np.nan

            s = rplt.show(glamour_dta, ax=ax, transform=geotransform_plot, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, zorder=2)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=src_glamour.crs.to_string(), zorder=1, attribution="")
        
        # ---------set the title
        ax.set_title(title, fontsize=2.0 * base_size)
        # ---------set the ticklabel
        ax.tick_params(axis="both", which="major", labelsize=1.5 * base_size)
        ax.set_xlabel("Longitude", fontsize=2.0 * base_size)
        ax.set_ylabel("Latitude", fontsize=2.0 * base_size)

        # ---------add the colorbar
        sym = ""
        if var == "footprint":
            sym = "$\lambda_p$ [$\mathrm{m}^2$/$\mathrm{m}^2$]"
        elif var == "height":
            sym = "$H_{\mathrm{ave}}$ [$\mathrm{m}$]"
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_tmp = matplotlib.colormaps.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap_tmp, norm=norm)
        sm.set_array([])

        position = ax.inset_axes([0.25, -0.17, 0.6, 0.03], transform=ax.transAxes)
        cbar = fig.colorbar(sm, cax=position, orientation="horizontal")
        _ = position.text(
            -0.2,
            0.5,
            sym,
            va="center",
            ha="center",
            transform=position.transAxes,
        )

        plt.savefig(output_path, dpi=400, bbox_inches="tight")
        os.remove(tmp_path)


def vis_glamour_by_extent_3d(output_path: str, glamour_base_dir: str, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], padding=0.02, bh_min=3.0, bh_max=30.0, zoom=10, angle=45, tmp_dir=".", mask_path=None):
    """
    Visualize a subset of building height files within a specific geospatial extent from the GLAMOUR dataset using 3D interactive maps.

    Args:
        output_path (str): The path to save the output HTML file generated by the `pydeck`.
        glamour_base_dir (str): The base directory of the GLAMOUR dataset.
        lon_min (Union[int, float]): The minimum longitude of the geospatial extent.
        lat_min (Union[int, float]): The minimum latitude of the geospatial extent.
        lon_max (Union[int, float]): The maximum longitude of the geospatial extent.
        lat_max (Union[int, float]): The maximum latitude of the geospatial extent.
        padding (float, optional): The padding around the geospatial extent. Defaults to 0.02.
        bh_min (float, optional): The minimum building height to include in the visualization. Defaults to 3.0.
        bh_max (float, optional): The maximum building height to include in the visualization. Defaults to 30.0.
        zoom (int, optional): The initial zoom level of the map. Defaults to 10.
        angle (int, optional): The initial pitch angle of the map. Defaults to 45.
        tmp_dir (str, optional): The temporary directory to store intermediate files. Defaults to ".".
        mask_path (str, optional): The path to the mask file where only pixels inside the mask will be included in the analysis. Defaults to None.
    """

    tmp_path = os.path.join(tmp_dir, "_tmp_BH.tif")
    flag = get_glamour_by_extent(tmp_path, glamour_base_dir, "height", lon_min, lat_min, lon_max, lat_max)

    if flag:
        lon_list, lat_list, val_list = geotiff_to_deckFeat(tmp_path, map_extent=[lon_min-padding, lat_min-padding, lon_max+padding, lat_max+padding], 
                                                           mask_shp=mask_path)

        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        val_list = np.array(val_list)
        value_mask = (val_list >= bh_min)

        val_list = val_list[value_mask]
        lon_list = lon_list[value_mask]
        lat_list = lat_list[value_mask]

        lon_center = (lon_min + lon_max) / 2.0
        lat_center = (lat_min + lat_max) / 2.0

        # ------define a data array for color mapping (which is not used in extruding polygons)
        value_rescale = np.clip(val_list, bh_min, bh_max)

        # ------generate some fake data to make sure the colorbar range is correct
        if bh_min < np.min(val_list):
            lon_fake = -lon_center
            lat_fake = -lat_center
            val_fake = bh_min
            lon_list = np.append(lon_list, lon_fake)
            lat_list = np.append(lat_list, lat_fake)
            value_rescale = np.append(value_rescale, val_fake)
            val_list = np.append(val_list, np.median(val_list))

        if bh_max > np.max(val_list):
            lon_fake = -lon_center
            lat_fake = -lat_center
            val_fake = bh_max
            lon_list = np.append(lon_list, lon_fake)
            lat_list = np.append(lat_list, lat_fake)
            value_rescale = np.append(value_rescale, val_fake)
            val_list = np.append(val_list, np.median(val_list))

        features = pd.DataFrame({"lon": lon_list, "lat": lat_list, "val": val_list, "val_rescale": value_rescale})

        # ------define the plotting settings
        cmap = "jet"
        alpha = 0.5
        GLAMOUR_RESOLUTION = 50.0  # the radius of the hexagon
        # ---------here are some empirical values for the height scale
        H_max = 100
        H_SCALE = 5
        
        jet_map = matplotlib.colormaps.get_cmap(cmap)
        values = np.linspace(0, 1, 100)
        jet_rgb = jet_map(values)
        jet_rgb[:, 3] = alpha
        jet_rgb = (jet_rgb * 255).astype(np.uint8)  # map to 0-255 range

        features = features.to_dict(orient='records')
        # ---Define the layer for the 3D building visualization
        layer = pdk.Layer(
            "HexagonLayer",
            data=features,
            get_position=['lon', 'lat'],
            get_elevation_weight='val',
            get_color_weight='val_rescale',
            elevation_range=[0, H_SCALE * H_max],
            elevation_scale=H_SCALE,
            radius=GLAMOUR_RESOLUTION,
            color_range=jet_rgb.tolist(),
            pickable=True,
            auto_highlight=True,
            extruded=True,
            aggregation=String('MEAN')
        )

        # ---Set the viewport location
        view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=zoom, pitch=angle, bearing=0)

        # ---Render the deck
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=False, map_style="light")
        deck.to_html(output_path)
        print("Output HTML file saved to: {0}".format(output_path))

        os.remove(tmp_path)


# ---[2] data analysis
def ana_glamour_joint_distribution(output_path: str, glamour_base_dir: str, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], bh_min=3.0, bh_max=30.0, bf_min=0.0, bf_max=1.0, w_h_ratio=1.0, base_size=4.0, tmp_dir=".", mask_path=None):
    """
    Derive the joint distribution of building height and footprint within a specific geospatial extent from the GLAMOUR dataset.

    Args:
        output_path (str): The path to save the output plot.
        glamour_base_dir (str): The base directory of the GLAMOUR dataset.
        lon_min (Union[int, float]): The minimum longitude of the geospatial extent.
        lat_min (Union[int, float]): The minimum latitude of the geospatial extent.
        lon_max (Union[int, float]): The maximum longitude of the geospatial extent.
        lat_max (Union[int, float]): The maximum latitude of the geospatial extent.
        bh_min (float, optional): The minimum building height. Defaults to 3.0.
        bh_max (float, optional): The maximum building height. Defaults to 30.0.
        bf_min (float, optional): The minimum building footprint. Defaults to 0.0.
        bf_max (float, optional): The maximum building footprint. Defaults to 1.0.
        w_h_ratio (float, optional): The width-to-height ratio of the plot. Defaults to 1.0.
        base_size (float, optional): The base size of the plot. Defaults to 4.0.
        tmp_dir (str, optional): The temporary directory to store intermediate files. Defaults to ".".
        mask_path (str, optional): The path to the mask file where only pixels inside the mask will be included in the analysis. Defaults to None.
    """
    tmp_bf_file = os.path.join(tmp_dir, "_tmp_BF.tif")
    tmp_bh_file = os.path.join(tmp_dir, "_tmp_BH.tif")
    
    flag = get_glamour_by_extent(tmp_bf_file, glamour_base_dir, "footprint", lon_min, lat_min, lon_max, lat_max)
    flag = get_glamour_by_extent(tmp_bh_file, glamour_base_dir, "height", lon_min, lat_min, lon_max, lat_max)
    
    if flag:
        if mask_path is not None:
            gdf_tmp = gpd.read_file(mask_path)
        else:
            gdf_tmp = None

        with rasterio.open(tmp_bh_file) as src:
            raster_data = src.read(1)
            # ----------------clip the data to the boundary of the city
            if gdf_tmp is not None:
                bh_clp, geotransform_clp = rmask.mask(src, gdf_tmp.geometry, crop=True, nodata=src.nodata)
            else:
                bh_clp = raster_data
            bh_clp[bh_clp == src.nodata] = np.nan

        with rasterio.open(tmp_bf_file) as src:
            # ----------------test nodata region
            raster_data = src.read(1)
            if gdf_tmp is not None:
                bf_clp, geotransform_clp = rmask.mask(src, gdf_tmp.geometry, crop=True, nodata=src.nodata)
            else:
                bf_clp = raster_data
            bf_clp[bf_clp == src.nodata] = np.nan
            bf_clp[bh_clp <= 0.0] = np.nan
        
        dta_mask = np.logical_and(~np.isnan(bf_clp), ~np.isnan(bh_clp))
        bf_dta = bf_clp[dta_mask].flatten()
        bh_dta = bh_clp[dta_mask].flatten()

        if len(bf_dta) < 1000:
            d_min = 1e-1
        elif len(bf_dta) < 100000:
            d_min = 1e-2
        elif len(bf_dta) < 10000000:
            d_min = 1e-3
        else:
            d_min = 1e-4

        # ------define the plotting settings
        num_bins_hist2d = 100
        cmap = "jet"

        fig, ax = plt.subplots(figsize=(base_size, base_size * w_h_ratio))
        s = ax.hist2d(bf_dta, bh_dta, 
                        range=[[bf_min, bf_max], [bh_min, bh_max]],
                        density=True, bins=num_bins_hist2d, 
                        norm=mcolors.LogNorm(vmin=d_min, vmax=1.0), cmap=cmap)
        
        # ------set the label
        ax.set_xlabel("$\lambda_p$ [$\mathrm{m}^2$/$\mathrm{m}^2$]")
        ax.set_ylabel("$H_{\mathrm{ave}}$ [$\mathrm{m}$]")
        # ------add the colorbar
        cb = fig.colorbar(s[3], ax=ax)
        cb.set_label("Density")

        plt.savefig(output_path, dpi=400, bbox_inches="tight")
        os.remove(tmp_bh_file)
        os.remove(tmp_bf_file)


if __name__ == "__main__":
    glamour_base_dir = "*** folder which contains the `BH_100m` and `BF_100` sub-folders of the GLAMOUR dataset ***"

    # ---define the geospatial extent
    lon_min = 116.56
    lat_min = 39.75
    lon_max = 116.84
    lat_max = 40.02
    mask_shp_path = "./data/tz_mask.gpkg"

    # ---[1] data retrieval
    bf_file = "./GLAMOUR_test_bf.tif"
    bh_file = "./GLAMOUR_test_bh.tif"
    get_glamour_by_extent(bf_file, glamour_base_dir, "footprint", lon_min, lat_min, lon_max, lat_max)
    get_glamour_by_extent(bh_file, glamour_base_dir, "height", lon_min, lat_min, lon_max, lat_max)

    # ---[2] data visualization
    vis_glamour_by_extent_2d("./GLAMOUR_test_bf.png", glamour_base_dir, "footprint", 
                             lon_min, lat_min, lon_max, lat_max,
                             vmin=0.0, vmax=1.0, w_h_ratio=1.0, base_size=5.0,
                             mask_path=mask_shp_path)
    
    vis_glamour_by_extent_2d("./GLAMOUR_test_bh.png", glamour_base_dir, "height",
                                lon_min, lat_min, lon_max, lat_max,
                                vmin=3.0, vmax=30.0, w_h_ratio=1.0, base_size=5.0,
                                mask_path=mask_shp_path)
    
    vis_glamour_by_extent_3d("./GLAMOUR_3d.html", glamour_base_dir,
                                lon_min, lat_min, lon_max, lat_max,
                                padding=0.02, bh_min=3.0, bh_max=30.0, zoom=10, angle=45,
                                mask_path=mask_shp_path)
    
    # ---[3] data analysis
    ana_glamour_joint_distribution("./GLAMOUR_test_joint.png", glamour_base_dir,
                                    lon_min, lat_min, lon_max, lat_max,
                                    bh_min=3.0, bh_max=30.0, bf_min=0.0, bf_max=1.0,
                                    w_h_ratio=1.0, base_size=5.0, mask_path=mask_shp_path)