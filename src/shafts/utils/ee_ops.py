import ee


SR_BAND_SCALE = 1e4
cloudScale = 30


def add_cloud_bands_infer(img: ee.Image, cloud_prob_threshold=50):
  cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").rename("cloudScore")
  # ------1 for cloudy pixels and 0 for non-cloudy pixels
  is_cloud = cloud_prob.gt(cloud_prob_threshold).rename("cloudFlag")

  return img.addBands(ee.Image([cloud_prob, is_cloud]))


def computeQualityScore_infer(img: ee.Image):
  # ------QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
  score = img.select(['cloudScore']).max(img.select(['shadowFlag']))

  score = score.reproject('EPSG:4326', None, 30).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(1))

  score = score.multiply(-1)

  return img.addBands(score.rename('cloudShadowScore'))


# ------recommended for exporting cloud-free Sentinel-2's images during inference
def add_shadow_bands_infer(img: ee.Image, nir_dark_threshold=0.15, cloud_proj_distance=1.5):
  # ------identify the water pixels from the SCL band
  not_water = img.select("SCL").neq(6)

  # ------identify dark NIR pixels that are not water (potential cloud shadow pixels)
  dark_pixels = img.select("B8").lt(nir_dark_threshold * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

  # ------determine the direction to project cloud shadow from clouds
  shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

  # ------project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
  # ---------`directionalDistanceTransform` function is used to calculate the distance between 
  # ---------the nearest cloudy pixels and current non-cloudy pixels along `shadow_azimuth` (<= CLD_PRJ_DIST*10)
  cld_proj = img.select('cloudFlag').directionalDistanceTransform(shadow_azimuth, cloud_proj_distance * 10) \
              .reproject(**{'crs': img.select(0).projection(), 'scale': 100}) \
              .select('distance') \
              .mask() \
              .rename('cloud_transform')

  # ------identify the intersection of dark pixels with cloud shadow projection.
  shadows = cld_proj.multiply(dark_pixels).rename('shadowFlag')

  # ------add dark pixels, cloud projection, and identified shadows as image bands.
  img = img.addBands(ee.Image([cld_proj, shadows]))

  return img


# ------recommended for exporting cloud-free Sentinel-2's images during inference
def add_cloud_shadow_mask_infer(img: ee.Image, cloud_prob_threshold=50):
  img_cloud = add_cloud_bands_infer(img, cloud_prob_threshold)
  img_cloud_shadow = add_shadow_bands_infer(img_cloud)

  #roi = ee.Geometry(img.get('ROI'))
  imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
            ee.Geometry(img.get("system:footprint")).coordinates()
  )

  cloudMask = img_cloud_shadow.select("cloudFlag").add(img_cloud_shadow.select("shadowFlag")).gt(0)
  cloudMask = (cloudMask.focalMin(1.5).focalMax(1.5).reproject(**{'crs': img.select([0]).projection(), 'scale': cloudScale}).rename('cloudmask'))
  cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

  stats = cloudAreaImg.reduceRegion(reducer=ee.Reducer.sum(), scale=cloudScale, maxPixels=1e14)

  cloudPercent = ee.Number(stats.get('cloudmask')).divide(imgPoly.area()).multiply(100)  
  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent)
  
  return img_cloud_shadow


if __name__ == "__main__":
  ee.Initialize()