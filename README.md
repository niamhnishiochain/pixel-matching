# Pixel matching using FAISS for Dynamic Baseline Development

This repo provides an implementation of pixel matching using FAISS (Facebook AI Similarity Search) for dynamic baseline. The goal is to match pixels (30m) in a project area with similar features in a buffer area (100km). The matching features are derived from satellite imagery and include distance to roads, distance to settlements, distance to deforestation in the past 5 years, distance to cropland, biome, human modification, elevation, and slope. The procedure to create of GeoTiffs with these features is detailed in the Earth Engine Scripts which can be found in the ee_scripts folder.

## Packages 
The packages used are faiss, rasterio, numpy, pandas, and sklearn. I recommend setting up a conda venv and conda-forge installing the faiss package. I hope the environment.yml file will do this correctly. 

## The Faiss Implementation 
This is quite simple. The [documentation](https://github.com/facebookresearch/faiss) for the faiss package is solid and I recommend reading the example tutorial to get a sense of what is going on. The steps are as follows:

1. Open the project area raster file with the 'read_geotiff_to_array_faiss' function that preprocesses it for FAISS execution and return 'sararar_array'
2. Randomly sample 1% of 'sarara_array' and stroe the indices in 'sample_indices'.
3. Open the buffer area raster file ready for FAISS execution. 
4. Set up the FAISS index with the 'buffer_array'. Once this has been done once you can write the index to memory and call it.
5. Perform the FAISS search with the 'sarara_sample' array as the query vectors and specify the number of neighbours 'k'. The function returns 'D' for distances and 'I' for indices.
6. Export the results by calling 'read_geotiff_to_array_results' for both the project area and buffer area raster files and the relevant indices.  
 

## The Data:
Every project and buffer search region pixel is represented by a vector of satellite-derived matching features. These features are as follows:

* Forest Cover Change: 2000-2022 annually, 30.92 meter resolution [link](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2022_v1_10)
* Distance to roads: 2018, 5 arcminutes resolution[link](https://gee-community-catalog.org/projects/grip/)
* Distance to settlements: 2015, 10m resolution [link](https://developers.google.com/earth-engine/datasets/catalog/DLR_WSF_WSF2015_v1)
* Distance to cropland: 2010, 1km resolution [link](https://developers.google.com/earth-engine/datasets/catalog/USGS_GFSAD1000_V1#description)
* Biome: 2001, 1km resolution [link](https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_PNV_PNV_BIOME-TYPE_BIOME00K_C_v01)
* Elevation: 2000, 30m resolution [link](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
* Slope: calculated from elevation image [documentation](https://developers.google.com/earth-engine/apidocs/ee-terrain-slope)
* Global Human Modification: 2016, 1km resolution  [link](https://developers.google.com/earth-engine/datasets/catalog/CSP_HM_GlobalHumanModification)

### Other Potential Features
Gross primary production: think of it as a forest characteristic, annual carbon uptake by photosynthesis [link](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MYD17A2H#description)

## Outstanding
* There are some features which are important to add: percentage forest cover, and climate variables (precipitation and mean temperature (Hewson)).
* Discuss whether we want to be weighting different matching features differently. Understand how this would be implemented.
* Unit tests...