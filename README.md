# Pixel matching using FAISS for Dynamic Baseline Development

This repo provides an implementation of pixel matching using FAISS (Facebook AI Similarity Search) for dynamic baseline. The goal is to match pixels in a project area with similar features in a buffer area (100km). The matching features are derived from satellite imagery and include distance to roads, distance to settlements, distance to deforestation in the past 5 years, distance to cropland, biome, elevation, and slope. The procedure to create of GeoTiffs with these features is detailed in the Earth Engine Scripts which can be found in the ee_scripts folder.

## Packages 
The packages used are faiss, rasterio, numpy, pandas, and sklearn. I recommend setting up a conda venv and conda-forge installing the faiss package. I hope the environment.yml file will do this correctly. 

## The Faiss Implementation 
This is quite simple. The [documentation](https://github.com/facebookresearch/faiss) for the faiss package is solid and I recommend reading the example tutorial to get a sense of what is going on. The steps are as follows:

1. 
 

## The Data:
Every project and buffer search region pixel is represented by a vector of satellite-derived matching features. These features are as follows:

* Distance to roads:
* Distance to settlements:
* Distance to deforestation (within past 5 years):
* Distance to cropland: 2010, 1km resolution [link](https://developers.google.com/earth-engine/datasets/catalog/USGS_GFSAD1000_V1#description)
* Biome: 
* Elevation:
* Slope: 