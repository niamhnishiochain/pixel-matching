# Pixel matching using FAISS for dynamic baseline development

The packages used are faiss, rasterio, numpy, pandas, and sklearn. I recommend setting up a venv and conda-forge installing the faiss package. I hope the environment.yml file will do this correctly.

For now the faiss implementation is very simple. The steps are as follows:

	1. Open the raster files for the project and buffer area. The buffer area does not include the full 100km around the project for this initial run but rather a sample of the buffer area.
	2. The raster files are restricted to the bands which do not contain majority missing values. The bands are then scaled between 0 and 1.
	3. The faiss setup involves setting the dimensions of to the number of bands, setting the index using this dimension, adding the base data (the buffer in our case) to the index. 
	4. Then, choose the number of neighbours we wish to return for each. Then conduct the search on the index with this k.
	5. Print the I for the ids of the matches, print the D for the distances. 

