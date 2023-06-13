import faiss
import rasterio
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

np.random.seed(42)
 
# Open the GeoTIFF file using rasterio and preprocess it 
def read_geotiff_to_array(file_path):
    """
    Parameters
    ----------
    file_path : str
        path to the tif file with the features to match on 
    Returns
    -------
    scaled : numpy array
        all values 0-1 and all missing values removed
    """
    with rasterio.open(file_path) as dataset:
    # Read all the bands into separate NumPy arrays
        band_data = {band: dataset.read(i   + 1) for i, band in enumerate(dataset.indexes)}
    # Reshape the band data arrays into 1D arrays
    reshaped_data = {band: data.flatten() for band, data in band_data.items()}
    # Create a pandas DataFrame from the reshaped band data dictionary
    df = pd.DataFrame(reshaped_data)
    #Way of dealing with missing values (for now)
    df = df[[10, 12, 13, 14]] 
    # define min max scaler
    scaler = MinMaxScaler()
    # transform data
    scaled = scaler.fit_transform(df)
    return scaled

#Project Area
sarara_file_path = r'C:\Users\35387\OneDrive\Documents\learning\sarara.tif'
sarara_array = read_geotiff_to_array(sarara_file_path)

#Randomly sample 1% from the Sarara array
# Calculate the number of elements for the 1% sample
sample_size = round(len(sarara_array) * 0.01)
print('sample size', sample_size)
# Randomly sample 
sarara_sample = np.random.choice(sarara_array.flatten(), size=sample_size, replace=False)
sarara_sample = sarara_sample.reshape(-1, sarara_array.shape[1])

#Buffer Area
buffer_file_path = r'C:\Users\35387\OneDrive\Documents\learning\band_subset_buffer.tif'
buffer_array = read_geotiff_to_array(buffer_file_path)

#FAISS Set Up
#faiss.write_index(index, "populated.index")
index = faiss.read_index("populated.index")
"""
#Train and add to the index
nlist= 100                     #number of Voronoi cells
d = sarara_array.shape[1]      #dimensions
quantizer = faiss.IndexFlatL2(d)  #the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(buffer_array)   
assert index.is_trained

index.add(buffer_array)        #add vectors to the index
print(index.ntotal)            #number of pixels indexed
"""
#Search Subsection
k = 1                         # we want to see k nearest neighbors
index.nprobe = 10             #number of nearby cells to search
D, I = index.search(sarara_sample[:5], k) # sanity check - want the distances to increase
print(I)                      # IDs
print(D)                      # Distances

#Full Search  
startTime = datetime.now()
D, I = index.search(sarara_sample, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
print(datetime.now() - startTime)
