import faiss
import rasterio
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
        band_data = {band: dataset.read(i + 1) for i, band in enumerate(dataset.indexes)}
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
sarara_file_path = r'C:\Users\35387\Downloads\trialTiff.tif'
sarara_array = read_geotiff_to_array(sarara_file_path)

#Buffer Area
buffer_file_path = r'C:\Users\35387\Downloads\bufferSub.tif'
buffer_array = read_geotiff_to_array(buffer_file_path)

#FAISS Set Up
d = sarara_array.shape[1]      #dimensions
index = faiss.IndexFlatL2(d)   #build the index
print(index.is_trained)
index.add(buffer_array)        #add vectors to the index
print(index.ntotal)

k = 3                         # we want to see X nearest neighbors
D, I = index.search(sarara_array[:5], k) # sanity check - want the distances to increase, want the first match to be itself
print(I)                      # IDs
print(D)                      # Distances

D, I = index.search(sarara_array, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries