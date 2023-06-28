"""
This script:
    1. Two functions to read in raster files
        a) to execute the FAISS
        b) to extract the coordinates from the FAISS results
    2. Execute the FAISS
        a) set up and train the index
        b) or load it
        c) do the search
    3. Export the results
@author: Niamh
"""
#%%
#Package imports
import faiss
import rasterio
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
np.random.seed(42)
 
# Open the GeoTIFF file using rasterio and preprocess it for FAISS execution
def read_geotiff_to_array_faiss(file_path):
    """
    Purpose
    ----------
    Read the raster files for the project area and the buffer to prepare them 
    for the faiss execution. That is, keep only the bands to match on, scale all bands,
    remove missing values and return a numpy array with dtype float32.
   
    Parameters
    ----------
    file_path : str
        path to the tif file with the features to match on 
    all_bands: boolean
        whether the raster file has all 17 bands or not
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
    array = np.transpose(np.array(list(reshaped_data.values())))
    
    #Way of dealing with missing values (exporting from R we set NAs to -999)
        #there are so many missing values because raster always square matrix and the shape are not
    array = array[~np.any(np.isnan(array), axis=1)]

    # Scale the values
    scaler = MinMaxScaler()
    # transform data
    array = scaler.fit_transform(array)
        
    return array

# Open the GeoTIFF file using rasterio and preprocess to analyse matching results
def read_geotiff_to_array_results(file_path, select_indices):
    """
    Purpose
    ----------
    Read the raster files for the project area and the buffer to anlyse the matching results.
    That is, 
    
    Parameters
    ----------
    file_path : str
        path to the tif file with the features to match on
    number_of_bands: int
        the number of bands in the raster file
    
    Returns
    -------
    array : numpy array
        all the original data which can be 
    """
    with rasterio.open(file_path) as dataset:
    # Read all the bands into separate NumPy arrays
        data = dataset.read(1)    
    # Reshape the band data arrays into 1D arrays
    reshaped_data = data.flatten()
    array = np.transpose(reshaped_data) #np.array(list(reshaped_data.values())))
    #add coordinate information
    x_coords = []
    y_coords = []   
    for row in range(dataset.height):
        for col in range(dataset.width):
            x, y = dataset.xy(row, col)
            x_coords.append(x)
            y_coords.append(y)
    array = np.c_[array, x_coords]
    array = np.c_[array, y_coords]
    
    array[array == -999] = np.nan
    array = array[~np.any(np.isnan(array), axis=1)]   
    #subset to the matches or sample 
    array = array[select_indices, 1:3]

    return array

#%%
#Project Area
sarara_file_path = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\matching\export_matching_data_sarara.tif'
sarara_array = read_geotiff_to_array_faiss(sarara_file_path)
#%%
#Randomly sample 1% from the Sarara array
# Calculate the number of elements for the 1% sample
sample_size = round(len(sarara_array) * 0.01)
print('sample size', sample_size)
# Randomly sample 
sample_indices = np.random.choice(range(sarara_array.shape[0]), sample_size, replace=False)
sarara_sample = sarara_array[sample_indices]                             
#sarara_sample = np.random.choice(sarara_array.flatten(), size=sample_size, replace=False)
#sarara_sample = sarara_sample.reshape(-1, sarara_array.shape[1])

#%%
#Buffer Area
buffer_file_path = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\matching\export_matching_data_buffer.tif'
buffer_array = read_geotiff_to_array_faiss(buffer_file_path)

#%%
#FAISS Set Up
index = faiss.read_index("populated.index")
k = 1                         # we want to see k nearest neighbors   
"""  
#Train and add to the index
nlist= 100                     #number of Voronoi cells
d = sarara_array.shape[1]      #dimensions
quantizer = faiss.IndexFlatL2(d)  #the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(buffer_array)   
assert index.is_trained

index.nprobe = 10             # number nearby of Voronoi cells to search
index.add(buffer_array)        #add vectors to the index
print(index.ntotal)            #number of pixels indexed
#faiss.write_index(index, "populated.index") #save the index to disk

#Search subsection as sense check       
D, I = index.search(sarara_sample[:5], k) # sanity check - want the distances to increase
print(I)                      # IDs
print(D)                      # Distances
"""
#%%
#Full Search  
startTime = datetime.now()
D, I = index.search(sarara_sample, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
print(datetime.now() - startTime)

#%%
#Results export and explore
sarara_sample_matches = read_geotiff_to_array_results(sarara_file_path, sample_indices)

#%%
buffer_matches = read_geotiff_to_array_results(buffer_file_path, I.flatten())

#%%
#add the match id to the project area sample array 
#sarara_sample_matched = np.concatenate((sarara_sample, I), axis = 1)
    
#matching_buffer_coords = np.column_stack((buffer_array[list(I.flatten())], np.array(match_coords_x)))
#matching_buffer_coords = np.column_stack((matching_buffer_coords, np.array(match_coords_y)))

#%%
#df.to_csv('buffer_matches_plot.csv', index = False)

#SENSE CHECKING IN PLOT (geopands package issues so just export and plot outside venv)
#X = pd.DataFrame(matching_buffer_coords, columns = ['distance_roads','distance_settlements','elevation', 'slope', 'x', 'y'])
#X.to_csv('file.csv', index = False)