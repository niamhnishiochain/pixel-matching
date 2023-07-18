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
import pandas as pd

np.random.seed(42)
 
#%%
#Functions
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

    # Scale the values - now in earth engine
    #scaler = MinMaxScaler()
    # transform data
    #array = scaler.fit_transform(array)
        
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
    array = array[select_indices, :] #, 1:3]

    return array

#%%
#Import Project Area
pa_file_path = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\matching\export_matching_data_pa_sub.tif'
pa_array = read_geotiff_to_array_faiss(pa_file_path)
#%%
#Randomly sample 1% from the project array
# Calculate the number of elements for the 1% sample
sample_size = round(len(pa_array) * 0.01)
print('sample size', sample_size)
# Randomly sample 
sample_indices = np.random.choice(range(pa_array.shape[0]), sample_size, replace=False)
sample_array = pa_array[sample_indices]                             

#%%
#Import Buffer Area
buffer_file_path = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\matching\export_matching_data_buffer_sub.tif'
buffer_array = read_geotiff_to_array_faiss(buffer_file_path)
#%%
#FAISS Set Up
#NOTE: When you first run or want to change the setup of the FAISS
# you need to run lines 131-142. You can then check the results using 
# 148-150 and write the index to memory with 145. 

#See note above
index = faiss.read_index("populated.index")

#Choose number of neighbours
k = 1                       # we want to see k nearest neighbors   

"""
#Train and add to the index
nlist= 100                     #number of Voronoi cells
d = pa_array.shape[1]      #dimensions
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
D, I = index.search(sample_array[:5], k) # sanity check - want the distances to increase
print(I)                      # IDs
print(D)                      # Distances
"""
#%%
#Full Search  
startTime = datetime.now()
D, I = index.search(sample_array, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
print(datetime.now() - startTime)
#%%
#SAMPLE results coordinates
#sarara_sample_matches = read_geotiff_to_array_results(pa_file_path, sample_indices)

#BUFFER results coordinates
#buffer_matches = read_geotiff_to_array_results(buffer_file_path, I.flatten())
#%%
#add the match id to the project area sample array 
#sarara_sample_matched = np.concatenate((sarara_sample, I), axis = 1)
    
#matching_buffer_coords = np.column_stack((buffer_array[list(I.flatten())], np.array(match_coords_x)))
#matching_buffer_coords = np.column_stack((matching_buffer_coords, np.array(match_coords_y)))

#%%
#Exporting the two matches to plot for visual sense-check 
"""
buffer_df = pd.DataFrame(buffer_matches, columns = ['x', 'y'])
buffer_df.to_csv('buffer_matches.csv', index = False)

mathews_df = pd.DataFrame(sarara_sample_matches, columns = ['x', 'y'])
mathews_df.to_csv('mathews_matches.csv', index = False)
"""
# %%
# Generate index values based on the number of rows
#make_index = np.arange(sample_array.shape[0]).reshape(-1, 1)

# Add the index column to the ndarray
#sample_array = np.hstack((sample_indices.reshape(-1, 1), sample_array))
#related_elements = dict(zip(sample_array[:, 0], buffer_array[I.flatten()]))

#%%
#Filter the buffer array to match
match_array = buffer_array[I.flatten()]

#%%
#Consider an absolute standardized mean difference of <0.25
# between treated and control samples across all covariates as
# acceptable (Stuart, E. A. (2010). Matching methods for causal inference: A review and a
# look forward. Statistical Science, 25, 1–21)
diff_array = np.empty([match_array.shape[0], match_array.shape[1]])
for loc in range(0, len(match_array)):
    for band in range(0, match_array.shape[1]):
        s = sample_array[loc, band]
        m = match_array[loc, band]
        diff = s - m
        diff_array[loc, band] = diff

#%%
#Means from all values
total_array = np.concatenate((buffer_array, pa_array), axis=0) 
means = np.mean(total_array, axis=0)
#%%
#Compare the differences to the means
absolute_differences = np.abs(diff_array - means)

#Understand the differences
mean_differences_from_mean = np.mean(absolute_differences, axis = 0)
range_differences_from_mean = np.ptp(absolute_differences, axis = 0)

#%%

