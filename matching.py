import faiss
import rasterio
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import xarray as xr

np.random.seed(42)
 
# Open the GeoTIFF file using rasterio and preprocess it 
def read_geotiff_to_array(file_path, full_bands = True, coordinates = False):
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
    if full_bands == True:
        df = df[[10, 12, 13, 14]] 
    else:
        pass
    # define min max scaler
    if coordinates == False:
        scaler = MinMaxScaler()
        # transform data
        df = scaler.fit_transform(df)
    else:
        df = df.to_numpy()
    x_coords = []
    y_coords = []
    for row in range(dataset.height):
        for col in range(dataset.width):
            x, y = dataset.xy(row, col)
            x_coords.append(x)
            y_coords.append(y)
    df = np.column_stack((df, np.array(x_coords)))    
    df = np.column_stack((df, np.array(y_coords)))    
    return df

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
#buffer_file_path = r'C:\Users\35387\OneDrive\Documents\learning\band_subset_buffer.tif'
#buffer_array = read_geotiff_to_array(buffer_file_path)

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
#D, I = index.search(sarara_sample[:5], k) # sanity check - want the distances to increase
#print(I)                      # IDs
#print(D)                      # Distances

#Full Search  
startTime = datetime.now()
D, I = index.search(sarara_sample[:, 0:4], k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
print(datetime.now() - startTime)

#Results export and explore
#add the match id to the project area sample array 
sarara_sample_matched = np.concatenate((sarara_sample, I), axis = 1)

buffer_file_path = r'C:\Users\35387\OneDrive\Documents\learning\band_subset_buffer.tif'
dataset = rasterio.open(buffer_file_path)
    
x_coords = []
y_coords = []
for row in range(dataset.height):
    for col in range(dataset.width):
        x, y = dataset.xy(row, col)
        x_coords.append(x)
        y_coords.append(y)

match_coords_x = [x_coords[i] for i in list(I.flatten())]
match_coords_y = [y_coords[i] for i in list(I.flatten())]

matching_buffer_coords = np.column_stack((buffer_array[list(I.flatten())], np.array(match_coords_x)))
matching_buffer_coords = np.column_stack((matching_buffer_coords, np.array(match_coords_y)))


#SENSE CHECKING IN PLOT (geopands package issues so just export and plot outside venv)
#X = pd.DataFrame(matching_buffer_coords, columns = ['distance_roads','distance_settlements','elevation', 'slope', 'x', 'y'])
#X.to_csv('file.csv', index = False)