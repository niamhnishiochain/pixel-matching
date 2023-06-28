"""
This script is to calculate the deforestation in the 
Sarara sample vs the matching pixels in the buffer.
@author: Niamh
"""
import pandas as pd
import rasterio

# Read the CSV files containing the coordinates
sample_coordinates_filepath = r'C:\Users\35387\OneDrive\Documents\learning\sample_matches_plot.csv'
buffer_match_coordinates_filepath = r'C:\Users\35387\OneDrive\Documents\learning\buffer_matches_plot.csv'
sample_coordinates = pd.read_csv(sample_coordinates_filepath)
buffer_match_coordinates = pd.read_csv(buffer_match_coordinates_filepath)
buffer_match_coordinates.columns = ['x', 'y']
# Load the raster files containing the binary deforestation data
sarara_hansen_filepath = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\hansen\export_hansen_data_sarara.tif'
buffer_hansen_filepath = r'C:\Users\35387\OneDrive\Documents\learning\earth_engine_export_data\hansen\export_hansen_data_buffer.tif'
sarara_hansen = rasterio.open(sarara_hansen_filepath)
buffer_hansen = rasterio.open(buffer_hansen_filepath)

def calc_deforestation_history(coordinate_df, def_raster):
    # Initialize the sum variable
    deforestation_sum = 0
    # Iterate over the rows in the CSV file
    for index, row in coordinate_df.iterrows():
        # Get the coordinates from the CSV file
        x = row['x']
        y = row['y']
        
        # Convert the coordinates to pixel coordinates
        row, col = def_raster.index(x, y)
        
        # Read the pixel value from the raster
        pixel_value = def_raster.read(1, window=((row, row+1), (col, col+1)))
        
        # Accumulate the pixel value if it's 1
        if pixel_value == 1:
            deforestation_sum += pixel_value

    # Print the sum of pixel values
    print("Sum of pixel values:", deforestation_sum)
    return deforestation_sum

