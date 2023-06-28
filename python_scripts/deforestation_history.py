"""
This script is to calculate the deforestation in the 
Sarara sample vs the matching pixels in the buffer.
@author: Niamh
"""
import pandas as pd
import rasterio

# Read the CSV file containing the coordinates
csv_file = 'coordinates.csv'
df = pd.read_csv()

# Load the raster file
raster_file = 'raster.tif'
raster = rasterio.open(raster_file)

# Initialize the sum variable
pixel_sum = 0

# Iterate over the rows in the CSV file
for index, row in df.iterrows():
    # Get the coordinates from the CSV file
    x = row['x_coordinate']
    y = row['y_coordinate']
    
    # Convert the coordinates to pixel coordinates
    row, col = raster.index(x, y)
    
    # Read the pixel value from the raster
    pixel_value = raster.read(1, window=((row, row+1), (col, col+1)))
    
    # Accumulate the pixel value if it's 1
    if pixel_value == 1:
        pixel_sum += pixel_value

# Print the sum of pixel values
print("Sum of pixel values:", pixel_sum)

