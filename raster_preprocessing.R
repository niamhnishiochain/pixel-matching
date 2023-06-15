#The purpose of this script:
# 1. To merge the output raster files from the earth engine export (4 separate tifs with same structure)
# 2. To preprocess that total raster file to only contain a subset of bands for the initial matching
#      For the time being the bands we match on are distance to roads, distance to settlements, slope, elevation

# Load required libraries
library(raster)

# Set working directory
setwd('C:/Users/35387/Downloads/buffer_tiff')

#Merging
# Read input raster files and merge them
#THIS IS MORE EFFICIENT CODE BUT IT TAKES LONGER TO RUN THAN FROM LINE 19
    #raster_files <- c("1.tif", "2.tif", "3.tif", "4.tif")
    #rasters <- lapply(raster_files, stack)
    #total <- do.call(merge, rasters)

one <- stack("1.tif")
two <- stack("2.tif")
three <- stack("3.tif")
four <- stack("4.tif")

#Merge step by step to avoid memory errors
one_two <- merge(one, two)
one_two_three <- merge(one_two, three)
total <- merge(one_two_three, four)

# Export the merged file 
output_file <- "merged.tif"
writeRaster(total, output_file)

#Subset the bands
#either use the total raster which you create above or read in the file
#total <- stack('merged.tif')
buffer_subset <- dropLayer(total, c(1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16 17))
writeRaster(buffer_coordinates, 'band_subset_buffer.tif')