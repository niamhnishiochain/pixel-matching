#The purpose of this script:
# 1. To merge the output raster files from the earth engine export (4 separate tifs with same structure)
# 2. To preprocess that total raster file to only contain a subset of bands for the initial matching
#      For the time being the bands we match on are distance to roads, distance to settlements, slope, elevation

# Load required libraries
library(raster)

# Set working directory
setwd('C:/Users/35387/Downloads/earth_engine_export_data/matching')

#Open the Earth Engine GeoTiff exports
buffer = stack('export_matching_data_buffer.tif')
sarara = stack('export_matching_data_sarara.tif')

################################################################################
#Merging
#    This is not necessary with the current earth engine exports
#     if the files are larger when we add more features this may become 
#     necessary again.
# Read input raster files and merge them
#THIS IS MORE EFFICIENT CODE BUT IT TAKES LONGER TO RUN THAN FROM LINE 19
    #raster_files <- c("1.tif", "2.tif", "3.tif", "4.tif")
    #rasters <- lapply(raster_files, stack)
    #total <- do.call(merge, rasters)
# one <- stack("1.tif")
# two <- stack("2.tif")
# three <- stack("3.tif")
# four <- stack("4.tif")
# #Merge step by step to avoid memory errors
# one_two <- merge(one, two)
# one_two_three <- merge(one_two, three)
# total <- merge(one_two_three, four)
# 
# # Export the merged file 
# output_file <- "merged.tif"
# writeRaster(total, output_file)

###############################
#Subset the bands
#    Also not now in use!
#either use the total raster which you create above or read in the file
#total <- stack('merged.tif')

#only keep bands we are matching on for now (which have no missings)
  #distance to roads, distance to settlements, slope, elevation
#buffer_subset <- dropLayer(total, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 16, 17))
#writeRaster(buffer_subset, 'band_subset_buffer_nan.tif', overwrite = TRUE, NAflag = -999)
################################################################################

#Sense Checking Missingness
band <- 6 #select a band from 1-8
stack <- sarara #select either 'buffer' or 'sarara'
print(stack[[band]])
to_plot <- stack[[band]] #select a band from one of the stacks
plot(to_plot, colNA = 'red')
table(is.na(to_plot[]))