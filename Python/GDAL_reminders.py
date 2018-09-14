"""
GDAL routines to translate
Created on Mon Sep 10 15:12:12 2018

@author: User
"""

# Convert albers shapefiles to nad83 for an easier rasterization
"gdalwarp -s_srs epsg:102008 -t_target epsg:4269 out.shp in.shp"

