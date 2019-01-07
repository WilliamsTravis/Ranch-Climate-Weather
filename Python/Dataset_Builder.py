"""
Pull in cattle production information and associate with climate values.
    It is time to consolidate all of this into python


Created on Mon Sep 10 14:32:10 2018

@author: User
"""
#import datetime as dt
import geopandas as gpd
import glob
import numpy as np
import os
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
os.chdir('C:/Users/User/github/Ranch-Climate-Weather/')
from functions import *
# In[]: Get or update cattle market data

#This step can also be done for the raster datasets if we can find the right
#    date format. For instance, alot of the .nc files from wwdt are grouped
#    such that all januaries dating back to 1895 are grouped in a single file.
#    That would work, too big, there are often individual tiffs, though.
#    This routine is found in the tifCollect function. 

# Get Present Cattle Market Data
    # Get what we have [pd.read_csv("data/tables/rmw/ams_cattle_data.csv")]
    # Get the most recent date
    # ...

# Pull in AMS text files
    # Use selenium for https://marketnews.usda.gov/mnp/ls-report-config
    # Start from the most recent date above
    # ...

# Convert to tables, potentially automate this step
    # ...

# Associate with coordinates
    # This was done manually originally. Is there a better way? 
    # Name recognition, geocoding packages?
    # ...

# Save to file
    # ...

# For now, just pull these fields from our one complete dataset
cows = pd.read_csv("data/tables/rmw/ams_cattle_data.csv")
cows['dateid'] = (cows['year'].astype(str) +
                  cows['month'].astype(str).apply(lambda x: x.zfill(2)) +
                  "_" + cows['polyid'].astype(str))

# In[]: Use climate datasets that are standardized and adjusted for outliers?

# Raster_Arrays Class
#rpath = "D:/data/droughtindices/pdsi/nad83/"
rpath = "C:/Users/User/github/data/rasters/albers/pdsisc/"

rasterArrays = RasterArrays(rpath,-9999)
test = rasterArrays.arraylist[0]
# In[]: Dataset Building
# Climate_Market_Builder class
#rpath = "C:/Users/User/github/data/rasters/albers/pdsisc/"
rpath = "D:/data/droughtindices/pdsi/albers/"

sfpaths = glob.glob("data/shapefiles/marketsheds/albers/central/fourradii/*shp")

for sp in sfpaths:
    radius = sp[-9:-6]
    palmerBuilder = Climate_Market_Builder(rpath,
                                           sp,
                                           "Locale",
                                           [-10, -4],
                                           [200001, 201704])
    palmerdf = palmerBuilder.aggregationMaker(cows, "dateid")
    palmerdf.to_csv("data/tables/rmw/pdsi/pdsi_" + radius +
                    "_standardized_central_agg.csv")
