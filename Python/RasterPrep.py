# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:48:09 2017

    This is to index, standardize, and account for outliers in whichever climate dataset.

@author: Travis
"""
import os
import sys
os.chdir('G:\\My Drive\\NOT THESIS\\Shrum-Williams\\project')
sys.path.insert(0,'Python')
from functions import *

# Let start with indexing - This is the raw data since 1948
indexlist,geom, proj = readRasters("data\\rasters\\tifs\\noaa_raw\\",-9999)

# Index according to baseline since 1948, choose only 2000-2018
indexlist = index(indexlist,1948,2018)

# Account for outliers
# Separate Rasters frome names
arrays = [a[1] for a in indexlist]

# Get standard deviation and adjust for outliers
sd = np.nanstd(arrays)
for a in arrays:
    a[a < -3*sd] = -3*sd
    a[a > 3*sd] = 3*sd

# Reassign names
indexlist = [[indexlist[i][0],arrays[i]] for i in range(len(indexlist))]

    
# This is the min/max standardiztion
indexlist = standardize(indexlist)

# filter for years
indexlist = [a for a in indexlist if int(a[0][-6:-2]) >= 2000 and int(a[0][-6:-2]) <=2018]

# Save to final folder!
toRasters(indexlist,"data\\rasters\\tifs\\noaa_indexed\\nad83\\",geom,proj)
