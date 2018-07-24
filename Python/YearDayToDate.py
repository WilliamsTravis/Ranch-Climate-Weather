# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:48:09 2017

    This is to extract rasters from the Daily Unified Gauge-Based Analysis over CONUS
        Precipitation datasets. Units are in mm per day and each yearly file contains 
        364 - 366 daily records. For this project we need these to be averaged monthly.

@author: Travis
"""
import os
import sys
os.chdir('G:\\My Drive\\NOT THESIS\\Shrum-Williams\\project')
sys.path.insert(0,'Python')
from functions import *


# Watch out for memory --  And don't try to open the list in variable explorer!!
rasterpath = "data\\rasters\\tifs\\noaa_full"
maskpath = "data\\rasters\\tifs\\masks\\nad83\\mask4.tif"
mask,geom,proj = readRaster(maskpath,1,-9999)
indexname = 'NOAA'

################################### New Function #############################
def tifCollect(rasterpath, indexname, mask, years):
    """
    Move to functions when complete
    
    raster path = path to folder with list of raster files
    indexname = string of name of index in question ("NOAA", "SPI",...)
    maskpath = path to raster of country mask
    """
    if rasterpath[:-2] != '\\':
        rasterpath = rasterpath + '\\'
    files = glob.glob(rasterpath+'*.tif')
    files = [file for file in files if int(file[-8:-4]) >= years[0] and int(file[-8:-4]) <= years[1]]
    mask,geo,proj = readRaster(maskpath,1,-9999.) # we want these spatial references, the dailies are upside down
    indexlist = [] 
    
    def ydToym(yd):
        date = datetime.datetime.strptime(yd[:4]+' ' +yd[4:],'%Y %j')
        ym = format(date,"%Y%m")
        return ym
        
    # Below will aggregate dailies into monthlies, watch out for memory
    for path in files:
        print("Working on: " + path)
        rastercount = gdal.Open(path).RasterCount
        year = path[-8:-4]
        arrays = ([[indexname+'_'+year+str(band).zfill(3), # this also flips the image right side up and applies a mask for the great lakes
                    readRaster(path, band, -9999)[0][::-1,:]*mask] for band in tqdm(range(1, rastercount+1),position=0)])
        months = [arrays[i][0][:-7]+ydToym(arrays[i][0][-7:]) for i in range(len(arrays))]
        arrays = [[months[i], arrays[i][1]] for i in range(len(arrays))]
        monthstrings = [str(m).zfill(2) for m in range(1,13)]
        arrays= ([[indexname+"_"+year+m,np.nanmean([arrays[i][1] for i in range(len(arrays)) if arrays[i][0][-2:] in m],axis = 0)]
                    for m in monthstrings])
        gc.collect()
        indexlist.append(arrays)
    
    indexlist = [l for lst in indexlist for l in lst]
    return indexlist 

        
indexlist = tifCollect(rasterpath,"NOAA", mask,[1948,2018]) # Careful the last year got omitted last time, somehow
toRasters(indexlist,"data\\rasters\\tifs\\noaa_raw\\",geom,proj)    
        
# indexlist2 = readRasters("data\\rasters\\tifs\\noaa_raw\\",-9999)[0]
        
        
        
        
        
     