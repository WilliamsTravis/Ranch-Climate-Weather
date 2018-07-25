# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:15:26 2018

@author: User
"""

################################# Switching to/from Ubuntu VPS ###################
from sys import platform
import os

if platform == 'win32':
    homepath = "G:\\My Drive\\NOT THESIS\\Shrum-Williams\\Ranch-Climate-Weather"
    os.chdir(homepath)
    from flask_cache import Cache # I have this one working on Windows but not Linux
    import gdal
    import rasterio
    import boto3
    import urllib
    import botocore
    def PrintException():
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    gdal.UseExceptions()
    print("GDAL version:" + str(int(gdal.VersionInfo('VERSION_NUM'))))
else:
    homepath = "/Ranch-Market-Weather/"
    os.chdir(homepath)
    from flask_caching import Cache # I have this one working on Linux but not Windows :)
    
##################### Libraries #############################################
import copy
import dash
from dash.dependencies import Input, Output, State, Event
import datetime
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import gc
import glob
import json
from flask import Flask
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import plotly
from textwrap import dedent
import time
from tqdm import *
import xarray as xr

##################### Functions ###############################################
###########################################################################
############## Indexing by Baseline  ######################################
###########################################################################  
def index(indexlist,baselinestartyear,baselinendyear):
    '''
        This will find the indexed value to the monthly average. 
    '''        
    indexname = indexlist[0][0][:-7] #Get index name
    baseline = [year for year in indexlist if int(year[0][-6:-2]) >= baselinestartyear and int(year[0][-6:-2]) <= baselinendyear]
    average = monthlies(baseline)
    normallist = []
    for i in range(len(indexlist)):
        for y in range(len(average)):
            if indexlist[i][0][-2:] == average[y][0][-2:]:
                index = indexlist[i][1] / average[y][1]
                normallist.append([indexlist[i][0],index]) 
    return(normallist)
    
###########################################################################
############## Creating monthly averages at each cell #####################
###########################################################################  
def monthlies(indexlist):
    '''
        This takes in the series of indemnity arrays  an RMA grid ID of choice and outputs
            average monthly payouts there.
    '''        
    indexname = indexlist[0][0][:-7] #Get index name
    intmax = np.max([int(item[0][-2:]) for item in indexlist])
    intervals = [format(int(interval),'02d') for interval in range(1,intmax + 1)] # formatted interval strings
    intervallist = [[index[1] for index in indexlist if index[0][-2:] == interval] for interval in intervals] # Just lists arrays for simplicity
    averageslist = [np.nanmean(interval,axis = 0) for  interval in intervallist] # Just averages for each interval
    averageslist = [[indexname + "_" + format(int(i+1),'02d'), averageslist[i]] for i in range(len(averageslist))] # Tack on new names for each interval
    return(averageslist)

###############################################################################
##################### Convert single raster to array ##########################
###############################################################################
def readRaster(rasterpath,band,navalue = -9999):
    """
    rasterpath = path to folder containing a series of rasters
    navalue = a number (float) for nan values if we forgot 
                to translate the file with one originally
    
    This converts a raster into a numpy array along with spatial features needed to write
            any results to a raster file. The return order is:
                
      array (numpy), spatial geometry (gdal object), coordinate reference system (gdal object)
    
    """
    raster = gdal.Open(rasterpath)
    geometry = raster.GetGeoTransform()
    arrayref = raster.GetProjection()
    array = np.array(raster.GetRasterBand(band).ReadAsArray())
    del raster
    array = array.astype(float)
    if np.nanmin(array) < navalue:
        navalue = np.nanmin(array)
    array[array==navalue] = np.nan
    return(array,geometry,arrayref)

###############################################################################
######################## Convert multiple rasters #############################
####################### into numpy arrays #####################################
###############################################################################
def readRasters(rasterpath,navalue = -9999):
    """
    rasterpath = path to folder containing a series of rasters
    navalue = a number (float) for nan values if we forgot 
                to translate the file with one originally
    
    This converts monthly rasters into numpy arrays and them as a list in another
            list. The other parts are the spatial features needed to write
            any results to a raster file. The list order is:
                
      [[name_date (string),arraylist (numpy)], spatial geometry (gdal object), coordinate reference system (gdal object)]
    
    The file naming convention required is: "INDEXNAME_YYYYMM.tif"

    """

    alist=[]
    if rasterpath[-1:] != '\\':
        rasterpath = rasterpath+'\\'
    files = sorted(glob.glob(rasterpath+'*.tif'))
    names = [files[i][len(rasterpath):] for i in range(len(files))]
    sample = gdal.Open(files[0])
#    mask = np.array(sample.GetRasterBand(1).ReadAsArray())
#    mask[mask == 0] = np.nan
#    mask = mask*0+1
    geometry = sample.GetGeoTransform()
    arrayref = sample.GetProjection()
    del sample
    for i in tqdm(range(len(files)),position=0): 
        rast = gdal.Open(files[i])
        array = np.array(rast.GetRasterBand(1).ReadAsArray())
        del rast
        array = array.astype(float)
        array[array==navalue] = np.nan
        name = str.upper(names[i][:-4]) #the file name excluding its extention (may need to be changed if the extension length is not 3)
        alist.append([name,array]) # It's confusing but we need some way of holding these dates. 
    return(alist,geometry,arrayref)
    
###########################################################################
###################### Read Arrays from NPZ or NPY format #################
###########################################################################
def readArrays(path):
    '''
    This will only work if the date files are in the same folder as the .np or .npz
        Otherwise it outputs the same results as the readRaster functions. 
        No other parameters required. 
    '''
    path = 'data\\indices\\noaa_arrays.npz'
    datepath = path[:-10]+"dates"+path[-4:]
    with np.load(path) as data:
        arrays = data.f.arr_0
        data.close()
    with np.load(datepath) as data:
        dates = data.f.arr_0
        data.close()
        dates = [str(d) for d in dates]
    arraylist = [[dates[i],arrays[i]] for i in range(len(arrays))]
    return(arraylist)
    
###########################################################################
############## Little Standardization function for differenct scales ######
###########################################################################  
# Min Max Standardization 
def standardize(indexlist):
    if type(indexlist[0][0])==str:
        arrays = [a[1] for a in indexlist]
    else:
        arrays = indexlist
    mins = np.nanmin(arrays)
    maxes = np.nanmax(arrays)
    def single(array,mins,maxes):    
        newarray = (array - mins)/(maxes - mins)
        return(newarray)
    standardizedlist = [[indexlist[i][0],single(indexlist[i][1],mins,maxes)] for i in range(len(indexlist))]
    return(standardizedlist)

# SD Standardization
def standardize2(indexlist):
    arrays = [indexlist[i][1] for i in range(len(indexlist))]
    mu = np.nanmean(arrays)
    sd = np.nanstd(arrays)
    def single(array,mu,sd):    
        newarray = (array - mu)/sd
        return(newarray)
    standardizedlist = [[indexlist[i][0],single(indexlist[i][1],mu,sd)] for i in range(len(indexlist))]
    return(standardizedlist)

###############################################################################
##################### Write single array to tiffs #############################
###############################################################################
def toRaster(array,path,geometry,srs):
    """
    path = target path
    srs = spatial reference system
    """
    xpixels = array.shape[1]    
    ypixels = array.shape[0]
    path = path.encode('utf-8')
    image = gdal.GetDriverByName("GTiff").Create(path,xpixels, ypixels, 1,gdal.GDT_Float32)
    image.SetGeoTransform(geometry)
    image.SetProjection(srs)
    image.GetRasterBand(1).WriteArray(array)
      
###############################################################################
##################### Write arrays to tiffs ###################################
###############################################################################
def toRasters(arraylist,path,geometry,srs):
    """
    Arraylist format = [[name,array],[name,array],....]
    path = target path
    geometry = gdal geometry object
    srs = spatial reference system object
    """
    if path[-2:] == "\\":
        path = path
    else:
        path = path + "\\"
    sample = arraylist[0][1]
    ypixels = sample.shape[0]
    xpixels = sample.shape[1]
    for ray in  tqdm(arraylist):
        image = gdal.GetDriverByName("GTiff").Create(path+"\\"+ray[0]+".tif",xpixels, ypixels, 1,gdal.GDT_Float32)
        image.SetGeoTransform(geometry)
        image.SetProjection(srs)
        image.GetRasterBand(1).WriteArray(ray[1])
          
