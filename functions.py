"""
Created on Sun Jul 15 11:15:26 2018

@author: User
"""
# In[]
################################# Switching to/from Ubuntu VPS ###################
from sys import platform
import os

if platform == 'win32':
    homepath = "C:/Users/User/github/Ranch-Climate-Weather"
    os.chdir(homepath)
    from flask_cache import Cache  # This works in Windows but not Linux
    import gdal
    import rasterio
    import boto3
    import urllib
    import botocore
    def PrintException():  # Honestly not sure how to use this yet :)
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
              line.strip(), exc_obj))

    gdal.UseExceptions()
    print("GDAL version:" + str(int(gdal.VersionInfo('VERSION_NUM'))))
else:
    homepath = "/Ranch-Climate-Weather/"
    os.chdir(homepath)
    from flask_caching import Cache  # This works on Linux but not Windows :)

# In[]: Libraries
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
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import plotly
import progress
import re 
import subprocess
import sys
from textwrap import dedent
import threading
import time
from tqdm import tqdm
import warnings
import xarray as xr
warnings.filterwarnings("ignore",category = RuntimeWarning)
# In[]: Functions 
###############################################################################
############## Indexing by Baseline  ##########################################
############################################################################### 
def index(indexlist, baselinestartyear, baselinendyear):
    '''This will find the indexed value to the monthly average. '''
    warnings.filterwarnings("ignore")
    indexname = indexlist[0][0][:-7]  
    baseline = [year for year in indexlist if
                int(year[0][-6:-2]) >= baselinestartyear and
                int(year[0][-6:-2]) <= baselinendyear]
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
        This takes in the series of indemnity arrays  an RMA grid ID of choice
            and outputs average monthly payouts there.
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
# Define Numpy array creation routine 
def readRasters(rasterpath, navalue=-9999):
    """
    Inputs:

        rasterpath = folder containing raster files
        navalue = a number (float) for nan values if needed

    Outputs:

        named list of numpy arrays  = [[name1,array1],[name2,array2],....]
        spatial geometry = (upper left coordinate,
                            x-dimension pixel size,
                            rotation,
                            lower right coordinate,
                            rotation,
                            y dimension pixel size)
        coordinate reference system = Well-Known Text Format

    This uses GDAL bindings to read and convert rasters into numpy arrays. The
        output is a list of numpy arrays and the information needed to write
        back to a raster format.

    """
    print("Reading in rasters and converting to numpy arrays...")

    # Fix possible path inconsistencies
    rasterpath = os.path.normpath(rasterpath)
    rasterpath = os.path.join(rasterpath, '')

    # List full path of all files in folder
    files = glob.glob(rasterpath + '*')

    # Save names for writing files or any useful embedded information
    names = [files[i][len(rasterpath):] for i in range(len(files))]

    # Now iterate over files and convert to numpy array and populate lists
    crs_list = []
    geom_list = []
    numpylist = []
    exceptions = []
    for i in tqdm(range(len(files))):
        try:
            raster = gdal.Open(files[i])
            array = np.array(raster.GetRasterBand(1).ReadAsArray())

            # gdal reads some files that aren't rasters, these won't be 2-D
            if array.ndim == 2:

                # Get spatial reference information to check consistency
                crs = raster.GetProjection()
                crs_list.append([names[i], crs])
                geom = raster.GetGeoTransform()
                geom_list.append([names[i], geom])
                del raster

                # Find  length of extension by the position of the last period
                extension_len = (names[i][::-1].index("."))*-1

                # Set NA Values
                array[array == navalue] = np.nan

                # Get the file name without the extension
                name = names[i][:extension_len-1]
                numpylist.append([name, array])

        except Exception as error:
            exceptions.append('{0}'.format(error))


    # Find locations of spatial mistmatches using the most common geometry
        # and reference system as the baselines.
    crses = [c[1] for c in crs_list]
    crs = max(set(crses), key=crses.count)
    crs_warning_indx = [i for i, s in enumerate(crses) if crs != s]

    geoms = [g[1] for g in geom_list]
    geometry = max(set(geoms), key=geoms.count)
    geom_err_indx = [i for i, s in enumerate(geoms) if geometry != s]

    # Below flags crs mismatches as warnings, but does not remove them from 
    # the array list because it is possible that they can be reprojected when 
    # saving back to a raster. It removes elements with mismatched geometries 
    # and flags these with an error, because reprojection is not as simple in 
    # this case. Most of the time,though, if one is mismatched the other will
    # be, too.
    numpylist = [a for i, a in enumerate(numpylist) if i not in geom_err_indx]
    crs_warnings = []
    for c in crs_warning_indx:
        crs_warnings.append("Warning! Coordinate reference mismatch at " +
                            "{0}".format(crs_list[c][0]))
    geom_errors = []
    for i in geom_err_indx:
        geom_errors.append("Error! Spatial geometry mismatch at " +
                           "{0}".format(geom_list[i][0]))

    # Add warnings and errors to exceptions list
    [exceptions.append(w) for w in crs_warnings]
    [exceptions.append(e) for e in geom_errors]

    return(numpylist, geometry, crs, exceptions)



###########################################################################
###################### Read Arrays from NPZ or NPY format #################
###########################################################################
def readArrays(path):
    '''
    This will only work if the date files are in the same folder as the .np or .npz
        Otherwise it outputs the same results as the readRaster functions. 
        No other parameters required. 
    '''
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
############## Collect tifs from PRISM multibands #############################
###############################################################################
def tifCollect(rasterpath, year, indexname):
    """
    raster path = path to folder with list of raster files
    year = year from each raster to extract after
    indexname = string of name of index in question
    
    Remember this only works for PRISM drought indices atm, and 
        for now its only tested with the PDSI.
    """
    if rasterpath[:-2] != '\\':
        rasterpath = rasterpath + '\\'
    files = glob.glob(rasterpath+'*.tif')
    sample = gdal.Open(files[0])
    navalue = sample.GetRasterBand(1).GetMetadata().get('_FillValue')
    geom = sample.GetGeoTransform()
    proj = sample.GetProjection()
    indexlist = [] 
    
    def daystoDate(raster, band, startyear):
        days = raster.GetRasterBand(band).GetMetadata().get('NETCDF_DIM_day')
        date = datetime.datetime(startyear,1,1,0,0) + datetime.timedelta(int(days) - 1)
        return date.strftime("%Y%m")

    # Below will get the names. We could do the same thing with the raster, however...memory problems
    for path in tqdm(files, position = 0):
        raster = gdal.Open(path)
        yearly = [[indexname+"_"+daystoDate(raster,band,1900),
                   np.array(raster.GetRasterBand(band).ReadAsArray())] for 
            band in range(1,raster.RasterCount+1) if 
            int(daystoDate(raster,band,1900)[:-2]) >= year]
        indexlist.append(yearly)
    indexlist = [index for sublist in indexlist for index in sublist] 
    namelist = [a[0] for a in indexlist]
    arraylist = [a[1] for a in indexlist]
    for a in arraylist:
        a[a==int(navalue)] = np.nan
        
    indexlist = [[namelist[i], arraylist[i]] for i in range(len(arraylist))]
    indexlist.sort()
    return([indexlist,geom,proj])
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
        
        
# In[]
############################## Classes ########################################
# def class to deal with Zonal Statistics and Lag finding
class Climate_Market_Builder:
    '''
    this class will take in a path to a monthly  time series of rasters and a
        path to a
        shapefile with vectors. It will then associate each vector with the
        mean raster values wihin for each time period in the raster folder
        It will provide methods to build datasets of variables that
        associate month names and seasons to lagged raster values for each
        observation month extending two years back

        inputs:

            raster_path = directory string to folder
            shapefile_path = directory string to folder
            ym_field (list) = positions of 6 digit year-month string in filenames
                        (however you want to specify, eg [-10,-4] for
                        "psdisc_194801.tif")
            year_range = optional specification of study period

        attributes:

            raster_files
            shapefile_files

        Things to do:
            This doesn't check for spatial reference inconsistencies like the
            RasterArray class does.

    '''
    def __init__(self, rasterpath, shapefilepath, loc_field, ym_field,
                 ym_range=None):
        # Fix possible path inconsistencies
        rasterpath = os.path.normpath(rasterpath)
        rasterpath = os.path.join(rasterpath, '')
        rfiles = glob.glob(rasterpath+"*")
        if ym_range:
            dx1 = ym_field[0]
            dx2 = ym_field[1]
            d1 = ym_range[0]
            d2 = ym_range[1]
            rfiles = [r for r in rfiles if int(r[dx1:dx2]) >= d1 and
                      int(r[dx1:dx2]) <= d2]
        self.shapefilepath = shapefilepath
        self.rasterfiles = rfiles
        self.ym_field = ym_field
        self.loc_field = loc_field

    def zonalStatMaker(self):
        '''
        Finds zonals stats of raster values within shapefiles
        '''
        print("Calculating mean raster values within shapefile elements...")
        shp = gpd.GeoDataFrame.from_file(self.shapefilepath)
        locations = shp[self.loc_field]
        stats = []
        for i in tqdm(range(len(self.rasterfiles)), position=0):
            stat = zonal_stats(self.shapefilepath,
                               self.rasterfiles[i],
                               stats="mean",
                               nodata=-9999)
            date = self.rasterfiles[i][self.ym_field[0]:self.ym_field[1]]
            df = pd.DataFrame(stat)
            df['date'] = date
            df['polyid'] = df.index
            df['dateid'] = df['date'] + "_" + df['polyid'].astype(str)
            df['year'] = df['date'].str[:4].astype(int)
            df['month'] = df['date'].str[4:].astype(int)
            df['locale'] = locations
            stats.append(df)
        bigdf = pd.concat(stats)
        return bigdf

    # Methods
    def monthMaker(self):
        '''
        associates lagged values for each observation with months of the year.
        '''
        bigdf = self.zonalStatMaker()
        print("Calculating lagged values by month...")

        # Lets get a date index of the full range of possible dates
        dates = [[str(y) + str(m).zfill(2) for y in range(min(bigdf['year']),
                      max(bigdf['year'])+1)] for m in range(1, 13)]
        dates = np.sort([int(lst) for sublist in dates for lst in sublist])
        monthdict = {str(dates[i]):i for i in range(len(dates))}

        # Now we have the time period number for each date, if we add an id
            # for each location we have the index needed for getting the months
        bigdf['datenum'] = bigdf['date'].map(monthdict)
        bigdf['datenumid'] = bigdf['datenum'].astype(str) + '_' + bigdf['polyid'].astype(str)

        # Each month lag has a particular predictable sequence
        # Below shows the number of months back to go to get to each month
            # from each month {obervation month : target month lags }
        lagnames = ['jan1', 'feb1', 'mar1', 'apr1', 'may1', 'jun1',
                    'jul1', 'aug1', 'sep1', 'oct1', 'nov1', 'dec1',
                    'jan2', 'feb2', 'mar2', 'apr2', 'may2', 'jun2',
                    'jul2', 'aug2', 'sep2', 'oct2', 'nov2', 'dec2']
        lagdict = {1 : [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                        24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13],
                   2 : [1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                        13, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14],
                   3 : [2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                        14, 13, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15],
                   4 : [3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4,
                        15, 14, 13, 24, 23, 22, 21, 20, 19, 18, 17, 16],
                   5 : [4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5,
                        16, 15, 14, 13, 24, 23, 22, 21, 20, 19, 18, 17],
                   6 : [5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6,
                        17, 16, 15, 14, 13, 24, 23, 22, 21, 20, 19, 18],
                   7 : [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7,
                        18, 17, 16, 15, 14, 13, 24, 23, 22, 21, 20, 19],
                   8 : [7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8,
                        19, 18, 17, 16, 15, 14, 13, 24, 23, 22, 21, 20],
                   9 : [8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9,
                        20, 19, 18, 17, 16, 15, 14, 13, 24, 23, 22, 21],
                   10 : [9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10,
                         21, 20, 19, 18, 17, 16, 15, 14, 13, 24, 23, 22],
                   11 : [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11,
                         22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 24, 23],
                   12 : [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12,
                         23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 24]}

        # For each of 12 months, map the corresponding lags to this column
        bigdf['monthlags'] = bigdf['month'].map(lagdict)

        # Split this column into spearate rows
        bigdf = bigdf.reset_index()
        bigdf[lagnames] = pd.DataFrame(bigdf['monthlags'].values.tolist(),
             index=bigdf.index)
        bigdf = bigdf.drop(columns=['monthlags'])

        # Now find the date number of the target lagged value and join with the location id
        for lag in tqdm(lagnames, position=0):
            bigdf[lag] = bigdf['datenum'] - bigdf[lag]
            bigdf[lag] = bigdf[lag].astype(str) + '_' + bigdf['polyid'].astype(str)

        # This step a takes the most computation. The latest idea is to query
            # a numpy array for the lagged value given a date number and id
        values = np.array(bigdf['mean'])
        ids = np.array(bigdf['datenumid'])
        lagdict2 = dict(zip(ids, values))

        # Simple dictionary query
        def getLag(lag):
            lag = lagdict2.get(lag)
            return lag

        # Apply dictionary query to each cell in each target lag column
        for lag in tqdm(lagnames, position=0):
            bigdf[lag] = bigdf[lag].map(getLag)

#       Get regular lags: this can be done in the formula of any panel data
#       package, though...might as well it's quick
        print("Calculating regular lagged values...")
        for i in tqdm(range(0, 25), position=0):
            bigdf['t' + str(i)] = bigdf.groupby(['polyid'])['mean'].shift(i)

        bigdf = bigdf.dropna()
        bigdf = bigdf.reset_index()
        return bigdf

    def seasonMaker(self):
        '''  Now use the lagged months to get lagged seasons '''
        bigdf = self.monthMaker()

        # make a dictionary of lagged months for each observation month
        dec = {'winter1' : ['t0'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec1', 'jan1', 'feb1'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        jan = {'winter1' : ['dec1', 't0'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan1', 'feb1'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        feb = {'winter1' : ['dec1', 'jan1', 't0'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb1'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        mar = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['t0'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar1', 'apr1', 'may1'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        apr = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 't0'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1'   : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr1', 'may1'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        may = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 't0'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may1'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        jun = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['t0'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun1', 'jul1', 'aug1'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        jul = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 't0'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul1', 'aug1'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        aug = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 't0'],
               'fall1' : ['sep1', 'oct1', 'nov1'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug1'],
               'fall2' : ['sep2', 'oct2', 'nov2']}
        sep = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1' : ['t0'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep1', 'oct1', 'nov1']}
        octo = {'winter1' : ['dec1', 'jan1', 'feb1'],
                'spring1' : ['mar1', 'apr1', 'may1'],
                'summer1' : ['jun1', 'jul1', 'aug1'],
                'fall1' : ['sep1', 't0'],
                'winter2' : ['dec2', 'jan2', 'feb2'],
                'spring2' : ['mar2', 'apr2', 'may2'],
                'summer2' : ['jun2', 'jul2', 'aug2'],
                'fall2' : ['sep2', 'oct1', 'nov1']}
        nov = {'winter1' : ['dec1', 'jan1', 'feb1'],
               'spring1' : ['mar1', 'apr1', 'may1'],
               'summer1' : ['jun1', 'jul1', 'aug1'],
               'fall1'   : ['sep1', 'oct1', 't0'],
               'winter2' : ['dec2', 'jan2', 'feb2'],
               'spring2' : ['mar2', 'apr2', 'may2'],
               'summer2' : ['jun2', 'jul2', 'aug2'],
               'fall2' : ['sep2', 'oct2', 'nov1']}

        # Okay, now we know which lagged months to average for each season
        seasondict = {1:jan, 2:feb, 3:mar, 4:apr, 5:may, 6:jun,
                      7:jul, 8:aug, 9:sep, 10:octo, 11:nov, 12:dec}

        seasonames = ['winter1', 'spring1', 'summer1', 'fall1',
                      'winter2', 'spring2', 'summer2', 'fall2']

        # what if I ran the function on each month separately?
        # Okay this will work, first create list of monthly dataframes
        print("Calculating lagged seasonal averages...")
        dflist = [bigdf[bigdf.month.isin([m])] for m in range(1, 13)]

        # now loop through each list and calculate lags
        lagdflist = []
        for df in tqdm(dflist, position=0):
            for season in seasonames:
                lags = seasondict.get(1).get(season)
                df[season] = df[lags].mean(axis=1)
            lagdflist.append(df)

        # then concatenate the list of altered dataframes into one!
        bigdf = pd.concat(lagdflist)
        
        return bigdf

    def completeMaker(self, df, join_field):
        '''join climate and other data if locations match'''
        bigdf = self.seasonMaker()
        
        # Go ahead and drop repeated columns
        reps = [bc for bc in bigdf.columns if
                bc in df.columns and bc != join_field]
        bigdf = bigdf.drop(reps, axis=1)
        bigdf = pd.merge(df, bigdf, on=join_field, how='left')
        
        # One last thing, log weight
        bigdf['logweight'] = np.log(bigdf['weight'])
        return bigdf
    
    def aggregationMaker(self, df, join_field):
        '''aggregate all cateogries by date'''      
        # Aggregate by all classes, framesizes, and muscle grades  
             # Use the arguments to determine how to group and aggregate 
        df = self.completeMaker(df, join_field)

        # Group the dataframe and recalculate
        group_list = ['locale','date']
        df['total_count'] = df.groupby(group_list)['count'].transform("sum")
        df['total_weight'] = df['weight'] * df['total_count']
        df['total_weight'] = df.groupby(group_list)['total_weight'].transform("sum")
        df['adj_price'] = df.groupby(group_list)['adj_price'].transform("mean")
        df['price'] = df.groupby(group_list)['price'].transform("mean")
        df['adj_revenue'] = df['adj_price']/100 * df['total_count'] * df['weight']
        df['adj_revenue'] = df.groupby(group_list)['adj_revenue'].transform("sum")
        df['revenue'] = df['price']/100 * df['total_count'] * df['weight']
        df['revenue'] = df.groupby(group_list)['revenue'].transform("sum")
        
        # now drop the grouping columns and duplicates
        df['weight'] = df['total_weight'] / df['total_count']
        df['logweight'] = np.log(df['weight'])
        df['count'] = df['total_count']
        df = df.drop(['class', 'framesize', 'grade'],
                      axis = 1).drop_duplicates()
        
        return df
    
######################## Define Raster Manipulation Class #####################
class RasterArrays:
    '''
    This class creates a series of Numpy arrays from a folder containing a
    series of rasters. With this you can retrieve a named list of arrays, a
    list of only arrays, and a few general statistics or fields. It also
    includes several sample methods that might be useful when manipulating or
    analysing gridded data.

        Initializing arguments:

            rasterpath(string) = directory containing series of rasters.
            navalue(numeric) = value used for NaN in raster, or user specified

        Attributes:

            namedlist (list) = [[filename, array],[filename, array]...]
            geometry (tuple) = (spatial geometry): (upper left coordinate,
                                          x-dimension pixel size,
                                          rotation,
                                          lower right coordinate,
                                          rotation,
                                          y-dimension pixel size)
            crs (string) = Coordinate Reference System in Well-Know Text Format
            arraylist (list) = [array, array...]
            minimumvalue (numeric)
            maximumvalue (numeric)
            averagevalues (Numpy array)

        Methods:

            standardizeArrays = Standardizes all values in arrays
            calculateCV = Calculates Coefficient of Variation
            generateHistogram = Generates histogram of all values in arrays
            toRaster = Writes a singular array to raster
            toRasters = Writes a list of arrays to rasters
    '''
    # Reduce memory use of dictionary attribute storage
    __slots__ = ('namedlist', 'geometry', 'crs', 'exceptions', 'arraylist',
                 'minimumvalue', 'maximumvalue', 'averagevalues', 'navalue')

    # Create initial values
    def __init__(self, rasterpath, navalue=-9999):
        [self.namedlist, self.geometry,
         self.crs, self.exceptions] = readRasters(rasterpath, navalue)
        self.arraylist = [a[1] for a in self.namedlist]
        self.minimumvalue = np.nanmin(self.arraylist)
        self.maximumvalue = np.nanmax(self.arraylist)
        self.averagevalues = np.nanmean(self.arraylist, axis=0)
        self.navalue = navalue

    # Establish methods
    def standardizeArrays(self):
        '''
        Min/Max standardization of array list, returns a named list
        '''
        print("Standardizing arrays...")
        mins = np.nanmin(self.arraylist)
        maxes = np.nanmax(self.arraylist)
        def singleArray(array, mins, maxes):
            '''
            calculates the standardized values of a single array
            '''
            newarray = (array - mins)/(maxes - mins)
            return newarray

        standardizedarrays = []
        for i in range(len(self.arraylist)):
            standardizedarrays.append([self.namedlist[i][0],
                                       singleArray(self.namedlist[i][1],
                                                   mins, maxes)])
        return standardizedarrays

    def calculateCV(self, standardized=True):
        '''
         A single array showing the distribution of coefficients of variation
             throughout the time period represented by the chosen rasters
        '''
        # Get list of arrays
        if standardized is True:
            numpyarrays = self.standardizeArrays()
        else:
            numpyarrays = self.namedlist

        # Get just the arrays from this
        numpylist = [a[1] for a in numpyarrays]

        # Simple Cellwise calculation of variance
        sds = np.nanstd(numpylist, axis=0)
        avs = np.nanmean(numpylist, axis=0)
        covs = sds/avs

        return covs

    def generateHistogram(self,
                          bins=1000,
                          title="Value Distribution",
                          xlimit=0,
                          savepath=''):
        '''
        Creates a histogram of the entire dataset for a quick view.

          bins = number of value bins
          title = optional title
          xlimit = x-axis cutoff value
          savepath = image file path with extension (.jpg, .png, etc.)
        '''
        print("Generating histogram...")
        # Get the unnamed list
        arrays = self.arraylist

        # Mask the array for the histogram (Makes this easier)
        arrays = np.ma.masked_invalid(arrays)

        # Get min and maximum values
        amin = np.min(arrays)
        if xlimit > 0:
            amax = xlimit
        else:
            amax = np.max(arrays)

        # Get the bin width, and the frequency of values within
        hists, bins = np.histogram(arrays, range=[amin, amax],
                                   bins=bins, normed=False)
        width = .65 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        # Make plotting optional
        plt.ioff()

        # Create Pyplot figure
        plt.figure(figsize=(8, 8))
        plt.bar(center, hists, align='center', width=width)
        title = (title + ":\nMinimum: " + str(round(amin, 2)) +
                 "\nMaximum: " + str(round(amax, 2)))
        plt.title(title, loc='center')

        # Optional write to image
        if len(savepath) > 0:
            print("Writing histogram to image...")
            savepath = os.path.normpath(savepath)
            if not os.path.exists(os.path.dirname(savepath)):
                os.mkdir(os.path.dirname(savepath))
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()


    def toRaster(self, array, savepath):
        '''
        Uses the geometry and crs of the rasterArrays class object to write a
            singular array as a GeoTiff.
        '''
        print("Writing numpy array to GeoTiff...")
        # Check that the Save Path exists
        savepath = os.path.normpath(savepath)
        if not os.path.exists(os.path.dirname(savepath)):
            os.mkdir(os.path.dirname(savepath))

        # Retrieve needed raster elements
        geometry = self.geometry
        crs = self.crs
        xpixels = array.shape[1]
        ypixels = array.shape[0]

        # This helps sometimes
        savepath = savepath.encode('utf-8')

        # Create file
        image = gdal.GetDriverByName("GTiff").Create(savepath,
                                                     xpixels,
                                                     ypixels,
                                                     1,
                                                     gdal.GDT_Float32)
        # Save raster and attributes to file
        image.SetGeoTransform(geometry)
        image.SetProjection(crs)
        image.GetRasterBand(1).WriteArray(array)
        image.GetRasterBand(1).SetNoDataValue(self.navalue)

    def toRasters(self, namedlist, savefolder):
        """
        namedlist (list) = [[name, array], [name, array], ...]
        savefolder (string) = target directory
        """
        # Create directory if needed
        print("Writing numpy arrays to GeoTiffs...")
        savefolder = os.path.normpath(savefolder)
        savefolder = os.path.join(savefolder, '')
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        # Get spatial reference information
        geometry = self.geometry
        crs = self.crs
        sample = namedlist[0][1]
        ypixels = sample.shape[0]
        xpixels = sample.shape[1]

        # Create file
        for array in tqdm(namedlist):
            image = gdal.GetDriverByName("GTiff").Create(savefolder+array[0] +
                                                         ".tif",
                                                         xpixels,
                                                         ypixels,
                                                         1,
                                                         gdal.GDT_Float32)
            image.SetGeoTransform(geometry)
            image.SetProjection(crs)
            image.GetRasterBand(1).WriteArray(array[1])
#            image.GetRasterBand(1).SetNoDataValue(self.navalue)