"""
Created on Sun Jul 15 11:15:26 2018

@author: User
"""

################################# Switching to/from Ubuntu VPS ###################
from sys import platform
import os

if platform == 'win32':
    homepath = "C:/Users/User/github/Ranch-Climate-Weather"
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
    homepath = "/Ranch-Climate-Weather/"
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
###########################################################################
############## Indexing by Baseline  ######################################
###########################################################################  
def index(indexlist,baselinestartyear,baselinendyear):
    '''
        This will find the indexed value to the monthly average. 
    '''        
    warnings.filterwarnings("ignore")
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
          
