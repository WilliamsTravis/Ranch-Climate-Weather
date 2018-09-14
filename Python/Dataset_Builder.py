"""
Pull in cattle production information and associate with climate values. 
    It is time to consolidate all of this into python


Created on Mon Sep 10 14:32:10 2018

@author: User
"""
import datetime as dt
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
os.chdir('C:/Users/User/github/Ranch-Climate-Weather/')

from functions import *

# In[]
# Get Cattle Market Data
# Pull in AMS text files and convert to tables, potentially automate this step
    # ...
# Aggregate monthly using weight means or sums
    # ...
# Associate with coordinates
    # ...
# Save to file
    # ...
    
# For now, just pull these fields from our one complete dataset
cows = pd.read_csv("data/tables/rmw/ams_cattle_data.csv")

# In[]
# def class to deal with Zonal Statistics and Lag finding
class Climate_Market_Builder:
    '''
    this class will take in a path to a monthly  time series of rasters and a 
        path to a 
        shapefile with vectors. It will then associate each vector with the 
        mean raster values wihin for each time period in the raster folder.
        It will provide methods to build datasets of variables that 
        associate month names and seasons to lagged raster values for each 
        observation month extending two years back.
        
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
    def __init__(self,rasterpath,shapefilepath, loc_field, ym_field, 
                 ym_range = None):
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
        
    def zonalStats(self):
        '''
        Finds zonals stats of raster values within shapefiles
        '''
        print("Calculating mean raster values within shapefile elements...")
        shp = gpd.GeoDataFrame.from_file(self.shapefilepath)
        locations = shp[self.loc_field]
        stats = []
        for i in tqdm(range(len(self.rasterfiles)), position = 0):
            stat = zonal_stats(self.shapefilepath,
                               self.rasterfiles[i],
                               stats = "mean",
                               nodata = -9999)
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
#        bigdf = palmerBuilder.zonalStats()
        bigdf = self.zonalStats()
        print("Calculating lagged monthly values...")
        
        # Lets get a date index of the full range of possible dates
        dates = [[str(y) + str(m).zfill(2) for y in range(min(bigdf['year']), 
                         max(bigdf['year'])+1)] for m in range(1,13)]
        dates = np.sort([int(lst) for sublist in dates for lst in sublist])
        monthdict = {str(dates[i]):i for i in range(len(dates))}
        
        # Now we have the time period number for each date, if we add an id 
            # for each location we have the index needed for getting the months
        bigdf['datenum'] = bigdf['date'].map(monthdict)
        bigdf['datenumid'] = bigdf['datenum'].astype(str) + '_' + bigdf['polyid'].astype(str)
        
        # This step might be repeated later...
#        valuearray = np.array(bigdf[['datenumid','mean']])
#        ids = [a[0] for a in valuearray]
#        values = [a[1] for a in valuearray]
        
        # Each month lag has a particular predictable sequence
        # Below shows the number of months back to go to get to each month
            # from each month {obervation month : target month lags }
        lagnames = ['jan1','feb1','mar1','apr1','may1','jun1','jul1','aug1',
                    'sep1','oct1','nov1','dec1',
                    'jan2','feb2','mar2','apr2','may2','jun2','jul2','aug2',
                    'sep2','oct2','nov2','dec2']     
        lagdict = {1 : [12,11,10,9,8,7,6,5,4,3,2,1,
                        24,23,22,21,20,19,18,17,16,15,14,13],
                   2 : [1,12,11,10,9,8,7,6,5,4,3,2,
                        13,24,23,22,21,20,19,18,17,16,15,14],
                   3 : [2,1,12,11,10,9,8,7,6,5,4,3,
                        14,13,24,23,22,21,20,19,18,17,16,15],
                   4 : [3,2,1,12,11,10,9,8,7,6,5,4,
                        15,14,13,24,23,22,21,20,19,18,17,16],
                   5 : [4,3,2,1,12,11,10,9,8,7,6,5,
                        16,15,14,13,24,23,22,21,20,19,18,17],
                   6 : [5,4,3,2,1,12,11,10,9,8,7,6,
                        17,16,15,14,13,24,23,22,21,20,19,18],
                   7 : [6,5,4,3,2,1,12,11,10,9,8,7,
                        18,17,16,15,14,13,24,23,22,21,20,19],
                   8 : [7,6,5,4,3,2,1,12,11,10,9,8,
                        19,18,17,16,15,14,13,24,23,22,21,20],
                   9 : [8,7,6,5,4,3,2,1,12,11,10,9,
                        20,19,18,17,16,15,14,13,24,23,22,21],
                   10 : [9,8,7,6,5,4,3,2,1,12,11,10,
                         21,20,19,18,17,16,15,14,13,24,23,22],
                   11 : [10,9,8,7,6,5,4,3,2,1,12,11,
                         22,21,20,19,18,17,16,15,14,13,24,23],
                   12 : [11,10,9,8,7,6,5,4,3,2,1,12,
                         23,22,21,20,19,18,17,16,15,14,13,24]}
        
        # For each of 12 months, map the corresponding lags to this column 
        bigdf['monthlags'] = bigdf['month'].map(lagdict)
        
        # Split this column into spearate rows
        bigdf = bigdf.reset_index()
        bigdf[lagnames] =  pd.DataFrame(bigdf['monthlags'].values.tolist(), 
                                        index = bigdf.index)
        bigdf = bigdf.drop(columns = ['monthlags'])
        
        # Now find the date number of the target lagged value and join with the
            # location id
        for lag in lagnames:
            bigdf[lag] = bigdf['datenum'] - bigdf[lag]
            bigdf[lag] = bigdf[lag].astype(str) + '_' + bigdf['polyid'].astype(str)
            
        # This step a takes the most computation. The latest idea is to query 
            # a numpy array for the lagged value given a date number and id
        values = np.array(bigdf['mean'])
        ids = np.array(bigdf['datenumid']) 
        lagdict2 = dict(zip(ids,values))

        # Simple dictionary query
        def getLag(lag):
            lag = lagdict2.get(lag)
            return(lag)
        
        # Apply dictionary query to each cell in each target lag column
        for lag in tqdm(lagnames,position = 0):
            bigdf[lag] = bigdf[lag].map(getLag) 

#       Get regular lags: this can be done in the formula of any panel data
#       package, though...might as well it's quick
        print("Grabbing the shifted lag values real quick...")
        for i in range(0,25):
            bigdf['t' + str(i)] = bigdf.groupby(['polyid'])['mean'].shift(i)
        
        bigdf = bigdf.dropna()
        bigdf = bigdf.reset_index()
        return bigdf
    
    def seasonMaker(self):
        '''  Now use the lagged months to get lagged seasons '''
#        bigdf = self.monthMaker()
        bigdf = palmerBuilder.monthMaker()
        
        # make a dictionary of lagged months for each observation month
        dec = {'winter1' : ['t0'],
                    'spring1' : ['mar1','apr1','may1'],
                    'summer1' : ['jun1','jul1','aug1'],
                    'fall1'   : ['sep1','oct1','nov1'],
                    'winter2' : ['dec1','jan1','feb1'],
                    'spring2' : ['mar2','apr2','may2'],
                    'summer2' : ['jun2','jul2','aug2'],
                   'fall2'   : ['sep2','oct2','nov2']}
        jan = {'winter1' : ['dec1','t0'],
                   'spring1' : ['mar1','apr1','may1'],
                   'summer1' : ['jun1','jul1','aug1'],
                   'fall1'   : ['sep1','oct1','nov1'],
                   'winter2' : ['dec1','jan1','feb1'],
                   'spring2' : ['mar2','apr2','may2'],
                   'summer2' : ['jun2','jul2','aug2'],
                   'fall2'   : ['sep2','oct2','nov2']}
        feb = {'winter1' : ['dec1','jan1','t0'],
                    'spring1' : ['mar1','apr1','may1'],
                    'summer1' : ['jun1','jul1','aug1'],
                    'fall1'   : ['sep1','oct1','nov1'],
                    'winter2' : ['dec2','jan2','feb2'],
                    'spring2' : ['mar2','apr2','may2'],
                    'summer2' : ['jun2','jul2','aug2'],
                    'fall2'   : ['sep2','oct2','nov2']}
        mar = {'winter1' : ['dec1','jan1','feb1'],
                 'spring1' : ['t0'],
                 'summer1' : ['jun1','jul1','aug1'],
                 'fall1'   : ['sep1','oct1','nov1'],
                 'winter2' : ['dec2','jan2','feb2'],
                 'spring2' : ['mar1','apr1','may1'],
                 'summer2' : ['jun2','jul2','aug2'],
                 'fall2'   : ['sep2','oct2','nov2']}
        apr = {'winter1' : ['dec1','jan1','feb1'],
                 'spring1' : ['mar1','t0'],
                 'summer1' : ['jun1','jul1','aug1'],
                 'fall1'   : ['sep1','oct1','nov1'],
                 'winter2' : ['dec2','jan2','feb2'],
                 'spring2' : ['mar1','apr1','may1'],
                 'summer2' : ['jun2','jul2','aug2'],
                 'fall2'   : ['sep2','oct2','nov2']}
        may = {'winter1' : ['dec1','jan1','feb1'],
               'spring1' : ['mar1','apr1','t0'],
               'summer1' : ['jun1','jul1','aug1'],
               'fall1'   : ['sep1','oct1','nov1'],
               'winter2' : ['dec2','jan2','feb2'],
               'spring2' : ['mar1','apr1','may1'],
               'summer2' : ['jun2','jul2','aug2'],
               'fall2'   : ['sep2','oct2','nov2']}
        jun = {'winter1' : ['dec1','jan1','feb1'],
               'spring1' : ['mar1','apr1','may1'],
               'summer1' : ['t0'],
               'fall1'   : ['sep1','oct1','nov1'],
               'winter2' : ['dec2','jan2','feb2'],
               'spring2' : ['mar2','apr2','may2'],
               'summer2' : ['jun1','jul1','aug1'],
               'fall2'   : ['sep2','oct2','nov2']}
        jul = {'winter1' : ['dec1','jan1','feb1'],
                'spring1' : ['mar1','apr1','may1'],
                'summer1' : ['jun1','t0'],
                'fall1'   : ['sep1','oct1','nov1'],
                'winter2' : ['dec2','jan2','feb2'],
                'spring2' : ['mar2','apr2','may2'],
                'summer2' : ['jun1','jul1','aug1'],
                'fall2'   : ['sep2','oct2','nov2']}
        aug = {'winter1' : ['dec1','jan1','feb1'],
                  'spring1' : ['mar1','apr1','may1'],
                  'summer1' : ['jun1','jul1','t0'],
                  'fall1'   : ['sep1','oct1','nov1'],
                  'winter2' : ['dec2','jan2','feb2'],
                  'spring2' : ['mar2','apr2','may2'],
                  'summer2' : ['jun1','jul1','aug1'],
                  'fall2'   : ['sep2','oct2','nov2']}
        sep = {'winter1' : ['dec1','jan1','feb1'],
                     'spring1' : ['mar1','apr1','may1'],
                     'summer1' : ['jun1','jul1','aug1'],
                     'fall1'   : ['t0'],
                     'winter2' : ['dec2','jan2','feb2'],
                     'spring2' : ['mar2','apr2','may2'],
                     'summer2' : ['jun2','jul2','aug2'],
                     'fall2'   : ['sep1','oct1','nov1']}
        october = {'winter1' : ['dec1','jan1','feb1'],
                   'spring1' : ['mar1','apr1','may1'],
                   'summer1' : ['jun1','jul1','aug1'],
                   'fall1'   : ['sep','t0'],
                   'winter2' : ['dec2','jan2','feb2'],
                   'spring2' : ['mar2','apr2','may2'],
                   'summer2' : ['jun2','jul2','aug2'],
                   'fall2'   : ['sep1','oct1','nov1']}
        nov = {'winter1' : ['dec1','jan1','feb1'],
                    'spring1' : ['mar1','apr1','may1'],
                    'summer1' : ['jun1','jul1','aug1'],
                    'fall1'   : ['sep','oct','t0'],
                    'winter2' : ['dec2','jan2','feb2'],
                    'spring2' : ['mar2','apr2','may2'],
                    'summer2' : ['jun2','jul2','aug2'],
                    'fall2'   : ['sep1','oct1','nov1']}
        
        # Okay, now we know which lagged months to average for each season
        seasondict = {1:jan,2:feb,3:mar,4:apr,5:may,6:jun,
                      7:jul,8:aug,9:sep,10:october,11:nov,12:dec}
        
        seasonames = ['winter1','spring1','summer1','fall1',
                       'winter2','spring2','summer2','fall2']

        def rowSet(row):
            winter1lags  = seasondict.get(row['month']).get('winter1')
            spring1lags  = seasondict.get(row['month']).get('spring1')
            row['winter1'] = row[winter1lags].mean()
            row['spring1'] = row[spring1lags].mean()
        
# In[]          
# testing...
rasterpath = "C:/Users/User/github/data/rasters/albers/pdsisc/"

shapefilepath = ("C:/Users/User/github/Ranch-Climate-Weather/data/" +
                 "shapefiles/marketsheds/albers/central/fourradii/" +
                 "CentralBuffer_700km.shp")

palmerBuilder = Climate_Market_Builder(rasterpath, 
                                       "C:/Users/User/github/" +
                                       "Ranch-Climate-Weather/data/" +
                                       "shapefiles/marketsheds/albers/" +
                                       "central/fourradii/" +
                                       "CentralBuffer_700km.shp",
                                       "Locale",
                                       [-10,-4], 
                                       [200001,201704])
#bigdf = palmerBuilder.zonalStats()
bigdf = palmerBuilder.monthMaker()
