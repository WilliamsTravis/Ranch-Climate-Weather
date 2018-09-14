# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:52:34 2018

@author: User
"""
os.chdir('C:/Users/User/github/Ranch-Climate-Weather/')
import matplotlib.pylab as plt
import datetime as dt
plt.style.use('seaborn-whitegrid')
from functions import *

# Create our date parser
dateparser = lambda dates: pd.datetime.strptime(dates, "%Y%m")

# Read in cows - note, dates field in originals are in zoo time from R, fix
cows = pd.read_csv("data/tables/rmw/timetester.csv",
                   parse_dates = ['date'],
                   index_col = 'date',
                   date_parser = dateparser)

# Get example location
ok = cows[cows['locale'] == "Cattleman's Livestock Commission Co., TX"]
ok = ok['weight']
fig = plt.figure()
plt.plot(ok)

# Get Moving average
mok = pd.rolling_mean(ok,12)
plt.plot(moving_avg)
plt.plot(mok)

# Difference between observations and moving average 
dok = ok - ok.shift(1)
plt.clf()
plt.plot(dok)
plt.plot(mok)
plt.plot(ok)

# Decomposition with statsmodel
from statsmodels.tsa.seasonal import seasonal_decompose

# Get whatever these are!
decomp = seasonal_decompose(ok)
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid

# Plot each
plt.clf()
plt.subplot(411)
plt.plot(ok)
plt.title("Original Cattle Weights - Oklahoma Stockyards")
plt.subplot(412)
plt.plot(trend)
plt.title("Trend")
plt.subplot(413)
plt.plot(seasonal)
plt.title("Seasonal")
plt.subplot(414)
plt.plot(residual)
plt.title("Residual")

