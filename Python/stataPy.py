# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:27:47 2018

@author: User


Playing around with STATA subroutines
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:15:26 2018

@author: User
"""

################################# Switching to/from Ubuntu VPS ###################
from sys import platform
import os

if platform == 'win32':
    homepath = "G:\\My Drive\\NOT THESIS\\Shrum-Williams\\"
    os.chdir(homepath + "project")
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
    homepath = "/home/ubuntu/"
    os.chdir(homepath+"project")
    from flask_caching import Cache # I have this one working on Linux but not Windows :)
    
############################# Extra Functions ############################
import glob
import os
import pandas as pd
import progress
from subprocess import Popen, PIPE, check_call
import sys
import threading

# The template model
dofile =r'G:\My Drive\NOT THESIS\Shrum-Williams\project\STATA\models\py_template.do'

# STATA subprocess function
def doStata(dofile, *params):
    cmd = ["C:\Program Files (x86)\Stata15\Stata-64","/e","do",dofile]#
    for param in params:
        cmd.append(param)
    process = subprocess.call(cmd,shell = False,stdout=PIPE,stderr = PIPE)

    
# Call on STATA
process = doStata(dofile,"xtreg logweight spring1 summer1", "500", "logweight")

# Read in model and results
model = pd.read_csv(r"STATA\outputs\py_temp\pyout.csv")
results = open(r"STATA\results\py_temp\py_result.tex").read()

# Convert text table to RMarkdown
with open(r"STATA\results\py_temp\py_result.txt","r") as f:
    latex_string = f.read() 

# Convert logfile to markdown
with open("py_template.log","r") as f:
    text = f.read()     


import re


def startEnd(phrase, text): 
    positions = [[match.start(), match.end()] for match in re.finditer(phrase,text)]
    return(positions)

positions = startEnd("------------------------------------------------------------------------------", text)
start = positions[0][0]
end = positions[len(positions)-1][1]
table = text[start:end]
table.replace("\n","")
