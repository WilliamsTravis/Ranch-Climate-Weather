import os 
import sys
import subprocess

dopath = "STATA/models/py_template.do"
which_df = "data/tables/rmw/noaa_500_standardized_central_all.csv"
y = 'logweight'
formula = 'logweight spring summer1'
# Stata subprocess call - with call()
def doStata(dopath, *params):
    cmd = ["stata","-b","do",dopath]
    for param in params:
        cmd.append(param)
    return(subprocess.call(cmd))
print("function defined")
doStata(dopath,formula, which_df, y)
print("Done.")
