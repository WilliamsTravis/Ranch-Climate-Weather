import os 
import sys
import subprocess

dopath = "STATA/models/py_template_linux.do"
which_df = "data/tables/rmw/noaa_500_standardized_central_all.csv"
y = 'logweight'
x1 = 'spring1'
x2 = ' '
# Stata subprocess call - with call()
def doStata(dopath, *params):
    cmd = ["stata","-b","do",dopath]
    for param in params:
        cmd.append(param)
    return(subprocess.call(cmd))
print("function defined")
doStata(dopath, which_df, y,x1,x2)
print("Done.")
