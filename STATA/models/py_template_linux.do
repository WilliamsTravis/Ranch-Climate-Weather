//clear everything out
clear all

//Includes function, up to comma
local which `1'
// Two files, one temp for aggregatation & one to stay the same

//Increase matrix size
set matsize 600

// import data
//import delimited `which'
import delimited "data/tables/rmw/noaa_500_standardized_central_all.csv"

// create formula 
local y `2'
local x1 `3'
local x2 `4'
local x3 `5'
local x4 `6'
local x5 `7'
local x6 `8'
local x7 `9'
local x8 `10'
local x9 `11'
local x10 `12'

// Set date and time 
egen id = group(polyid)
g time = ym(year,month)
xtset id time

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #1 //////////////////////////////////////	
////////////////////////////////////////////////////////////////////////////////
eststo: xtreg `y' `x1' `x2' `x3' `x4' `x5' `x6' `x7' `x8' `x9' `x10', fe vce(robust) 

esttab using "STATA\results\py_temp\py_result.csv", cells("b(fmt(4)) se(fmt(4)) p(fmt(4)star)") replace r2 plain

predict predictions, xb 
//predictions 
predict predictions_u, xbu
//predictions plus the fixed effect
predict residuals, residual
//residuals from standardized errors?
predict stnderror, stdp
//standardized error
predict u, u
predict e, e
export delimited locale date month year dateid x y `y' weight count price adj_price adj_revenue lat lon predictions predictions_u residuals stnderror u e using "STATA\outputs\py_temp\pyout.csv", replace

clear
