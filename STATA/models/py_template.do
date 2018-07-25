//clear everything out
clear all

//Set variables from subroutine calls
local formula `1' //Includes function, up to comma
local which `2' // Two files, one temp for aggregatation & one to stay the same

//Increase matrix size
set matsize 600

// import data
//import delimited `which'
import delimited "data/tables/rmw/noaa_500_standardized_central_all.csv"
local y `3'
display y


// Set date and time 
egen id = group(polyid)
g time = ym(year,month)
xtset id time

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #1 //////////////////////////////////////	
////////////////////////////////////////////////////////////////////////////////

eststo: `formula' , fe vce(robust) 

esttab using "STATA\results\py_temp\py_result.csv", cells("b(fmt(4)) se(fmt(4)) p(fmt(4)star)") replace r2 plain

predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date month year dateid x y `y' weight count price adj_price adj_revenue lat lon predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\py_temp\pyout.csv", replace

clear
