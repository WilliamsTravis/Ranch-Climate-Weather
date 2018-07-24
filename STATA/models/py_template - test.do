

//set working directory
cd "G:/my drive/not thesis/shrum-williams/project"

//clear everything out
clear all

// create cluster
//parallel setclusters 1, f


//Increase matrix size
//set matsize 600

// import data
import delimited "data\tables\rmw\noaa_500_standardized_central_all.csv"

// Set date and time 
egen id = group(polyid)
g time = ym(year,month)
xtset id time

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #1 //////////////////////////////////////	
////////////////////////////////////////////////////////////////////////////////

//parallel 
eststo: xtreg price winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 i.time, fe vce(robust) 
 
esttab using "STATA\results\py_temp\py_result.csv", cells("b(fmt(3)) se(fmt(2)) p(fmt(3)star)") replace r2 plain

predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date month year dateid x y price weight count price adj_price adj_revenue lat lon predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\py_temp\pyout.csv", replace

clear
