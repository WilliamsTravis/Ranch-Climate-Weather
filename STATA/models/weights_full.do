//Increase matrix size
set matsize 600

// set working directory
cd "G:\my drive\shrum-williams\project"

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #1 //////////////////////////////////////	
////////////////////////////////////////////////////////////////////////////////
//clear everything out
clear

//Load data
import delimited "data\tables\noaa_500_standardized.csv"

//Filter by a month and set that month as a variable for titles and savepaths
egen id = group(polyid)
g time = ym(year,month)
xtset id time

// Run model 
areg logweight t0 L12.logweight price t0 jan1 feb1 mar1 apr1 may1 jun1 jul1 aug1 sep1 oct1 nov1 dec1 jan2 feb2 mar2 apr2 may2 jun2 jul2 aug2 sep2 oct2 nov2 dec2 i.time, absorb(id)	
	ereturn list
	gen r2 = e(r2)
	gen ar2 = e(r2_a)
xtreg logweight t0 L12.logweight price t0 jan1 feb1 mar1 apr1 may1 jun1 jul1 aug1 sep1 oct1 nov1 dec1 jan2 feb2 mar2 apr2 may2 jun2 jul2 aug2 sep2 oct2 nov2 dec2 i.time, fe vce(robust) 
outreg2 using "STATA\results\weight_full.tex", append tex() dec(4) title("\huge 500 km Regression Results - Two-Way Fixed Effects") cttop("\Large Rainfall") ///
	addstat(Rho, e(rho),RMSE, e(rmse), Within R2, e(r2_w), Between R2, e(r2_b), Overall R2, e(r2_o), Full-Model R2, r2) ///
	keep(L12.logweight price t0 jan1 feb1 mar1 apr1 may1 jun1 jul1 aug1 sep1 oct1 nov1 dec1 jan2 feb2 mar2 apr2 may2 jun2 jul2 aug2 sep2 oct2 nov2 dec2) 
predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date dateid x y logweight predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\weight_full.csv", replace

// Store the model 
estimate store fe

	
