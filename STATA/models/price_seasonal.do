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
// For the Full Model R2
areg price usmeans winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 i.time, absorb(id)	
	ereturn list
	gen r2 = e(r2)
	gen ar2 = e(r2_a)
xtreg price usmeans winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 i.time, fe vce(robust) 
outreg2 using "STATA\results\price\price_seasonal.tex", append tex() dec(4) title("\huge 500 km Regression Results - Two-Way Fixed Effects") cttop("\Large Rainfall") ///
	addstat(Rho, e(rho),RMSE, e(rmse), Within R2, e(r2_w), Between R2, e(r2_b), Overall R2, e(r2_o), Full-Model R2, r2) ///
	keep(usmeans winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 ) 
predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date dateid x y weight predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\price_seasonal.csv", replace

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #2 //////////////////////////////////////	
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
// For the Full Model R2
areg price usmeans winter1 fall1 winter2 fall2 i.time, absorb(id)	
	ereturn list
	gen r2 = e(r2)
	gen ar2 = e(r2_a)
xtreg price usmeans winter1 fall1 winter2 fall2 i.time, fe vce(robust) 
outreg2 using "STATA\results\price\price_seasonal.tex", append tex() dec(4) title("\huge 500 km Regression Results - Two-Way Fixed Effects") cttop("\Large Rainfall") ///
	addstat(Rho, e(rho),RMSE, e(rmse), Within R2, e(r2_w), Between R2, e(r2_b), Overall R2, e(r2_o), Full-Model R2, r2) ///
	keep(usmeans winter1 fall1 winter2 fall2 ) 
predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date dateid x y weight predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\price_winter_fall.csv", replace
	
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Model #3 //////////////////////////////////////	
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
// For the Full Model R2
areg price usmeans spring1 summer1 spring2 summer2 i.time, absorb(id)	
	ereturn list
	gen r2 = e(r2)
	gen ar2 = e(r2_a)
xtreg price usmeans spring1 summer1 spring2 summer2 i.time, fe vce(robust) 
outreg2 using "STATA\results\price\price_seasonal.tex", append tex() dec(4) title("\huge 500 km Regression Results - Two-Way Fixed Effects") cttop("\Large Rainfall") ///
	addstat(Rho, e(rho),RMSE, e(rmse), Within R2, e(r2_w), Between R2, e(r2_b), Overall R2, e(r2_o), Full-Model R2, r2) ///
	keep(usmeans spring1 summer1 spring2 summer2) 
predict predictions, xb //predictions 
predict predictions_u, xbu //predictions plus the fixed effect
predict residuals, residual //residuals from standardized errors?
predict stnderror, stdp //standardized error
predict u, u
predict e, e
export delimited locale date dateid x y weight predictions predictions_u residuals stnderror u e /// 
	using "STATA\outputs\price_spring_summer.csv", replace

	
