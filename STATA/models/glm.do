//Increase matrix size
set matsize 600

// set working directory
cd "G:\my drive\shrum-williams\project"

//clear everything out
clear

clear
import delimited "data\tables\noaa_500_standardized_balanced.csv"

//Filter by a month and set that month as a variable for titles and savepaths
egen id = group(polyid)
g time = ym(year,month)
xtset id time

// Run GLS for the likelihood of auction closing given weather
xtgee open t0 jan1 feb1 mar1 apr1 may1 jun1 jul1 aug1 sep1 oct1 nov1 dec1 i.time if continuous==1, family(binomial) link(probit) vce(robust) 




// The Panel Model //usmean
//xtreg weight price  t0 winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 i.time, fe vce(robust) 
//outreg2 using "G:\My Drive\THESIS\Market Project\STATA\FINAL\models\results\FINAL.tex", append tex() dec(4) title("\huge 500 km Regression Results - Two-Way Fixed Effects") cttop("\Large `index'") ///
//	addstat(Rho, e(rho),RMSE, e(rmse)) ///
//	keep(price usmean t0 winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2) 
// Store the model 
//estimate store fe

	
