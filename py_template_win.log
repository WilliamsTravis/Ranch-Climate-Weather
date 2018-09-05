
  ___  ____  ____  ____  ____ (R)
 /__    /   ____/   /   ____/
___/   /   /___/   /   /___/   15.1   Copyright 1985-2017 StataCorp LLC
  Statistics/Data Analysis            StataCorp
                                      4905 Lakeway Drive
                                      College Station, Texas 77845 USA
                                      800-STATA-PC        http://www.stata.com
                                      979-696-4600        stata@stata.com
                                      979-696-4601 (fax)

Single-user Stata license expires 19 Mar 2019:
       Serial number:  301509290780
         Licensed to:  Travis Williams
                       University of Colorado Boulder

Notes:
      1.  Stata is running in batch mode.
      2.  Unicode is supported; see help unicode_advice.

. do STATA/models/py_template_win.do "logweight L12.logweight winter1 spring1 s
> ummer1 fall1 winter2 spring2 summer2 fall2 i.month" data/tables/rmw/noaa_500_
> standardized_central_all.csv logweight 

. //clear everything out
. clear all

. 
. //Set variables from subroutine calls
. local formula `1'

. //Includes function, up to comma
. 
. local which `2'

. // Two files, one temp for aggregatation & one to stay the same
. 
. //Increase matrix size
. set matsize 600

. 
. // import data
. import delimited `which'
(77 vars, 19,550 obs)

. //import delimited "data/tables/rmw/noaa_500_standardized_central_all.csv"
. local y `3'

. display y
103340.52

. 
. 
. 
. // Set date and time 
. egen id = group(polyid)

. g time = ym(year,month)

. xtset id time
       panel variable:  id (unbalanced)
        time variable:  time, 504 to 702, but with gaps
                delta:  1 unit

. 
. /////////////////////////////////////////////////////////////////////////////
> ///
> //////////////////////////////// Model #1 ///////////////////////////////////
> ///        
> /////////////////////////////////////////////////////////////////////////////
> ///
> eststo: xtreg `formula', fe vce(robust) 

Fixed-effects (within) regression               Number of obs     =     17,381
Group variable: id                              Number of groups  =        139

R-sq:                                           Obs per group:
     within  = 0.4337                                         min =          6
     between = 0.9961                                         avg =      125.0
     overall = 0.7845                                         max =        187

                                                F(20,138)         =     135.98
corr(u_i, Xb)  = 0.7905                         Prob > F          =     0.0000

                                   (Std. Err. adjusted for 139 clusters in id)
------------------------------------------------------------------------------
             |               Robust
   logweight |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
   logweight |
        L12. |   .5380161   .0187704    28.66   0.000     .5009014    .5751309
             |
     winter1 |   .0096819   .0070514     1.37   0.172    -.0042607    .0236246
     spring1 |  -.0038584   .0075114    -0.51   0.608    -.0187107    .0109938
     summer1 |   .0330296   .0093877     3.52   0.001     .0144673     .051592
       fall1 |  -.0174148   .0062607    -2.78   0.006    -.0297941   -.0050355
     winter2 |  -.0129323   .0078504    -1.65   0.102    -.0284548    .0025903
     spring2 |    .007765   .0080301     0.97   0.335    -.0081131     .023643
     summer2 |  -.0344714   .0077388    -4.45   0.000    -.0497733   -.0191695
       fall2 |  -.0048539   .0072309    -0.67   0.503    -.0191516    .0094438
             |
       month |
          2  |   .0056242   .0014167     3.97   0.000     .0028229    .0084255
          3  |   .0048343    .001683     2.87   0.005     .0015065    .0081622
          4  |   .0007778   .0022997     0.34   0.736    -.0037693    .0053249
          5  |   .0087864   .0028089     3.13   0.002     .0032323    .0143406
          6  |   .0196542   .0029022     6.77   0.000     .0139157    .0253928
          7  |   .0254068   .0035152     7.23   0.000     .0184562    .0323575
          8  |   .0242321    .003938     6.15   0.000     .0164455    .0320188
          9  |   .0124513   .0038497     3.23   0.002     .0048392    .0200634
         10  |  -.0235259   .0028707    -8.20   0.000    -.0292021   -.0178497
         11  |  -.0311657   .0024261   -12.85   0.000    -.0359628   -.0263686
         12  |  -.0223277    .001678   -13.31   0.000    -.0256456   -.0190098
             |
       _cons |   2.963747   .1205008    24.60   0.000      2.72548    3.202013
-------------+----------------------------------------------------------------
     sigma_u |  .06541695
     sigma_e |  .06209248
         rho |  .52605457   (fraction of variance due to u_i)
------------------------------------------------------------------------------
(est1 stored)

. 
. esttab using "STATA/results/py_temp/py_result.csv", cells("b(fmt(4)) se(fmt(4
> )) p(fmt(4)star)") replace r2 plain
(output written to STATA/results/py_temp/py_result.csv)

. 
. predict predictions, xb 
(2,169 missing values generated)

. //predictions 
. predict predictions_u, xbu
(2,169 missing values generated)

. //predictions plus the fixed effect
. predict residuals, residual
(2,169 missing values generated)

. //residuals from standardized errors?
. predict stnderror, stdp
(2,169 missing values generated)

. //standardized error
. predict u, u
(2,169 missing values generated)

. predict e, e
(2,169 missing values generated)

. export delimited locale date month year dateid x y `y' weight count price adj
> _price adj_revenue lat lon predictions predictions_u residuals stnderror u e 
> using "STATA/outputs/py_temp/pyout.csv", replace
file STATA/outputs/py_temp/pyout.csv saved

. 
. clear

. 
end of do-file
