"
A new list of functions unique to the econometric analysis. 

The old list is a jumbled mess, but it may still have some things that we need. 

"
###############################################################################
##################### Packages We may need ####################################
###############################################################################
if(!require(tabularaster)){install.packages("tabularaster")}; require(tabularaster)
if(!require(colorRamps)){install.packages("colorRamps")}; require(colorRamps)
if(!require(cowplot)){install.packages("cowplot")}; require(cowplot)
if(!require(data.table)){install.packages("data.table")}; require(data.table)
if(!require(doParallel)){install.packages("doParallel")}; require(doParallel)
if(!require(dismo)){install.packages("dismo")}; require(dismo)
if(!require(dplyr)){install.packages("dplyr")}; require(dplyr)
if(!require(ggplot2)){install.packages("ggplot2")}; require(ggplot2)
if(!require(grid)){install.packages("grid")}; require(grid)
if(!require(gstat)){install.packages("gstat")}; require(gstat)
if(!require(Hmisc)){install.packages("Hmisc")}; require(Hmisc)
if(!require(kableExtra)){install.packages("kableExtra")}; require(kableExtra)
if(!require(knitr)){install.packages("knitr")}; require(knitr)
if(!require(latticeExtra)){install.packages("latticeExtra")}; require(latticeExtra)
if(!require(leaflet)){install.packages("leaflet")}; require(leaflet)
if(!require(lmtest)){install.packages("lmtest")}; require(lmtest)
if(!require(lmtest)){install.packages("lmtest")}; require(lmtest)
if(!require(magrittr)){install.packages("magrittr")}; require(magrittr)
if(!require(maps)){install.packages("maps")}; require(maps)
if(!require(maptools)){install.packages("maptools")}; require(maptools)
if(!require(parallel)){install.packages("parallel")}; require(parallel)
if(!require(pbapply)){install.packages("pbapply")}; require(pbapply)
if(!require(plm)){install.packages("plm")}; require(plm)
if(!require(progress)){install.packages("progress")}; require(progress)
if(!require(raster)){install.packages("raster")}; require(raster)
if(!require(RColorBrewer)){install.packages("RColorBrewer")}; require(RColorBrewer)
if(!require(rgdal)){install.packages("rgdal")}; require(rgdal)
if(!require(rgeos)){install.packages("rgeos")}; require(rgeos)
if(!require(scales)){install.packages("scales")}; require(scales)
# if(!require(sf)){install.packages("sf")}; require(sf)
if(!require(sp)){install.packages("sp")}; require(sp)
if(!require(splm)){install.packages("splm")}; require(splm)
if(!require(spdep)){install.packages("spdep")}; require(spdep)
if(!require(stringr)){install.packages("stringr")}; require(stringr)
if(!require(xts)){install.packages("xts")}; require(xts)
# if(!require(velox)){install.packages("velox")}; require(velox)
if(!require(zoo)){install.packages("zoo")}; require(zoo)
if(!require(pglm)){install.packages("pglm")}; require(pglm)

###############################################################################
##################### functions we may need ###################################
###############################################################################

################## Cell Number Switch ###############################
cellnumberSwitch = function(shapes,rasters){
  cells = tabularaster::cellnumbers(rasters,shapes)
  return(cells)
}

#################### Display Model Summaries] #######################
chartBuilder = function(summary,title,adjr2){
  require(stringr)
  options(knitr.table.format = "html") 

  formula = summary$formula
  
  coefficients = as.data.frame(summary$coefficients)
  
  r2 = summary$r.squared
  
  residuals = summary$residuals
  
  ftest = summary$fstatistic
  
  dfree = summary$df
  
  summary$model$mean = mean(summary$model$logweight)
  summary$model$diff = summary$model$mean - summary$model$logweight
  summary$model$sqdiff = summary$model$diff**2
  tss = sum(summary$model$sqdiff)
  o = text_spec("$*",'html')
  oo = text_spec("**",'html')
  ooo = text_spec("***",'html')
  coefficients$Significance = ifelse(coefficients$`Pr(>|t|)` < .001, '\\***',
                                     ifelse(coefficients$`Pr(>|t|)` < .01,'\\**',
                                            ifelse(coefficients$`Pr(>|t|)` < .05,'\\*',
                                                                                  "")))
  
  coefficients$`Pr(>|t|)` = format(coefficients$`Pr(>|t|)`,scientific = TRUE)
  
  pval = ifelse(ftest[[4]]>0,ftest[[4]],"< 2.2e-16")
  
  
  table = kable(coefficients,
                caption = title)%>%
    kable_styling(bootstrap_options = c("striped","hover"))%>%

    footnote(general_title = "Summary ",
             alphabet = "Signif. codes: \\*** 0.001, \\** 0.01, \\* 0.05",
             general = c(paste0("Projected Model Total Sum of Squares:   ",round(tss,2)),
                    paste0("Projected Model Residual Sum of Squares:   ",round(sum(summary$residuals**2),2)),
                    paste0('Projected Model Adj. R-Squared:   ',round(r2[2],4)),
                    paste0('Full Model Adj. R-Squared:   ',round(adjr2,4)),
                    paste0('F-Statistic:   ',round(ftest[[2]],2),
                           ' on ',ftest[[3]][1],' and ',ftest[[3]][2],' DF'),              
                    paste0('P-Value:  ',pval)),
             footnote_order = c('alphabet','general'),
             escape=F)
      # scroll_box(width = "800px", height = "500px")
  return(table)
}
######################### Create five shades of a color ###################
fiveShade = function(color){
  # To RGB
  rgb = readhex(file = textConnection(paste(color, collapse = "\n")),
                class = "RGB")
  
  # To HLS
  hls = as(rgb, "HLS")
  hls2 = hls
  hls3 = hls
  hls4 = hls
  hls5 = hls
  
  # Change Lightness
  hls@coords[1, "L"] = hls@coords[1, "L"] + .12 
  hls2@coords[1, "L"] = hls2@coords[1, "L"] + .09
  hls3@coords[1, "L"] = hls3@coords[1, "L"] + .05
  hls4@coords[1, "L"] = hls4@coords[1, "L"] + .03
  hls5@coords[1, "L"] = hls5@coords[1, "L"] 
  
  fiveshades = unlist(lapply(c(hls,hls2,hls3,hls4,hls5),hex))
  return(fiveshades)
}


#################### One-way clustered Errors #######################
errorCluster = function(model, group){
  " 
  A function that takes a fitted model and a group column and calculates the 
  coefficients and standard errors as clustered by that group. 
  
  (One-way clustering)

  from:
    Arai, M. (2015). Cluster-robust standard errors using R, (January 2009), 1-6.
  "
  # Adjust degrees of freedom to account for the number of groups
  n = length(unique(group))
  N = length(group)
  dfrmcw = model$df/ (model$df - (n - 1))
  dfc = (n/(n-1))*((N-1)/(N-model$rank))
  u  = apply(estfun(model),2, function(x) tapply(x, group, sum))
  vcovCL = dfc*sandwich(model, meat=crossprod(u)/N)*dfrmcw
  coeftest(model, vcovCL) 
  
  
} 
#################### Two-way clustered Errors #######################
errorCluster2 = function(model, group1, group2){
  " 
  A function that takes a fitted model and a group column and calculates the 
  coefficients and standard errors as clustered by a set of two groups. 
  
  (Two-way clustering)

  from:
    Arai, M. (2015). Cluster-robust standard errors using R, (January 2009), 1-6.
  "
  n = length(unique(group))
  N = length(group)
  dfrmcw = model$df/ (model$df - (n - 1))
  
  group12 = paste(group1,group2, sep="")
  M1  = length(unique(group1))
  M2  = length(unique(group2))
  M12 = length(unique(group12))
  N   = length(group1)
  K   = model$rank
  dfc1  = (M1/(M1-1))*((N-1)/(N-K))
  dfc2  = (M2/(M2-1))*((N-1)/(N-K))
  dfc12 = (M12/(M12-1))*((N-1)/(N-K))
  u1   = apply(estfun(model), 2, function(x) tapply(x, group1,  sum))
  u2   = apply(estfun(model), 2, function(x) tapply(x, group2,  sum))
  u12  = apply(estfun(model), 2, function(x) tapply(x, group12, sum))
  vc1   =  dfc1*sandwich(model, meat=crossprod(u1)/N)
  vc2   =  dfc2*sandwich(model, meat=crossprod(u2)/N)
  vc12  =  dfc12*sandwich(model, meat=crossprod(u12)/N)
  vcovMCL = (vc1 + vc2 - vc12)*dfrmcw
  coeftest(model, vcovMCL)
} 
############### Get lags with Month names #############################
getMonths = function(dfrm,cl){
  "
  Here we want to pickup the lag prior to an observation that corresponds 
  with a particular month of the year one and two years back. 
  
  column = source column from which to calculate a month
  monthnum = the number that corresponds with a month
  january == 0 btw
  lag = 1 or 2. For one or two years back
  "
  lags = vector(mode = 'list',length = 2)
  indx = 0
  dfrm%<>%na.omit()
  t0indx = match('t0',colnames(dfrm))
  t24indx = match('t24',colnames(dfrm))
  didx = match('dateid',colnames(dfrm))
  
  ################ function ##################
  fillIn = function(i,dfrm,ips){
    return(dfrm[[i,ips[i,2]]])
  }
  ############################################
  for(tlag in 1:2){
    # print("First lag...")
    # for(monthnum in 0:11){
    ################ function ##################
    monthNumbers = function(monthnum,dfrm,tlag){
      t0indx = match('t0',colnames(dfrm))
      t24indx = match('t24',colnames(dfrm))
      didx = match('dateid',colnames(dfrm))
      month = monthnum/12
      dfrm2 = cbind(as.numeric(dfrm$date),dfrm[(t0indx+1):t24indx])
      colnames(dfrm2)[1] = "date"
      
      # Sometimes the year will be the same, sometimes less
      prioryear = ifelse(as.integer(str_sub(as.character(dfrm2[[1]]),1,4)) + month < dfrm2$date,
                         as.integer(str_sub(as.character(dfrm2[[1]]),1,4)),
                         as.integer(str_sub(as.character(dfrm2[[1]]),1,4))-1)
      
      
      # The date will be the target month and the prioryear
      priordate = prioryear+month
      
      # What is the difference in months?
      difference = round((dfrm2[[1]] - priordate)*12,0)
      if(tlag == 2){
        difference = difference + 12
      }
      
      # Set the symbol for the lag
      colen = ncol(dfrm2)
      dfrm2[[colen+1]] = paste0('t',difference)
      
      # Set colnames
      colnames(dfrm2)[colen+1] = paste0(format(as.yearmon(2000+month),'%b'),'1')  
      if(tlag == 2){
        colnames(dfrm2)[colen+1] = paste0(format(as.yearmon(2000+month),'%b'),'2')  
      }
      
      # Match the lag symbol with the column name containining the variable value
      indexpositions = cbind(1:nrow(dfrm2),match(dfrm2[[colen+1]],names(dfrm2)))
      dfrm2 = data.table(dfrm2)


      #  Fill in appropriate values row by row
      cols = unlist(lapply(1:nrow(dfrm2),fillIn,dfrm2,indexpositions))
      return(cols)
    }
    ############################################
    
    # cl = makeCluster(detectCores())
    # registerDoParallel(cl)
    # clusterExport(cl,c('as.yearmon','cbind','data.table','fillIn','lapply','str_sub'),envir=environment())
    lags[[tlag]] = lapply(0:11,monthNumbers,dfrm,tlag)#,cl = cl
    # stopCluster(cl)
    # cl = NULL
  }
  
  lags[[1]] = as.data.frame(do.call('cbind',lags[[1]]))
  lags[[2]] = as.data.frame(do.call('cbind',lags[[2]]))
  
  columns = cbind(lags[[1]],lags[[2]])
  
  df2 = data.frame(columns)
  names(df2) =  c('jan1','feb1', 'mar1','apr1','may1',
                  'jun1','jul1','aug1','sep1','oct1','nov1',
                  'dec1','jan2','feb2','mar2','apr2','may2',
                  'jun2','jul2','aug2','sep2','oct2','nov2',
                  'dec2')
  
  dfrm3 = cbind(dfrm,df2)
  return(dfrm3)
}

############### Get seasons of lags #################################
# Test df
# setwd("g:/My Drive/Shrum-Williams/project/")
# df = readRDS("data/tables/noaa_500_standardized.rds")%>%na.omit()


getSeasons = function(df){
  # Dataframe month request template
  request_template = data.frame(winter1 = c("dec1","jan1","feb1"), 
                                spring1 = c("mar1","apr1","may1"),
                                summer1 = c("jun1","jul1","aug1"),
                                fall1   = c("sep1","oct1","nov1"),
                                winter2 = c("dec2","jan2","feb2"),
                                spring2 = c("mar2","apr2","may2"),
                                summer2 = c("jun2","jul2","aug2"),
                                fall2   = c("sep2","oct2","nov2"),
                                stringsAsFactors=FALSE)
  
  # month conversion
  numberGet = c(jan = 1,feb = 2, mar = 3, apr = 4, may = 5, jun = 6, jul = 7, aug = 8, sep = 9, oct = 10, nov = 11, dec = 12)
  monthGet = list("01"="jan","02"="feb","03"="mar","04"="apr","05"="may","06"="jun","07"="jul","08"="aug","09"="sep","10"="oct","11"="nov","12"="dec") 

  
  
  
  
  
  
  
  getRow = function(row){
    # example request
    # row = df[i,]
    monthnum = row[['month']] 
    monthabbr = monthGet[as.integer(monthnum)]
  
    # find columns 1 & 2 containing monthabbr
    request = request_template
    
    # only the season that the month is in will change, identify that season
    return = apply(request, 2, function(x) grep(paste0(monthabbr,"1"),x))
    col = which(return>0)
    
    ######### first year column ###############
    # get a copy of the column
    season1 = request[col]
    
    # Replace month of with "t0"
    request[col][request[col] == paste0(monthabbr,"1")] = "t0"
    
    # Replace months after observation month with NA?
    monthnumber = numberGet[paste0(monthabbr)]   # get number of observation month
    season1["month"] = gsub("[^a-z]","",season1[[1]]) # get just month abbr of first col
    season1["month"] = numberGet[season1$month]       # get number of each month in first col
    
    # Now, if it's winter december will always checkout as after the observation month, but its not so don't drop it ever
    season1["month"]= ifelse(season1[["month"]]==12,0,season1[["month"]])   # set 12 to 0
    positions = which(season1[["month"]] > monthnumber)              # Position of months to drop
    # Okay, now do it
    request[col][positions,] = NA #Any spot besides december that represents a month after the observation month is dropped
    
    
    ########## Second year ###################
    col2 = paste0(gsub("[^a-z]","",names(col)),"2")
    season2 = request[col2]

    # This one doesn't drop anything, but does add the first lag of the target month in where "t0" was placed before
    season2[positions,] = paste0(str_sub(season2[positions,],1,3),"1") # Using the positions found above, replace
    season2[which(request[col]=="t0"),] = paste0(str_sub(season2[which(request[col]=="t0"),],1,3),"1") # Using just the "t0" position, replace
    request[col2] = season2

    
    ############## Return the seasonal list of months! ############
    # So this will be unique for each row...we need to identify the column locations of each string, and combine them for each season
    request2 = apply(request, 2, function(x) which(names(row)%in%x)) # results in missing rows!
    if(class(request2)=='matrix'){
      request2=as.data.frame(request2)
      }

    pos = request2["winter1"]
    row[['winter1']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos)) 
    
    pos = request2["spring1"]
    row[['spring1']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["summer1"]
    row[['summer1']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["fall1"]
    row[['fall1']]   = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["winter2"]
    row[['winter2']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["spring2"]
    row[['spring2']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["summer2"]
    row[['summer2']] = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))

    pos = request2["fall2"]
    row[['fall2']]   = sum(as.numeric(row[unlist(pos)]))/length(unlist(pos))
    return(row)
  }
  
  df2 = pbapply(df,1,FUN = getRow)
  df3 = t(df2)
  df3 = data.frame(df3, stringsAsFactors = FALSE)
  
  # When it is converted to a matrix everything becomes a character string
  df3[1:3] = lapply(df3[1:3],as.character)
  df3[4:6] = lapply(df3[4:6],as.integer)
  df3[7:14] = lapply(df3[7:14], as.numeric)
  df3$region = as.integer(df3$region)
  df3$dateid = as.character(df3$dateid)
  df3$date = as.yearmon(df3$date)
  df3$polyid = as.integer(df3$polyid)
  df3[19:ncol(df3)] = lapply(df3[19:ncol(df3)], as.numeric)
  
  names(df3) = c(names(df),"winter1","spring1","summer1","fall1","winter2","spring2","summer2","fall2")
  return(df3)
  }

# df2 = getSeasons(df)

############### Group mean center ###################################
gcenter = function(dfrm,group) {
  "
  Group mean center for calculating fixed effects models. 

  Found this in:
    https://www.researchgate.net/publication/251965897_Cluster-robust_standard_errors_using_R 

  Group here is a column containing the group you want to mean center by. So it would like:
      dfrm$columname
  "
  variables = paste0(rep("c", ncol(dfrm)),  colnames(dfrm))
  dfrm2 = dfrm
  for (i in 1:ncol(dfrm)) {
    if(class(dfrm[[i]]) != "factor"){
      dfrm2[,i] = dfrm[,i] - ave(dfrm[,i], group,FUN=mean)
    }else{
      dfrm2[,i] = dfrm[,i]
    }
  }
  colnames(dfrm2) = variables
  return(cbind(dfrm,dfrm2))
} 

####################### Grow Get ####################################
# Get top 3 months by climate effect for each ecoregion and add these 
  # months to each observation
growGet = function(dfrmfull){
  dfrm = marketAgMonth3(dfrmfull)
  dfrm%<>%na.omit()
  ecoregions = unique(dfrm$ecoregion)
  get3 = function(i){
    df = dfrm[dfrm$ecoregion == ecoregions[[i]],]
    df$logweight = log(df$weight)
    
    idx1 = match("t0",colnames(df))
    idx2 = match("dec1",colnames(df))
    fm = as.formula(paste("logweight~count+price+hayindex+",paste0(colnames(df[idx1:idx2]),collapse = "+")))
    if(length(unique(df$locale))>1){
      pdf = pdata.frame(df,index = c("locale","date"))
      
      ################### Problem Zone #######################################
      fixedt = try(plm(fm,model = 'within',
                       effect = 'twoway', 
                       index=c("polyid","date"),
                       data = pdf),silent = T)
      if(class(fixedt) !=  "try-error"){
        fixed = plm(fm,model = 'within',
                    effect = 'twoway', 
                    index=c("locale","date"),
                    data = pdf)
      }else{
        pdf = df
        fixed = lm(fm,data = pdf)
      }
      ################### Problem Zone #######################################
    }else{
      pdf = df
      fixed = lm(fm,data = pdf)
    }
    
    # Now we want the three intervals whose effects add up to the most...consecutive?
    # Access the estimates
    # get the locations of the variables so we can change things without messing it up
    monindx1 = match("jan1",names(fixed$coefficients))
    monindx2 = match("dec1",names(fixed$coefficients))
    
    # Get the Coefficients for year 1 and 2
    effectsyr1 = fixed$coefficients[monindx1:monindx2]
    
    # Sort each and choose the top 3 strongest - absolute value or positive? Let's start with positive
    top3 = sort(effectsyr1,decreasing = T)[1:3]
    growmonths = c()
    for(m in 1:3){
      growmonths[m] =  gsub("[^[:alpha:]]","",names(top3)[m])
    }
    return = as.character(paste0(ecoregions[[i]]," ", paste(growmonths,collapse = ",")))
    return(return)
  }
  ecogrow = lapply(seq(length(ecoregions)),FUN = get3)
  ecogrow = unlist(ecogrow)
  ecogrow = data.frame(ecogrow)
  ecogrow$ecogrow = as.character(ecogrow$ecogrow)
  ecogrow = str_split_fixed(ecogrow$ecogrow, " ", 2)
  ecogrow = as.data.frame(ecogrow)
  names(ecogrow) = c("ecoregion","growmonths")
  ecogrow$ecoregion = as.numeric(as.character(ecogrow$ecoregion))
  ecogrow$growmonths = as.character(ecogrow$growmonths)
  dfrm2 = inner_join(ecogrow,dfrm,by = "ecoregion")
  
  # Get the numbers
  getCol = function(row,dfrm,which){
    idxs = vector(mode = 'list',length = 6)
    l = 0
    growmonths = dfrm$growmonths[row]
    
    for(c in c("1","2")){
      for(i in 1:3){
        l = l + 1
        idxs[[l]] = match(paste0(strsplit(growmonths,",")[[1]][i],c),names(dfrm))
      }
    }
    return = dfrm[row,idxs[[which]]]
    return(return)
  }
  
  # print("Getting first eco-optimized month for year one...")
  dfrm2$y1g1 =  unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,1))
  # print("Getting second eco-optimized month for year one...")
  dfrm2$y1g2 =unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,2))
  # print("Getting third eco-optimized month for year one...")
  dfrm2$y1g3= unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,3))
  # print("Getting first eco-optimized month for year two...")
  dfrm2$y2g1 =unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,4))
  # print("Getting second eco-optimized month for year two...")
  dfrm2$y2g2 = unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,5))
  # print("Getting third eco-optimized month for year two...")
  dfrm2$y2g3=unlist(lapply(seq(nrow(dfrm2)),getCol,dfrm2,6))
  
  
  return(dfrm2)
}

####################### IDW Weights Matrix ##########################
inverseDWM = function(spdf,d){
  # Inverse Distance Weights Matrix
  neighbors = dnearneigh(spdf,0,d)
  # neighbors = knn2nb(neighbors, row.names = NULL, sym = FALSE)
  dlist = nbdists(neighbors, spdf)
  idlist = lapply(dlist, function(x) 1/x)
  W = nb2listw(neighbors, glist=idlist, style="W",zero.policy=TRUE)
  
  # plot to check
  plot(usa,lty = 5, main = paste0('IDW Matrix of neighbors within ', d/1000, ' km'))
  plot(W,coordinates(spdf), add = T)
  plot(sites,cex= .75, pch = 20, add = T, col='orange')
  return(W)
}
################# Use Lags to get Month Values ######################
# Possible obselete
# makeMonths = function(dfrm){
#   cl = makeCluster(detectCores())
#   registerDoParallel(cl)
#   clusterExport(cl,c('str_sub','as.yearmon','data.table','pblapply'),envir=environment())
#   monthsback1 = pblapply(0:11,getLag,dfrm,1,cl=cl)
#   monthsback2 = pblapply(0:11,getLag,dfrm,2,cl=cl)
#   stopCluster(cl)
#   
#   monthsback = cbind(setDF(monthsback1),setDF(monthsback2))
#   dfrmnew = cbind(dfrm,monthsback)
#   colnames(dfrmnew)[(ncol(dfrm)+1):(ncol(dfrm)+24)] = c('jan1','feb1', 'mar1','apr1','may1',
#                                                         'jun1','jul1','aug1','sep1','oct1','nov1',
#                                                         'dec1','jan2','feb2','mar2','apr2','may2',
#                                                         'jun2','jul2','aug2','sep2','oct2','nov2',
#                                                         'dec2')
#   return(dfrmnew)
# }
####################### Aggregation by Lag ##########################
marketAgLag = function(dfrm){
  "
  This will take in a singular market data, assuming the naming scheme doesnt change, and aggregate the data into single monthly
  values by combining the different classes, grades, and framesizes with means or sums. 

  We may need to change some of the column names so watch out!
  "
  # xdex = which(tolower(colnames(dfrm)) == "x")
  # if(colnames(dfrm)[xdex] == "x"){ 
  #   colnames(dfrm)[c(xdex,xdex+1)] = c("X","Y")
  # }
  
  dfrm%<>%group_by(locale,
                      year,
                      month)%>%
  mutate(count = sum(monthlyheadcount), 
         weight = mean(monthlyweightavg), 
         price = mean(monthlypriceavg), 
         sales = sum(totalsales),
         revenue = sum(revenue))%>%
  ungroup()%>%
  dplyr::select(x, y, locale, year, month, date, dateid, count, 
                weight, price, sales, revenue, polyid, 
                t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,
                t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,
                t24,ecoregion)%>%
  distinct(x,y,locale,year,month, date, dateid, count, 
           weight, price, sales, revenue, polyid, 
           t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,
           t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,
           t24,ecoregion)
  
  return(dfrm)
}
####################### Aggregation by prior Month ##################
marketAgMonth = function(dfrm){
  "
  This will take in a singular market data, assuming the naming scheme doesnt change, and aggregate the data into single monthly
  values by combining the different classes, grades, and framesizes with means or sums. 
  
  We may need to change some of the column names so watch out!
  "
  names(dfrm) = tolower(names(dfrm))
  dfrm$grade = as.character(dfrm$grade)
  dfrm$frameSize = as.character(dfrm$framesize)
  dfrm$class = as.character(dfrm$class)

  dfrm2 = dfrm%>%group_by(locale,
                      year,
                      month)%>%
          mutate(count = sum(monthlyheadcount), 
                 weight = mean(monthlyweightavg), 
                 price = mean(monthlypriceavg),
                 sales = sum(totalsales),
                 revenue = sum(revenue))%>%
    ungroup()%>%
    dplyr::select(x, y, locale, year, month, date, dateid, count, 
                  weight, price, sales, revenue, polyid, 
                  t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
                  sep1,oct1,nov1,dec1,jan2,feb2,mar2,apr2,may2,
                  jun2,jul2,aug2,sep2,oct2,nov2,dec2)%>%
    distinct(x,y,locale,year,month, date, dateid, count, 
             weight, price, sales, revenue, polyid, 
             t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
             sep1,oct1,nov1,dec1,jan2,feb2,apr2,mar2,
             may2,jun2,jul2,aug2,sep2,oct2,nov2,dec2)

  return(dfrm2)
}

####################### MarketAgMonth2 ##############################
marketAgMonth2 = function(dfrm){
  "
  This does as above, but also balances the dataset and adds a detrended
    price field. We may also need to detrend cattle weights. 
  "
  dfrm2 = dfrm%>%group_by(locale,
                          year,
                          month)%>%
    mutate(count = sum(monthlyheadcount), 
           weight = mean(monthlyweightavg), 
           price = mean(monthlypriceavg), 
           sales = sum(totalsales),
           revenue = sum(revenue))%>%
    ungroup()%>%
    dplyr::select(x, y, locale, year, month, date, dateid, count, 
                  weight, price, sales, revenue, polyid, 
                  t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
                  sep1,oct1,nov1,dec1,jan2,feb2,mar2,apr2,may2,
                  jun2,jul2,aug2,sep2,oct2,nov2,dec2,ecoregion)%>%
    distinct(x,y,locale,year,month, date, dateid, count, 
             weight, price, sales, revenue, polyid, 
             t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
             sep1,oct1,nov1,dec1,jan2,feb2,apr2,mar2,
             may2,jun2,jul2,aug2,sep2,oct2,nov2,dec2,ecoregion)
  
  # Balance dataset.
  dfrm2 = pdata.frame(dfrm2, index = c("polyid","date"))
  dfrm3 = make.pbalanced(dfrm2, c('polyid','date'), balance.type = "shared.individuals")
  
  # Now detrend by location
  # This will have to be done for each individual location. Probably write a 
    # small function that generates a data frame for a location by polyid, 
    # creates a new field in the new df with the differenced price data, 
    # cbinds it to df[2:], dropping the first date, and then return. We can then rbind
    # the resulting dfs back together to recreate the original but with detrended
    # price. Yeah.
  
  # The single location function
  detrend = function(polyid){
    place = dfrm3[dfrm3$polyid == polyid,]
    dates = as.yearmon(place$date)
    price = ts(place[,c("date","price")], start=as.yearmon(dates[[1]]), frequency = 12)
    dprice = diff(price)  
    ts.plot(dprice, main = unique(place$locale),
            gpars = list(xlab = "Date",ylab = "Price ($ cwt)"))
    place = place[2:nrow(place),]
    place$dprice = as.data.frame(dprice)$price
    return(place)
  }
  
  ids = unique(dfrm3$polyid)
  detrended = lapply(ids, detrend)
  dfrm4 = do.call('rbind', detrended)
  return(dfrm4)
}

####################### Aggregation by prior Month ##################
marketAgMonth3 = function(dfrm){
  "
  This one adds a indexed values for some of the variables, along with a feed index
    to try and account for national price trends. 
  
  We may need to change some of the column names so watch out!
  "
  dfrm$grade = as.character(dfrm$grade)
  dfrm$frameSize = as.character(dfrm$framesize)
  dfrm$class = as.character(dfrm$class)
  
  dfrm%<>%group_by(locale,month,class)%>%
    mutate(baselineweight = mean(monthlyweightavg),# Consider a moving average for these two
           baselineprice = mean(monthlypriceavg),
           baselinecount = mean(monthlyheadcount),
           groupcount = sum(monthlyheadcount),
           weightindex = monthlyweightavg/baselineweight,
           priceindex = monthlypriceavg/baselineprice,
           countindex = monthlyheadcount/baselinecount)%>%
    ungroup()
  
  dfrm2 = dfrm%>%group_by(locale,
                          year,
                          month)%>%
    mutate(weightindex = sum(weightindex*groupcount)/sum(groupcount),
           priceindex = sum(priceindex*groupcount)/sum(groupcount),
           countindex = sum(countindex*groupcount)/sum(groupcount),
           count = sum(monthlyheadcount), 
           weight = mean(monthlyweightavg), 
           price = mean(monthlypriceavg),
           hayindex = mean(hayindex), 
           sales = sum(totalsales),
           revenue = sum(revenue))%>%
    ungroup()%>%
    dplyr::select(x, y, locale, year, month, date, dateid, count,countindex, 
                  weight,weightindex, price, priceindex, hayindex, sales, revenue, polyid, 
                  t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
                  sep1,oct1,nov1,dec1,jan2,feb2,mar2,apr2,may2,
                  jun2,jul2,aug2,sep2,oct2,nov2,dec2,ecoregion)%>%
    distinct(x,y,locale,year,month, date, dateid, count,countindex, 
             weight,weightindex, price, priceindex, hayindex, sales, revenue, polyid, 
             t0,jan1,feb1,mar1,apr1,may1,jun1,jul1,aug1,
             sep1,oct1,nov1,dec1,jan2,feb2,apr2,mar2,
             may2,jun2,jul2,aug2,sep2,oct2,nov2,dec2,ecoregion)
  
  return(dfrm2)
}

####################### Mode ########################################
Mode = function(x) {
  u = unique(x)
  return(u[which.max(tabulate(match(x, u)))])
}

####################### Moving Avg ##################################

######################### NAD to Albers #############################
nadAlbers = function(x,y){
  nad83 = CRS('+init=epsg:4269')
  albers = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
  df = data.frame(lon = x,lat = y)
  coordinates(df) = c("lon","lat")
  proj4string(df) = nad83
  df2 = spTransform(df,albers)
  print(paste0("Alber's coordinates are: ",round(df2@coords[1],4),", ",round(df2@coords[2],4)))
  
  }
######################### interpolation #############################
simpleKrig = function(spdataset, fieldnumber, usa, regionmask, variogramtype,model,title){
  "
  This will perform a simple krig to make a surface out of the econometric 
    model results. 


    spdataset = an sp dataset of residuals, coefficient, or whichever you want
    fieldnumber = column number the column with the desired variable
    usapath = path to contiguous USA shapefile
    plainsmaskpath = mask of the great plains
    southeastmaskpath = mask of the southeast
  "
  
  ###### Read in the USA shapefile and masks, and project ##################
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
  usa = spTransform(usa, srs)
  regionmask = spTransform(regionmask,srs)
  # spdataset = raster::intersect(spdataset,regionmask)
  
  ############ Plot title business ########################################
  if (str_sub(tolower(names(spdataset)[fieldnumber]),1,4) == 'ols_') {
    modeltype = "Ordinary Least Squares "
  } else if (str_sub(tolower(names(spdataset)[fieldnumber]),1,4) == 'sar_'){
    modeltype = "Spatial Autoregressive"
  }else if(str_sub(tolower(names(spdataset)[fieldnumber]),1,4) == 'sar2'){
    modeltype = "Two-Staged Least Squares Estimation"
  }
  
  medianresiduals = as.character(round(summary(model$residuals)[3],2))
  Title = paste0(title,"\n", unique(spdataset$date))
  ######### Create an empty grid ###########################################
  grid = as.data.frame(spsample(spdataset,bb = usa@bbox, "regular", n=50000))
  names(grid) = c("x", "y")
  coordinates(grid) = c("x", "y")
  gridded(grid) = TRUE  # Create SpatialPixel object
  fullgrid(grid) = TRUE  # Create SpatialGrid object
  proj4string(grid) = srs
  
  ####### Create a Variogram ###############################################
  # http://gsp.humboldt.edu/OLM/R/04_01_Variograms.html
  # This part can be done with or without transformations - how to parameterize this?
  # weightgram = variogram(log(spdataset[[fieldnumber]])~1, spdataset) # variogramST for space-time data
  weightgram = variogram(spdataset[[fieldnumber]]~1, spdataset) # variogramST for space-time data
  # gram = autofitVariogram(meanres~1+,spdataset)
  
  # Determining sill range and nugget by fitting it to the different model shapes
  # Gaussian is the only one that worked in this case
  # fit = fit.variogram(weightgram,vgm(model = 'Sph',psill = 5,range = 1000,nugget =0))
  
  # Plot to check - Good enough!
  plot(weightgram,fit, pch = 16, lwd = 3)
  
  ####### Create a Kriged Surface! #########################################
  kweight = krige(spdataset[[fieldnumber]]~1, spdataset, grid, model = fit)
  kweightr = raster(kweight)
  kweight1 = mask(kweightr,regionmask)
  
  
  # krig = autoKrige(meanres~1,plotsp, grid,miscFitOptions = list(min.np.bin = 10000, merge.small.bins = T))
  # automap::automapPlot(krig$krige_output[1])
  ####### Plot #############################################################
  colors = colorRampPalette(c("yellow", "red"))
  plot(kweight1,
       col = colors(30), 
       axes=FALSE,
       box=TRUE,
       legend=TRUE,
       legend.width=1,
       legend.shrink=0.75,
       main = paste0(Title,"\n",modeltype," Model Residuals"))
  plot(usa, add = T,lty = 4,lwd= .75)
  plot(regionmask,border = "brown",lwd = 2,add = T)
  text(x = 1100000, y =1200000, labels = paste0("Median Residual Value: ", medianresiduals), family = "serif")
}

######################### Spatialize ################################
# Maks by the same region
spatialize = function(dfrm){
  dfrm%<>%na.omit()
  names(dfrm) = tolower(names(dfrm))
  xindx = match("x",colnames(dfrm))
  yindx = match("y",colnames(dfrm))
  xy = dfrm[,c(xindx,yindx)]
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
  spdfrm = SpatialPointsDataFrame(coords = xy, data = dfrm, proj4string = CRS(srs))
  return(spdfrm)
}
######################### standardize ###############################
standard = function(x){(x-min(x))/(max(x)-min(x))}

########################## voronoi prices ###### ####################
voronoiPrice = function(dfrmpath, usa, regionmask, mnth, yr, grd, title, zlimit = c(0,1)){
  options(scipen = 999)
  # Spatial Reference System
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"  
  
  # Get dataframe and spatialize it
  dfrm = readRDS(dfrmpath)
  
  # Choose one of the market radii
  dfrm = dfrm[[1]]
  
  # Fix grade spaces
  dfrm$grade = str_trim(dfrm$grade, "left") 
  
  # Filter by grade chosen
  dfrm = dfrm[dfrm$grade == as.character(grd),]
  
  # Spatialize it
  dfrm = marketAgMonth(dfrm)
  dfrm%<>%na.omit()
  xy = dfrm[,c(1,2)]
  spdfrm = SpatialPointsDataFrame(coords = xy, data = dfrm, proj4string = CRS(srs))
  spdfrm = raster::intersect(spdfrm,regionmask)
  
  # Take a variety of average residual values for each location
  plot = spdfrm@data%>%filter(month == mnth, year == yr)%>%group_by(locale)%>%
            mutate(meanprice = mean(price))%>%
            distinct(x,y,locale,meanprice)
  
  # Turn it backinto a spatial object
  plotsp = SpatialPointsDataFrame(coords = plot[,c(1,2)], data = plot,proj4string = CRS(srs))
  
  # Make voronois polygons
  v = suppressWarnings(voronoi(plotsp, ext = extent(usa)))
  r = raster(v,ncol=180, nrow=180)
  vr = suppressWarnings(raster:::.p3r(v, r, 'meanprice', fun = 'first'))
  vr = raster::mask(vr,regionmask)
  
  # Plot everything
  colors = colorRampPalette(c("yellow","darkred"))
  plot(vr,
     main = paste0("Price By Market Area - ",mnth, "/", yr,"\n Grade: ",grd),
     sub = paste0("Mean Price: Total Study Period and Area: $", round(mean(plot$meanprice),2), ' per cwt'),
     col = colors(30), 
     axes=FALSE,
     box=TRUE,
     legend=TRUE,
     # zlim = zlimit,
     legend.width=1,
     legend.shrink=0.75)
  plot(usa,lty = 4,lwd= .75,add = T)
  plot(plotsp,add = T, pch = 16,col = 'black',cex = .9)
  plot(plotsp,add = T, pch = 16,col = 'yellow',cex = .3)
  plot(regionmask,border = "black",lwd = 4,add = T)
  points(-2150000,-1525000, pch = 16,col = 'black',cex = 2)
  points(-2150000,-1525000, pch = 16,col = 'yellow',cex = 1)
  text(x = -1400000, y = -1500000, labels = 'Auction Sites', cex = 1.25, family = "serif")

}

########################## Voronoi interpolation ###############################
voronoiInt = function(spdataset, usa, regionmask, title, zlimit = c(0,1)){
  options(scipen = 999)
  # Spatial Reference System
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"  
  
  # Take a variety of average residual values for each location
  plot = spdataset@data%>%mutate(totalmeanabsres = mean(abs(residuals)),
                                 totalmeanres = mean(residuals))
  
  # Get these for the plot - Absolute values because something's making the mean
  # very negative, atm
  totalmeanabsres = round(unique(plot$totalmeanabsres),2)
  # totalmeanres = formatC(unique(plot$totalmeanres), format = "e", digits = 2)
  totalmeanres = round(unique(plot$totalmeanres),2)
  
  
  # Make the dataframe simple and plottable
  plot%<>%group_by(locale)%>%
    mutate(meanres = mean(abs(residuals)))%>%
    distinct(X,Y,locale,meanres)
  
  # Turn it backinto a spatial object
  plotsp = SpatialPointsDataFrame(coords = plot[,c(1,2)], data = plot,proj4string = CRS(srs))
  
  # Make voronois polygons
  v = suppressWarnings(voronoi(plotsp, ext = extent(usa)))
  r = raster(v,ncol=180, nrow=180)
  vr = suppressWarnings(raster:::.p3r(v, r, 'meanres', fun = 'first'))
  vr = raster::mask(vr,regionmask)
  
  # Plot everything
  colors = colorRampPalette(c("yellow","darkred"))
  plot(vr,
       main = title,
       sub = paste0("Mean Absolute Residual Value: ", totalmeanabsres, ' (lbs) \n Mean Residual Values: ', totalmeanres,' (lbs)\n\n'),
       col = colors(30), 
       axes=FALSE,
       box=TRUE,
       legend=TRUE,
       zlim = zlimit,
       legend.width=1,
       legend.shrink=0.75,
       legend.args=list(text='Residual Absolute Value (lbs)', side=4, font=2, line=2.5, cex=0.8, pad = .8))
  plot(usa,lty = 4,lwd= .75,add = T)
  plot(plotsp,add = T, pch = 16,col = 'black',cex = .9)
  plot(plotsp,add = T, pch = 16,col = 'yellow',cex = .3)
  plot(regionmask,border = "black",lwd = 4,add = T)
  points(-2150000,-1525000, pch = 16,col = 'black',cex = 2)
  points(-2150000,-1525000, pch = 16,col = 'yellow',cex = 1)
  text(x = -1400000, y = -1500000, labels = 'Auction Sites', cex = 1.25, family = "serif")
  
}

######################### Voronoi int, no plot #################################
voronoiInt2 = function(spdataset, usa, regionmask){
  options(scipen = 999)
  # Spatial Reference System
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"  
  
  # Take a variety of average residual values for each location
  plot = spdataset@data%>%mutate(totalmeanresab = mean(abs(residuals)),
                                 totalmeanres = mean(residuals))
  
  # Get these for the plot - Absolute values because something's making the mean
  # very negative, atm
  totalmeanresab = round(unique(plot$totalmeanresab),2)
  # totalmeanres = formatC(unique(plot$totalmeanres), format = "e", digits = 2)
  totalmeanres = round(unique(plot$totalmeanres),2)
  
  
  # Make the dataframe simple and plottable
  plot%<>%group_by(locale)%>%
    mutate(meanres = mean(abs(residuals)))%>%
    distinct(locale,x,y,meanres)
  
  # Turn it backinto a spatial object
  plotsp = SpatialPointsDataFrame(coords = plot[,c(2,3)], data = plot,proj4string = CRS(srs))
  
  # Make voronois polygons
  v = suppressWarnings(voronoi(plotsp, ext = extent(usa)))
  r = raster(v,ncol=180, nrow=180)
  vr = suppressWarnings(raster:::.p3r(v, r, 'meanres', fun = 'first'))
  vr = raster::mask(vr,usa)
  
  # Return raster
  return(vr)
}

########################## splm interpolation #######################
splmKrig = function(spdataset,fieldnumber,usa,regionmask,variogramtype,model,title){
  '
    This will perform a simple krig to make a surface out of the spatial panel econometric
    model results.


  spdataset = an sp dataset of residuals, coefficient, or whichever you want
  fieldnumber = column number the column with the desired variable
  usapath = path to contiguous USA shapefile
  plainsmaskpath = mask of the great plains
  southeastmaskpath = mask of the southeast
  '

  srs = proj4string(spdataset)
  usa = spTransform(usa, srs)
  regionmask = spTransform(regionmask,srs)
  # spdataset = raster::intersect(spdataset,regionmask)

  grid = as.data.frame(spsample(spdataset,bb = usa@bbox, "regular", n=50000))
  names(grid) = c("x", "y")
  coordinates(grid) = c("x", "y")
  gridded(grid) = TRUE  # Create SpatialPixel object
  fullgrid(grid) = TRUE  # Create SpatialGrid object
  proj4string(grid) = srs

  # http://gsp.humboldt.edu/OLM/R/04_01_Variograms.html
  # This part can be done with or without transformations - how to parameterize this?
  weightgram = variogram((spdataset[[fieldnumber]]*100)~1, spdataset) # variogramST for space-time data

  # Determining sill range and nugget by fitting it to the different model shapes
  # Gaussian is the only one that worked in this case
  fit = fit.variogram(weightgram,vgm('Gau'))

  fit = fit.variogram(weightgram, vgm(psill=max(weightgram$gamma)*.9, model = "Pow", range=4.4, nugget = 3))


  # Plot to check - Good enough!
  plot(weightgram,fit, pch = 16, lwd = 3)

  kweight = krige(spdataset[[fieldnumber]]~1, spdataset, grid, model = fit)
  kweightr = raster(kweight)
  kweight1 = mask(kweightr,regionmask)

  # colors = colorRampPalette(c("yellow", "red"))
  # plot(kweight1,
  #      col = colors(30),
  #      axes=FALSE,
  #      box=TRUE,
  #      legend=TRUE,
  #      legend.width=1,
  #      legend.shrink=0.75,
  #      main = paste0("Model Residuals"))
  # plot(usa, add = T,lty = 4,lwd= .75)
  # plot(regionmask,border = "brown",lwd = 2,add = T)
  # text(x = 1100000, y =1200000, labels = paste0("Median Residual Value: ", medianresiduals), family = "serif")
}
################# Variable Correlation ############################
varCor = function(dfrm,sig){  
  "
  From:

  http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
  "

  require(corrplot)
  dfrm%<>%na.omit()
  df = dfrm[8:37]
  res = cor(df)
  # col<- colorRampPalette(c("blue","yellow"))(20)
  
  
  
  cor.mtest = function(mat, ...) {
    mat = as.matrix(mat)
    n = ncol(mat)
    p.mat = matrix(NA, n, n)
    diag(p.mat) = 0
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        tmp = cor.test(mat[, i], mat[, j], ...)
        p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
      }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
  }
  
  if(sig==FALSE){
    par(xpd=NA)
    par(oma = c(1,1,3,1))
    corrplot(res, type = "upper",
             tl.col = "black",
             col=c("blue","yellow"),
             bg = "lightblue",
             method = "circle",
             tl.srt = 45,
             main = "AMS Variable Correlations\n")
  }
  
  if(isTRUE(sig)){
    # matrix of the p-value of the correlation
    p.mat = cor.mtest(df)

    par(xpd=NA)
    par(oma = c(1,1,3,1))
    corrplot(res, type = "upper",
             tl.col = "black",
             col=c("blue", "yellow"),
             bg = "lightblue",
             method = "circle",
             tl.srt = 45,
             main = "AMS Variable Correlations\n",
             p.mat = p.mat,
             sig.level = .05,
             insig = "pch")
  }
}



