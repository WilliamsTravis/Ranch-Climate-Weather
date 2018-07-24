 "
#~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
###~~~~~~~~~~~~~~~~~
Grid averaging tools for Drought and Cattle Sale Responses 
Earth Lab Risk Team 
Travis Williams

The goal is to take a series of shapefiles and find the average 
  values of a series of rasters within them. Then take those average 
  values and bind them to the cattle market data by location, (Locale)
  along with the previous monthly values 24 months back.


###~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~
"
setwd("G:/My Drive/NOT THESIS/Shrum-Williams/project/")
source('R/econometric_functions.R')

########################### Calling Big Mondo Function ###############################################
indexpaths= c("data/rasters/tifs/noaa_indexed/albers/")
indexnames = "noaa"
lagname = "t"
datarange = c(0,1)
ninclusion = FALSE 
central = TRUE

########################### Big Mondo Function #######################################################
bigRMWFunction = function(indexpath,bufferdists,lagname,datarange,ninclusion,central = TRUE){
  '
  This takes in a path to a set of climate rasters and outputs a dataset of market information of climate data averaged within 5 different sized radii around each cattle auction location.
      Steps this functions takes:
        1) Reads in the cattle market dataset built with:
            *usda_dataimport.R*
            *toTibble() in RMW_support_functions.R*
            *RMWDataPrep.R*
        2) Reads in and projects the market area shapefiles built in:
            *marketsheds.R*
        3) Averages the climate data within the marketsheds with built in function.
        4) Morphs the above into a dataset 
        5) Combines the result from above with the cattle market data.
        6) Outputs a list of datasets for each market area.        
                            
                            ~~~~ Variables~~~~~

        path    = full path data if it is not in the working folder 
                     (should be in "C:/Users/Travis/Desktop/data/PRISM/" somewhere)
        lagname = The letter for each of the time lagged climate columns (so "T" would result in "T1",
                    "T2", etc for temperature one month back and two months back, etc.)
        datarange = vector of min and max data values (Takes too long to extract from rasterstack)
  '
  
  ########################### Ok, go #######################################################
  start = proc.time()[3]
  
  ############################# load data #########################################################
  print("Loading data...")
  
  # This is the organized cattle market data
  if(central == TRUE){
    CompressedCows = readRDS("data/AMS/RData/AMS_monthly_central.rds")%>%na.omit() # Stored as CompressedCows
    buffertype = "Central"
  }else{
    CompressedCows = readRDS("data/AMS/RData/AMS_monthly.rds")%>%na.omit() # Stored as CompressedCows
    buffertype = ""
  }
  
  CompressedCows$date = paste0(as.character(CompressedCows$year),"-",
                               sprintf("%02d",CompressedCows$month))
  CompressedCows$date = as.yearmon(CompressedCows$date,format = "%Y-%m")
  
  
  # Date matching between climate and market data
  startyear = min(CompressedCows$year)
  lastyear = max(CompressedCows$year)
  startmonth = min(CompressedCows$month)
  if(str_sub(indexpath,-1)!= "/"){
    indexpath = paste0(indexpath,"/")
    }
  files = list.files(indexpath, pattern ="*tif",full.names = TRUE) # climate variable  
  names = list.files(indexpath,pattern = "*tif")
  year1 = as.numeric(str_sub(names[1],-10,-7))
  year2 = as.numeric(str_sub(names[length(names)],-10,-7))
  if(startyear > year1){
    year1 = startyear
    files = files[as.numeric(str_sub(files,-10,-7))>=year1]
  }
  if(lastyear < year2){
    year2 = lastyear
    files = files[as.numeric(str_sub(files,-10,-7))<=year2]
  }
  
  colname = str_sub(names[1],0,-12)
  interval1 = as.numeric(str_sub(files[1],-6,-5))
  interval2 = as.numeric(str_sub(files[length(files)],-6,-5))
  date1 = as.Date(paste0(year1,"-",interval1,"-1"),"%Y-%m-%d")
  date2 = as.Date(paste0(year2,"-",interval2,"-1"),"%Y-%m-%d")
  dates = seq.Date(date1,date2,by = "month")
  dates = unlist(lapply(dates,as.yearmon))
  intervaltype = max(unique(as.integer(str_sub(names,-6,-5))))
  
  # Create the list of climate rasters
  rasters = raster::stack(files)
  plot(rasters[[1]],zlim = datarange,col = topo.colors(35), legend = FALSE,main = toupper(colname), axes = F,box = F)
  
  ############################# get market areas ###################################################
  '
  This a new attempt to streamline this 
  '
  # Get info 
  print("Reading shapefiles, reprojecting and getting cell numbers...")
  marketpath = "data/shapefiles/buffers"
  srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0"
  
  # These are the market radii we are testing
  bufferlist = vector(mode = 'list',length = length(bufferdists))
  names(bufferlist) = bufferdists
  
  # They've been saved as shapefiles
  for(i in seq(length(bufferdists))){
    bufferlist[[i]] = readOGR(marketpath,paste0(buffertype,"Buffer_",bufferdists[i],"km"))
    bufferlist[[i]] = spTransform(bufferlist[[i]],srs)
    plot(rasters[[1]],zlim = datarange,col = topo.colors(35), 
         legend = FALSE,main = paste0(toupper(colname),"\n",bufferdists[[i]]," km"), axes = F,box = F)
    plot(bufferlist[[i]], lwd = 1.5,lty = 12,add = T)
  }
  
  ########################## Get Cellnumbers within each buffer ################
  # Next gets all of the raster cell ids that correspond to each market area shapefile, making the extractions faster
  # First, I need the iterating variable to be first in the argument list for lapply
  cellnumberSwitch = function(shapes,rasters){
    cells = tabularaster::cellnumbers(rasters,shapes)
    return(cells)
  }
  
  # Next, let's check if the file exists to possibly save time
  cellrange = paste0(min(bufferdists),"_",max(bufferdists))
  if(!file.exists(paste0("data/tables/",buffertype,"cellslist",cellrange,".rds"))){
    print("Getting cell numbers for each radius")
    cl = makeCluster(detectCores())
    registerDoParallel(cl)
    clusterExport(cl,c('cellnumbers'),envir=environment())
    cellslist = pblapply(bufferlist,cellnumberSwitch, rasters, cl = cl)
    stopCluster(cl)
    saveRDS(cellslist,file = paste0("data/tables/",buffertype,"cellslist",cellrange,".rds"))
  }else{
    cellslist = readRDS(paste0("data/tables/",buffertype,"cellslist",cellrange,".rds"))
  }
  ########################## function 1 ########################################
  # Experimental new functions for speed
  # This uses the prefab cell ids to extract raster values within each market areas, includes circles with nans
  zonalMeans1 = function(iterate, rasters,cells){
    means = cells%>%mutate(v = raster::extract(rasters[[iterate]],cell_))%>%
      group_by(object_)%>%
      na.omit()%>%
      summarise(meanvalue = mean(v))  
    return(means)
  }
  
  # As above, but throws out observations with any nan values
  zonalMeans2 = function(iterate, rasters,cells){
    means = cells%>%dplyr::mutate(v = raster::extract(rasters[[iterate]],cell_))%>%
      group_by(object_)%>%
      summarise(meanvalue = mean(v))  
    return(means)
  }
  
  
  # This builds the climate dataset with one month worth of climate data for each observation
  gridToTable = function(market,km,rasters,year1,year2,gridname,cells,ninclusion){
    "
    Author: Travis
    
    This takes in a set of market polygons, a weather dataset (preconditioned), and a year range.  
    It uses the gridStackAvg function to build a tibble with averages of weather values within each          
    market polygon and out puts the value for each month within the specified years into a tibble.
    
    markets = shapefile
    weather = raster
    year1, year2 = year range
    gridname = 'string' (column name for grid output)
    
    This is a good location to experiment with clusters, I think.
    "    
    
    # Second get climate values 
    if(ninclusion == TRUE){
      print(paste0("Getting climate means for the ",km," km radius..."))
      cl = makeCluster(detectCores())
      registerDoParallel(cl)
      clusterExport(cl,c('%>%','extract','mutate','group_by','group_by','summarise'),envir=environment())
      marketmeans = pblapply(seq(nlayers(rasters)), zonalMeans1, rasters, cells, cl = cl)
      stopCluster(cl)
    }else{
      print(paste0("Getting climate means for the ",km," km radius..."))
      cl = makeCluster(detectCores())
      registerDoParallel(cl)
      clusterExport(cl,c('%>%','extract','mutate','group_by','group_by','summarise'),envir=environment())
      marketmeans = pblapply(seq(nlayers(rasters)), zonalMeans2, rasters, cells, cl = cl)
      stopCluster(cl)
    }
    
    
    # Iterate through and build the dataset chunk by chunk
    pb = progress_bar$new(total = year2 - year1) 
    pb$tick(0)
    cows = data.frame(matrix(ncol = 5))
    names(cows) = c("year","month","polyid","locale",gridname)
    start = 1
    N = nrow(market)
    end = N
    ########################## Problem Zone ##########################
    # The polygon IDs change when there are fewer polygons...duh
    # How to associate these with the original polygon IDs?
    
    # Generating Unique IDs    
    names(market) = c("locale","region")
    IDs = as.numeric(sapply(slot(market, "polygons"), function(x) slot(x, "ID")))

    
    #################### Problem Zone ################################
    print(paste0("Building dataset for the ",km,"km market area..."))
    r=1
    intervaltype = 12
    for (y in year1:year2){
      pb$tick()
      if(y == year2){intervaltype = interval2}
      for(i in 1:intervaltype){    
        cows[start:end,1]=y
        cows[start:end,2]=i
        cows[start:end,3]=IDs
        cows[start:end,4]=as.character(market$locale)
        cows[start:end,5]=marketmeans[[r]][,2]
        r = r+1 
        start= start + N
        end = end + N
      }
    }
    # polyids = unique(readRDS("data/Market Project/AMS/polyids.rds"))# original pairings
    # marketnames = market@data$Locale
    # IDs = data.frame(polyid = IDs,locale = marketnames)
    # cows = full_join(cows,IDs,by = "locale")
    # cows$PolyID = cows$PolyID.x
    # cows$PolyID.x = NULL
    cows%<>%na.omit
    return(cows)
  }
    
  ############################  Get Climate Data ##############################
  # Call Above function
  print("Averaging climate values within each market area...")  
  climates = vector(mode = 'list',length = length(bufferlist))
  
  for(i in seq(length(climates))){
    climates[[i]] = gridToTable(bufferlist[[i]],names(bufferlist)[[i]],rasters,year1,year2,colname,cellslist[[i]],ninclusion)
  }
  
  ############################ function 3  #####################################
  print("Creating lagged monthly climate observations...")
  setDateID = function(climatedf){
    '
    Create a Date-ID field for market averaged weather dataset
    '
    climatedf = climatedf%>%dplyr::mutate(date = as.yearmon(paste(year,month,sep = "-")))
    climatedf$dateid = paste0(climatedf$date,"_",climatedf$polyid)
    climatedf%<>%dplyr::select(year,month,polyid,locale,colname,date,dateid)
    names(climatedf) = c("year","month","polyid","locale",colname,"date","dateid")
    return(climatedf)
  }
  
  climatewids = pblapply(climates,setDateID)
  ############################ function 4  #####################################
  print("Organizing numbers...")
  longToTime = function(x){
    "
    This Creates a list of Dataframes for each radius from the long form data into 
    a time series format. It creates a dataframe with 24 extra columns, and in each 
    one the appropriate DateID for the number of corresponding months back. This is 
    then used to match with the precip info, which also has a DateID now.
  
    "
    empty = x%>%dplyr::select(date,polyid,dateid)%>%filter(date>=min(date)+2)
    
    # This fills the DateIDs in    
    #empty[[colname]] = empty$DateID
    pb = progress_bar$new(total = 24)      
    for(i in 0:24){
      empty[[paste0(lagname,as.character(i))]] = paste0(empty$date - i/12,"_",empty$polyid)
      pb$tick()
    }
    return(empty)
  }
  empty = longToTime(climatewids[[i]])
  
  IDs = vector(mode = 'list',length = length(climatewids))
  
  for(i in seq(length(IDs))){
    IDs[[i]] = empty
  }
  ############################ function 5 ######################################
  # This function will, hopefully, match each DateID with the corresponding precip value and replace
  # each with the precip value.
  getClimate = function(x,data){x = data[[5]][which(data[[7]]%in%x)]}
  for(y in seq(length(IDs))){
      for(i in 4:28){IDs[[y]][i] = lapply(empty[i],FUN = getClimate,data = climatewids[[y]])}
  }
  
  ############################ function 6  #####################################
  # Now we attach the timeseries grid data to the timeseries cattle data
  # So far I can only apply this to one column at a time, but it is so fast it doesn't matter
  # We can make this generic for either  
  polyids = as.data.frame(climates[[1]]%>%select(locale, polyid)%>%unique())
  CompressedCows = left_join(CompressedCows,polyids,by = "locale")
  CompressedCows$dateid = paste0(CompressedCows$date,"_",CompressedCows$polyid)
  
  print("Combining market and climate data...")
  joinData = function(data){
    data%<>%select(-date,-polyid)
    DataSet = full_join(CompressedCows,data,by = "dateid")%>%na.omit()
    return(DataSet)
  }
  #################################### Filling everything in ############################
    # Ok filling everything in!
  RMWs = vector(mode = 'list',length = length(IDs))
  for(i in seq(length(RMWs))){
    RMWs[[i]] = joinData(IDs[[i]])
  }
  
  ############################ function 7  Ecoregions! ##################################
  # Attach majority Level III ecoregions - This is really quite slow, the cell numbers 
    # from before don't work here, it has a different resolution. Also, the amount of time
    # it would take to create new cell ids wouldn't compensate for the time saved because
    # there is only one extraction for each radius. What would be fastest is to create the
    # cell numbers in advance, save them to file, and then read them in here to skip that step...
    # Seeing how long this takes, I really need to do that...
  # This uses the prefab cell ids to extract raster values within each market areas, includes circles with nans
  # zonalMode1 = function(raster,cells, market){
  #     locales = as.character(market@data$Locale)
  #     locales = tbl_df(locales)
  #     locales$object_ = as.numeric(row.names(locales))
  #     modes = cells%>%mutate(v = raster::extract(raster,cell_))%>%
  #                     group_by(object_)%>%
  #                     na.omit()%>%
  #                     summarise(modevalue = Mode(v))
  #     modes = full_join(modes,locales,by = c("object_"))
  #     modes$object_ = NULL
  #     colnames(modes) = c("ecoregion","Locale")
  #   return(modes)
  # }
  # 
  # print("Attaching modal Ecoregion Categories...")
  # ecoregions = raster('data/rasters/albers/ecoregionsIII_125.tif')
  # 
  # # When tabularaster decides to work again we can speed this part up with cell numbers
  #   # However, it only seem to want to work sometimes, atm.
  # if(!file.exists(paste0("data/Market Project/ecocellslist",cellrange,".rds"))){
  #   print("Getting cell numbers for ecoregion raster")
  #   cl = makeCluster(detectCores())
  #   registerDoParallel(cl)
  #   clusterExport(cl,c('cellnumbers'),envir=environment())
  #   ecocellslist = pblapply(bufferlist,cellnumberSwitch, ecoregions, cl = cl)
  #   stopCluster(cl)
  #   saveRDS(ecocellslist,paste0("data/Market Project/ecocellslist",cellrange,".rds"))
  # }else{
  #   ecocellslist = readRDS(paste0("data/Market Project/ecocellslist",cellrange,".rds"))
  #   }
  # 
  # 
  # ecoRegion = function(ecoregions,cells,market,RMWdf,distance){
  #   majority = zonalMode1(ecoregions,cells,market)
  #   polyids = unique(readRDS("data/Market Project/AMS/polyids.rds"))# original pairings
  #   majority = full_join(majority, polyids,by = "Locale" )%>%na.omit()
  #   colnames(majority)[3] = "PolyID"
  #   df = full_join(RMWdf,majority,by = 'PolyID',copy = T)%>%na.omit()
  #   colnames(df)[1] = "Locale"
  #   plot(ecoregions, main = paste0("EPA Ecoregions Level III - ", distance, " km radius"))
  #   plot(market, add = T)
  #   return(df)
  # }
  # 
  # pb = progress_bar$new(total = length(bufferlist))
  # for(i in seq(length(RMWs))){
  #   RMWs[[i]] = ecoRegion(ecoregions,ecocellslist[[i]],bufferlist[[i]],RMWs[[i]], bufferdists[[i]])
  #   pb$tick()
  #   }
  # 
  # #################################### Renaming Columns #################################
  # for(i in seq(length(RMWs))){
  #   names(RMWs[[i]]) = tolower(names(RMWs[[i]]))
  # }
  
  ####################### Associate Lags with Names of prior months  ###################
  print("Associating Months with lags...")
  lags = pblapply(RMWs,getMonths)
  
  ####################### Associate Lags with prior seasons  ###########################
  print("Associating seasons with months...")
  lags = pblapply(lags,getSeasons)
  
  ####################### Get Ecoregion optimized months ###############################
  # # print("Finding the months with strongest effects...")
  # Out = pblapply(lags,marketAgMonth)
  Out = lags
  ####################### Attach National Drought Index ################################
  usa = readOGR("data/shapefiles/USACoutline.shp")
  usa = spTransform(usa,srs)
  
  usmeans = vector(mode = 'list',length = nlayers(rasters))
  pb = progress_bar$new(total = nlayers(rasters))
  for(i in seq(nlayers(rasters))){
    usmeans[[i]] = as.numeric(data.frame(usmean = cellStats(rasters[[i]],'mean')))
    pb$tick()
  }
  
  usmeans2 = cbind(dates,unlist(usmeans))
  usmeans3 = data.frame(usmeans2)
  colnames(usmeans3)[2] = "usmeans"
  usmeans3$date = as.yearmon(usmeans3$date)
  for(r in seq(length(Out))){
    Out[[r]]$date = as.yearmon(Out[[r]]$date)
    Out[[r]]$logweight = log(Out[[r]]$weight)
    
  } 
  
  for(r in seq(length(Out))){
    Out[[r]] = full_join(Out[[r]], usmeans3, by = "date")%>%na.omit()
  }
  
  ############################### Go ahead and the Lagged Weight #########################
  # Because I am rebelling against any additional STATA steps and would like not to use 
    # it ever again after this master's thesis

  
  
  print("Done.")
  end =  proc.time()[3] - start
  print(paste0("Processing Time: ",round(end/60,2)," minutes"))
  return(Out)
  }
  

################################## Function call ####################################
bufferdists = c(100,300,500,700)
lagname = "t"
datarange = c(0,1)
ninclusion = TRUE
central = TRUE

# Save it all into the main rds file
for(i in seq(length(indexpaths))){
  df = bigRMWFunction(indexpaths[i],bufferdists,lagname,datarange,ninclusion,central)
  savepath = paste0("data/tables/rmw/",indexnames[i],"_",bufferdists[i],"_standardized_central.rds")
  saveRDS(df,file = savepath)
}

df = readRDS("data/tables/rmw/noaa_standardized_central.rds")

# Save each radius separately
library(tidyr)
for(i in seq(length(df))){
  df2 = df[[i]]
  df2$grade = as.character(df2$grade)
  split = function(string){
    if(grepl("and",string)){
      framesize = str_sub(string, 1, gregexpr("_",string)[[1]][3]-1)
      grade = str_sub(string,gregexpr("_",string)[[1]][3]+1, nchar(string))
    }else{
      framesize = str_sub(string, 1, gregexpr("_",string)[[1]][1]-1)
      grade = str_sub(string,gregexpr("_",string)[[1]][1]+1, nchar(string))
    }
    return(paste0(framesize,",",grade))
  }
    
  df2[[3]] = pblapply(df2[[3]],FUN = split)
  df2 = separate(data = df2, col = "grade", into = c("framesize", "grade"), sep = ",")
  df[[i]] = df2
  
  write.csv(df[[i]], paste0("data/tables/rmw/noaa_",bufferdists[i],"_standardized_central.csv"))
}

# Save each in STATA format
library(haven)
for(i in seq(length(df))){ 
  x = read.csv(paste0("data/tables/rmw/noaa_",bufferdists[i],"_standardized_central_all.csv"))
  write_dta(x,paste0("data/tables/rmw/noaa_",bufferdists[i],"_standardized_central_all.dta"),version = 15)
  }

# Save each with each class and grade aggregated
for(i in seq(length(df))){
  df[[i]] = df[[i]]%>%group_by(locale,month,year)%>%
              mutate(totalweight = weight*count,
                     count = sum(count),
                     weight = sum(totalweight/count),
                     logweight = log(weight),
                     price = mean(price),
                     adj_price = mean(adj_price),
                     adj_revenue = (weight/100)*price*count)%>%
              select(-class,-grade,-totalweight)%>%
              distinct_(.dots = names(.))
  write.csv(df[[i]], paste0("data/tables/rmw/noaa_",bufferdists[i],"_standardized_central_all.csv"))
  
}
 
