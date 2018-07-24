"
Here we flagging auctions that don't close between whenever the start and finish dates are.
"
setwd("G:/My Drive/shrum-williams/project")
source("R/econometric_functions.R")


# Read in df
df0 = read.csv("data/tables/precip_500km_standardized.csv", stringsAsFactors = FALSE)
df = df0[1:41]
df$date = as.yearmon(df$date)

# What is the most efficient way to check for continuously open locations
  # Well, all we need is the date field for each polyid

# Unique locaations
polyids = unique(df$polyid)

# Check each location
pb = progress_bar$new(total = length(polyids))
for(pid in polyids){
  pb$tick()
  timeseries = df$date[df$polyid == pid]
  start = min(timeseries)
  end = max(timeseries)
  len = length(seq.Date(as.Date(start, frac = 1),as.Date(end, frac = 1),"1 month"))
  if(abs(len - length(timeseries))==0){
    df$continuous[df$polyid==pid] = 1
  }else{
    df$continuous[df$polyid==pid] = 0
  }
}

# Locations that stay open continuously
open_always = unique(df$polyid[df$continuous == 1])
length(open_always)

# Recalculate Revenue, here because...
df$revenue = df$count*df$weight*.01*df$price

# Save csv and rds
saveRDS(df,"data/tables/noaa_500_standardized.rds")
write.csv(df,"data/tables/noaa_500_standardized.csv")
