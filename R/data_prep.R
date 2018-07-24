"
Take a standardized 500 km radius climate-market dataset and balance by filling
  with NAs.
"
setwd("G:/My Drive/shrum-williams/project")
source("R/econometric_functions.R")

# Read in df
list.files("G:/My Drive/THESIS/data/Market Project/RMWDatasets/")


df = readRDS("data/tables/noaa_500_standardized.rds")

# Associate Lag with particular months
df2 = getMonths(df)

# Create FUll date-id list
datemin = min(df$date)
datemax = max(df$date)
dates1 = seq.Date(as.Date(datemin), as.Date(datemax),by = "1 month")
id1 = unique(df$polyid)
id2 = rep(id1,each = length(dates1))
dates2 = rep(dates1,length(id1))
idf = data.frame(polyid = id2,date = dates2)
idf$date = as.yearmon(idf$date)
idf$dateid = paste0(idf$date,"_",idf$polyid)
df2 = left_join(idf,df,by = "dateid")
names(df2)[c(1,2)] = c("polyid","date")
df2$date.y = NULL
df2$polyid.y = NULL

# Add a field for open or closed
df2$open = ifelse(is.na(df2$locale),0,1)
df3 = full_join(df2,justrain,by = "dateid")
names(df3)[c(1,2)] = c("polyid","date")
df3$t0.x = NULL
df3$date.y = NULL
df3$polyid.y = NULL
names(df3)[match('t0.y',colnames(df3))] = "t0"
df3 = df3[1:25380,]
df3$year = format(df3$date, "%Y") 
df3$month = format(df3$date, "%m") 

# Save
saveRDS(df3,"data/tables/noaa_500_standardized_balanced.rds")

# Now, Balance by decimation
pdf = pdata.frame(df0,index = c('polyid','date'))
is.pbalanced(pdf)
pdf%<>%na.omit()
pdf = make.pbalanced(pdf,balance.type = "shared.individuals")
write.csv(pdf,"data/tables/precip_balanced_decimated.csv")

