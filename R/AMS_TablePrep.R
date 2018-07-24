"
#~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
###~~~~~~~~~~~~~~~~~
Preparing the cattle market data for association with climate

1) Aggregate data in monthly observations
2) Add a revenue, open/closed, and ... field
3) Adjust prices for inflation

Author: Travis
###~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~
"

setwd("G:/My Drive/NOT THESIS/Shrum-Williams/project/")
source("R/econometric_functions.R")

######################### Read in the weekly data ##################################
df = readRDS("data/ams/RData/USDA_CattleData.rds")
names(df) = c("location","date","class","selling_basis", "grade","count","weight_low",  
              "weight_high","weight_avg","price_low","price_high","price_avg","day")

####################### Aggregate into monthly values ##############################
# Add month and year
df$month = as.integer(format(df$date,"%m"))
df$year = as.integer(format(df$date,"%Y"))

# No bulls
df2 = df[!df$class == "Feeder_Bulls",]%>%na.omit()
df2%<>%dplyr::select(location, class, selling_basis, grade, count, weight_avg, price_avg, month, year)

# Aggregate
df2$totalweight = df2$weight_avg*df2$count
df3 = df2%>%group_by(location, year, month,class,grade)%>%
          mutate(monthlycount = sum(count),
                 monthlyweight = sum(totalweight) / monthlycount,
                 monthlyprice = mean(price_avg))%>%
          distinct(location,year,month, class,grade, monthlycount,monthlyweight, monthlyprice)
names(df3) = c("locale","class","grade","month","year","count","weight","price")

################### Adjust for inflation and add revenue ###########################
rates = read.csv("data/tables/USBLS_inflation.csv")
rates$Annual_avg[rates$Year == 2018] = sum(rates[106,2:13],na.rm = T)/6
rates%<>%select(Year,Annual_avg)
names(rates) = c("year","index")
df3$inflation_rate= unlist(lapply(df3$year, function(x) rates$index[rates$year==2018]/rates$index[rates$year==x]))
df3$adj_price = df3$price*df3$inflation_rate
df3$adj_revenue = df3$count * .01 * df3$weight * df3$adj_price
df3$inflation_rate = NULL

##################### Add Coordinates? ############################################
coords = read.csv("data/tables/US_auctions.csv")
names(coords) = c("locale","lat","lon")

# Oops got rid of commas earlier, need those for location names
df3$locale = paste0(str_sub(df3$locale ,1,-4),",",str_sub(df3$locale ,-3))

# Git rid of certain types of auctions - we don't know exactly where they are or how they work
df3 = df3[grepl("Video",df3$locale) == FALSE,]
df3 = df3[grepl("Direct",df3$locale) == FALSE,]
df3 = df3[grepl("Internet",df3$locale) == FALSE,]
df3 = df3[grepl("N,ONE",df3$locale) == FALSE,]
df3 = df3[grepl("Texas Panhandle And Western Oklahoma Feedlots, TX",df3$locale) == FALSE,]


# Rename some auctions
df3$locale[df3$locale =="Lajunta Livestock Commission Co., CO"] = "Winter Livestock Inc Lajunta, CO"
df3$locale[df3$locale =="Cattlemans Livestock Auction (Bowling Green), KY"] = "Bowling Green, KY"
df3$locale[df3$locale =="Farmers Regional Livestock Market - Glasgow, KY"] = "Glasgow, KY"
df3$locale[df3$locale =="Wythe County Feeder Cattle Weighted Average, VA"] = "Wythe County, VA"
df3$locale[df3$locale =="St. Joseph Mo., MO"] = "St. Joseph,  Mo., MO"
df3$locale[df3$locale =="Stockmen's Livestock Exchange, ND"] = "Stockman's Livestock Exchange, ND"
df3$locale[df3$locale =="Loup City Commission Co., NE"] = "Loup City, NE"
df3$locale[df3$locale =="Kingsville Missouri, MO"] = "Kingsville, Missouri, MO"
df3$locale[df3$locale =="Cuba Missouri, MO"] = "Cuba, Missouri, MO"
df3$locale[df3$locale =="Torrington Livestock Comm. Co., WY"] = "Torrington Livestock Commission Co., WY"
df3$locale[df3$locale =="Sioux Falls Feeder Cattle, SD"] = "Sioux Falls Regional Livestock Auction, SD"

# Join coordinates 
df4 = full_join(df3,coords,by = "locale")

# Drop the mystery or eastern auctions we don't have time for
df4%<>%na.omit()

# Make Spatial object and reproject
spdf = SpatialPointsDataFrame(coords = df4[c("lon","lat")],
                             proj4string = CRS("+proj=longlat +datum=NAD83 +ellps=GRS80 +no_defs"),
                             data = df4 )
srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0"
spdf2 = spTransform(spdf, CRS(srs))

# Add projected coordinates to dataframe
spdf2@data$x = spdf2@coords[,1]
spdf2@data$y = spdf2@coords[,2]
df5 = spdf2@data

# Filter for central plains and add a new field for the north or southern areas
plains = readOGR("data/shapefiles/greatplains.shp")
plains = spTransform(plains,srs)
central_df = spdf2[plains,]
central_df = raster::intersect(central_df, plains)
names(central_df)[15] = "region"
central_df$region = ifelse(central_df$region ==0,"plain","mountain")

# Save File for later
saveRDS(spdf2@data, "data/ams/rdata/AMS_monthly.rds")
saveRDS(central_df@data, "data/ams/rdata/AMS_monthly_central.rds")

# Create single layer shapefile
points = central_df
points@data = points@data%>%select(locale,region)%>%unique()
points$region = as.character(points$region)
points@data = points@data[order(points@data$locale),]
rownames(points@data) = seq(1,nrow(points@data))
writeOGR(points,dsn = "data/shapefiles/market_points.shp",layer="market_points",
         driver = "ESRI Shapefile",overwrite_layer = TRUE)

test = readOGR("data/shapefiles/market_points.shp")
