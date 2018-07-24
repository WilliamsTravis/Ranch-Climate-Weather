"
#~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
###~~~~~~~~~~~~~~~~~
Grid averaging tools (II) for Drought and Cattle Sale Responses 
Earth Lab Risk Team 
Travis Williams

Making buffers around the market points.

###~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~
"
setwd("G:/My Drive/NOT THESIS/Shrum-Williams/project/")
source("R/econometric_functions.R")

# Spatial Reference String
srs = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0"

# Read in Shapefiles
points = readOGR("data/shapefiles/market_points.shp")
points = spTransform(points,srs)


# Sample buffer
sample = gBuffer(points, byid = T,width = 200000)
plot(sample)


# Make list of buffer radii
buffers = c(100,200,300,400,500,600,700,800)

# Create new buffer shapefiles
for (b in buffers){
  filename = paste0("CentralBuffer_",b,"km")
  buffer = gBuffer(points, byid = T,width = b*1000)
  writeOGR(buffer,dsn = "data/shapefiles/buffers",layer = filename, driver="ESRI Shapefile")
}
