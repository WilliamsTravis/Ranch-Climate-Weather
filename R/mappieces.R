"
This is to save space and make all of the map elements needed
"
setwd('G:/My Drive/shrum-williams/project/')
source('R/econometric_functions.R')
source('R/themes.R')

################### Spatial Data ###################
##### Coordinate System #####
srs =  "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

####### Mask ################
mask = raster("data/rasters/mask4.tif")
mask2 = as(mask,"SpatialPixelsDataFrame")
maskdf = as.data.frame(mask2)
colnames(maskdf) = c("value","x","y")
studyarea = readOGR("data/shapefiles/greatplains.shp")
studyarea = spTransform(studyarea,srs)
studyarea@data$ids = row.names(studyarea@data)
studyareadf = fortify(studyarea,region = "ids")

#### US Outline #############
usoutline = readOGR("data/shapefiles/usaoutline.shp")
usaoutline = spTransform(usoutline,srs)
usoutline@data$ids = row.names(usoutline@data)
outlinedf = fortify(usoutline,region = "ids")

########### States ##########
states = readOGR("data/shapefiles/usacontiguous.shp")
states = spTransform(states,srs)
states@data$ids = row.names(states@data)
statesdf = fortify(states,region = "ids")

########### Relief ##########
# DEM - how do I exaggerate this???
if(!exists("dem")){
  dem = raster("data/rasters/usadem125mask.tif")
}
dem2 = as(dem, "SpatialPixelsDataFrame")
demdf = as.data.frame(dem2)
colnames(demdf) = c("value", "x", "y")

## Compute shaded relief
slope = terrain(dem, 'slope')
aspect = terrain(dem, 'aspect')
relief = hillShade(slope=slope, aspect=aspect,
                   angle=160, direction=90)
relief2 = as(relief, "SpatialPixelsDataFrame")
reliefdf = as.data.frame(relief2)
colnames(reliefdf) = c("value","x","y")

# Use DEM for boundary
boundary = raster::boundaries(dem,type = 'inner',
                              directions = 4,
                              asNA = T)
boundary = rasterToPolygons(boundary,
                            fun = NULL,
                            n = 16,
                            na.rm = T,
                            digits = 12,
                            dissolve = T)
boundarydf = fortify(boundary, by = "ids")           

# Create grey background
background = dem*0+1
background = as(background, "SpatialPixelsDataFrame")
backgroundf = as.data.frame(background)
colnames(backgroundf) = c("value", "x", "y")

# US With alternating shades of grey#
colors = as.data.frame(cbind(unique(statesdf$id),gray.colors(49,start = .55,end = .95)))
names(colors) = c("id","colors")
colors$id = as.character(colors$id)
colors$colors = as.character(colors$colors)
usadf = full_join(statesdf,colors,by = "id")

