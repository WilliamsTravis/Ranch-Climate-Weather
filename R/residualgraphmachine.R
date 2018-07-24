"
This will take in a polyid and output a scatter plot of residuals over time and a point on a map. 
  I tried before to make a four panel version of this, but the fonts were difficult to scale all together. 
  Perhaps doing it one at a time is a better way to start.


Things to do:
  1) Turn into a function
  2) Make a quick development map and a publishing map
  3) Make multiple sites optional 
"
setwd('G:/My Drive/shrum-williams/project/')
source('R/econometric_functions.R')
source('R/themes.R')
source('R/mappieces.R')
library(ggsn)
library(extrafont)
library(gridGraphics)
font_import()
loadfonts(device = "win")
windowsFonts()

# The model outputs are in the outputs folder in STATA
list.files("STATA/outputs","*csv", full.names = T)
files = list.files("STATA/outputs","*csv", full.names = T)

# Choose the model to retrieve 
model = read.csv(files[5])
# model = read.csv("G:/My Drive/THESIS/Market Project/STATA/FINAL/models/outputs/precipoutput.csv")

# Get list of sites and polyids
sites = read.csv('data/tables/sites.csv',stringsAsFactors = FALSE)
print(sites[order(sites$locale),])

# Choose one
polyid = 131

residualMachine = function(model,polyid, quick = TRUE){
  # Set up
  model$date = as.yearmon(model$date)
  model$date = as.Date(model$date)
  model$predictions = exp(model$predictions_u)
  model$weight = exp(model$logweight)
  model$residuals = model$weight - model$predictions
  
  # Filter model by location
  site = sites$locale[sites$polyid%in%polyid]
  local = model[model$locale == site,]
  sitecolors = data.frame(locale = unique(site),
                          color = rainbow(length(unique(local$locale))),
                          stringsAsFactors = FALSE)
  sitecolors$locale = as.character(sitecolors$locale)
  local = full_join(local,sitecolors, by = "locale")
  
  # Get coordinates for the map
  point = local%>%select(x,y,locale, color)%>%
    distinct(x,y,locale,color)
  
  
  # Make Map - quick
  if(quick==TRUE){
    dev.new()
    # par(xpd = NA, # switch off clipping, necessary to always see axis labels
    #     bg = "transparent", # switch off background to avoid obscuring adjacent plots
    #     oma = c(2, 2, 0, 0)) # move plot to the right and up
    plot(states,lwd = 2,col = colors$colors)
    points(point$x,point$y,cex = 1.5, pch = 21, bg = as.character(point$color))
    map = recordPlot()
    dev.off()
  }else{
    #Make map - publishable
    map = ggplot()+
      geom_polygon(data = usadf,aes(x = long,y = lat,
                                    id = group, # It says it ignores this but it's not
                                    fill = colors),
                   size = 1,
                   alpha = .75,
                   show.legend = F) +  
      geom_raster(data = reliefdf, 
                  aes(x=x,y=y,value=value,alpha=value),
                  show.legend = F)+ 
      geom_path(data=usadf,aes(x = long, y = lat, group = group),
                color = "grey30",
                lwd = 1.1)+
      geom_path(data = outlinedf,aes(x = long, y = lat,group = group),
                lwd = 2.5, color = "grey20")+
      geom_path(data = boundarydf,aes(x = long, y = lat, group = group),
                color = "black",
                lwd = 1.25)+
      geom_point(data =point,aes(x = x,y = y),
                 size = 4,
                 color = "black",
                 show.legend = FALSE)+
      geom_point(data =point,aes(x = x,y = y,color = color),
                 size = 2.5,
                 show.legend = FALSE)+
    mapTheme()+
      scale_fill_identity()+
      scale_alpha(name = "", range = c(1, 0), guide = F)  +
      scale_color_identity()+
      labs(title = "Sample Auction Sites")+
      coord_equal()+
      scalebar(usadf,
               dist = 500,
               anchor = c(x = 1200000,y= -1700000),
               st.size = 4
      )
  }
  # Now get residual plots, just time series for now
  # Average monthly residuals over time
  # Group by month
  time = local%>%na.omit()%>%group_by(date,locale)%>%
    mutate(meanresidual = mean(residuals))%>%
    distinct(date,locale,meanresidual,color)
  
  time = local%>%na.omit()%>%group_by(date,locale)%>%
    distinct(date,locale,residuals,color)
  
  title = unique(time$locale)
  
  timeseries = ggplot(data = time,aes(x = date,
                       y = residuals, 
                       color = color))+
    geom_point()+
    labs(title = title)+
    ylab("Mean Monthly Residual (lbs)")+
    xlab(NULL)+
    scale_x_date(
      labels = date_format("%Y"),
      date_breaks='24 months'
    )+
    scale_color_identity()+
    lineTheme()+
    theme(axis.text.x = element_text(hjust = 1),
          panel.grid.minor.x = element_line())
    plot_grid(map,timeseries,rel_widths = c(1,2))
  }
residualMachine(model,28)  
  
