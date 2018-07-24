"
A place holder for the themes
"

barTheme = function(...){
  theme_minimal() + 
    theme( 
      text = element_text(family = "Cambria",
                          face = "bold", color = "#050344",
                          size = 14),
      axis.line = element_blank(),
      plot.title = element_text(size = 17),
      axis.text.x = element_text(
        face = "bold",
        angle = 45,
        hjust = 1,
        size = 12,
        margin = margin(t = 0,b = 0,l = 0, r =0)),
      axis.text.y = element_text(face = "bold",
                                 margin = margin(l = 8),
                                 size = 12),
      
      panel.grid.minor = element_line(color = "#d8d8a9", size = 0.2),
      panel.grid.major = element_line(color = "#d8d8a9", size = 0.2),
      plot.background = element_rect(fill = "white", color = NA), 
      panel.background = element_rect(fill = "white", color = NA), 
      legend.background = element_rect(fill = "white", color = NA),
      legend.text = element_text(size=12),
      legend.key.size =  unit(1.25,"line"),
      panel.border = element_blank(),
      
      ...
    )
}
lineTheme = function(...){
  theme_minimal() + 
    theme( 
      text = element_text(family = "Cambria",
                          face = "bold", color = "#050344",
                          size = 14),
      axis.line = element_blank(),
      plot.title = element_text(size = 17),
      plot.subtitle = element_text(size = 12),
      axis.text.x = element_text(
        face = "bold",
        angle = 45,
        hjust = 1,
        size = 12,
        margin = margin(t = 0,b = 0,l = 0, r =0)),
      axis.text.y = element_text(face = "bold",
                                 margin = margin(l = 8),
                                 size = 12),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_line(color = "#d8d8a9", size = 0.2),
      panel.grid.major = element_line(color = "#d8d8a9", size = 0.2),
      plot.background = element_rect(fill = "white", color = NA), 
      panel.background = element_rect(fill = "white", color = NA), 
      legend.background = element_rect(fill = "white", color = NA),
      legend.text = element_text(size=10),
      legend.key.size =  unit(1.25,"line"),
      panel.border = element_blank(),
      
      ...
    )
}
mapTheme = function(...){
  theme_minimal() + 
    theme( 
      text = element_text(family = "Cambria",
                          face = "bold", 
                          color = "#050344",
                          size = 14),
      plot.title = element_text(size = 19),
      axis.line = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_blank(),
      # panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = NA), 
      panel.background = element_rect(fill = "white", color = NA), 
      legend.background = element_rect(fill = "white", color = NA),
      legend.text = element_text(size=14),
      legend.key.size =  unit(10.55,"line"),
      legend.key.height =  unit(3,"line"),
      legend.key.width = unit(1,"line"),
      # legend.direction = "vertical",
      # legend.justification = "center",
      panel.border = element_blank(),
      ...
    )
}

