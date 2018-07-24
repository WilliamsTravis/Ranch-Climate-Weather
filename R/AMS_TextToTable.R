"
#~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
###~~~~~~~~~~~~~~~~~
Converting Text Files from the Agricultural Marketing Service Custom Report tool to table.
https://marketnews.usda.gov/mnp/ls-report-config

Author: Travis
###~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~
"
setwd("G:/My Drive/not thesis/shrum-williams/project")
library(stringr)
library(dplyr)

file <- "data/AMS/AMS_text_files/AMS_Heifers_2000.txt"
file.names <- Sys.glob(file.path("data/AMS/AMS_text_files", "*txt"))  # Pulling all files from USDA folder

#Replaces all spaces with underscores, splits if there is 2 underscores in a row
parse_cattle <- function(text) {
  text_vec <- gsub(pattern = ' ', replacement = '_', x = text)
  unlist(strsplit(text_vec, split = '[_]{2,}'))
}

#Converts numeric to text while preserving non-numeric strings
maybe_as.numeric <- function(x) {
  num_x <- suppressWarnings(as.numeric(x))
  if (all(is.na(num_x))) {
    res <- x
  } else {
    res <- num_x
  }
  res
}

GetCattleData <- function(file) {
  x <- readLines(file)  # Read file
  x <- x[grepl("[[:alpha:]]", x)]  # Remove blank lines
  x <- gsub(",","",x)  # remove commas
  xl <- t(do.call(rbind, list(x))) %>%  # Converts x to a length(x) by 1 data.frame 
    apply(1, parse_cattle) %>%  # Parses each line, produces a list of parsed data character vectors
    unique()  # Removes non-unique lines (Headers)
  df <- do.call(rbind.data.frame, xl[-1])  # Converts list to dataframe and takes out first line (header)
  names(df) <- xl[[1]][!(xl[[1]] %in% c('Comments', 'Pricing_Point'))]  # Naming df vars with header line except comments and pricing_point
  df[] <- lapply(df, as.character)  # Converting all columns to character 
  df %>%
    lapply(maybe_as.numeric) %>%  # Converting numeric to numeric and keeping strings as strings
    data.frame(stringsAsFactors = FALSE) %>%  # Converting Factors to character vectors
    select(-starts_with('NA'))  # This takes out columns named NA. 
}

# single test case
testdf = GetCattleData(file)


# Parse textfiles and bundle output ---------------------------------------
# Note: with a lot of data files, this list is likely to grow quite large.
# Maybe consider saving these out as csv files as they are processed

library(parallel)
list_of_dfs_test <- mclapply(file.names[1:1045], GetCattleData)
# or in serial (with regular old lapply)
# list_of_dfs <- lapply(file.names[1:], GetCattleData)


# merge into a master data frame
merged_df <- do.call(rbind, list_of_dfs_test)

merged_df$Report_Date <- as.Date(merged_df$Report_Date, format = "%m/%d/%Y")  # Converting Date 
merged_df$Day <- merged_df$Report_Date - as.Date("2000-01-01")

# rename vars in merged_df
merged_df <- rename(merged_df, location = Location, date = Report_Date, 
       class = Class_Description, selling_basis = Selling_Basis_Description, 
       grade = Grade_Description, count = Head_Count, 
       wtrange_low = Weight_Range_Low, wtrange_high = Weight_Range_High,
       wt_avg = Weighted_Average, p_low = Price_Low, p_high = Price_High,
       p_avg = Average_Price, day = Day)

# Take out underscores in locations
merged_df$location <- gsub(pattern = "_", replacement = " ", x = merged_df$location)

# Save csv and RData file
write.table(merged_df, file = "USDA_CattleData.csv", sep = ",", row.names = FALSE)
saveRDS(merged_df, file = "USDA_CattleData.rds")


