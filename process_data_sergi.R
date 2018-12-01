
library(ggplot2)
library(data.table)
#library(prophet)
library(lubridate)
library(readxl)

root <- '/Users/sergigomezpalleja/Downloads/'
#path_daw_data <- paste0(root, "raw_data/")
path_daw_data <- copy(root)

# Import data
input_data <- read_excel(paste0(path_daw_data, "Data_Novartis_Datathon-Participants.xlsx"), skip = 3)  
input_data <- data.table(Data_Novartis_Datathon_Participants)

cols <- names(input_data)
cols_in <- cols[!grepl("X_", cols)]

dt_wide <- input_data[, cols_in, with = F]
dt_long <- melt(dt_wide, id=c("Cluster", "Brand Group", "Country", "Function"))

names(dt_long) <- gsub(" ", "_",names(dt_long))

dt_master <- reshape(dt_long, idvar = c('Cluster', 
                                        'Brand_Group',
                                        'Country',
                                        'variable'), 
                     timevar =  'Function',
                     direction = "wide")

setnames(dt_master, c("value.Sales 1", "value.Sales 2", 
                      "value.Investment 1", "value.Investment 2",
                      "value.Investment 3", "value.Investment 4", 
                      "value.Investment 5", "value.Investment 6"),
         c("sales1", "sales2", 
           "inv1", "inv2",
           "inv3", "inv4", 
           "inv5", "inv6"))

setnames(dt_master, "variable", "date")

setcolorder(dt_master, c("Cluster",
                         "Brand_Group",
                         "Country",
                         "date",
                         "sales1", "sales2", 
                         "inv1", "inv2",
                         "inv3", "inv4", 
                         "inv5", "inv6"))

dt_master[, date := paste0("01 ", date)]
dt_master[, date := dmy(date)]

setorder(dt_master, Cluster, Brand_Group, Country, date)
#dt_long[, id := paste(Cluster, Brand_Group, sep = "_")]
dt_master[, id := 1:.N]

path_res <- paste0(root, "raw_data_master.csv")
#path_res_rdata <- paste0(root, "data/clean_input.RData")

write.csv(dt_master, path_res, row.names = FALSE)
# saveRDS(dt_long, path_res_rdata)
