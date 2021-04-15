# DATA PREPARATION part I 

# HUMAN READABLE DATA PREPARATION  = ex make bar charts understandable with real name in axis ----
#Librairies
library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(stringr)

# Load Data ----
path_train <- "00_Data/telco_train.xlsx"
path_data_definitions <-  "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE) %>% 
    rename_all(~str_glue("X__{1:length(.)}"))

# Processing Pipeline

# combine both tibbles 
source("00_Scripts/data_processing_pipeline.R")

train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)

# Ex Effect the data_processing_pipeline : 
# Eduaction 
# Before 
train_raw_tbl %>%
    ggplot(aes(Education)) +
    geom_bar()
# After 
train_readable_tbl %>%
    ggplot(aes(Education)) +
    geom_bar()

# STEP1 - Tidying the data ----
definitions_tbl <- definitions_raw_tbl %>%
    # fill the NA above the Education column by the same name Education 
    fill(X__1, .direction = "down") %>%
    filter(!is.na(X__2)) %>%
    separate(X__2, into = c("key", "value"), sep = " '", remove = TRUE) %>%
    rename(column_name = X__1) %>%
    mutate(key = as.numeric(key)) %>%
    mutate(value = value %>% str_replace(pattern = "'", replacement = ""))
definitions_tbl

# Split() -> splits a data frame into multiple data frame contained within a list ---- 
# supply a column name as a vector (.$column_name) ----

# map() -> map a function a a vector or a list, the output is a list 
definitions_list <- definitions_tbl %>%
    split(.$column_name) %>%
    map(~ select(., -column_name)) %>%
    map(~ mutate(., value = as_factor(value)))

# seq_along() -> generates a numeric sequence (1,2,3, ..) along the length of an object ----

for (i in seq_along(definitions_list)) {
    
    list_name <- names(definitions_list)[i]
    
    #colname() -> used to get or set the column names of a data frame ----
    #pasteà()  -> combines multiple strings with no sepraration between the two ----
    
    colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
}

definitions_list

# JOIN THE DATA FRAMES WITHIN THE DEFINITIONS LIST WITH THE MAIN DATA FRAME (TRAINING DATA) ----

# Step01. transform the data frame into a list type ----
data_mergerd_tbl <- list(HR_Data = train_raw_tbl) %>%
    # append () -> ajouter des éléments liste à la suite d'éléments liste ---- 
    append(definitions_list, after = 1) %>%
    # reduce() -> iteratively applies a user specified function to successive binary ----
    # sets of objects. f(x1, x2, X3) => (f(x1), f(x2), f(x3), f(x4), f(xn))
    reduce(left_join) %>%
    # Remove all the doublon rows in the output (ex: Education & EducationField = code & words)
    # one_of() -> 
    select(-one_of(names(definitions_list))) %>%
    set_names(str_replace_all(names(.), pattern = "_value", replacement = "")) %>%
    # sort() arrange vector alphanumerically
    select(sort(names(.)))


data_processed_tbl <- data_mergerd_tbl %>% 
    # transform all the character data in factor type + map(levels)
    mutate_if(is.character, as.factor) %>%
    # Reorder the level of these column
    mutate(
        BusinessTravel = BusinessTravel %>% fct_relevel("Non-Travel", "Travel_Rarely", "Travel_Frequently"), 
        MaritalStatus = MaritalStatus %>% fct_relevel("Single", "Married", "Divorced"))
        )

# Processing Pipeline (make a function) ----

data  <-  train_raw_tbl
definitions_tbl <- definitions_raw_tbl

 process_hr_data_readable_F <- function(data, definitions_tbl) {
     
definitions_list <- definitions_tbl  %>%
         fill(X__1, .direction = "down") %>%
         filter(!is.na(X__2)) %>%
         separate(X__2, into = c("key", "value"), sep = " '", remove = TRUE) %>%
         rename(column_name = X__1) %>%
         mutate(key = as.numeric(key)) %>%
         mutate(value = value %>% str_replace(pattern = "'", replacement = "")) %>%
         split(.$column_name) %>%
         map(~ select(., -column_name)) %>%
         map(~ mutate(., value = as_factor(value)))
     
     for (i in seq_along(definitions_list)) {
         list_name <- names(definitions_list)[i]
         colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
     }

     data_merged_tbl <- list(HR_Data = data) %>%
     append(definitions_list, after = 1) %>%
     reduce(left_join) %>%
         select(-one_of(names(definitions_list))) %>%
         set_names(str_replace_all(names(.), pattern = "_value", replacement = "")) %>%
         select(sort(names(.))) %>% 
         mutate_if(is.character, as.factor) %>%
         mutate(
             BusinessTravel = BusinessTravel %>% fct_relevel("Non-Travel", "Travel_Rarely", "Travel_Frequently"), 
             MaritalStatus = MaritalStatus %>% fct_relevel("Single", "Married", "Divorced"))
     
     return(data_merged_tbl)
 }

 process_hr_data_readable_F(train_raw_tbl, definitions_tbl = definitions_raw_tbl) %>% glimpse()


