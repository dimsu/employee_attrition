# STEP2 in CRISP-DM : Data Understanding 
library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(GGally)
library(skimr)
library(DataExplorer)

# Load Data ----
path_train <- "00_Data/telco_train.xlsx"
path_data_definitions <-  "00_Data/telco_data_definitions.xlsx"
train_raw_tbl <- read_excel(path_train, sheet = 1)
# read_excel(     col_names = FALSE) => specified that the first is not for col_names 
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE)

# Exploratory data ----

# 01. package DataExplorer ----
create_report(train_raw_tbl)
# 02. library (dataMaid)----
makeDataReport(train_raw_tbl)
# 03. library (skim)----
skim(train_raw_tbl)

# CHARACTER DATA  ----
# view the unique values of you dataset 
train_raw_tbl %>%
    select_if(is.character)%>%
    map(unique)

# map(~table) => map the table ----
# prop.table() -> modifies the output of table() to proportions ----
train_raw_tbl %>%
select_if(is.character)%>%
    map(~table(.) %>% prop.table())

# NUMERIC DATA ----
## Variables with only one level are non-essential variables (zero-variance features) ----
## these features are not useful to when modeling | Numeric variables that are lower in levels are ----
## likely to be discrete and numeric variables that are higher in levels are continous ----

train_raw_tbl %>%
    select_if(is.numeric)%>%
    # this combination create a tibble for seeing how many unique observations are contained in numeric data ----
    map_df(~unique(.) %>% length()) %>%
    gather() %>%
    # this allow to see features/variables that have many levels 
    arrange(desc(value)) %>%
    filter(value > 10)

# step 2: Data visualisation ----
train_raw_tbl %>%
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18, DistanceFromHome) %>%
    ggpairs()

# Customization the ggpairs graph : 
train_raw_tbl %>%
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18, DistanceFromHome) %>%
    ggpairs(aes(color = Attrition), lower = "blank", legend = 1, 
            diag = list(continous = wrap("densityDiag"))) + 
    theme(legend.position = "bottom")

# ggpairs plots function ----

plot_ggpairs <- function(data, color = NULL, density_alpha = 0.5) {
    
    color_expr <- enquo(color)
    
    if (rlang::quo_is_null(color_expr)) {
        
        g <- data %>%
            ggpairs(lower = "blank")
        
    } else{
        
        color_name <- quo_name(color_expr)
        
        g <- data %>%
            ggpairs(mapping = aes_string(color = color_name), 
                    lower = "blank", legend = 1, 
                    diag = list(continous = wrap("densityDiag"))) + 
            theme(legend.position = "bottom")
    }
    
    return(g)
}


# Explore feature by category 

#   1. Descriptive features: age, gender, marital status 
train_raw_tbl %>%
    select(Attrition, Age, Gender, MaritalStatus, NumCompaniesWorked, Over18, DistanceFromHome) %>%
    plot_ggpairs(Attrition)

#   2. Employment features: department, job role, job level
train_raw_tbl %>%
    select(Attrition, contains("employee"), contains("department"), contains("job")) %>%
    plot_ggpairs(Attrition) 

#   3. Compensation features: HourlyRate, MonthlyIncome, StockOptionLevel 
train_raw_tbl %>%
    select(Attrition, contains("income"), contains("rate"), contains("salary"), contains("stock")) %>%
    plot_ggpairs(Attrition)

#   4. Survey Results: Satisfaction level, WorkLifeBalance 
train_raw_tbl %>%
    select(Attrition, contains("satisfaction"), contains("life")) %>%
    plot_ggpairs(Attrition)

#   5. Performance Data: Job Involvment, Performance Rating
train_raw_tbl %>%
    select(Attrition, contains("performance"), contains("involvement")) %>%
    plot_ggpairs(Attrition)

#   6. Work-Life Features 
train_raw_tbl %>%
    select(Attrition, contains("overtime"), contains("travel")) %>%
    plot_ggpairs(Attrition)

#   7. Training and Education 
train_raw_tbl %>%
    select(Attrition, contains("training"), contains("education")) %>%
    plot_ggpairs(Attrition)

#   8. Time-Based Features: Years at company, years in current role
train_raw_tbl %>%
    select(Attrition, contains("years")) %>%
    plot_ggpairs(Attrition)






