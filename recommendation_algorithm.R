# RECOMMENDATION ALGORITHM ----

# 1.0 Setup ----

# Libraries
library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(stringr)
library(recipes)
library(correlationfunnel)

# Load Data ----
path_train <- "00_Data/telco_train.xlsx"
path_test <- "00_Data/telco_test.xlsx"
path_data_definitions <-  "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1)
test_raw_tbl <- read_excel(path_test, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE) %>% 
    rename_all(~str_glue("X__{1:length(.)}"))

# Processing Pipeline 
source("00_Scripts/data_processing_pipeline.R")
train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)
test_readable_tbl <- process_hr_data_readable(test_raw_tbl, definitions_raw_tbl)


# 2.0 Correlation Analysis - Machine Readable ----
source("00_Scripts/plot_cor.R")

# 2.1 Recipes ----

# variables which are not really numeric but factor ----
factor_names <- c("JobLevel", "StockOptionLevel")

# Instead of applying different transformation on the numeric features, 
# we can binned them to compare cohorts within the population 
#    step_center(all_numeric()) %>%  STEP_DISCRETIZE
#    step_scale(all_numeric()) %>%

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
# Turns factor_names in real factor type ----
    step_mutate_at(factor_names, fn=as.factor) %>%
    step_discretize(all_numeric(), options = list(min_unique = 1, cuts = 4)) %>%
# step_dummy(): one_hot = TRUE -> provides a column for every category when dummying
# Vs the default of one less column than the number of categories -> correlation analysis interpretability
# step_dummy() -> discretize output 0/1    
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    prep()

recipe_obj

train_corr_tbl <- bake(recipe_obj, new_data = train_readable_tbl)

train_corr_tbl %>% glimpse()

tidy(recipe_obj)
# tidy() -> returns the step level details that define the step strategy ----
tidy(recipe_obj, number = 3)

# 2.2 Correlation Visualization ----

# Manipulate Data -> Separate the features into different feature base
correlation_results_tbl <- train_corr_tbl %>%
    # supprimer la colonne attrition_No qui ne nous intéresse pas 
    select(-Attrition_No) %>%
    # get_cor() -> function du (RGlobalEnvironnemnt) 
    get_cor(Attrition_Yes, fct_reorder = T, fct_rev = T) %>%
    # choose a random correlation threshold (according to the result
    # to apply to the correlation (0.02)
    filter(abs(Attrition_Yes) >= 0.02) %>%
    # all coeff > 0 -> support the prediction 
    # all coeff < 0 -> contradicts the prediction 
    mutate(
        relationship = case_when(
            Attrition_Yes > 0 ~ "Supports", 
            TRUE ~ "Contradicts"
        )
    ) %>%
    # transform the column in character type to apply the stringr packages 
    mutate(feature_text = as.character(feature)) %>%
    separate(feature_text, into = "feature_base", sep = "_", extra = "drop")%>%
    mutate(feature_base = as_factor(feature_base) %>% fct_rev())

length_unique_groups <- correlation_results_tbl %>%
    pull(feature_base) %>%
# unique() = distinct() pour les éléments de classe factor
    unnique() %>%
    length()

correlation_results_tbl %>%
    ggplot(aes(Attrition_Yes, feature_base, color = relationship)) + 
    geom_point() +
    geom_label(aes(label = feature), vjust = -0.1) +
    expand_limits(x = c(-0.3, 0.3), y = c(1, length_unique_groups+2))+
    theme_tq()+
    scale_color_tq()+
    labs(
        title = "Correlation Analysis : Recommendation Strategy Development", 
        subtitle = "Discretization features"
    )

# 3.0 Recommendation Strategy Development Worksheet ----




# 4.0 Recommendation Algorithm Development ----

# 4.1 Personal Development (Mentorship, Education) ----


# 4.2 Professional Development (Promotion Readiness) ----


# 4.3 Work Life Balance ----




