# DATA PREPARATION part II 

#  MACHINE READABLE DATA PREPARATION  = ex make bar charts understandable with real name in axis ----
#Librairies
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


data <- train_raw_tbl 
bins <- 10
plot_hist_facet <- function(data, bins = 10, ncol = 5, 
                            fct_reorder = FALSE, fct_rev = FALSE, 
                            fill = palette_light()[[3]], 
                            color = "white", scale = "free") {
    
    data_factored <- data %>%
        # Pour passer de character à numeric, il faut préalablement le transformer en factor ----
        mutate_if(is.character , as.factor) %>%
        mutate_if(is.factor, as.numeric) %>%
        # Pour tracer avec ggplot, les données doivent être en format (col1, col2), gather permets cela ----
        gather(key = key, value = value, factor_key = T)
    
    g <- data_factored %>% 
        ggplot(aes(x = value, group = key)) + 
        geom_histogram(bins = bins, fill = fill, color = color) +
        facet_wrap(~ key, ncol = ncol, scale = scale) +
        theme_tq()
    
    return(g)
}
# When make these histogram, it's better to have the target (Attrition) at the beginning ---
# of the graph ----
train_raw_tbl %>%
    select(Attrition, everything()) %>%
    plot_hist_facet(bins = 10, ncol = 5, fct_rev = F)

# #### RECIPES LIBRARY ###### ----
# recipe (formula, ....) -> create a template assigning roles to the variables within the data ----
# formula (>) crates the outcomes and predictors roles assigining each variable or feature to a role ----
# step_*() -> step functions add preprocessiing steps to the reipe as instructions in a sequential order ----
# all_predictors() -> enables selecting by specific roles assigned by the recipe() function ----
# EX : step_knnimpute(all_predictors()) 
# all_outcomes() -> for outcomes ----
# all_numeric() -> for numeric data ----
# all_nominal() -> for factor data ----
# prep() -> prepares the recipe by calculating the transformations (not modified the data) ----
# bake() -> performs the prepared recipe on data. This transforms data ( performed on both the train and test data) ----


# Order of preprocessing : 

# Plan : Correlation Analysis 

#1. Impute (zero variance features = features not useful for prediction) ----
#2. Individual transformations ----
#   2.1 skewness -> log, box cox, stabilize variance, make stationary
#   2.2 Normality -> requirement for linear models that depend on correlation (ex:regression)
#3. Discretize (if needed - can hurt correlations | not recommended) ----
#4. Normalization steps (put data onto consistent scale : center, scaling) ----
#5. Create dummy variables (turn categorical data into separate columns of 0 or 1) ----
#6. Multivariate transformation (PCA, Umap - dimensionality reduction when data is wide and susceptible to overfit) ---- 
#7. Create interactions (advanced topic, ex: height weight related to obesity) ----

#S1. Impute (zero variance features)  ----

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors())

recipe_obj

#S2. skewness transformations ----
# We look the skewness in numeric features and the factor features.
#numeric_skewed_features_names and factor_skewed_features_names

numeric_skewed_features_names <- train_readable_tbl %>% 
    select_if(is.numeric)%>%
    map_df(skewness) %>%
    gather(factor_key = T) %>%
    arrange(desc(value)) %>%
    # We fix a threeshold for the skewness in our data
    filter(value >= 0.8) %>%
    pull(key) %>% 
    # gather transforme en key colonne = factor et l'autre = value 
    as.character()

# We call the readable training dataset and checking the histogram
train_readable_tbl %>%
    select(numeric_skewed_features_names) %>%
    plot_hist_facet()

# we see 2 features has to pass from numeric to factors (JobLevel", "StockOptionLevel)
numeric_skewed_features_names <- train_readable_tbl %>% 
    select_if(is.numeric)%>%
    map_df(skewness) %>%
    gather(factor_key = T) %>%
    arrange(desc(value)) %>%
    # We fix a threeshold for the skewness in our data
    filter(value >= 0.8) %>%
    filter(!key %in% c("JobLevel", "StockOptionLevel")) %>%
    pull(key) %>% 
    as.character()

factor_names <- c("JobLevel", "StockOptionLevel")

# Transform the skewed data with the YeoJohnson transformation 
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_YeoJohnson(numeric_skewed_features_names) %>%
    step_mutate_at(factor_names, fn=as.factor)

# Test the RECIPE (applying on the train dataset = No skew function)
recipe_obj %>%
    prep() %>%
    bake(train_readable_tbl) %>%
    select(numeric_skewed_features_names) %>%
    plot_hist_facet()

#4. Normalization steps(center, scaling) on numeric features----
# center the data before scaling, if 
# Algorithm that require feature scaling : kmeans, deep learning, PCA, SVMs

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_YeoJohnson(numeric_skewed_features_names) %>%
    step_mutate_at(factor_names, fn=as.factor)%>%
    #all_numeric inside de step_center make applying the function on the numeric feature only ----
    step_center(all_numeric()) %>%
    step_scale(all_numeric())

#check the 4th step in the recipe
recipe_obj$steps[[4]] 

prepared_recip <- recipe_obj %>% prep()
# we see what happened when we apply center on the data
prepared_recip$steps[[4]]

# check the recipe if it works 
prepared_recip %>%
    bake(new_data = train_readable_tbl) %>%
    select_if(is.numeric) %>% plot_hist_facet()


#5. Create dummy variables (categorical -> 0 or 1) ----
# if a fct has 3 levels -> the feature is expanded into 2 columns

dummied_recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_YeoJohnson(numeric_skewed_features_names) %>%
    step_mutate_at(factor_names, fn=as.factor)%>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    step_dummy(all_nominal())




# FINAL RECIPE ---

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_YeoJohnson(numeric_skewed_features_names) %>%
    step_mutate_at(factor_names, fn=as.factor)%>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    step_dummy(all_nominal()) %>%
    prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)

test_tbl <- bake(recipe_obj, new_data = test_readable_tbl)

writexl::write_xlsx(train_tbl, path = "00_Data/train_tbl.xlsx")
writexl::write_xlsx(test_tbl, path = "00_Data/test_tbl.xlsx")


train_corr_tbl <- train_tbl %>% correlate(target = Attrition_Yes)

train_corr_tbl %>% plot_correlation_funnel(interactive = TRUE)



