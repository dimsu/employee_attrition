# LIME FEATURE EXPLANATION ----

# 1. Setup ----

# Load Libraries 

library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(lime)

# Load Data
path_train            <- "00_Data/telco_train.xlsx"
path_test             <- "00_Data/telco_test.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl       <- read_excel(path_train, sheet = 1)
test_raw_tbl        <- read_excel(path_test, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE) %>% 
    rename_all(~str_glue("X__{1:length(.)}"))

# Processing Pipeline
source("00_Scripts/data_processing_pipeline.R")
train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)
test_readable_tbl  <- process_hr_data_readable(test_raw_tbl, definitions_raw_tbl)

# ML Preprocessing Recipe 
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>%
    prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

# 2. Models ----

automl_leader <- h2o.loadModel("04_Modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20210327_033210")

automl_leader

# 3. LIME ----

# 3.1 Making Predictions ----

predictions_tbl <- automl_leader %>%
    h2o.predict(newdata = as.h2o(test_tbl)) %>%
    as_tibble() %>%
    bind_cols(
        test_tbl %>%
            select(Attrition, EmployeeNumber)
    )

# 3.2 Single Explanation ----
#predictions_tbl -> we see that the the employeeNumber #1767 effectively left the company
#We want to investigate it more this single case (Single Explanation)

# Remove the Target Feature : the h2o model does not use the "Target" -> Attrition column 
# within the prediction set. 

# EXPLAINER: the "recipe" for creating an explanation. It contains the ML model & 
# feature distributions (bins) for training data 
explainer <- train_tbl %>%
    select(-Attrition) %>%
    lime(
        model = automl_leader,
        bin_continuous = TRUE, 
        n_bins  = 4,
        quantile_bins = TRUE
    # bin_continuous -> bin features, it makes it easy to detect what causes the continuous 
    # feature to have a high feature weight in the explanation (converts numeric values into bins)
    # quantile_bins to tell how to distribute observations with the bins. if TRUE, 
    # cuts will be selected to evenly distribute the total observations within each of the bins
    # bin_cuts$Age -> (nbins = 4 -> lime() = 5 cuts : bin_cuts$Age (0%, 25%, 50%, 75%, 100%))
    )

# EXPLANATION : (just one observation -> one prediction)
# slice(5) -> we select the observation we want to explain 
# the data argument must match the format that the model requires to predict 
# since h2o.predict() requires the predictors to be without the target variable
explanation <- test_tbl %>%
    slice(5) %>%
    select(-Attrition) %>%
    lime::explain(
        explainer = explainer, 
        # n_labels = 1 --> classification problem (Yes or No) we're looking for Yes
        n_labels = 1, 
        n_features = 8, 
        n_permutations = 5000, 
        # kernel_width -> affect the lime linear model fit (R2) and therefore 
        # should be tuned to get the highest R2
        kernel_width = 1)

# lime::explain() -> performs the LIME algorithm that produces explanations for
# which features have the highest impact (weight) on the localized prediction 

explanation %>% 
    as_tibble() %>%
    select(feature:prediction)
# graph for one explanatio
plot_features(explanation = explanation, ncol = 1) 

# 3.3 Multiple Explanations ----
set.seed(123)
explanations <- test_tbl %>%
    slice(1:20) %>%
    select(-Attrition) %>%
    lime::explain(
        explainer = explainer, 
        # n_labels = 1 --> classification problem (Yes or No) we're looking for Yes
        n_labels = 1, 
        n_features = 8, 
        n_permutations = 4000, 
        # kernel_width -> affect the lime linear model fit (R2) and therefore 
        # should be tuned to get the highest R2
        kernel_width = 1)

plot_explanations(explanations)
# Positive values in weight indicates it supports the prediction.





