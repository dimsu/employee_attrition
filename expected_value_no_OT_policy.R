# EVALUATION: EXPECTED VALUE OF POLICY CHANGE ----
# NO OVERTIME POLICY ----

# 1. Setup ----

# Load Libraries 

library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)


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

#3. Expected Value ----

source("00_Scripts/assess_attrition (1).R")

# 3.1 Calculating Expected Value With OT ----

predictions_with_OT_tbl <- automl_leader %>%
    h2o.predict(newdata = as.h2o(test_tbl)) %>%
    as_tibble() %>%
    bind_cols(
        test_tbl %>%
            select(EmployeeNumber, MonthlyIncome, OverTime)
    )

predictions_with_OT_tbl


expected_value_with_OT_tbl <- predictions_with_OT_tbl %>%
    mutate(
        attrition_cost = calculate_attrition_cost(
            n = 1, 
            salary = MonthlyIncome * 12, 
            net_revenue_per_employee = 250000
        ) 
    ) %>%
    mutate(cost_of_policy_change = 0
    ) %>% 
    mutate(expected_attrition_cost = 
               Yes *(attrition_cost + cost_of_policy_change) + 
               No * (cost_of_policy_change))

# total expected value with over-time (INITIAL STATE)
total_ev_with_OT_tbl <- expected_value_with_OT_tbl %>%
    summarise(
        total_expected_attrition_cost_0 = sum(expected_attrition_cost))
    

# 3.2 Calculating Expected Value without Over_time (new policy) ----

# new state -> no OT means in the OvertTime column "Yes" => "No"

test_without_OT_tbl <- test_tbl %>%
    mutate(OverTime = fct_recode(OverTime, "No" = "Yes"))

# New prediction with the new policy (no OT), change in the test_data (test_without_OT_tbl)

predictions_without_OTp_tbl <- automl_leader %>%
    h2o.predict(newdata = as.h2o(test_without_OT_tbl)) %>%
    as.tibble() %>%
    bind_cols(
        test_tbl %>%
            select(EmployeeNumber, MonthlyIncome, OverTime), 
        test_without_OT_tbl %>% 
            select(OverTime)
    )

# New columns with the past and the actual policy (OT_0 = init state, OT_1 = new state)
predictions_without_OT_tbl <- predictions_without_OT_tbl %>% 
    rename(
        OverTime_0 = OverTime...6, 
        OverTime_1 = OverTime...7) 

# Compute expected value without Over Time
# we suppose employee make 10% overtime in average : 
avg_overtime_pct <- 0.10

expected_value_without_OT_tbl <- predictions_without_OT_tbl %>%
    mutate(
        attrition_cost = calculate_attrition_cost(
            n = 1, 
            salary = MonthlyIncome * 12, 
            net_revenue_per_employee = 250000
        ) 
    ) %>%
    mutate(cost_of_policy_change = case_when(
        OverTime_0 == "Yes" & OverTime_1 == "No" ~ avg_overtime_pct * attrition_cost,
        TRUE ~ 0
    )
    ) %>% 
    mutate(expected_attrition_cost = 
               Yes *(attrition_cost + cost_of_policy_change)
           + 
               No * (cost_of_policy_change))

# total expected value without over-time (NEW STATE)
total_ev_without_OT_tbl <- expected_value_without_OT_tbl %>%
    summarise(
        total_expected_attrition_cost_1 = sum(expected_attrition_cost))

# 3.3 Savings Calculation ----


bind_cols(
    total_ev_with_OT_tbl, 
    total_ev_without_OT_tbl
) %>%
    mutate(
        savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1, 
        pct_savings = savings / total_expected_attrition_cost_0
    )
# For the test data (220 employees savings = 379K - the pct of saving is 12%)
# For the whole employees (train + test data) -> train + test data / test data = 6.7 





