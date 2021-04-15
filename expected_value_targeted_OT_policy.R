# EVALUATION: EXPECTED VALUE OF POLICY CHANGE ----
# TARGETED OVERTIME POLICY ----

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

h2o.init()

split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.90), seed = 1234)

train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

y <- "Attrition"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
    x=x, 
    y=y, 
    training_frame = train_h2o, 
    # normally we can use the data validation_frame = , leaderboard_frame = ----
    # max_runtime_secs() -> minimize modeling time initially. Once results look ----
    # promissing increase the run time to get more models with highly tuned params
    max_runtime_secs = 30, 
    # nfolds() -> K-Fold cross validation (nfolds = k) ----
    nfolds = 5, 
    seed = 1234)

slotNames(automl_models_h2o)

automl_models_h2o@leaderboard
automl_models_h2o@leader

automl_models_h2o@leaderboard %>%
    as_tibble() %>%
    slice(2) %>% 
    pull(model_id) %>% h2o.saveModel("04_Modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20210412_002149")

h2o.getModel("StackedEnsemble_AllModels_AutoML_20210412_171529") %>%
    h2o.saveModel(path = "04_Modeling/h2o_models/")

automl_leader <- h2o.loadModel("04_Modeling/h2o_models/StackedEnsemble_AllModels_AutoML_20210412_171529")
# 3. Primer: Working With Threshold & Rates ----

performance_h2o <- automl_leader %>%
    h2o.performance(newdata = as.h2o(test_tbl))

performance_h2o %>% h2o.confusionMatrix()

rates_by_threshold_tbl <- performance_h2o %>%
    h2o.metric() %>%
    as_tibble()

rates_by_threshold_tbl %>%
    select(threshold, f1, tnr:tpr) %>%
    filter(f1 == max(f1))

rates_by_threshold_tbl %>%
    select(threshold, f1, tnr:tpr) %>%
    # gather here to plot on ggplot easily --- stacked variables 
    gather(key = "key", value = "value", tnr:tpr, factor_key = T) %>%
    mutate(key = fct_reorder2(key, threshold, value)) %>%
    ggplot(aes(threshold, value, color = key)) +
    geom_point() + 
    geom_smooth() + 
    theme_tq() + 
    scale_color_tq() + 
    labs(
        title = "Expected rates", 
        y = "Value", x= "Threshold"
    )
    
# 4. Expected Value ----

# 4.1 Calculating Expected Value With OT ----

source("00_Scripts/assess_attrition (1).R")

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
total_ev_with_OT_tbl


# 4.2 Calculating Expected Value With Targeted OT ----

# f1_Max = maximum f1 (not the optimal one)
max_f1_tbl <- rates_by_threshold_tbl %>%
    select(threshold, f1, tnr:tpr) %>%
    filter(f1 == max(f1))

tnr <- max_f1_tbl$tnr
fnr <- max_f1_tbl$fnr
fpr <- max_f1_tbl$fpr
tpr <- max_f1_tbl$tpr

threshold <- max_f1_tbl$threshold

# Targeted Policy based on the threshold (obtain from the init state)
# if the prediction for an employee to leave is >= threshold --> NO OverTime policy 
# for this employee 

test_targeted_OT_tbl <- test_tbl %>%
    add_column(Yes = predictions_with_OT_tbl$Yes) %>%
    mutate(
        
        # Overitme_1: new column with the targeted employees
        OverTime = case_when(
            # transform the "No" to a factor 
            Yes >= threshold ~ factor("No", levels = levels(test_tbl$OverTime)), 
            TRUE ~ OverTime
        )
    ) %>%
    select(-Yes)

predictions_targeted_OT_tbl <- automl_leader %>%
    h2o.predict(newdata = as.h2o(test_targeted_OT_tbl)) %>%
    as_tibble() %>%
    bind_cols(
        test_tbl %>%
            select(EmployeeNumber, MonthlyIncome, OverTime), 
        test_targeted_OT_tbl %>%
            select(OverTime)) %>%
    rename(
        OverTime_0 = OverTime...6, 
        OverTime_1 = OverTime...7) 
        
avg_overtime_pct <- 0.10

ev_targeted_OT_tbl <- predictions_targeted_OT_tbl %>%
    mutate(
        attrition_cost = calculate_attrition_cost(
            n = 1, 
            salary = MonthlyIncome * 12, 
            net_revenue_per_employee = 250000
        )
    ) %>%
    mutate(
        cost_of_policy_change = case_when(
            OverTime_0 == "Yes" & OverTime_1 == "No" ~ attrition_cost * avg_overtime_pct,
            TRUE ~ 0
        )
    ) %>%
    mutate(
        cb_tn = cost_of_policy_change, 
        cb_fp = cost_of_policy_change, 
        cb_tp = cost_of_policy_change + attrition_cost, 
        cb_fn = cost_of_policy_change + attrition_cost, 
        expected_attrition_cost = 
            Yes * (tpr*cb_tp + fnr*cb_fn) + 
            No * (tnr*cb_tn + fpr*cb_tp)
    )
        
total_ev_targeted_OT_tbl <- ev_targeted_OT_tbl %>%
    summarise(
        total_expected_attrition_cost_1 = sum(expected_attrition_cost)
    )
    
# 4.3 Savings Calculation ----

savings_tbl <- bind_cols(
    total_ev_with_OT_tbl, 
    total_ev_targeted_OT_tbl) %>%
    mutate(
        savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1, 
        pct_savings = savings / total_expected_attrition_cost_0)

savings_tbl

# 5. Optimizing By Threshold ----

# 5.1 Function that returns the saving according the threshold
# calculate_savings_by_threshold() ----

calculate_savings_by_threshold <- function(data, h2o_model, threshold = 0,
                                           tnr = 0, fpr = 1, fnr = 0, tpr = 1) {
    
    
    data_0_tbl <- as.tibble(data)
    
    # 4. Expected Value 
    
    # 4.1 Calculating Expected Value With OT 
    
    pred_0_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_0_tbl)) %>%
        as.tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime)
        )
    
    ev_0_tbl <- pred_0_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                net_revenue_per_employee = 250000)
        ) %>%
        mutate(
            cost_of_policy_change = 0
        ) %>%
        mutate(
            expected_attrition_cost = 
                Yes * (attrition_cost + cost_of_policy_change) +
                No *  (cost_of_policy_change)
        )
    
    
    total_ev_0_tbl <- ev_0_tbl %>%
        summarise(
            total_expected_attrition_cost_0 = sum(expected_attrition_cost)
        )
    
    # 4.2 Calculating Expected Value With Targeted OT
    
    data_1_tbl <- data_0_tbl %>%
        add_column(Yes = pred_0_tbl$Yes) %>%
        mutate(
            OverTime = case_when(
                Yes >= threshold ~ factor("No", levels = levels(data_0_tbl$OverTime)),
                TRUE ~ OverTime
            )
        ) %>%
        select(-Yes) 
    
    pred_1_tbl <- h2o_model %>%
        h2o.predict(newdata = as.h2o(data_1_tbl)) %>%
        as.tibble() %>%
        bind_cols(
            data_0_tbl %>%
                select(EmployeeNumber, MonthlyIncome, OverTime),
            data_1_tbl %>%
                select(OverTime)
        ) %>%
        rename(
            OverTime_0 = OverTime...6, 
            OverTime_1 = OverTime...7
        )
 
    avg_overtime_pct <- 0.10
    
    ev_1_tbl <- pred_1_tbl %>%
        mutate(
            attrition_cost = calculate_attrition_cost(
                n = 1,
                salary = MonthlyIncome * 12,
                net_revenue_per_employee = 250000)
        ) %>%
        mutate(
            cost_of_policy_change = case_when(
                OverTime_1 == "No" & OverTime_0 == "Yes" 
                ~ attrition_cost * avg_overtime_pct,
                TRUE ~ 0
            ))%>%
        mutate(
            cb_tn = cost_of_policy_change,
            cb_fp = cost_of_policy_change,
            cb_fn = attrition_cost + cost_of_policy_change,
            cb_tp = attrition_cost + cost_of_policy_change,
            expected_attrition_cost = Yes * (tpr*cb_tp + fnr*cb_fn) + 
                No * (tnr*cb_tn + fpr*cb_fp)
        )
    
    
    total_ev_1_tbl <- ev_1_tbl %>%
        summarise(
            total_expected_attrition_cost_1 = sum(expected_attrition_cost)
        )
    
    
    # 4.3 Savings Calculation
    
    savings_tbl <- bind_cols(
        total_ev_0_tbl,
        total_ev_1_tbl
    ) %>%
        mutate(
            savings = total_expected_attrition_cost_0 - total_expected_attrition_cost_1,
            pct_savings = savings / total_expected_attrition_cost_0
        )
    
    return(savings_tbl$savings)
    
}

rates_by_threshold_tbl %>%
    select(threshold, f1, tnr:tpr) %>%
    filter(f1 == max(f1))

# Test the function for certain parameters, (tpr+fpr = 1, fnr+tnr = 1)

# Saving corresponding to the max F1
max_f1_savings <- calculate_savings_by_threshold(test_tbl, automl_leader, 
                               threshold = max_f1_tbl$threshold, 
                               tnr = max_f1_tbl$tnr, 
                               fnr = max_f1_tbl$fnr, 
                               fpr = max_f1_tbl$fpr, 
                               tpr = max_f1_tbl$tpr
                               )
# 5.2 Optimization ----

#Threshold Optimization -> Determine the threshold that maximizes saving

smpl <- seq(1, 220, length = 20) %>% round(digits = 0)
# round () -> integer 
# when we perform an iterative process, this often takes a lot of time 
# we can sample the indices to reduce the number of iterations from 220 to 20 
# which reduces our iteration time 

#partial() -> apply a function, filling in some arguments by a already fixed/precised 
#arguments (ex: the function calculate_savings_by_threshold() I fixed the 
#arguments data = test_tbl et h2o_model = automl_leader)

partial(calculate_savings_by_threshold, data = test_tbl, h2o_model = automl_leader)

rates_by_threshold_optimized_tbl <- rates_by_threshold_tbl %>% 
    select(threshold, tnr:tpr) %>%
    slice(smpl) %>%
    mutate(
        # mutate + pmap_dbl (purr package) allow the row_rise iteration ----
        savings = pmap_dbl(
            .l = list(
                # .l -> pass the list 
                threshold = threshold, 
                tnr = tnr, 
                fnr = fnr, 
                fpr = fpr, 
                tpr = tpr
            ), 
            .f = partial(calculate_savings_by_threshold, data = test_tbl, h2o_model = automl_leader)
            )
        )

rates_by_threshold_optimized_tbl %>%
    ggplot(aes(threshold, savings)) +
    geom_line(color = palette_light()[[1]]) +
    geom_point(color = palette_light()[[1]]) +
    
    # Optimal Point (MAX Savings obtain from the iterative process)
    geom_point(shape = 21, size = 5, color = palette_light()[[3]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(savings == max(savings))) +
    geom_label(aes(label = scales::dollar(savings)), 
               vjust = -1, color = palette_light()[[3]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(savings == max(savings))) +
    
    # F1 Max -> default threshold from the classifier (classification) but not the optimal 
    geom_vline(xintercept = max_f1_tbl$threshold, 
               color = palette_light()[[5]], size = 2) +
    # annotate() -> generate elements (text, labels using cartesian coordinates)
    annotate(geom = "label", label = scales::dollar(max_f1_savings),
             x = max_f1_tbl$threshold, y = max_f1_savings, vjust = -1,
             color = palette_light()[[1]]) + 
    
    # No OT Policy (Minimum threshold)
    geom_point(shape = 21, size = 5, color = palette_light()[[2]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(threshold == min(threshold))) +
    geom_label(aes(label = scales::dollar(savings)), 
               vjust = -1, color = palette_light()[[2]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(threshold == min(threshold))) +
    
    # Do Nothing Policy (Maximum threshold)
    geom_point(shape = 21, size = 5, color = palette_light()[[2]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(threshold == max(threshold))) +
    geom_label(aes(label = scales::dollar(round(savings, 0))), 
               vjust = -1, color = palette_light()[[2]],
               data = rates_by_threshold_optimized_tbl %>%
                   filter(threshold == max(threshold))) +
    
    # Aesthestics
    theme_tq() +
    expand_limits(x = c(-.1, 1.1), y = 8e5) +
    scale_x_continuous(labels = scales::percent, 
                       breaks = seq(0, 1, by = 0.2)) +
    scale_y_continuous(labels = scales::dollar) +
    labs(
        # % savings -> filter(savings = max(savings))
        title = "Optimization: Expected Savings Maximized : 12.7%",
        subtitle = "We were trying to optimize the company savings. We did it by iterating \nto find the best threshold. \n$430,326 = initial savings (New policy) \n$438,446 = Savings (considering the max threshold but not the optimal) \n$572,994 = Optimal savings (Optimal Threshold) \n$2,344 = No change (Doing nothing)",
        caption = "Business Optimization problem - Maximize savings \nThreshold optimization using the expected value framework \nAuthor: Ralph D. Tasing",
        x = "Threshold (%)", y = "Savings"
    )

# 6 Sensitivity Analysis ----

# Sensitivty analysis consist in choosing more than 1 parameters and play with that 

# 6.1 Create calculate_savings_by_threshold_2() ----

# 6.2 Sensitivity Analysis ----

# CHALLENGE : 

# People with no stock options are leaving the company ----

# Challenge Combination OverTime & Stock-Option ----

# Part 1: Find optimal threshold 

