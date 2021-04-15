# H20 modeling ----

#1.0 SETUP ----

# Load Librairies
library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(cowplot)
library(fs)
library(glue)
library(stringr)
library(forcats)

# Load Data
path_train <- "00_Data/telco_train.xlsx"
path_test <- "00_Data/telco_test.xlsx"
path_data_definitions <- "00_Data/telco_data_definitions.xlsx"

train_raw_tbl <- read_excel(path_train, sheet = 1) 
test_raw_tbl <- read_excel(path_test, sheet = 1)
definitions_raw_tbl <- read_excel(path_data_definitions, sheet = 1, col_names = FALSE) %>% 
    rename_all(~str_glue("X__{1:length(.)}"))

# Processing Pipeline 
source("00_Scripts/data_processing_pipeline.R")
train_readable_tbl <- process_hr_data_readable(train_raw_tbl, definitions_raw_tbl)
test_readable_tbl <- process_hr_data_readable(test_raw_tbl, definitions_raw_tbl)

# ML Preprocessing 

recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>%
    prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl <- bake(recipe_obj, new_data = test_readable_tbl)


# 2.0 Modeling ----
h2o.init()

# as.h2o() => imports a data frame to an h2o cloud (H2o frame) ----
as.h2o(train_tbl)

#split data
# split the training data into 2 tibbles following the the ration 85/15 ----
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 123)

train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o <- as.h2o(test_tbl)

# Fix the target variables and the predictors -----
# setdiif() => compute the difference between two columns and return the result ----
y <- "Attrition"
x <- setdiff(names(train_h2o), y)

#key concepts in h2o_autoML : 
# training Frame -> used to develop model 
# Validation Fame -> used to tune hyperparameters via grid search
# Leaderboard Frame -> Test set completely held out from model training & tuning 

automl_models_h2o <- h2o.automl(
    x= x, 
    y=y, 
    training_frame = train_h2o, 
    # normally we can use the data validation_frame = , leaderboard_frame = ----
    # max_runtime_secs() -> minimize modeling time initially. Once results look ----
    # promissing increase the run time to get more models with highly tuned params
    max_runtime_secs = 40, 
    # nfolds() -> K-Fold cross validation (nfolds = k) ----
    nfolds = 5)

# typeof () -> returns the object type (aka class) ----
# typeof(automl_models_h2o) = S4 -----
# S4 = data type object in h2o automl ----
# slotnames() -> returns the names of slots in an S4 class object ----
# S4 objects use the @ symbol to select slots, slots are like entries in a a list ----

slotNames(automl_models_h2o)

# Models performance metrics are contained in leaderboard and are based on the 
# leaderboard frame which is held out during modeling ----
automl_models_h2o@leaderboard

automl_models_h2o@leader

# ligne #** reported on training data ** -> training results during the modeling process, 
# which are not representative of new data 

# ligne #** Reported on validation data ** -> validation results during the tuning process,
# which are not representative of new data 

# ligne #** reported on cross-validation data ** -> these are the results during the 5-fold
# cross validation performed on the training data 

# h2o.getModel() -> give a detail of a model when given a model ID (OUTPUT) ----
h2o.getModel("GLM_1_AutoML_20210327_033210")

automl_models_h2o@leaderboard %>%
    as_tibble() %>%
    slice(1) %>% 
    pull(model_id) %>%
    h2o.getModel()

# FUNCTION TO EXTRACT A MODEL ID FROM LEADERBOARD ----
extract_h2o_model_name_by_position <- function(h2o_leaderboard, n = 1, verbose = TRUE){
    
    model_name <- h2o_leaderboard %>%
        as_tibble() %>%
        slice(n) %>%
        pull(model_id)
    if(verbose) message(model_name)
    
    return(model_name)
}

automl_models_h2o@leaderboard %>% extract_h2o_model_name_by_position(3)

# h2o.saveModel() -> function to save a model ----
# h2o.loadModel() -> function to load a model ----
h2o.getModel("GBM_2_AutoML_20210328_235922") %>%
h2o.saveModel(path = "04_Modeling/h2o_models/")

h2o.loadModel(path = "")

h2o.getModel("StackedEnsemble_BestOfFamily_AutoML_20210327_033210") %>%
    h2o.saveModel(path = "04_Modeling/h2o_models/")

h2o.getModel("DeepLearning_grid__1_AutoML_20210327_033210_model_1") %>%
    h2o.saveModel(path = "04_Modeling/h2o_models/")

# PREDICTIONS ON THE TEST DATA AFTER THE MODEL HAVE BEEN TRAINED ON TRAINING DATA ----

stacked_ens_h2o <- h2o.loadModel("04_Modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20210327_033210")


# h2o.predict( , newdata = h2o_table_object) -> Generate predictions using an h2o Model and new data ----

predictions <- h2o.predict(stacked_ens_h2o, newdata = as.h2o(test_tbl))

# Result = binary classification : 
    # 1 - Class Prediction 
    # 2 - 1st Class Probability 
    # 3 - 2nd Class Probability 


typeof(predictions)

predictions_tbl <- predictions %>% as.tibble()


# Modified model parameters ----

# Deep Learning ----

deep_learning_mod_h2o <- h2o.loadModel("04_Modeling/h2o_models/DeepLearning_grid__1_AutoML_20210327_033210_model_1")

deep_learning_mod_h2o@allparameters

# CROSS VALIDATION OF A MODEL ---- 
# Cross validation is performed on the training dataset ----

h2o.cross_validation_models()

# h2o.auc() -> retrieve the area under the curve (AUC) for the classifier ----
# can use the xval argument to retrieve the average cross validation AUC. 
# In this part if you want to retrieve the cross-validation 
# models remember to fix keep_cross_validation_models = TRUE in h2o::h2o.automl

h2o.auc(deep_learning_mod_h2o, train = T, xval = T )

# GRID SEARCH ----

# h2o.grid() -> perform grid search in h2o ----


# 3. Visualizing The Leaderboard ----

data_transformed <- automl_models_h2o@leaderboard %>%
    as_tibble() %>%
    mutate(model_type = str_split(model_id, "_", simplify = T)[,1]) %>%
    slice(1:15) %>%
    
    # rownames_to_column() -> Adds the rownames of a data frame to a column ----
    rownames_to_column() %>%
    mutate(
        model_id = as_factor(model_id) %>% reorder(auc),
        # transformed the column model_type from character to factor 
        model_type = as.factor(model_type)) %>%
    # gather to stacked the auc and logloss columns to make a long format table for ggplot 
    gather(key = key, value = value, -c(model_id, model_type, rowname), factor_key = T) %>%
    
    # combine the names to form a column with the function paste0
    mutate(model_id = paste0(rowname, ".", model_id) %>% as_factor() %>% fct_rev())


h2o_leaderboard <- automl_models_h2o@leaderboard

# GGPLOT Function to plot the leaderboard  ----

plot_h2o_leaderboard <- function (h2o_leaderboard, order_by = c("auc", "logloss"), 
                                  n_max = 6, size = 4, include_lbl = TRUE, scales = "free_x"){
    
    # Setup inputs (select auc or logloss)
    order_by <- tolower(order_by[[1]])
    
    leaderboard_tbl <- h2o_leaderboard %>%
        as_tibble() %>%
        mutate(model_type = str_split(model_id, "_", simplify = T)[,1]) %>%
        rownames_to_column(var = "rowname") %>%
        mutate(model_id = paste0(rowname, ". ", as.character(model_id)) %>% as.factor())
    
    # Transformation 
    if (order_by == "auc") {
        
        data_transformed_tbl <- leaderboard_tbl %>%
            slice(1:n_max) %>%
            mutate(
                model_id = as_factor(model_id) %>% reorder(auc), 
                model_type = as.factor(model_type)) %>%
            gather(key = key, value = value, 
                   -c(model_id, model_type, rowname), factor_key = T)
        
    } else if (order_by == "logloss") {
        
        data_transformed_tbl <- leaderboard_tbl %>%
            slice(1 : n_max) %>%
            mutate(
                model_id = as_factor(model_id)%>% reorder (logloss) %>% fct_rev(), 
                model_type = as.factor(model_type)) %>%
            gather(key = key, value = value, -c(model_id, model_type, rowname), factor_key = T)
        
    } else {
        # stop () -> stop the execution of the function ----
        stop(paste0("order_by =  '", order_by, "' is not a permitted option."))
    }
    
    # Visualisation 
    
    g <- data_transformed_tbl %>%
        ggplot(aes(value, model_id, color = model_type)) + 
        geom_point(size = size) + 
        facet_wrap(~ key, scales = scales) +
        theme_tq() +
        scale_color_tq() + 
        labs(title = "Leaderboard Metrics",
             subtitle = paste0 ("Ordered by: ", toupper(order_by)), 
             y = "Model Position, Model ID", x = "")
    
    if (include_lbl) g <- g + geom_label(aes(label = round(value, 2), hjust = "inward"))
    
    return(g)
}
   
    
automl_models_h2o@leaderboard %>% plot_h2o_leaderboard(order_by = "logloss")

# APPLY GRID SEARCH ON OUR MODEL ----

# Check the variable importance (h2o.imp(model_id) -> get the features importance ----

gbm_model <- h2o.loadModel("04_Modeling/h2o_models/GBM_2_AutoML_20210328_235922")
    
var.imp_gbm <- h2o.varimp(gbm_model) %>%
    filter(var.imp_gbm$scaled_importance > 0.05)
    
var.imp_gbm$scaled_importance

test_tbl

# h2o_performance() -> Create an h2o performance object using new data ----
# once the model have been trained on the training data, the performance are 
# check on the test data 
h2o.performance(gbm_model, newdata = as.h2o(test_tbl))

# The result are not so good for the : AUC = 0.67, 

# Two method of grid search in h2o : 1) Cartesian Grid Search and 2) Random Grid Search ----

gbm_gridsearch_01 <- h2o.grid(
    # 1. arg from h2o.grid
    algorithm = "gbm", 
    grid_id = "gbm_gridsearch_01", 
    
    # 2. arg from the model use here : h2o.gbm (?h2o.gbm())
    x = x, 
    y = y, 
    training_frame = train_h2o, 
    validation_frame = valid_h2o, 
    nfolds = 5, 
    # Enter the gbm_model@allparameters to see the hyperparamters that we can change tune
    # list(hyper_params) -> combination between both parameters we'v changed  
    hyper_params = list(
        max_depth = c(3, 8, 13, 18), 
        ntrees = c(10, 20, 44, 65)
    )
)
# h2o.getGrid(grid_id, ..) -> result of the hyper-parameters tuning, arrange by auc ----

h2o.getGrid("gbm_gridsearch_01", sort_by = "auc", decreasing = T)

# Récupère le model le plus performant après le triage ci-haut 

gbm_grid_search_auc <- h2o.getModel("gbm_gridsearch_01_model_19")

# the new best model on train & valid data set (AUC)

gbm_grid_search_auc %>% h2o.auc(train = T, valid = T, xval = T)

# result : train = 0.97, valid = 0.85, xval = 0.79 -> Overfitting ----
# the AUC on train higher than the AUC on validation/Xcross validation 

gbm_grid_search_auc %>%
    h2o.performance(newdata = as.h2o(test_tbl))



# UNDERSTAND THE BINARY CLASSIFIER PERFORMANCE ----

# 1. on the test data set 
performance_h2o <- h2o.performance(gbm_model, newdata = as.h2o(test_tbl))

typeof(performance_h2o)

performance_h2o %>% slotNames()

performance_h2o@metrics

# 1.1 auc
h2o.auc(performance_h2o)
# 1.2 logloss
h2o.logloss(performance_h2o)

# 2. Confusion Matrix 
# on the whole model
h2o.confusionMatrix(gbm_model)
# on the trained model 
h2o.confusionMatrix(performance_h2o)

# Metric 
performance_tbl <- performance_h2o %>% 
    h2o.metric() %>%
    as_tibble()

# By this we pick the optimizing F1 (max (F1)) for a certain threshold ----
performance_tbl %>%
    filter(f1 == max(f1))
# Visualize the precision - recall trade-off and get the best threshold ----
performance_tbl %>% 
    ggplot(aes(x = threshold)) +
    geom_line(aes(y = precision), color = "blue", size = 2) + 
    geom_line(aes(y = recall), color = "red", size = 2) + 
    #h2o.find_threshold_by_max_metric() -> find the max metric to model
    geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) + 
    theme_tq() +
    labs(title = "Model metrics : precision vs recall", y = "value", 
         subtitle = "vertical line is the optimal threshold that maximizes the F1")

# ROC plot ----
path <- "04_Modeling/h2o_models/DeepLearning_grid__1_AutoML_20210327_033210_model_1"

load_model_performance_metrics <- function(path, test_tbl) {
    
    model_h2o <- h2o.loadModel(path)
    perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    
   tt <-  perf_h2o %>%
        h2o.metric() %>%
        as.tibble()%>%
        mutate(auc = h2o.auc(perf_h2o)) %>%
        select(tpr, fpr, auc)
   
   return(tt)
}

# Iterate On all the models saved in the paths (fs + purr)

model_metrics_tbl <- fs::dir_info(path = "04_Modeling/h2o_models/") %>%
    select(path) %>%
    mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
    unnest()

model_metrics_tbl %>%
    mutate(path = str_split(path, pattern = "/", simplify = T)[,3] %>% as_factor(), 
           auc = auc %>% round(3) %>% as.character() %>% as_factor()) %>%
    ggplot(aes(fpr, tpr, color = path, linetype = auc)) + 
    geom_line(size = 1 ) + 
    theme_tq() + 
    scale_color_tq() +
    theme(legend.direction = "vertical") + 
    labs(
        title = "ROC Plot", 
        subtitle = "Performance of 4 ML models \nWe see that the deep_learning model seem to be the best performing \nAuthor: Ralph D. Tasing")
        
    
# Precision VS Recall Plot ---- 

load_model_performance_metrics <- function(path, test_tbl) {
    
    model_h2o <- h2o.loadModel(path)
    perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    
    tt <-  perf_h2o %>%
        h2o.metric() %>%
        as.tibble()%>%
        mutate(auc = h2o.auc(perf_h2o)) %>%
        select(tpr, fpr, auc, precision, recall)
    
    return(tt)
}

# Iterate on the whole saved models fs + purr

model_metrics_tbl <- fs::dir_info(path = "04_Modeling/h2o_models/") %>%
    select(path) %>%
    mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
    unnest()

model_metrics_tbl %>%
    mutate(path = str_split(path, pattern = "/", simplify = T)[,3] %>% as_factor(), 
           auc = auc %>% round(3) %>% as.character() %>% as_factor()) %>%
    ggplot(aes(precision, recall, color = path, linetype = auc)) + 
    geom_line(size = 1 ) + 
    theme_tq() + 
    scale_color_tq() +
    theme(legend.direction = "vertical") + 
    labs(
        title = "Precision Vs Recall Plot", 
        subtitle = "Performance of 4 ML models \nWe see that the deep_learning model seem to be the best performing \nAuthor: Ralph D. Tasing")


# the Precision vs Recall plot : Business Application -> 
# False Negative (FN) are what we typically care most about. Recall indicates susceptibility 
# to FN's (lower recall, more susceptible). 
# In other words, we want to accurately predict which employes will leave (lower FN's) at 
# the expense of over predicting employees to leave (FP's) 
# The precision vs recall curve shows us which models will give up less FP's as we optimize the
# threshold  for FN's 


# GAIN & LIFT : BUSINESS OUTCOME OF MODELING (gain/lift chart) ----

# Yes = class probability for churn or how likely the employee is to churn.
# Attrition = actual response (reality)
# By ranking by call probability of Yes, we assess the models ability to truly detect someone
# that is leaving 
## Grouping into cohorts of most likely groups is as the heart of the Gain/Lift Chart. 
# when we do this, we can immediately show that if a candidate has a high probability of leaving
# how likely they are to leave. 

ranked_predictions_tbl <- predictions_tbl %>%
    bind_cols(test_tbl) %>%
    select(predict:Yes, Attrition) %>%
    arrange(desc(Yes))

# ntile() -> breaks a continuous value into "n" buckets or groups 
# group by ntile = split the continuous variable into "n" buckets this allows 
# to group the response (Attrition) based on the ntile column
# Ex : 10 groups (ntile ..., n = 10)

gain_lift_function_manual <- ranked_predictions_tbl %>% 
    mutate(ntile = ntile(Yes, n = 10)) %>% 
    group_by(ntile) %>%
    summarise(
        cases = n(), 
        responses = sum(Attrition == "Yes")
    ) %>%
    arrange(desc(ntile)) %>% 
    # We see that for the 10th tile, 18 person on total 22 is leaving the company
    # The groups are the prediction risk - they are sorted by highest risk, we can them 
    # into 10 (10 tiles) groups. the first group has the highest risk. This allows us to 
    # assess how well the algorithm doing when it predicts ==> call gain
    # Comparing the gain of expanding the groups - as we expand the groups. the gain will 
    # add but it wont be as impactuful since group 2 has lower prediction risk and likely 
    # higher mis-classification 
    # Algorithms will spike in the first group because it's the highest prediction risk. 
    # We can use this information to targets high risk individuals (first 2 groups)
     mutate(group = row_number()) %>%
    select(group, cases, responses) %>%
    mutate(
        cumulative_response = cumsum(responses), 
        pct_responses       = responses / sum(responses), 
        gain = cumsum(pct_responses),
        cumulative_pct_cases = cumsum(cases) / sum(cases), 
        lift = gain / cumulative_pct_cases, 
        gain_baseline = cumulative_pct_cases, 
        lift_baseline = gain_baseline / cumulative_pct_cases
    )

# h2o.gainsLift() -> h2o integrates gain lift function ----



performance_DeepL_h2o <- h2o.performance(deep_learning_mod_h2o, newdata = as.h2o(test_tbl))


gain_lift_function_h2o_tbl <- performance_DeepL_h2o %>%
    h2o.gainsLift() %>%
    as_tibble()

#cumulative_data_fraction = cumulative percentage case (cumulative_pct_cases)
#cumulative lift = lift
#cumulative_capture_rate = gain 

gain_transformed_tbl <- gain_lift_function_h2o_tbl %>%
    select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
    select(-contains("lift")) %>%
    mutate(baseline = cumulative_data_fraction) %>%
    mutate(gain = cumulative_capture_rate) %>%
    # Gather to use the ggplot facet_wrap (separate 2 keys on the plot)
    gather(key = key, value = value, gain, baseline)

gain_transformed_tbl %>%  
    ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) + 
    geom_line(size = 2) + 
    theme_tq() + 
    scale_color_tq() + 
    labs(
        title = "Gain Chart",
        subtitle = "the straight blue line is the baseline \nthis graph shows that we have a ROI by strategically choosing \n the customers we send an Email for a marketing campain \n from: 25% (baseline) to 75% (Model) ",
        x = "Cumulative Data Fraction", 
        y = "Gain (model)") 
    
lift_transformed_tbl <- gain_lift_function_h2o_tbl %>%
    select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
    select(-contains("capture")) %>%
    mutate(baseline = 1) %>%
    mutate(lift = cumulative_lift) %>%
    # Gather to use the ggplot facet_wrap (separate 2 keys on the plot)
    gather(key = key, value = value, lift, baseline)

lift_transformed_tbl %>%  
    ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) + 
    geom_line(size = 2) + 
    theme_tq() + 
    scale_color_tq() + 
    labs(
        title = "Lift Chart",
        subtitle = "Gain & Lift go hand in hand, the two charts work together to show the results \nof using the modeling approach Versus just targeting people at random \nfrom : 1x (baseline) to : 3x (model)", 
        x = "Cumulative Data Fraction", 
        y = "lift")

# 5. Performance Visualization ----

h2o_leaderboard <- automl_models_h2o@leaderboard
newdata <- test_tbl
order_by <- "auc"
max_models <- 4
size <- 1

plot_h2o_performance <- function(h2o_leaderboard, newdata, order_by = c("auc", "logloss"),
                                  max_models = 4, size = 1){

    # Inputs

    leaderboard_tbl <- h2o_leaderboard %>%
        as_tibble() %>%
        slice(1:max_models) #%>% select("model_id") %>% h2o.getModel(model_id)

    newdata_tbl <- newdata %>%
        as_tibble()

    order_by <- tolower(order_by[[1]])
    # rlang::sym() -> converts a string stored within a variable to a column name (symbol)
    # that is unevaluated and can be evaluated in tidyverse functions later using the !!
    order_by_expr <- rlang::sym(order_by)
    
    # h2o.no_progress() -> remove the process bar
    h2o.no_progress()

    # 1. Model metrics

    get_model_performance_metrics <- function(model_id, test_tbl) {

        model_h2o <- h2o.getModel(model_id)
        perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))

        perf_h2o %>%
            h2o.metric() %>%
            as.tibble() %>%
            select(threshold, tpr, fpr, precision, recall)

    }


    model_metrics_tbl <- leaderboard_tbl %>%
        mutate(metrics = map(model_id, get_model_performance_metrics, newdata_tbl)) %>%
        unnest() %>%
        mutate(
            model_id = as_factor(model_id) %>%

                fct_reorder(!! order_by_expr, .desc = ifelse(order_by == "auc", TRUE, FALSE)),
            auc  = auc %>%
                round(3) %>%
                as.character() %>%
                as_factor() %>%
                fct_reorder(as.numeric(model_id)),
            logloss = logloss %>%
                round(4) %>%
                as.character() %>%
                as_factor() %>%
                fct_reorder(as.numeric(model_id))
        )

    # 1A. ROC Plot

    p1 <- model_metrics_tbl %>%
        ggplot(aes_string("fpr", "tpr", color = "model_id", linetype = order_by)) +
        geom_line(size = size) +
        theme_tq() +
        scale_color_tq() +
        labs(title = "ROC", x = "FPR", y = "TPR") +
        theme(legend.direction = "vertical")

    # 1B. Precision vs Recall

    p2 <- model_metrics_tbl %>%
        ggplot(aes_string("recall", "precision", color = "model_id", linetype = order_by)) +
        geom_line(size = size) +
        theme_tq() +
        scale_color_tq() +
        labs(title = "Precision Vs Recall", x = "Recall", y = "Precision") +
        theme(legend.position = "none")


    # 2. Gain / Lift

    get_gain_lift <- function(model_id, test_tbl) {

        model_h2o <- h2o.getModel(model_id)
        perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))

        perf_h2o %>%
            h2o.gainsLift() %>%
            as.tibble() %>%
            select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift)

    }

    gain_lift_tbl <- leaderboard_tbl %>%
        mutate(metrics = map(model_id, get_gain_lift, newdata_tbl)) %>%
        unnest() %>%
        mutate(
            model_id = as_factor(model_id) %>%
                fct_reorder(!! order_by_expr, .desc = ifelse(order_by == "auc", TRUE, FALSE)),
            auc  = auc %>%
                round(3) %>%
                as.character() %>%
                as_factor() %>%
                fct_reorder(as.numeric(model_id)),
            logloss = logloss %>%
                round(4) %>%
                as.character() %>%
                as_factor() %>%
                fct_reorder(as.numeric(model_id))
        ) %>%
        rename(
            gain = cumulative_capture_rate,
            lift = cumulative_lift
        )

    # 2A. Gain Plot

    p3 <- gain_lift_tbl %>%
        ggplot(aes_string("cumulative_data_fraction", "gain",
                          color = "model_id", linetype = order_by)) +
        geom_line(size = size) +
        geom_segment(x = 0, y = 0, xend = 1, yend = 1,
                     color = "black", size = size) +
        theme_tq() +
        scale_color_tq() +
        expand_limits(x = c(0, 1), y = c(0, 1)) +
        labs(title = "Gain",
             x = "Cumulative Data Fraction", y = "Gain") +
        theme(legend.position = "none")

    # 2B. Lift Plot

    p4 <- gain_lift_tbl %>%
        ggplot(aes_string("cumulative_data_fraction", "lift",
                          color = "model_id", linetype = order_by)) +
        geom_line(size = size) +
        geom_segment(x = 0, y = 1, xend = 1, yend = 1,
                     color = "black", size = size) +
        theme_tq() +
        scale_color_tq() +
        expand_limits(x = c(0, 1), y = c(0, 1)) +
        labs(title = "Lift",
             x = "Cumulative Data Fraction", y = "Lift") +
        theme(legend.position = "none")

    # Combine using cowplot
    p_legend <- get_legend(p1)
    p1 <- p1 + theme(legend.position = "none")

    p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)

    p_title <- ggdraw() +
        draw_label("H2O Model Metrics", size = 18, fontface = "bold",
                   colour = palette_light()[[1]])

    p_subtitle <- ggdraw() +
        draw_label(glue("Ordered by {toupper(order_by)}"), size = 10,
                   colour = palette_light()[[1]])
    
        ret <- plot_grid(p_title, p_subtitle, p, p_legend,
                     ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.03 * max_models))
    #
     h2o.show_progress()

    return(ret)

}

automl_models_h2o@leaderboard %>%
    plot_h2o_performance(newdata = test_tbl, order_by = "auc",
                         size = 1, max_models = 3)


































