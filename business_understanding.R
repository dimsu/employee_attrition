#Librairies
library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(stringr)

# Load Data
path_train <- "00_Data/telco_train.xlsx"
train_raw_tbl <- read_excel(path_train, sheet = 1)

#Data Subset 
dept_job_role_tbl <- train_raw_tbl %>%
    select(EmployeeNumber, Department, JobRole, PerformanceRating, Attrition)

# 1. BUsiness Science Problem Framework ----

# 1A. view business As Machine ----

# Define objectives : 
# Assess Outcomes : TBD
# Calculate the number of attribution YES or NO
dept_job_role_tbl %>% 
    group_by(Attrition) %>%
    summarise(n = n()) %>%
    ungroup() %>%
    mutate(pct = n / sum(n))

# 1B. Understand The Drivers ----
# step1. Investigate Objectives : 16 % attrition 
# Synthesize Outcomes : 
# Hypothesize Drivers : Job Role and Departements

## By Departements ----
dept_job_role_tbl %>%
    group_by(Department, Attrition) %>%
    summarise(n = n(), .groups = "drop_last") %>%
    #ungroup() %>%
    mutate(pct = n/sum(n))
    

## By job ROle ----
dept_job_role_tbl %>%
    
    group_by(Department, JobRole, Attrition) %>%
    summarise(n = n()) %>%
    ungroup() %>%
    
    group_by(Department, JobRole) %>%
    mutate(pct = n/sum(n)) %>% 
    ungroup() %>%
    filter(Attrition == "Yes")

# 1C. Understand The Drivers ----
# Collect Information on Employee Attrition : On 
# Develop KPI's : Industry KPIs : 8.8 %
dept_job_role_tbl %>%
    
    group_by(Department, JobRole, Attrition) %>%
    summarise(n = n()) %>%
    ungroup() %>%
    
    group_by(Department, JobRole) %>%
    mutate(pct = n/sum(n)) %>% 
    ungroup() %>%
    filter(Attrition == "Yes") %>%
    arrange(desc(pct)) %>%
    mutate(
        above_indu_avg = case_when(
            pct > 0.088 ~ "Yes", 
            TRUE ~ "No"
        )
    )

# 1D. Uncover Problems and Opportunities ----

calculate_attrition_cost <- function(
    
    # Employee
    n                    = 1,
    salary               = 80000,
    
    # Direct Costs
    separation_cost      = 500,
    vacancy_cost         = 10000,
    acquisition_cost     = 4900,
    placement_cost       = 3500,
    
    # Productivity Costs
    net_revenue_per_employee = 250000,
    workdays_per_year        = 240,
    workdays_position_open   = 40,
    workdays_onboarding      = 60,
    onboarding_efficiency    = 0.50
    
) {
    
    # Direct Costs
    direct_cost <- sum(separation_cost, vacancy_cost, acquisition_cost, placement_cost)
    
    # Lost Productivity Costs
    productivity_cost <- net_revenue_per_employee / workdays_per_year * 
        (workdays_position_open + workdays_onboarding * onboarding_efficiency) 
    
    # Savings of Salary & Benefits (Cost Reduction)
    salary_benefit_reduction <- salary / workdays_per_year * workdays_position_open
    
    # Estimated Turnover Per Employee
    cost_per_employee <- direct_cost + productivity_cost - salary_benefit_reduction
    
    # Total Cost of Employee Turnover
    total_cost <- n * cost_per_employee
    
    return(total_cost)
    
}
# Calculate Cost By Job Role ---- 

dept_job_role_tbl %>% 
    
    group_by(Department, JobRole, Attrition) %>%
    summarize(n = n()) %>%
    ungroup() %>%
    
    group_by(Department, JobRole) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup() %>%
    
    filter(Attrition %in% c("Yes")) %>%
    arrange(desc(pct)) %>%
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    ) %>%
    
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    )

# Worflow of Attrition ----

# Calculate Cost By Job Role ---- 

dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>%
    
    
    count_to_pct(Department, JobRole) %>%
    
    filter(Attrition %in% c("Yes")) %>%
    arrange(desc(pct)) %>%
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    ) %>%
    
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    )
# Function Tidy Eval 
count_to_pct <- function(data, ..., col = n){
        
        grouping_var_expr <- quos(...)
        col_expr <- enquo(col)
        
        ret <- data %>%
        group_by(!!! grouping_var_expr) %>%
            mutate(pct = !!col_expr/sum(!!col_expr)) %>%
            ungroup()
        return(ret)
        
    }

dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>%
    
    count_to_pct(Department, JobRole) %>%
    
    filter(Attrition %in% c("Yes")) %>%
    arrange(desc(pct)) %>%
    mutate(
        above_industry_avg = case_when(
            pct > 0.088 ~ "Yes",
            TRUE ~ "No"
        )
    ) %>%
    
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    )

assess_attrition <- function (data, attrition_col, attrition_value, baseline_pct) {
    
    attrition_col_expr <- enquo(attrition_col)
    
    data %>%
        filter((!! attrition_col_expr) %in% attrition_value) %>%
    arrange(desc(pct)) %>%
    mutate(
        above_industry_avg = case_when(
            pct > baseline_pct ~ "Yes",
            TRUE ~ "No"
        )
    )
        
}

dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>%
    count_to_pct(Department, JobRole) %>%
    assess_attrition(Attrition, attrition_value = "Yes", baseline_pct = 0.088) %>%
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    ) 
    
    # Visualization of Attrition Cost ----

dept_job_role_tbl %>% 
    
    count(Department, JobRole, Attrition) %>%
    count_to_pct(Department, JobRole) %>%
    assess_attrition(Attrition, attrition_value = "Yes", baseline_pct = 0.088) %>%
    mutate(
        cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
    ) %>%
    
    # Data Manipulation
    mutate(name = str_c(Department, JobRole, sep = ": ") %>% as_factor()) %>% 
    mutate(name = fct_reorder(name, cost_of_attrition)) %>%
    mutate(cost_text = str_c("$", format(cost_of_attrition / 1e6, digits = 2), 
                             "M", sep = "")) %>%
    
    # Plotting
    ggplot(aes(x = cost_of_attrition, y = name)) +
    geom_segment(aes(xend = 0, yend = name), color = palette_light()[[1]]) +
    geom_point(aes(size = cost_of_attrition), color = palette_light()[[1]]) +
    scale_x_continuous(labels = scales::dollar) +
    geom_label(aes(label = cost_text, size = cost_of_attrition), 
               hjust = "inward", color = palette_light()[[1]]) +
    theme_tq() +
    scale_size(range = c(3, 5)) +
    labs(title = "Estimated Cost of Attrition: By Dept and Job Role",
         y = "", x = "Cost of Attrition") +
    theme(legend.position = "none")

#save the function ----
c("assess_attrition", "count_to_pct", "calculate_attrition_cost", "plot_attrition") %>%
    dump(file = "06_Challenge/base_function.R", append = TRUE)

