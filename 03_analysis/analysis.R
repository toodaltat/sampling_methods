###################################
# Setup
###################################
library(tidyverse)
library(lubridate)
df <- read.csv("../02_output/smoothed_occupancy_log.csv")




###################################
# Zero  count
###################################
zero_summary_clean <- df %>%
  summarise(
    zero_occupancy = sum(occupancy == 0),
    occupied_rows = sum(occupancy > 0),
    total_rows = n(),
    percent_zero = round(mean(occupancy == 0) * 100, 2)
  ) %>%
  rename(
    `Zero occupancy rows` = zero_occupancy,
    `Occupied rows` = occupied_rows,
    `Total rows` = total_rows,
    `Percent zero (%)` = percent_zero
  )

###################################
# Temp occupancy graph
###################################
df <- df %>%
  mutate(
    timestamp_dt = dmy_hms(timestamp),
    day = as.Date(timestamp_dt)
  )

temp_summary <- df %>%
  group_by(temp, day) %>%
  summarise(
    mean_occupancy = mean(occupancy),
    total_occupancy = sum(occupancy),
    n = n(),
    .groups = "drop"
  )

temp_occupancy_plot <- ggplot(temp_summary, aes(x = temp, y = mean_occupancy, colour = factor(day))) +
  geom_point() +
  geom_line(aes(group = day)) +
  labs(
    title = "Mean Occupancy by Temperature and Day",
    x = "Temperature",
    y = "Mean Occupancy",
    colour = "Day"
  )

###################################
# Close to entry occupancy graph
###################################
entry_occupancy_plot <- ggplot(df, aes(x = close_to_entry, y = occupancy)) +
  geom_boxplot() +
  geom_jitter(width = 0.15, alpha = 0.3) +
  labs(
    title = "Occupancy by Entry Point",
    x = "Close to Entry",
    y = "Occupancy Count"
  )

###################################
# Length of occupancy
###################################
df_runs <- df %>%
  mutate(
    timestamp_dt = dmy_hms(timestamp),
    date = as.Date(timestamp_dt),
    occupied = occupancy > 0
  ) %>%
  arrange(table, date, timestamp_dt) %>%
  group_by(table, date) %>%
  mutate(
    new_run = occupied != lag(occupied, default = first(occupied)),
    run_id = cumsum(new_run)
  ) %>%
  group_by(table, date, close_to_entry, run_id, occupied) %>%
  summarise(
    start_time = min(timestamp_dt),
    end_time = max(timestamp_dt),
    n_intervals = n(),
    duration_minutes = n_intervals * 0.5,
    mean_occupancy = mean(occupancy),
    max_occupancy = max(occupancy),
    .groups = "drop"
  )


occupied_runs <- df_runs %>%
  filter(occupied == TRUE)

occupied_runs %>%
  group_by(close_to_entry) %>%
  summarise(
    n_runs = n(),
    mean_duration_minutes = mean(duration_minutes),
    median_duration_minutes = median(duration_minutes),
    mean_occupancy_during_runs = mean(mean_occupancy),
    max_duration_minutes = max(duration_minutes),
    .groups = "drop"
  )

occupied_runs %>%
  group_by(close_to_entry) %>%
  slice_max(duration_minutes, n = 1, with_ties = TRUE) %>%
  select(
    close_to_entry,
    table,
    date,
    start_time,
    end_time,
    duration_minutes,
    mean_occupancy,
    max_occupancy
  )

run_summary_table <- occupied_runs %>%
  group_by(close_to_entry) %>%
  summarise(
    `Runs` = n(),
    `Mean duration (min)` = mean(duration_minutes),
    `Median duration (min)` = median(duration_minutes),
    `Mean occupancy` = mean(mean_occupancy),
    `Max duration (min)` = max(duration_minutes),
    .groups = "drop"
  ) %>%
  mutate(
    `Close to entry` = ifelse(close_to_entry, "Yes", "No"),
    `Mean duration (min)` = round(`Mean duration (min)`, 2),
    `Median duration (min)` = round(`Median duration (min)`, 2),
    `Mean occupancy` = round(`Mean occupancy`, 2),
    `Max duration (min)` = round(`Max duration (min)`, 2)
  ) %>%
  select(
    `Close to entry`,
    Runs,
    `Mean duration (min)`,
    `Median duration (min)`,
    `Mean occupancy`,
    `Max duration (min)`
  )

###################################
# Logistic model
###################################
df <- df %>%
  mutate(
    timestamp_dt = dmy_hms(timestamp),
    date = as.Date(timestamp_dt),
    hour = hour(timestamp_dt),
    occupied = occupancy > 0,
    in_shadow = factor(in_shadow),
    close_to_entry = factor(close_to_entry)
  )

logistic_model_time <- glm(
  occupied ~ in_shadow + temp + dist_from_road + close_to_entry + factor(date) + hour,
  data = df,
  family = binomial
)

logistic_results <- as.data.frame(summary(logistic_model_time)$coefficients)

logistic_results <- logistic_results %>%
  mutate(
    Odds_ratio = exp(Estimate),
    CI_lower = exp(Estimate - 1.96 * `Std. Error`),
    CI_upper = exp(Estimate + 1.96 * `Std. Error`),
    `95% CI` = paste0(
      round(CI_lower, 2),
      " - ",
      round(CI_upper, 2)
    )
  ) %>%
  rename(
    `p-value` = `Pr(>|z|)`
  ) %>%
  select(
    `Odds ratio` = Odds_ratio,
    `95% CI`,
    `p-value`
  ) %>%
  mutate(
    `Odds ratio` = round(`Odds ratio`, 2),
    `p-value` = format.pval(`p-value`, digits = 3, eps = 0.001)
  )

###################################
# Poisson model
###################################
occupied_df <- df %>%
  filter(occupancy > 0)

count_model <- glm(
  occupancy ~ in_shadow + temp + dist_from_road + close_to_entry + factor(date) + hour,
  data = occupied_df,
  family = poisson
)

dispersion <- sum(residuals(count_model, type = "pearson")^2) / count_model$df.residual

count_results <- as.data.frame(summary(count_model)$coefficients)

count_results <- count_results %>%
  mutate(
    Rate_ratio = exp(Estimate)
  ) %>%
  rename(
    `Std. error` = `Std. Error`,
    `Rate ratio` = Rate_ratio,
    `p-value` = `Pr(>|z|)`
  ) %>%
  select(
    Estimate,
    `Rate ratio`,
    `Std. error`,
    `p-value`
  ) %>%
  mutate(
    Estimate = round(Estimate, 3),
    `Rate ratio` = round(`Rate ratio`, 3),
    `Std. error` = round(`Std. error`, 3),
    `p-value` = format.pval(`p-value`, digits = 3, eps = 0.001)
  )
