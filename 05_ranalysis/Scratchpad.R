library(tidyverse)

df = read.csv("combined.csv")

df$timestamp[1:532] <- sub(
  "^13-05-2026",
  "12-05-2026",
  df$timestamp[1:532]
)

df <- df %>%
  mutate(
    close_to_entry = case_when(
      table %in% c("table_3", "table_4") ~ TRUE,
      table %in% c("table_1", "table_2") ~ FALSE
    ),
    close_to_entry = factor(close_to_entry)
  )

df$datetime_parsed <- as.POSIXct(df$timestamp, format = "%d-%m-%Y %H:%M:%OS")
df$date <- as.Date(df$datetime_parsed) + 1
df$time <- format(df$datetime_parsed, "%H:%M:%S")

correct_starts <- data.frame(
  date = as.Date(c("2026-05-12", "2026-05-13", "2026-05-15")),
  correct_start = as.POSIXct(c("2026-05-12 12:16:39",
                               "2026-05-13 12:03:45",
                               "2026-05-15 12:13:58"))
)

for (i in seq_len(nrow(correct_starts))) {
  d <- correct_starts$date[i]
  rows <- which(df$date == d)
  
  current_start <- min(df$datetime_parsed[rows])
  offset <- difftime(correct_starts$correct_start[i], current_start, units = "secs")
  
  df$datetime_parsed[rows] <- df$datetime_parsed[rows] + as.numeric(offset)
  df$time[rows] <- format(df$datetime_parsed[rows], "%H:%M:%S")
}

df = subset(df, select = -c(timestamp, datetime_parsed))
write.csv(df, file="timecorrected.csv")

plot(df$time, df$occupancy)

df$time_plot <- as.POSIXct(df$time, format = "%H:%M:%S")

ggplot(df, aes(x = time_plot, y = (occupancy != 0), color = as.factor(close_to_entry))) +
  geom_line() +
  scale_x_datetime(date_labels = "%H:%M") +
  labs(x = "Time of Day", y = "Occupancy", color = "Close to entry") +
  theme_minimal()

close_subset = subset(df, close_to_entry == TRUE)
far_subset = subset(df, close_to_entry == FALSE)

mean(subset(df, close_to_entry == TRUE)$occupancy)
mean(subset(df, close_to_entry == TRUE)$occupancy != 0)
mean(subset(df, close_to_entry == FALSE)$occupancy)
mean(subset(df, close_to_entry == FALSE)$occupancy != 0)

model1 = glm(occupied ~ close_to_entry, data=df, family="binomial")

ggplot(df, aes(x = time_plot, y = temp, color = as.factor(date))) +
  geom_line() +
  scale_x_datetime(date_labels = "%H:%M") +
  labs(x = "Time of Day", y = "Temp", color = "Date") +
  theme_minimal()

hist(far_subset$occupancy, breaks=seq(-0.5, 8.5,1))
hist(close_subset$occupancy)
