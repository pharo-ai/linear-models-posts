source('theme.R')
source('colors.R')

library(ggplot2)
library(ggthemes)
library(grid)
library(gridExtra)
library(scales)

line_color <- red
data <- read.csv(file='../data/houses.csv')

mse <- function(func, data) {
  sum <- 0
  
  for (i in seq(nrow(data))) {
    row <- data[i,]
    #prediction <- predict(model, row)[[1]]
    prediction <- func(row$Area)
    error <- (prediction - row$Price) / 1000
    sum <- sum + (error^2)
  }
  return (sum / nrow(data));
}

line_func <- function(slope, intercept)
  function(area)
    slope * area + intercept

model <- lm(Price ~ Area, data)
best.intercept <- model$coefficients[1][[1]]
best.slope <- model$coefficients[2][[1]]

slopes <- c(2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000)

houses <- ggplot(data, aes(x=Area, y=Price)) +
  geom_point() +
  labs(x="Area (sq. m)", y="Price (euros)") +
  theme_Publication() +
  theme(
    aspect.ratio=1,
    panel.grid.major=element_blank(),
    text=element_text(family="Avenir"))

for (slope in slopes) {
  houses <- houses +
    geom_abline(
      intercept=best.intercept,
      slope=slope,
      color=line_color,
      lwd=.8)
}

costs <- lapply(slopes, function(slope) {
  func <- line_func(slope, best.intercept)
  cost <- mse(func, data)
  return(cost);
})

df <- data.frame(
  Slope = slopes,
  Cost = costs)

ggplot(df, aes(x=Slope, y=Cost)) +
  geom_point()

