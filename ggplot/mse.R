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
    error <- (prediction - row$Price)# / 1000
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

slopes <- seq(1880, 3000, 20)

costs <- sapply(slopes, function(slope) {
  func <- line_func(slope, best.intercept)
  cost <- mse(func, data)
  return(cost);
})

df <- data.frame(
  Slope = slopes,
  Cost = costs)

slope.mse <- ggplot(df, aes(x=Slope, y=Cost)) +
  geom_line(color=line_color, lwd=.8) +
  labs(x="Slope (parameter k)", y="Mean Squared Error (MSE)") +
  theme_Publication() +
  theme(
    aspect.ratio=0.8,
    panel.grid.major=element_blank(),
    text=element_text(family="Avenir"))

intercepts <- seq(70000, 92300, 100)

costs <- sapply(intercepts, function(intercept) {
  func <- line_func(best.slope, intercept)
  cost <- mse(func, data)
  return(cost);
})

df <- data.frame(
  Intercept = intercepts,
  Cost = costs)

intercept.mse <- ggplot(df, aes(x=Intercept, y=Cost)) +
  geom_line(color=line_color, lwd=.8) +
  labs(x="Intercept (parameter b)", y="Mean Squared Error (MSE)") +
  theme_Publication() +
  theme(
    aspect.ratio=0.8,
    panel.grid.major=element_blank(),
    text=element_text(family="Avenir"))

result <- grid.arrange(arrangeGrob(slope.mse, intercept.mse, ncol=2))

ggsave(
  '../img/mseParabolas.png',
  plot=result,
  width=50, height=18, units="cm",
  scale=.7,
  dpi="retina")