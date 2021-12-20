source('theme.R')
source('colors.R')

library(ggplot2)
library(ggthemes)
library(grid)
library(gridExtra)
library(scales)

line_color <- blue

n <- 50
areas <- runif(n) * 40 + 10
prices <- areas * 10 + 200 + rnorm(length(areas)) * 50

data <- data.frame(
  area = areas,
  price = prices)

line_func <- function(slope, intercept)
  function(area)
    slope * area + intercept

error_lines <- function(pred_func)
  geom_segment(aes(
    x=area,
    xend=area,
    y=price,
    yend=pred_func(area)),
    color=red)

error_squares <- function(data, pred_func) {
  squares <- c()
  
  for (i in seq(nrow(data))) {
    area <- data$area[i]
    true_price <- data$price[i]
    pred_price <- pred_func(area)
    
    ymin <- min(true_price, pred_price)
    ymax <- max(true_price, pred_price)
    xmin <- area
    xmax <- area + (ymax - ymin) / (max(data$price) - min(data$price)) * (max(data$area) - min(data$area))
    
    square <- ggplot2::annotate(
      "rect",
      xmin=xmin,
      xmax=xmax,
      ymin=ymin,
      ymax=ymax,
      fill=red, alpha=0.1)
    
    squares <- c(squares, square)
  }
  
  return(squares);
}

draw_regression <- function(data, slope, intercept, title)
  ggplot(data, aes(x=area, y=price)) +
    #error_squares(data, line_func(slope, intercept)) +
    error_lines(line_func(slope, intercept)) +
    geom_abline(intercept=intercept, slope=slope, color=line_color, lwd=.8) +
    geom_point() +
    coord_cartesian(xlim=range(data$area)) +
    ggtitle(title) +
    labs(x="Area (sq. m)", y="Price (euros)") +
    theme_Publication() +
    theme(
      aspect.ratio=1,
      panel.grid.major=element_blank(),
      text=element_text(family="Avenir"))

model <- lm(price ~ area, data)

p1 <- draw_regression(data, -5, 700, "Bad Line")
p2 <- draw_regression(data, 4, 320, "Better Line")
p3 <- draw_regression(data, model$coefficients[2], model$coefficients[1], "Best Line")

result <- grid.arrange(arrangeGrob(p1, p2, p3, ncol=3))

ggsave(
  '../img/goodRegressionLine.png',
  plot=result,
  width=50, height=18, units="cm",
  scale=.7,
  dpi="retina")
