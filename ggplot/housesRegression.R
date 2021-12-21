source('theme.R')
source('colors.R')

library(ggplot2)
library(ggthemes)
library(grid)
library(gridExtra)
library(scales)

line_color <- blue

data <- read.csv(file='../data/houses.csv')

houses <- ggplot(data, aes(x=Area, y=Price)) +
  geom_point() +
  labs(x="Area (sq. m)", y="Price (euros)") +
  theme_Publication() +
  theme(
    aspect.ratio=1,
    panel.grid.major=element_blank(),
    text=element_text(family="Avenir"))

ggsave(
  '../img/houses.png',
  plot=houses,
  width=9, height=9, units="cm",
  scale=1.2,
  dpi="retina")

model <- lm(Price ~ Area, data)

housesRegression <- houses +
  geom_abline(
   intercept=model$coefficients[1],
   slope=model$coefficients[2],
   color=line_color,
   lwd=.8) 

ggsave(
  '../img/housesRegression.png',
  plot=housesRegression,
  width=9, height=9, units="cm",
  scale=1.2,
  dpi="retina")
