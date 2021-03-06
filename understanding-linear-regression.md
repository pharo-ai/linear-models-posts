# Understanding Linear Regression

Linear regression is one of the most well-known and most commonly used machine learning algorithms.
It allows us to learn the linear dependency between the input variables $x_1, \dots, x_n$ and the output variable $y$.
Then we can use the trained model to predict the previously unseen values of y.

**A note about math:** This chapter is intended primarily for software engineers who want to understand machine learning.
It does not require deep mathematical expertise.
That being said, linear regression is one of those topics that is hard to explain without plots or formulas.
We tried to minimize the amount of math and keep it as simple as possible without sacrificing the information that is essential to understanding the algorithm.
Here is a tip on how to read this chapter:

- _Option 1: You want a simple explanation that requires no math._ Read everything except the section _"Linear Regression: Finding the Best Line to Fit the Data"_. Just consider that we find the best line through mathematical optimization.
- _Option 2: You want to understand how things work inside._ Read everything including the abovementioned section. The only mathematical knowledge that it requires is the understanding of a [derivative](https://en.wikipedia.org/wiki/Derivative). It will give you a clear understanding of how do we train the model (and why there is no magic), what is the learning rate (and why is it important to choose it well). 

## Simple Example: Predicting the Price of a House Based on its Area

We start with a simple example of estimating the price of a house that is put on sale.
In practice, many factors can influence the price: area, number of rooms, floor, distance from the city center, neighborhood, distance from the metro, market economy, etc.
But in this simple example, we will be only considering one factor: the area of a house.
We start with a trivial assumption: houses with small area are cheaper than houses with large area.
Our goal is to find a mathematical expression that would allow us to estimate as well as we can the price of the house based only on its area.

### Collecting the Data

First, we go to [Leboncoin](https://www.leboncoin.fr/) and search for houses in Lille.
We select top 50 houses into our datasets, recording the value and price of each house.
You can see them visualized in the picture below.

<img src="img/houses.png" width="500"/>

### Estimating the Price

When we need to estimate the price for a new house, the good strategy would be to look at the houses with the similar area.
Intuitively, based on the data that we have collected, we can understand that the green points are better price estimations for the house with 55 sq.m than the red points.

<img src="img/estimations.png" width="500"/>

We can also see that the data follows a certain linear pattern.
If we draw the line through those points, it will give us the best estimation of a house price based on its area.

<img src="img/housesRegression.png" width="500"/>

Such line can be defined with a line equation: $\hat y=kx+b$.
It allows us to estimate the output $y$ (in our case, the price) based on input value $x$, in:

* Parameter $k$ is called the _"slope"_, it defines the angle at which the line is rotated.
* Parameter $b$ is the intercept, it defines the distance from origin $(0,0)$ to the point $(0,y_0)$ at which the line intersects the y-axis (in other words,_"how high is the line raised from origin?"_).

By finding the optimal values of those two parameters, we find the best line.

### Which Line is the Best?

To understand what is the best line (or the best price estimation), we need to select a function that will allow us to measure the goodness of fit for every given line (or the amount of errors that it makes).
In machine learning such function is called the **cost function** --- it defines the cost of mistakes that a machine learning model makes when predicting the output on a given dataset.

The simple cost function that is commonly used for linear regression models is the _mean squared error_ (MSE).
It is calculated in the following way.

First, for every house in our dataset, we calculate the _estimation error_ ??? the difference between the price estimated by the line and the real price.

$$ e_i = \hat y_i - y_i $$

($y_i$ is the real price of the $i^{th}$ house that was posted on Leboncoin, $\hat y_i$ is the price of that same house predicted by our line).

The good line would have as small errors as possible.
For a better intuitive understanding, take a look at the image below.
It demonstrates three line estimates of the dataset: the bad line which makes large errors, the better line which makes smaller errors, and the best line which makes as little errors as possible.

<img src="img/goodRegressionLine.png" width="1000"/>

The mean squared error can now be calculated by raising each error to the power of 2 and finding the average:

$$J(w) = \frac{1}{m}\sum_{i=1}^m (\hat y_i - y_i)^2$$

($m$ is the number of houses in our dataset and $J(w)$ is the common denomination for the cost function in machine learning).

_Why did we square the errors?_
Because this ensures that they will not cancel out when we sum them.
For example, look at the _"Bad Line"_ in the picture above.
It has about the same amount of positive errors (when predicted value is greater than the real one) and negative errors (when predicted value is smaller).
The sum of those values will be close to 0 (low cost = good line), although we can see that the line is clearly making bad predictions.
Raising the errors to the power of 2 ensures that all of values are positive.

_OK, but why didn't we use absolute errors?_
We can.
Such cost function is called the Mean Absolute Error (MAE).
However, as you will see in the next section, to find the best line, we will be calculating the derivative of the cost function.
And derivating the square function $f(x) = x^2$ is much easier than the absolute function $f(x) = |x|$.

By finding the line that gives us the smallest value cost function (the lowest mean squared error), we will find the best model to predict the housing prices.

### Linear Regression: Finding the Best Line to Fit the Data

In this section, we explain how we can find the best line to fit the data with the help of mathematical optimization.
Unlike the rest of the chapter, this section requires the basic knowledge of highschool calculus, more precisely, the understanding of a [derivative](https://en.wikipedia.org/wiki/Derivative).
If you would like to skip all the math, feel free to jump directly to the next section (although we encourage you to give it a try).

The MSE will tell us how the model is performing. If the errors in the prediction are too high, the MSE function will have a high value that will indicate that the performance of the model is not good. In mathematical terms it is called the *cost function*. From now on, we will call the MSE error as the cost function.

To find the optimal value of parameters $k$ and $b$ that minimize the mean squared error, we need to differentiate Cost function with respect to $k$ and $b$ as different equations.
The result of the derivative will tell us in which direction to go to *climb the curve*. So, we need to take the negative value to go to the inverse direction.

The Cost function is a quadratic function (parabola) for both those parameters:

<img src="img/mseParabolas.png" width="10000"/>

We can expand the formula for the Cost function using the line equation $\hat y_i = kx_i + b$:

$$ Cost(k,b) = \frac{1}{m}\sum_{i=1}^m (kx_i + b - y_i)^2 $$

Now, we take the partial derivatives of both $b$ and $k$ to find their optimal value. We will skip the steps of the derivation process.

The partial derivatives of the Cost function are:

* partial derivative of the Cost function with respect of $k$:
$$ \frac{\partial}{\partial k} Cost(k,b) = \frac{2}{m} \sum_{i=1}^{m} (kx_i + b - y_i) x_i $$

* partial derivative of the Cost function with respect of $b$:
$$ \frac{\partial}{\partial b} Cost(k,b) = \frac{2}{m} \sum_{i=1}^{m} (kx_i + b - y_i) $$

We want to reach the lowest point of the parabola to minimize the error. As we said, the partial derivatives help us to reach that point.

<img src="img/OptimalValue.png" width="500"/>

At the beginning, we will start at a random point of the parabola and with the derivative we will start moving.
To reach the optimal point we should jump in the direction. But there is the risk of either jumping too much or too low. If we jump too much the model will diverge. If we jump too slow the training process will be too slow.

<img src="img/GradientDescent.png" width="500"/>

To jump efficiently we will use a **learning rate** $  \alpha $ . It is a number that typically is between 1e-6 and 1.
The learning rate will help us to control the speed of the jumping.

As we said before, we need to use the negative of the partial derivative. The derivative will tell us where to climb the most the curve, the negative where to descend the most.

After each iteration, we update both $b$ and $k$ values with the new best found point of the curve. In that way, we will by descending the curve step-by-step.

$$ k := k - \alpha \frac{\partial}{\partial k} Cost(k, b) $$

$$ b := b - \alpha \frac{\partial}{\partial b} Cost(k, b) $$

<img src="img/LearningRate.png" width="1000"/>

After reaching the maximum number of iterations or if the model converged (If it found the optimal value), the model is able to predict what would be the output of new data.
