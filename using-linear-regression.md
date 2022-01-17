# Using linear regression

Linear regression is a machine learning model that learns the linear dependencies between the independent variables and the dependent variable. It is capable of making predictions for previously unseen values once it has learned.

Here we will use the renowned [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) to train the linear regression model.

## Table of Contents  

- [Preprocessing the data](#Preprocessing-the-data)
- [Training the machine learning model](#Training-the-machine-learning-model)
- [Measuring the accuracy and other metrics of the model](#Measuring-the-accuracy-and-other-metrics-of-the-model)
- [Workflow summary](#Workflow-summary)

## [Preprocessing the data](#Preprocessing-the-data)

We will use [Pharo Datasets](https://github.com/pharo-ai/Datasets) to load the dataset into the Pharo image. The library contains several datasets ready to be loaded. Pharo Datasets will return a [Pharo DataFrame](https://github.com/PolyMathOrg/DataFrame) object. To install Pharo Datasets you only need to run the code sniped of the Metacello script available on the [README](https://github.com/pharo-ai/Datasets)

First, we load the iris dataset into the Pharo image.

```st
"Loading the dataset"
irisDataset := AIDatasets loadIris.
```

If we inspect the data, we will see that the target variable, the irises, has 3 different types of values: `setosa`, `versicolor` or `virginica`.

| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | species |
| ----------------- | ---------------- | ----------------- | ---------------- | ------- |
| 5.1               | 3.5              | 1.4               | 0.2              | setosa  |
| 6.0               | 2.9              | 4.5               | 1.5              | versicolor |
| 6.5               | 3.0              | 5.2               | 2.0              | virginica |

Our linear regression model works only with numbers. So, we need to do a mapping. We will change the value of `setosa` with 1, `versicolor` with 2, and `virginica` with 3. As we inspected the DataFrame, we know that each row has 5 columns. We need to change the last one which is the target or the dependent variable.

```st
"Mapping the target variable"
mapper := Dictionary new.
mapper add: 'setosa' -> 1.
mapper add: 'versicolor' -> 2.
mapper add: 'virginica' -> 3.

mappedIris := irisDataset collect: [ :row | 
	| mappedDataRow |
	mappedDataRow := row copy.
	mappedDataRow atIndex: 5 put: (mapper at: mappedDataRow fifth).
	mappedDataRow ].
```

Now, to train the machine model we need to separate the dataset into at least two parts: one for training and the other for testing it. We have already a library in Pharo that does that: [Random partitioner](https://github.com/pharo-ai/random-partitioner). It is already included be default if you load the Pharo Datasets library.

We will separate besides the two sets: test and train, into X (independent, input variables) and Y (dependent, output variable).

```st
"Dividing into test and training"
partitioner := AIRandomPartitioner new.
subsets := partitioner split: mappedIris withProportions: #(0.75 0.25).
irisTrainDF := subsets first.
irisTestDF := subsets second.


"Separating between X and Y"
irisTrainDF columnNames. "an OrderedCollection('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)' 'species')"

xTrain := irisTrainDF columns: #('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)').
yTrain := irisTrainDF column: 'species'.

xTest := irisTestDF columns: #('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)').
yTest := irisTestDF column: 'species'.
```

Finally, as our linear regression model only accepts array and not DataFrame objects (for now!), we need to convert the DataFrame into an array. We can do that only sending the message `asArray`.

```st
"Converting the DataFrame into an array of arrays For using it in the linear model.
For now, the linear model does not work on a DataFrame."
xTrain := xTrain asArray.
yTrain := yTrain asArray.
xTest := xTest asArray.
yTest := yTest asArray.
```

## [Training the machine learning model](#Training-the-machine-learning-model)

We have everything that is needed to start training the linear regression model. We need to load the [Linear models library](https://github.com/pharo-ai/linear-models) from pharo-ai. That library contains both the logistic regression and linear regression algorithms.

We instantiated the model, set the learning rate and the max iterations (if not set the model will be the default values). After that, we train the model with the `trainX` and `trainY` collections that we have obtained.

```st
"Training the linear regression model"
linearRegression := AILinearRegression
	learningRate: 0.001
	maxIterations: 5000.
    
linearRegression fitX: xTrain y: yTrain.
```

Now we can make predictions for previously unseen values.

```st
yPredicted := linearRegression predict: xTest.
```

## [Measuring the accuracy and other metrics of the model](#Measuring-the-accuracy-and-other-metrics-of-the-model)

We want to see how well our model is performing. In Pharo we also have a library for measuring the metrics of machine learning models: [Machine learning metrics!](https://github.com/pharo-ai/metrics). As usual, you will find the Metacello script for installing it on the README file.

For a linear regression model we have several metrics implemented:
- Mean Squared Error (AIMeanSquaredError)
- Mean Absolute Error (AIMeanAbsoluteError)
- Mean Squared Logarithmic Error (AIMeanSquaredLogarithmicError)
- R2 Score (AIR2Score)
- Root Mean Squared Error (AIRootMeanSquaredError)
- Max Error (AIMaxError)
- and Explained Variance Score (AIExplainedVarianceScore)

For this exercise will we use the R2 score metric (coefficient of determination). It is a coefficient that determinates the proportion of the variation in the dependent variable that is predictable from the independent variables. If the value of r2 is 1 means that the model predicts perfectly. You can read more information on this Wikipedia article [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).

```st
"Computing the accuracy of the logistic regression model"
metric := AIR2Score new.

r2Score "0.945929126485175" := (metric computeForActual: yTest predicted: yPredicted) asFloat.
```

We obtained a 94% as a coefficient of correlation which is quite acceptable.

## [Workflow summary](#Workflow-summary)

Here is the complete workflow of the exercise in which we have worked today. You can run everything in a Pharo Playground to play with the model.

Do not forget that you need to install the libraries for this to work.

```st
"Loading the dataset"
irisDataset := AIDatasets loadIris.


"PREPROCESSING THE DATA"
"Mapping the target variable"
mapper := Dictionary new.
mapper add: 'setosa' -> 1.
mapper add: 'versicolor' -> 2.
mapper add: 'virginica' -> 3.

mappedIris := irisDataset collect: [ :row | 
	| mappedDataRow |
	mappedDataRow := row copy.
	mappedDataRow atIndex: 5 put: (mapper at: mappedDataRow fifth).
	mappedDataRow ].


"SEPARATING THE DATA"
"Dividing into test and training"
partitioner := AIRandomPartitioner new.
subsets := partitioner split: mappedIris withProportions: #(0.75 0.25).
irisTrainDF := subsets first.
irisTestDF := subsets second.


"Separating between X and Y"
irisTrainDF columnNames. "an OrderedCollection('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)' 'species')"

xTrain := irisTrainDF columns: #('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)').
yTrain := irisTrainDF column: 'species'.

xTest := irisTestDF columns: #('sepal length (cm)' 'sepal width (cm)' 'petal length (cm)' 'petal width (cm)').
yTest := irisTestDF column: 'species'.


"Converting the DataFrame into an array of arrays For using it in the linear model.
For now, the linear model does not work on a DataFrame."
xTrain := xTrain asArray.
yTrain := yTrain asArray.
xTest := xTest asArray.
yTest := yTest asArray.


"TRAINING THE MACHINE LEARNING MODEL"
"Training the linear regression model"
linearRegression := AILinearRegression
	learningRate: 0.001
	maxIterations: 5000.

linearRegression fitX: xTrain y: yTrain.
yPredicted := linearRegression predict: xTest.


"COMPUTING METRICS"
"Computing the accuracy of the logistic regression model"
metric := AIR2Score new.

r2Score "0.945929126485175" := (metric computeForActual: yTest predicted: yPredicted) asFloat.
```