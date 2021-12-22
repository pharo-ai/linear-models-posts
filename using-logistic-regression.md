# Using Logistic Regression

Write something here

## Preprocessing the data

In Pharo, we have a library for loading several dataset directly into Pharo as DataFrame objects. [Pharo Datasets](https://github.com/pharo-ai/Datasets). It contains the well-know examples like the [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) and other ones.
Here we will use a dataset from the [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.kaggle.com/uciml/pima-indians-diabetes-database) for predicting if a patient has or not diabetes.

First we need to install the library using the [Metacello script](https://github.com/pharo-ai/Datasets) available in the README. The method `loadXXX` will return a [DataFrame object](https://github.com/PolyMathOrg/DataFrame).

[Pharo DataFrame](https://github.com/PolyMathOrg/DataFrame) can be seen as the Pharo equivalent of [Python Pandas](https://pandas.pydata.org/).

```st
"Loading the dataset"
diabetesPima := AIDatasets loadDiabetesPima.
```

Now, for training our model, we need at least two partitions of the data set: one for training and the other for measuring the accurancy of the model. Also in Pharo, we have a small library to help you with that task! [Random Partitioner](https://github.com/pharo-ai/random-partitioner). First random shuffle the data and then partitions the data to the given proportions. This is included by default when you load the [Pharo Datasets](https://github.com/pharo-ai/Datasets) or [Pharo DataFrame](https://github.com/PolyMathOrg/DataFrame). So, you do not need to install it again. We will partiton our data into two sets: training and test with a proportion of 75%-25%.

```st
"Dividing into test and training"
partitioner := AIRandomPartitioner new.
subsets := partitioner split: normalizedDF withProportions: #(0.75 0.25).
diabetesPimaTrainDF := subsets first.
diabetesPimaTestDF := subsets second.
```

As a next step, we can separate the features between X and Y. That means: into the independent variables `X` and the dependent variable `Y`. As we have the data loaded in a DataFrame object, we can select the desire columns.

If we inspect the variable `diabetesPima columnNames` we will see: `an OrderedCollection('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age' 'Outcome')`. Where `Outcome` is the dependent variable and all the rest are the independent ones.

The method `DataFrame>>columns:` will return a new DataFrame with the specify columns.

```st
"Separating between X and Y"
diabetesPimaTrainDF columnNames. "an OrderedCollection('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age' 'Outcome')"

xTrain := diabetesPimaTrainDF columns: #('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age').
yTrain := diabetesPimaTrainDF column: 'Outcome'.

xTest := diabetesPimaTestDF columns: #('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age').
yTest := diabetesPimaTestDF column: 'Outcome'.
```

Now we have everything that we need to start training our machine learning model!

## Training the machine learning model



The summary of all the workflow is:

```st
"Loading the dataset"
diabetesPima := AIDatasets loadDiabetesPima.


"Normalizing the data frames"
normalizedDF := diabetesPima normalized: AIMinMaxNormalizer.


"Dividing into test and training"
partitioner := AIRandomPartitioner new.
subsets := partitioner split: normalizedDF withProportions: #(0.75 0.25).
diabetesPimaTrainDF := subsets first.
diabetesPimaTestDF := subsets second.


"Separating between X and Y"
diabetesPimaTrainDF columnNames. "an OrderedCollection('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age' 'Outcome')"

xTrain := diabetesPimaTrainDF columns: #('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age').
yTrain := diabetesPimaTrainDF column: 'Outcome'.

xTest := diabetesPimaTestDF columns: #('Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI' 'DiabetesPedigreeFunction' 'Age').
yTest := diabetesPimaTestDF column: 'Outcome'.


"Converting the DataFrame into an array of arrays For using it in the linar model.
For now, the linear model does not work on a DataFrame."
xTrain := xTrain asArray.
yTrain := yTrain asArray.
xTest := xTest asArray.
yTest := yTest asArray.


"Training the logistic regression model"
logisticRegression := AILogisticRegression
	learningRate: 3
	maxIterations: 5000.

logisticRegression f~itX: xTrain y: yTrain.
yPredicted := logisticRegression predict: xTest.


"Computing the accuracy of the logistic regression model"
metric := AIAccuracyScore new.
{ (metric computeForActual: yTest predicted: yPredicted) asFloat . logisticRegression iterationsPerformed }
```

git status