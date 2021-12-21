# Logistic Regression

Script for the entire workflow for the logistic regression:

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