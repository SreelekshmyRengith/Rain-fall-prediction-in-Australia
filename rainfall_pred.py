import pandas as pd
import numpy as np
import sklearn

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt





#loading the dataset
df=pd.read_csv(r'C:\Fall 23\ENPM 808L\Final Project\weatherAUS.csv')


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
px.histogram(df, x='Location', title='Location vs. Rainy Days', color='RainToday').show()

px.histogram(df, 
             x='Temp3pm', 
             title='Temperature at 3 pm vs. Rain Tomorrow', 
             color='RainTomorrow').show()

px.histogram(df, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today').show()

px.scatter(df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday').show()

px.scatter(df.sample(2000), 
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow').show()

print("Initial # of rows: ", len(df))

#----Data Cleaning and Preprocessing----

#getting details about the columns
print(df.info())
#dropping the rows where the value of RainToday and RainTomorrow are missing
df.dropna(subset=['RainToday','RainTomorrow'],inplace=True)

# Fill numeric rows with the mean value where all elements are NaN
# Identify numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Initialize the SimpleImputer with the "mean" strategy
imputer = SimpleImputer(strategy="mean")
# Fit the imputer on the numeric columns and transform the DataFrame
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

#Filling in the NaN elements for categorical data 
# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Initialize the SimpleImputer with the "most_frequent" strategy for categorical columns
categorical_imputer = SimpleImputer(strategy="most_frequent")
# Fit the imputer on the categorical columns and transform the DataFrame
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
#counting the NaN values in the Dataset
nan_counts = df.isna().sum()


#viewing the numeric columns
print(numeric_cols)
#encoding categorical columns
#identifying the categorical columns
print(df[categorical_cols].nunique()[1:-1])

new_target = LabelEncoder()
df['target_RainTomorrow'] = new_target.fit_transform(df['RainTomorrow'])
Location_label=LabelEncoder()
df['Location_new']=Location_label.fit_transform(df['Location'])
WindGustDir_label=LabelEncoder()
df['WindGustDir_new']=WindGustDir_label.fit_transform(df['WindGustDir'])
WindDir9am_label=LabelEncoder()
df['WindDir9am_new']=WindDir9am_label.fit_transform(df['WindDir9am'])
WindDir3pm_label=LabelEncoder()
df['WindDir3pm_new']=WindDir3pm_label.fit_transform(df['WindDir3pm'])
RainToday_label=LabelEncoder()
df['RainToday_new']=RainToday_label.fit_transform(df['RainToday'])
inputs = df.drop(['Date','RainTomorrow','Location','WindGustDir','WindDir9am','WindDir3pm','RainToday'],axis="columns",inplace=True)
inputs_columns = ['Location_new','WindGustDir_new','WindDir9am_new','WindDir3pm_new','RainToday_new','MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
# Create the 'inputs' DataFrame with the specified columns
inputs = df[inputs_columns]
target = df['target_RainTomorrow']

print("Columns: >>",inputs)
print("target :>>",target)
print(df.shape)
print(nan_counts)


#splitting into training and testing sets
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(inputs,target,test_size=0.25,random_state=42)
# Initialize models
lr_model = LogisticRegression(random_state=0, max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=0)
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()
lda_model = LinearDiscriminantAnalysis()

# Models to evaluate
models = [lr_model, dt_model, knn_model, nb_model, lda_model]

# Evaluate each model using k-fold cross-validation
k = 10
for model in models:
    model_name = model.__class__.__name__
    cross_val_scores = cross_val_score(model, inputs, target, cv=k)
    
    # Display the cross-validation results
    for fold, score in enumerate(cross_val_scores, 1):
        print(f'{model_name} - Fold {fold}: Accuracy = {score:.2f}')
    
    # Calculate the mean accuracy of all folds
    mean_accuracy = np.mean(cross_val_scores)
    print(f'{model_name} - Mean Accuracy = {mean_accuracy:.2f}')

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test)


    # Constructing a confusion matrix to calculate the accuracy of the model
    cm = confusion_matrix(y_test, y_pred)
    print(f'{model_name} - Confusion Matrix:\n', cm)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} - Accuracy: {accuracy:.2f}\n')

    #print the precision, recall and F1 score for each model
    print(classification_report(y_test, y_pred))
    
    # Check if the model supports predict_proba
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right")
        plt.show()