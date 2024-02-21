import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import pickle

# Importing the dataset
data = pd.read_csv("/content/drive/MyDrive/pet_finder/petfinder-mini.csv")

# Checking for null values
data.isnull().sum()

# Separating features and target variable
X = data.drop(['AdoptionSpeed', 'Description'], axis=1)
Y = data['AdoptionSpeed']
D = data['Description'].fillna('')  # Filling null values in Description column

# Vectorizing text data
cv = CountVectorizer()
D_transformed = cv.fit_transform(D)

# Encoding categorical features
categorical_features = ["Type", "Breed1", "Gender", "Color1", "Color2", "MaturitySize", "FurLength", "Vaccinated", "Sterilized", "Health"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
OH_X = transformer.fit_transform(X)

# Combining text and encoded features
OH_X_with_text = hstack([OH_X, D_transformed])

# Splitting data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(OH_X_with_text, Y, test_size=0.2)

# Model 1: Decision Tree Classifier
model_1 = DecisionTreeClassifier()
model_1.fit(xtrain, ytrain)
accuracy_model_1 = model_1.score(xtest, ytest)

# Evaluating Model 1
y_preds_model_1 = model_1.predict(xtest)
metrics_model_1 = evaluate_preds(ytest, y_preds_model_1)

# Confusion Matrix for Model 1
plot_confusion_matrix(model_1, xtest, ytest, "Confusion Matrix Model 1")

# Model 2: Random Forest Classifier
model_2 = RandomForestClassifier(n_jobs=-1, verbose=1)
model_2.fit(xtrain, ytrain)
accuracy_model_2 = model_2.score(xtest, ytest)

# Evaluating Model 2
y_preds_model_2 = model_2.predict(xtest)
metrics_model_2 = evaluate_preds_weighted(ytest, y_preds_model_2)

# Confusion Matrix for Model 2
plot_confusion_matrix(model_2, xtest, ytest, "Confusion Matrix Model 2")

# Grid Search for Model 2
grid = {'n_estimators': [10, 100, 1000], 'max_depth': [None, 5, 10, 20], 'max_features': ["sqrt", "log2"], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]}
gs_clf = GridSearchCV(estimator=model_2, param_grid=grid, cv=2, verbose=1)
gs_clf.fit(xtrain, ytrain)

# Best parameters for Model 2
best_params_model_2 = gs_clf.best_params_

# Saving Model 2 with Grid Search best parameters
with open("/content/drive/MyDrive/pet_finder/model_2_random_forest_Grid_search_best_param.pkl", 'wb') as file:
    pickle.dump(gs_clf, file)

# Loading Model 2 with Grid Search best parameters
model_2_GS_params_loaded = pickle.load(open('/content/drive/MyDrive/pet_finder/model_2_random_forest_Grid_search_best_param.pkl', 'rb'))

# Predictions with loaded model
model_2_GS_params_loaded_y_preds = model_2_GS_params_loaded.predict(xtest)

# Evaluating loaded model
model_2_GS_params_loaded_metrics = evaluate_preds_weighted(ytest, model_2_GS_params_loaded_y_preds)

# Confusion Matrix for loaded model
plot_confusion_matrix(model_2_GS_params_loaded, xtest, ytest, "Confusion Matrix Model 2 (Grid Search)")

# Saving Model 2 with Grid Search best parameters (88% accuracy)
with open("/content/drive/MyDrive/pet_finder/model_2_random_forest_Grid_search_best_param_88%.pkl", 'wb') as file:
    pickle.dump(model_2_GS_params_loaded, file)
