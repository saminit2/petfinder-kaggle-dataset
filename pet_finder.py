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

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def check_null_values(data):
    """Check for null values in the dataset."""
    return data.isnull().sum()

def preprocess_data(data):
    """Preprocess the data."""
    X = data.drop(['AdoptionSpeed', 'Description'], axis=1)
    Y = data['AdoptionSpeed']
    D = data['Description'].fillna('')
    return X, Y, D

def vectorize_text_data(D):
    """Vectorize text data using CountVectorizer."""
    cv = CountVectorizer()
    return cv.fit_transform(D)

def encode_categorical_features(X):
    """Encode categorical features using OneHotEncoder."""
    categorical_features = ["Type", "Breed1", "Gender", "Color1", "Color2", "MaturitySize", "FurLength", "Vaccinated", "Sterilized", "Health"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
    return transformer.fit_transform(X)

def combine_features(OH_X, D_transformed):
    """Combine encoded features and text data."""
    return hstack([OH_X, D_transformed])

def split_data(OH_X_with_text, Y, test_size=0.2):
    """Split data into train and test sets."""
    return train_test_split(OH_X_with_text, Y, test_size=test_size)

def train_decision_tree_classifier(xtrain, ytrain):
    """Train a Decision Tree classifier."""
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    return model

def evaluate_classifier(model, xtest, ytest):
    """Evaluate the classifier and return metrics."""
    y_preds = model.predict(xtest)
    accuracy = accuracy_score(ytest, y_preds)
    precision = precision_score(ytest, y_preds, average='weighted')
    recall = recall_score(ytest, y_preds, average='weighted')
    f1 = f1_score(ytest, y_preds, average='weighted')
    cm = confusion_matrix(ytest, y_preds)
    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(model, xtest, ytest, title):
    """Plot the confusion matrix for the classifier."""
    y_preds = model.predict(xtest)
    cm = confusion_matrix(ytest, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def train_random_forest_classifier(xtrain, ytrain, n_estimators=100, max_depth=None, max_features='sqrt', min_samples_split=2, min_samples_leaf=1):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1, verbose=1)
    model.fit(xtrain, ytrain)
    return model

def grid_search_model(model, grid, xtrain, ytrain):
    """Perform Grid Search to find the best parameters."""
    gs_clf = GridSearchCV(estimator=model, param_grid=grid, cv=2, verbose=1)
    gs_clf.fit(xtrain, ytrain)
    return gs_clf

def save_model(model, file_path):
    """Save the trained model."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    """Load a trained model."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    # Load dataset
    data = load_dataset("/content/drive/MyDrive/pet_finder/petfinder-mini.csv")

    # Check for null values
    check_null_values(data)

    # Preprocess data
    X, Y, D = preprocess_data(data)

    # Vectorize text data
    D_transformed = vectorize_text_data(D)

    # Encode categorical features
    OH_X = encode_categorical_features(X)

    # Combine features
    OH_X_with_text = combine_features(OH_X, D_transformed)

    # Split data
    xtrain, xtest, ytrain, ytest = split_data(OH_X_with_text, Y)

    # Model 1: Decision Tree Classifier
    model_1 = train_decision_tree_classifier(xtrain, ytrain)
    accuracy_model_1, precision_model_1, recall_model_1, f1_model_1, cm_model_1 = evaluate_classifier(model_1, xtest, ytest)
    plot_confusion_matrix(model_1, xtest, ytest, "Confusion Matrix Model 1")

    # Model 2: Random Forest Classifier
    model_2 = train_random_forest_classifier(xtrain, ytrain)
    accuracy_model_2, precision_model_2, recall_model_2, f1_model_2, cm_model_2 = evaluate_classifier(model_2, xtest, ytest)
    plot_confusion_matrix(model_2, xtest, ytest, "Confusion Matrix Model 2")

    # Grid Search for Model 2
    grid = {'n_estimators': [10, 100, 1000], 'max_depth': [None, 5, 10, 20], 'max_features': ["sqrt", "log2"], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]}
    gs_clf = grid_search_model(model_2, grid, xtrain, ytrain)

    # Best parameters for Model 2
    best_params_model_2 = gs_clf.best_params_

    # Saving Model 2 with Grid Search best parameters
    save_model(gs_clf, "/content/drive/MyDrive/pet_finder/model_2_random_forest_Grid_search_best_param.pkl")

    # Loading Model 2 with Grid Search best parameters
    model_2_GS_params_loaded = load_model('/content/drive/MyDrive/pet_finder/model_2_random_forest_Grid_search_best_param.pkl')

    # Predictions with loaded model
    model_2_GS_params_loaded_accuracy, model_2_GS_params_loaded_precision, model_2_GS_params_loaded_recall, model_2_GS_params_loaded_f1, model_2_GS_params_loaded_cm = evaluate_classifier(model_2_GS_params_loaded, xtest, ytest)

    # Confusion Matrix for loaded model
    plot_confusion_matrix(model_2_GS_params_loaded, xtest, ytest, "Confusion Matrix Model 2 (Grid Search)")

    # Saving Model 2 with Grid Search best parameters (88% accuracy)
    save_model(model)

