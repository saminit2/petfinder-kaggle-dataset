# Pet Adoption Speed Prediction

This repository contains code for predicting pet adoption speed using machine learning models. The dataset used for training and testing the models can be downloaded from [this link](https://drive.google.com/file/d/1Ae3X8X2tmCDVkfk_ev9i8Nm7ByytUZ-h/view).

## About the Dataset
The adoption speed values are determined as follows:
- **0**: Pet was adopted on the same day as it was listed.
- **1**: Pet was adopted between 1 and 7 days (1st week) after being listed.
- **2**: Pet was adopted between 8 and 30 days (1st month) after being listed.
- **3**: Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
- **4**: No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days.)

## Code Overview
The code is organized into several functions to maintain clarity and modularity. Here's an overview of the main functionalities:
- **Data Loading and Preprocessing**: The dataset is loaded from the provided Google Drive link. Null values are checked and processed accordingly.
- **Feature Engineering**: Categorical features are one-hot encoded, and text data is vectorized using CountVectorizer.
- **Model Training and Evaluation**: Two classifiers, Decision Tree and Random Forest, are trained and evaluated. Grid Search is performed to optimize hyperparameters for the Random Forest model.
- **Model Persistence**: Trained models are saved using pickle for future use.
- **Visualization**: Confusion matrices are plotted to visualize the performance of the trained models.

## Instructions
1. **Download the Dataset**: Click [here](https://drive.google.com/file/d/1Ae3X8X2tmCDVkfk_ev9i8Nm7ByytUZ-h/view) to download the dataset from Google Drive.
2. **Clone the Repository**: Clone this repository to your local machine using `git clone`.
3. **Install Dependencies**: Install the required dependencies listed in `requirements.txt` using `pip install -r requirements.txt`.
4. **Run the Code**: Execute the `pet_finder.py` script to train and evaluate the models.

## File Structure
- `pet_finder.py`: Main Python script containing the code for data loading, preprocessing, model training, evaluation, and visualization.
- `petfinder-mini.csv`: Dataset containing information about pets and their adoption speed.
- `README.md`: This file contains information about the project.

## Tech Stack
The project is implemented using:
- Python
- pandas
- scikit-learn
- matplotlib

## Contributors
- [Your Name](https://github.com/yourusername)

Feel free to contribute to this project by submitting issues or pull requests!


