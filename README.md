# petfinder-kaggle-dataset
## ***Training a model for Petfinder dataset from Kaggle***

I tried to train a model for petfinder dataset from Kaggle for my first project. The obligation was to predict how fast a pet is adopted in 
an animal shelter.
I used Goole Colab since I'm low on resource. 
Type of data i dealt with was a `.csv` file.

### *Addoptation speed values are determined in the following way:*
0 - Pet was adopted on the same day as it was listed.

1 - Pet was adopted between 1 and 7 days (1st week) after being listed.

2 - Pet was adopted between 8 and 30 days (1st month) after being listed.

3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.

4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100


I used [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)  and [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn-model-selection-gridsearchcv) from [scikit-learn](https://scikit-learn.org/)

In this notebook i reached the accuracy of 88.51%; feel free to enhance it.
I have uploaded the `.pkl` file and the link to the [dataset](https://drive.google.com/file/d/1Ae3X8X2tmCDVkfk_ev9i8Nm7ByytUZ-h/view?usp=drive_link)
