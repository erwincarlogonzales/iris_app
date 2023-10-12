# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# import datasets
from sklearn import datasets

# preprocessing
from sklearn.model_selection import train_test_split

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import accuracy_score

# create title page
st.title('Iris Prediction Model')

# write a text for your app
st.write("""
         # Explore this classifier
         ## Iris and Wine dataset classifier sample
         """)

# create a selectbox as a sidebar with the Iris dataset and assign it to a variable
dataset = st.sidebar.selectbox('Select Dataset', ('Iris', 'Wine'))

# select a built-in ML model as a sidebar selectbox
model = st.sidebar.selectbox('Select Classifier', ('KNN', 'Random Forest'))

# create a fuction to get dataset from sklearn
def get_dataset(dataset):
    if dataset == 'Iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()
    
    # identify target y and features X
    y = data.target
    X = data.data
    
    return X, y

X, y = get_dataset(dataset)

# display the shape of the dataset
st.write(f'the dataset contains {len(X)} columns and {X.shape[1]} rows')

# display the number of classes in the dataset
st.write('Number of classes', len(np.unique(y)))

# create a function that selects the parameters for the ML models using a slider
def add_param_ui(classifier_name):
    
    # create an empty dictionary
    params = dict()
    
    # create a slider for the parameters
    if classifier_name == 'KNN':
        K = st.sidebar.slider('K-neighbors', 1, 10)
        
        # append the dictionary
        params['K'] = K
        
    else:
        # random forest parameters
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        
         # append the dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators    
        
    return params

# call the function with the model
params = add_param_ui(model)

# create a function that calls the ML classifier
def get_classifier(classifier_name, params):
    
    # create a slider for the parameters
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=params['K'])   
    else:
        # random forest parameters
        classifier = RandomForestClassifier(n_estimators = params['n_estimators'],
                                            max_depth = params['max_depth'], random_state = 42)
    
    return classifier

# call the function
classifier = get_classifier(model, params)
        
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# fit the dataset
classifier.fit(X_train, y_train)

# make prediction
y_pred = classifier.predict(X_test)

# evaluate model accuracy
acc = accuracy_score(y_test, y_pred)

# display accuracy
st.write(f'Classifier {model}')
st.write(f'Accuracy {acc*100:.2f}%')