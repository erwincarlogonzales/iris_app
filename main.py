# import dependencies
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st

# import datasets
from sklearn import datasets

# preprocessing
from sklearn.model_selection import train_test_split

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

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
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'Random Forest'))

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
def add_param_ui(model_name):
    
    # create an empty dictionary
    params = dict()
    
    # create a slider for the parameters
    if model_name == 'KNN':
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
params = add_param_ui(classifier_name)

# create a function that calls the ML classifier
def get_classifier(model_name, params):
    
    # create a slider for the parameters
    if model_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params['K'])   
    else:
        # random forest parameters
        model = RandomForestClassifier(n_estimators = params['n_estimators'],
                                            max_depth = params['max_depth'], random_state = 42)
    
    return model

# call the function
model = get_classifier(classifier_name, params)
        
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# fit the dataset
model.fit(X_train, y_train)

# make prediction
y_pred = model.predict(X_test)

# evaluate model accuracy
acc = accuracy_score(y_test, y_pred)

# display accuracy
st.write(f'Classifier {classifier_name}')
st.write(f'Accuracy {acc*100:.2f}%')

# create visualization
# PCA for dimensionality reduction
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# # Get the feature names
# feature_names = datasets.load_iris().feature_names if dataset == 'Iris' else datasets.load_wine().feature_names

# # Identify the most influential feature for each principal component
# most_influential_feature_x1 = feature_names[np.argmax(np.abs(pca.components_[0]))]
# most_influential_feature_x2 = feature_names[np.argmax(np.abs(pca.components_[1]))]

# # Displaying the most influential features for x1 and x2
# st.write(f'Most influential feature for PCA Feature 1: {most_influential_feature_x1}')
# st.write(f'Most influential feature for PCA Feature 2: {most_influential_feature_x2}')

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# # Scatter plot for PCA features x1 and x2 using Seaborn
# fig, ax = plt.subplots()
# sns.scatterplot(x=x1, y=x2, hue=y, palette='viridis', ax=ax)
# ax.set_xlabel(f'PCA Feature 1 ({most_influential_feature_x1})')
# ax.set_ylabel(f'PCA Feature 2 ({most_influential_feature_x2})')
# st.pyplot(fig)

# # Histogram for PCA feature x1 using Seaborn
# fig, ax = plt.subplots()
# sns.histplot(x1, kde=True, ax=ax)
# ax.set_title(f'PCA Feature 1 ({most_influential_feature_x1}) Histogram')
# st.pyplot(fig)

# # Histogram for PCA feature x2 using Seaborn
# fig, ax = plt.subplots()
# sns.histplot(x2, kde=True, ax=ax)
# ax.set_title(f'PCA Feature 2 ({most_influential_feature_x2}) Histogram')
# st.pyplot(fig)
