import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['DESCR'])
print(cancer['target'])

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# step 3 Data Visualisation
vis = df_cancer.iloc[:,0:6]
#plot all the correlations
sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
#count target
sns.countplot(df_cancer['target'])
#correlation btn two points
sns.scatterplot(data = df_cancer, x = 'mean area', y ='mean smoothness', hue = 'target') 
#show correlation btn features
sns.heatmap(df_cancer.corr(), annot=True)

#step 4: Model Training
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
classification = SVC()

#fitting data
classification.fit(X_train, y_train)

# ecaluating the model
from sklearn.metrics import confusion_matrix, classification_report

