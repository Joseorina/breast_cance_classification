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

#predicting
y_pred = classification.predict(X_test)

# ecaluating the model
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm ,annot=True)

# improving the model
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train
#unscaled x_train
sns.scatterplot(x =X_train['mean area'], y =X_train['mean smoothness'], hue =y_train )

# scaled values
sns.scatterplot(x =X_train_scaled['mean area'], y =X_train_scaled['mean smoothness'], hue =y_train )

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

classification.fit(X_train_scaled, y_train)

y_pred = classification.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))

# Improcing model part 2
param_grid = {'c':[0.1,1,10,100], 'gamma':['1, 0.1,0.01,0.001'], 'kerne':['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid, refit = True, verbose = 4)
grid.fit(X_train_scaled, y_train)

grid_prediction = grid.fit(X_test_scaled)

cm = confusion_matrix(y_test, grid_prediction)