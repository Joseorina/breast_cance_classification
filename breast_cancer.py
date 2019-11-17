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