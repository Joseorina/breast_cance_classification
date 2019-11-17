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