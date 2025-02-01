import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

scoutium_attributes = pd.read_csv("/kaggle/input/scoutium-datasets/scoutium_attributes.csv", sep=";")

scoutium_potential_labels = pd.read_csv("/kaggle/input/scoutium-datasets/scoutium_potential_labels.csv", sep=";")

df = pd.merge(scoutium_attributes, scoutium_potential_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"] )
df.head()

# Data Analysis and First Insights

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(scoutium_attributes)

# Remove the class (1) in position_id from the data set.

df = df[df['position_id'] != 1]

# Remove the below_average class in potential_label from the dataset.

df = df[df['potential_label'] != 'below_average']

# Pivot Table

pivot_table = df.pivot_table(index= ["player_id","position_id","potential_label"], columns=["attribute_id"], values="attribute_value")
pivot_table_df = pivot_table.reset_index()
df['attribute_id'] = df['attribute_id'].astype(str)
pivot_table_df.head()

pivot_table_df.columns = pivot_table_df.columns.astype(str)
pivot_table_df.info()

labelencoder = LabelEncoder()
pivot_table_df["potential_label"] = labelencoder.fit_transform(pivot_table_df["potential_label"])

pivot_table_df.head()

num_cols = pivot_table_df.select_dtypes(include=['number']).columns.tolist()

# Scaling Numeric Columns

scaler = StandardScaler()
new_df = pd.DataFrame(scaler.fit_transform(pivot_table_df), columns=num_cols)
new_df.head()

# Label Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

labelEncoderCols = ["potential_label"]

for col in labelEncoderCols:
    new_df = label_encoder(new_df, col)

# Fitting Models

y = new_df["potential_label"]
X = new_df.drop(["potential_label","player_id"], axis=1)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))
    print("##############################")


# Feature Importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')



model = LGBMClassifier()
model.fit(X, y)
plot_importance(model, X)
