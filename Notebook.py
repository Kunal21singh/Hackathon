import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


import missingno as msno
import warnings
warnings.filterwarnings('ignore')
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)
pd.set_option("display.max.rows",None)
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings

# Suppress specific warnings for LGBMClassifier and CatBoostClassifier
import logging
logging.getLogger("catboost").setLevel(logging.ERROR)  # Suppress CatBoost logs
logging.getLogger("lightgbm").setLevel(logging.ERROR)  # Suppress LightGBM logs

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import label_binarize
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

df = pd.read_csv('Fraud.csv')

numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])

fig = px.imshow(numeric_data.corr(),text_auto=True,aspect="auto")

encoder = {}
for i in df.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    df[i] = encoder[i].fit_transform(df[i])

x = df.drop(['isFraud'], axis = 1)
y = df['isFraud']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)

pt = PowerTransformer(method='yeo-johnson')

x_train_scaled = pt.fit_transform(x_train)
x_test_scaled = pt.transform(x_test)

train_accuracy_scores = []
train_precision_scores = []
train_recall_scores = []
train_f1_scores = []

test_accuracy_scores = []
test_precision_scores = []
test_recall_scores = []
test_f1_scores = []

def evaluate_classification_performance(model, x_train, y_train, x_test, y_test, score_append=False):
    """
    Evaluates Accuracy, Precision, Recall, F1-score, AUC, and Confusion Matrix for a given classification model 
    on training and testing data using Plotly for visualizations.
    
    Parameters:
    - model: The machine learning model to evaluate
    - x_train: Training feature set
    - y_train: Training target values
    - x_test: Testing feature set
    - y_test: Testing target values
    """

    # Fit the model
    model.fit(x_train, y_train)

    # Predictions for training and testing data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics for training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

    # Calculate metrics for testing data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    # AUC (Area Under Curve) - for binary or multiclass
    if len(np.unique(y_train)) == 2:  # Binary Classification
        train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    else:  # Multiclass Classification
        train_auc = roc_auc_score(label_binarize(y_train, classes=np.unique(y_train)), 
                                  model.predict_proba(x_train), average='macro', multi_class='ovr')
        test_auc = roc_auc_score(label_binarize(y_test, classes=np.unique(y_test)), 
                                 model.predict_proba(x_test), average='macro', multi_class='ovr')

    # Append scores to respective lists
    if score_append == True:
        train_accuracy_scores.append(train_accuracy)
        train_precision_scores.append(train_precision)
        train_recall_scores.append(train_recall)
        train_f1_scores.append(train_f1)
        
        test_accuracy_scores.append(test_accuracy)
        test_precision_scores.append(test_precision)
        test_recall_scores.append(test_recall)
        test_f1_scores.append(test_f1)
    else:
        pass
        
    # Confusion Matrix for Training and Testing Data
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    # Function to plot confusion matrix using Plotly
    def plot_confusion_matrix(cm, title):
        labels = [f"Class {i}" for i in range(len(cm))]
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            reversescale=False
        )
        fig.update_layout(
            title_text=title,
            xaxis=dict(title='Predicted Labels'),
            yaxis=dict(title='True Labels')
        )
        fig.show()

    print(f"{model.__class__.__name__} Performance Metrics:")
    print(f"Training Data: Accuracy = {train_accuracy:.2f}, Precision = {train_precision:.2f}, Recall = {train_recall:.2f}, F1-score = {train_f1:.2f}, AUC = {train_auc:.2f}")
    print(f"Testing Data : Accuracy = {test_accuracy:.2f}, Precision = {test_precision:.2f}, Recall = {test_recall:.2f}, F1-score = {test_f1:.2f}, AUC = {test_auc:.2f}\n")

    # Display Confusion Matrices
    plot_confusion_matrix(train_cm, title='Training Confusion Matrix')
    plot_confusion_matrix(test_cm, title='Testing Confusion Matrix')

evaluate_classification_performance(
    model=LogisticRegression(n_jobs=-1),
    x_train=x_train_scaled,
    y_train=y_train,
    x_test=x_test_scaled,
    y_test=y_test,
    score_append = True
)