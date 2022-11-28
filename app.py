import os
HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
import pandas as pd

import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from scipy import stats as st
from random import randrange
from matplotlib import pyplot as plt
from scipy.special import softmax

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import Pool
from sklearn.ensemble import RandomForestClassifier


import optuna

import shap

import gradio as gr

import random

#Read and redefine data.

from datasets import load_dataset
data = load_dataset("mertkarabacak/NSQIP-PCF", data_files="pcf_imputed.csv", use_auth_token = HF_TOKEN)

data = pd.DataFrame(data['train'])
variables = ['SEX', 'INOUT', 'TRANST', 'AGE', 'SURGSPEC', 'HEIGHT', 'WEIGHT', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'VENTILAT', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRWBC', 'PRHCT', 'PRPLATE', 'ASACLAS', 'READMISSION1', 'BMI', 'RACE', 'LEVELS', 'LOS', 'DISCHARGE']
data = data[variables]

data['SEX'] = data['SEX'].replace(['male'], 'Male')
data['SEX'] = data['SEX'].replace(['female'], 'Female')

print(data.columns)

#Define outcomes.

x = data
y1 = data.pop('LOS')
y2 = data.pop('DISCHARGE')
y3 = data.pop('READMISSION1')
y1 = (y1 == "Yes").astype(int)
y2 = (y2 == "Yes").astype(int)
y3 = (y3 == "Yes").astype(int)

categorical_columns = list(x.select_dtypes('object').columns)

x = x.astype({col: "category" for col in categorical_columns})

#Prepare data for LOS (y1).
y1_data_xgb = xgb.DMatrix(x, label=y1, enable_categorical=True)
y1_data_lgb = lgb.Dataset(x, label=y1) 
y1_data_cb = Pool(data=x, label=y1, cat_features=categorical_columns)

#Prepare data for DISCHARGE (y2).
y2_data_xgb = xgb.DMatrix(x, label=y2, enable_categorical=True)
y2_data_lgb = lgb.Dataset(x, label=y2)
y2_data_cb = Pool(data=x, label=y2, cat_features=categorical_columns)

#Prepare data for READMISSION (y3).
y3_data_xgb = xgb.DMatrix(x, label=y3, enable_categorical=True)
y3_data_lgb = lgb.Dataset(x, label=y3)
y3_data_cb = Pool(data=x, label=y3, cat_features=categorical_columns)

#Prepare data for Random Forest models.
x_rf = x
categorical_columns = list(x_rf.select_dtypes('category').columns)
x_rf = x_rf.astype({col: "category" for col in categorical_columns})
le = sklearn.preprocessing.LabelEncoder()
for col in categorical_columns:
        x_rf[col] = le.fit_transform(x_rf[col].astype(str))
d1 = dict.fromkeys(x_rf.select_dtypes(np.int64).columns, str)
x_rf = x_rf.astype(d1)

#Assign unique values as answer options.

unique_sex = list(data["SEX"].unique())
unique_inout = ['Outpatient', 'Inpatient']
unique_transt = list(data["TRANST"].unique())
unique_surgspec = ['Neurosurgery', 'Orthopedics']
unique_diabetes = list(data["DIABETES"].unique())
unique_smoke = ['No', 'Yes']
unique_dyspnea = ['No', 'Yes']
unique_fnstatus2 = list(data["FNSTATUS2"].unique())
unique_ventilat = ['No', 'Yes']
unique_hxcopd = ['No', 'Yes']
unique_ascites = ['No', 'Yes']
unique_hxchf = ['No', 'Yes']
unique_hypermed = ['No', 'Yes']
unique_renafail = ['No', 'Yes']
unique_dialysis = ['No', 'Yes']
unique_discancr = ['No', 'Yes']
unique_wndinf = ['No', 'Yes']
unique_steroid = ['No', 'Yes']
unique_wtloss = ['No', 'Yes']
unique_bleeddis = ['No', 'Yes']
unique_transfus = ['No', 'Yes']
unique_asaclas = ['1-No Disturb', '2-Mild Disturb','3-Severe Disturb']
unique_race = ['White', 'Black or African American', 'Hispanic', 'Asian', 'Other', 'Unknown']
unique_levels = ['Single', 'Multi']

#Assign hyperparameters.

y1_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 0.3506223423303318, 'alpha': 0.010119011691233883, 'max_depth': 9, 'eta': 0.965102554386191, 'gamma': 1.119572508228617e-08, 'grow_policy': 'depthwise'}
y2_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 9.07012338181008e-05, 'alpha': 0.002016615391016473, 'max_depth': 9, 'eta': 0.2128926555612135, 'gamma': 0.022388121082152507, 'grow_policy': 'lossguide'}
y3_xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'lambda': 1.9663606797816134e-05, 'alpha': 0.003929127375775657, 'max_depth': 9, 'eta': 0.3048025839317336, 'gamma': 0.0005177082154188227, 'grow_policy': 'depthwise'}

y1_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 0.008110972525548755, 'lambda_l2': 0.004008190618744454, 'num_leaves': 215, 'feature_fraction': 0.7016737503035954, 'bagging_fraction': 0.8722740851467577, 'bagging_freq': 6, 'min_child_samples': 38}
y2_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 1.0642674158535966e-07, 'lambda_l2': 0.02743327939933644, 'num_leaves': 148, 'feature_fraction': 0.5466594731848615, 'bagging_fraction': 0.6064213992508849, 'bagging_freq': 3, 'min_child_samples': 11}
y3_lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 3.6639804466263284e-05, 'lambda_l2': 2.3837193969323747e-06, 'num_leaves': 180, 'feature_fraction': 0.9280040724233041, 'bagging_fraction': 0.44687588072885676, 'bagging_freq': 3, 'min_child_samples': 8}

y1_cb_params = {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09852257532036239, 'depth': 11, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.22087836884513118}
y2_cb_params = {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08053436532735482, 'depth': 12, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 2.5017720587004737}
y3_cb_params = {'objective': 'Logloss', 'colsample_bylevel': 0.08772247044816832, 'depth': 11, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 7.083815408987153}

y1_rf_params = {'criterion': 'entropy', 'bootstrap': 'auto', 'max_features': 'sqrt', 'max_depth': 82, 'n_estimators': 1200, 'min_samples_leaf': 4, 'min_samples_split': 4}
y2_rf_params = {'criterion': 'entropy', 'bootstrap': 'sqrt', 'max_features': 'auto', 'max_depth': 93, 'n_estimators': 600, 'min_samples_leaf': 2, 'min_samples_split': 2}
y3_rf_params = {'criterion': 'gini', 'bootstrap': 'auto', 'max_features': 'auto', 'max_depth': 95, 'n_estimators': 200, 'min_samples_leaf': 1, 'min_samples_split': 2}


#Modeling for y1/LOS.

y1_model_xgb = xgb.train(params=y1_xgb_params, dtrain=y1_data_xgb)
y1_explainer_xgb = shap.TreeExplainer(y1_model_xgb)

y1_model_lgb = lgb.train(params=y1_lgb_params, train_set=y1_data_lgb)
y1_explainer_lgb = shap.TreeExplainer(y1_model_lgb)

y1_model_cb = cb.train(pool=y1_data_cb, params=y1_cb_params)
y1_explainer_cb = shap.TreeExplainer(y1_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y1_rf = rf(**y1_rf_params)
y1_model_rf = y1_rf.fit(x_rf, y1)
y1_explainer_rf = shap.TreeExplainer(y1_model_rf)


#Modeling for y2/COMP.

y2_model_xgb = xgb.train(params=y2_xgb_params, dtrain=y2_data_xgb)
y2_explainer_xgb = shap.TreeExplainer(y2_model_xgb)

y2_model_lgb = lgb.train(params=y2_lgb_params, train_set=y2_data_lgb)
y2_explainer_lgb = shap.TreeExplainer(y2_model_lgb)

y2_model_cb = cb.train(pool=y2_data_cb, params=y2_cb_params)
y2_explainer_cb = shap.TreeExplainer(y2_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y2_rf = rf(**y2_rf_params)
y2_model_rf = y2_rf.fit(x_rf, y2)
y2_explainer_rf = shap.TreeExplainer(y2_model_rf)


#Modeling for y3/DISCHARGE.

y3_model_xgb = xgb.train(params=y3_xgb_params, dtrain=y3_data_xgb)
y3_explainer_xgb = shap.TreeExplainer(y3_model_xgb)

y3_model_lgb = lgb.train(params=y3_lgb_params, train_set=y3_data_lgb)
y3_explainer_lgb = shap.TreeExplainer(y3_model_lgb)

y3_model_cb = cb.train(pool=y3_data_cb, params=y3_cb_params)
y3_explainer_cb = shap.TreeExplainer(y3_model_cb)

from sklearn.ensemble import RandomForestClassifier as rf
y3_rf = rf(**y3_rf_params)
y3_model_rf = y3_rf.fit(x_rf, y3)
y3_explainer_rf = shap.TreeExplainer(y3_model_rf)


#Define predict for y1/LOS.

def y1_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"Prolonged LOS": float(pos_pred[0]), "Not Prolonged LOS": 1 - float(pos_pred[0])}

def y1_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_lgb.predict(df)
    return {"Prolonged LOS": float(pos_pred[0]), "Not Prolonged LOS": 1 - float(pos_pred[0])}

def y1_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y1_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"Prolonged LOS": float(pos_pred[0][1]), "Not Prolonged LOS": float(pos_pred[0][0])}

def y1_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y1_model_rf.predict_proba(df)
    return {"Prolonged LOS": float(pos_pred[0][1]), "Not Prolonged LOS": float(pos_pred[0][0])}


#Define predict for y2/DISCHARGE.

def y2_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"Non-home Discharge": float(pos_pred[0]), "Home Discharge": 1 - float(pos_pred[0])}

def y2_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_lgb.predict(df)
    return {"Non-home Discharge": float(pos_pred[0]), "Home Discharge": 1 - float(pos_pred[0])}

def y2_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y2_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"Non-home Discharge": float(pos_pred[0][1]), "Home Discharge": float(pos_pred[0][0])}

def y2_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y2_model_rf.predict_proba(df)
    return {"Non-home Discharge": float(pos_pred[0][1]), "Home Discharge": float(pos_pred[0][0])}


#Define predict for y3/READMISSION.

def y3_predict_xgb(*args):
    df_xgb = pd.DataFrame([args], columns=x.columns)
    df_xgb = df_xgb.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_xgb.predict(xgb.DMatrix(df_xgb, enable_categorical=True))
    return {"No Readmission": float(pos_pred[0]), "Readmission": 1 - float(pos_pred[0])}

def y3_predict_lgb(*args):
    df = pd.DataFrame([args], columns=data.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_lgb.predict(df)
    return {"No Readmission": float(pos_pred[0]), "Readmission": 1 - float(pos_pred[0])}

def y3_predict_cb(*args):
    df_cb = pd.DataFrame([args], columns=x.columns)
    df_cb = df_cb.astype({col: "category" for col in categorical_columns})
    pos_pred = y3_model_cb.predict(Pool(df_cb, cat_features = categorical_columns), prediction_type='Probability')
    return {"No Readmission": float(pos_pred[0][1]), "Readmission": float(pos_pred[0][0])}

def y3_predict_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)
    pos_pred = y3_model_rf.predict_proba(df)


#Define interpret for y1/LOS.

def y1_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y1_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y1_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m


#Define interpret for y2/DISCHARGE.

def y2_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y2_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y2_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m


#Define interpret for y3/READMISSION.

def y3_interpret_xgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_xgb.shap_values(xgb.DMatrix(df, enable_categorical=True))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_lgb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_lgb.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_cb(*args):
    df = pd.DataFrame([args], columns=x.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_cb.shap_values(Pool(df, cat_features = categorical_columns))
    scores_desc = list(zip(shap_values[0], x.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

def y3_interpret_rf(*args):
    df = pd.DataFrame([args], columns=x_rf.columns)
    df = df.astype({col: "category" for col in categorical_columns})
    shap_values = y3_explainer_rf.shap_values(df)
    scores_desc = list(zip(shap_values[0][0], x_rf.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(facecolor='white')
    fig_m.set_size_inches(14, 10)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values", fontsize = 24, pad = 20, fontweight = 'bold')
    plt.yticks(fontsize=12)
    plt.xlabel("Shap Value", fontsize = 16, labelpad=8, fontweight = 'bold')
    plt.ylabel("Feature", fontsize = 16, labelpad=14, fontweight = 'bold')
    return fig_m

with gr.Blocks(title = "NSQIP-PCF") as demo:
    
    gr.Markdown(
        """ 
    """
    )
        
    gr.Markdown(
        """
    # Prediction Tool

    ## Short-Term Postoperative Outcomes Following Posterior Cervical Fusion

    **The publication describing the details of this predictive tool will be posted here upon the acceptance of publication.**

    ### Disclaimer

    The American College of Surgeons National Surgical Quality Improvement Program and the hospitals participating in the ACS NSQIP are the source of the data used herein; they have not been verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors.

    The predictive tool located on this web page is for general health information only. This prediction tool should not be used in place of professional medical guidance, diagnosis, or treatment for any disease or concern. Users of the prediction tool shouldn't base their decisions about their own health issues on the information presented here. You should ask any questions to your own doctor or another healthcare professional. 

    The authors of the study mentioned above make no guarantees or representations, either express or implied, as to the truth, completeness, timeliness, comparative or contentious nature, or utility of any information contained in or referred to in this prediction tool. The risk associated with using this prediction tool or the information in this predictive tool is not at all assumed by the authors. The information contained in the prediction tools may be outdated, uncompleted, or incorrect because health-related information is subject to frequent change.

    No express or implied doctor-patient relationship is established by using the prediction tool. The prediction tools on this website are not recommended or validated by the authors. Users of the tool are not contacted by the authors, who also do not record any specific information about them.

    You are hereby advised to seek the advice of a doctor or other qualified healthcare provider before making any decisions, acting, or refraining from acting in response to any healthcare problem or issue you may be experiencing at any time, now or in the future. By using the prediction tool, you acknowledge and agree that neither the authors nor any other party are or will be liable or otherwise responsible for any decisions you make, actions you take, or actions you choose not to take as a result of using any information presented here.

    By using this tool, you accept all of the above terms or any further use of this service.
    """
    )

    with gr.Tab('Length of Stay'):
        
        gr.Markdown(
            """
         
        ### Prolonged Length of Stay Prediction Model for PCF Surgery
        
            """
        )
            
    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y1_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y1_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y1_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y1_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y1_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y1_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y1_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y1_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y1_predict_btn_xgb.click(
                    y1_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_lgb.click(
                    y1_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_cb.click(
                    y1_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_predict_btn_rf.click(
                    y1_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y1_interpret_btn_xgb.click(
                    y1_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_lgb.click(
                    y1_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_cb.click(
                    y1_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y1_interpret_btn_rf.click(
                    y1_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )
                
    with gr.Tab('Non-home Discharge'):
        
        gr.Markdown(
            """
         
        ### Non-home Discharge Prediction Model for PCF Surgery
        """
        )

    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y2_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y2_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y2_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y2_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y2_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y2_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y2_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y2_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y2_predict_btn_xgb.click(
                    y2_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_lgb.click(
                    y2_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_cb.click(
                    y2_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_predict_btn_rf.click(
                    y2_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y2_interpret_btn_xgb.click(
                    y2_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_lgb.click(
                    y2_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_cb.click(
                    y2_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y2_interpret_btn_rf.click(
                    y2_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )
                
    with gr.Tab('Readmission'):
        
        gr.Markdown(
            """
         
        ### Readmission Prediction Model for PCF Surgery
        """
        )

    
        with gr.Row():

            with gr.Column():

                AGE = gr.Slider(label="Age", minimum=17, maximum=99, step=1, randomize=True)

                SEX = gr.Radio(
                    label="Sex",
                    choices=unique_sex,
                    type='index',
                    value=lambda: random.choice(unique_sex),
                )

                RACE = gr.Radio(
                    label="Race",
                    choices=unique_race,
                    type='index',
                    value=lambda: random.choice(unique_race),
                )

                HEIGHT = gr.Slider(label="Height (in meters)", minimum=1.0, maximum=2.25, step=0.01, randomize=True)

                WEIGHT = gr.Slider(label="Weight (in kilograms)", minimum=20, maximum=200, step=1, randomize=True)

                BMI = gr.Slider(label="BMI", minimum=10, maximum=70, step=1, randomize=True)

                TRANST = gr.Radio(
                    label="Transfer Status",
                    choices=unique_transt,
                    type='index',
                    value=lambda: random.choice(unique_transt),
                )
                
                INOUT = gr.Radio(
                    label="Inpatient or Outpatient",
                    choices=unique_inout,
                    type='index',
                    value=lambda: random.choice(unique_inout),
                )

                SURGSPEC = gr.Radio(
                    label="Surgical Specialty",
                    choices=unique_surgspec,
                    type='index',
                    value=lambda: random.choice(unique_surgspec),
                )

                SMOKE = gr.Radio(
                    label="Smoking Status",
                    choices=unique_smoke,
                    type='index',
                    value=lambda: random.choice(unique_smoke),
                )

                DIABETES = gr.Radio(
                    label="Diabetes",
                    choices=unique_diabetes,
                    type='index',
                    value=lambda: random.choice(unique_diabetes),
                )

                DYSPNEA = gr.Radio(
                    label="Dyspnea",
                    choices=unique_dyspnea,
                    type='index',
                    value=lambda: random.choice(unique_dyspnea),
                )
                
                VENTILAT = gr.Radio(
                    label="Ventilator Dependency",
                    choices=unique_ventilat,
                    type='index',
                    value=lambda: random.choice(unique_ventilat),
                )

                HXCOPD = gr.Radio(
                    label="History of COPD",
                    choices=unique_hxcopd,
                    type='index',
                    value=lambda: random.choice(unique_hxcopd),
                )

                ASCITES = gr.Radio(
                    label="Ascites",
                    choices=unique_ascites,
                    type='index',
                    value=lambda: random.choice(unique_ascites),
                )

                HXCHF = gr.Radio(
                    label="History of Congestive Heart Failure",
                    choices=unique_hxchf,
                    type='index',
                    value=lambda: random.choice(unique_hxchf),
                )

                HYPERMED = gr.Radio(
                    label="Hypertension Despite Medication",
                    choices=unique_hypermed,
                    type='index',
                    value=lambda: random.choice(unique_hypermed),
                )

                RENAFAIL = gr.Radio(
                    label="Renal Failure",
                    choices=unique_renafail,
                    type='index',
                    value=lambda: random.choice(unique_renafail),
                )

                DIALYSIS = gr.Radio(
                    label="Dialysis",
                    choices=unique_dialysis,
                    type='index',
                    value=lambda: random.choice(unique_dialysis),
                )

                STEROID = gr.Radio(
                    label="Steroid",
                    choices=unique_steroid,
                    type='index',
                    value=lambda: random.choice(unique_steroid),
                )

                WTLOSS = gr.Radio(
                    label="Weight Loss",
                    choices=unique_wtloss,
                    type='index',
                    value=lambda: random.choice(unique_wtloss),
                )

                BLEEDDIS = gr.Radio(
                    label="Bleeding Disorder",
                    choices=unique_bleeddis,
                    type='index',
                    value=lambda: random.choice(unique_bleeddis),
                )

                TRANSFUS = gr.Radio(
                    label="Transfusion",
                    choices=unique_transfus,
                    type='index',
                    value=lambda: random.choice(unique_transfus),
                )
                
                WNDINF = gr.Radio(
                    label="Wound Infection",
                    choices=unique_wndinf,
                    type='index',
                    value=lambda: random.choice(unique_wndinf),
                )

                DISCANCR = gr.Radio(
                    label="Disseminated Cancer",
                    choices=unique_discancr,
                    type='index',
                    value=lambda: random.choice(unique_discancr),
                )

                FNSTATUS2 = gr.Radio(
                    label="Functional Status",
                    choices=unique_fnstatus2,
                    type='index',
                    value=lambda: random.choice(unique_fnstatus2),
                )

                PRSODM = gr.Slider(label="Sodium", minimum=min(x['PRSODM']), maximum=max(x['PRSODM']), step=1, randomize=True)

                PRBUN = gr.Slider(label="BUN", minimum=min(x['PRBUN']), maximum=max(x['PRBUN']), step=1, randomize=True)

                PRCREAT = gr.Slider(label="Creatine", minimum=min(x['PRCREAT']),maximum=max(x['PRCREAT']), step=0.1, randomize=True)

                PRWBC = gr.Slider(label="WBC", minimum=min(x['PRWBC']), maximum=max(x['PRWBC']), step=0.1, randomize=True)

                PRHCT = gr.Slider(label="Hematocrit", minimum=min(x['PRHCT']), maximum=max(x['PRHCT']), step=0.1, randomize=True)

                PRPLATE = gr.Slider(label="Platelet", minimum=min(x['PRPLATE']), maximum=max(x['PRPLATE']), step=1, randomize=True)

                ASACLAS = gr.Radio(
                    label="ASA Class",
                    choices=unique_asaclas,
                    type='index',
                    value=lambda: random.choice(unique_asaclas),

                )

                LEVELS = gr.Radio(
                    label="Levels",
                    choices=unique_levels,
                    type='index',
                    value=lambda: random.choice(unique_levels),
                )

            with gr.Column():

                with gr.Row():
                    y3_predict_btn_xgb = gr.Button(value="Predict (XGBoost)")
                    y3_predict_btn_lgb = gr.Button(value="Predict (LightGBM)")
                    y3_predict_btn_cb = gr.Button(value="Predict (CatBoost)")
                    y3_predict_btn_rf = gr.Button(value="Predict (Random Forest)")
                label = gr.Label()

                with gr.Row():
                    y3_interpret_btn_xgb = gr.Button(value="Explain (XGBoost)")
                    y3_interpret_btn_lgb = gr.Button(value="Explain (LightGBM)")
                    y3_interpret_btn_cb = gr.Button(value="Explain (CatBoost)")
                    y3_interpret_btn_rf = gr.Button(value="Explain (Random Forest)") 

                plot = gr.Plot()

                y3_predict_btn_xgb.click(
                    y3_predict_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_lgb.click(
                    y3_predict_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_cb.click(
                    y3_predict_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_predict_btn_rf.click(
                    y3_predict_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[label]
                )

                y3_interpret_btn_xgb.click(
                    y3_interpret_xgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_lgb.click(
                    y3_interpret_lgb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_cb.click(
                    y3_interpret_cb,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

                y3_interpret_btn_rf.click(
                    y3_interpret_rf,
                    inputs=[SEX, INOUT, TRANST, AGE, SURGSPEC, HEIGHT, WEIGHT, DIABETES, SMOKE, DYSPNEA, FNSTATUS2, VENTILAT, HXCOPD, ASCITES, HXCHF, HYPERMED, RENAFAIL, DIALYSIS, DISCANCR, WNDINF, STEROID, WTLOSS, BLEEDDIS, TRANSFUS, PRSODM, PRBUN, PRCREAT, PRWBC, PRHCT, PRPLATE, ASACLAS, BMI, RACE, LEVELS,],
                    outputs=[plot],
                )

demo.launch()  
