import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, roc_auc_score
# set a class_weight which will use in smote
class_weight = {0:1,1:1.4}
# read the data and initialize a timestamp
start_time = time.time()
data = pd.read_csv(r'Scaled_Encoded_Final_dataset.csv') # your own path
end_time = time.time()
print("Time taken to read data:", end_time - start_time, "seconds")
# selected feature and the object variable
X = data[['Absenteeism', 'Age', 'BusinessTravel_Travel_Frequently', 'DistanceFromHome', 'Gender', 'JobLevel', 'JobInvolvement',
          "JobSatisfaction", 'MaritalStatus_Single', 'NumCompaniesWorked', 'OverTime', 'StockOptionLevel',
          'TotalWorkingYears', "TrainingTimesLastYear", "YearsWithCurrManager", "BusinessTravel_Non-Travel",
          "JobRole_P2", "JobRole_P3", "JobRole_P5", "JobRole_P8", "JobRole_P9", "MaritalStatus_Divorced",
          "MaritalStatus_Married", "YearsSinceLastPromotion", "Job_mode_Part Time"]]
y = data['Attrition']

# split the dataset into train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# normalization
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# oversampling
smote = SMOTE(random_state=427, sampling_strategy=0.3, k_neighbors=9)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_normalized, y_train)

# definite the parameters scope and step
param_grid = {'C': np.arange(0.001, 500, 10), 'gamma': np.arange(0.001, 500, 10)}

# initialize the model
svm_model = SVC()

# use gridsearchcv
start_time = time.time()
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)
end_time = time.time()
print("Time taken for grid search:", end_time - start_time, "seconds")

# find the best combination and test the result
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_c = best_params['C']
best_gamma = best_params['gamma']
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# initialize the model
model = SVC(C=best_c, gamma=best_gamma, kernel='sigmoid', class_weight=class_weight,probability=True)

# fit the model
model.fit(X_train_resampled, y_train_resampled)

# predict
y_pred = model.predict(X_test)

# calculate the confused matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# visualize the confused matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
y_pred_proba = model.predict_proba(X_test)[:,1]
# calculate the auc and f_1 and accuracy.
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# put out the results
print("best_c:", best_c)
print("best_gamma:", best_gamma)
print("AUC:", auc)
print("F1-score:", f1)
print("Accuracy:", accuracy)


# gain the probability of the predication
y_pred_proba = model.predict_proba(X_test)[:,1]

# The true and false positive rates of the ROC curve were calculated
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# calcuate auc
auc = roc_auc_score(y_test, y_pred_proba)

# draw roc curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc, color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# gain the performance of different parameter combination
scores = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

# plot the parameter combination and draw the heat map.
plt.figure(figsize=(10, 8))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.title('Grid Search Mean Test Scores')
plt.show()
