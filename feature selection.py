import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
data = pd.read_csv(r'Scaled_Encoded_Final_dataset.csv') # pardon you should change it into your own path. and this path is standardized data
X = data.drop(columns=['Attrition'])
y = data['Attrition']
# create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=527)
k_fold = KFold(n_splits=5, shuffle=True, random_state=527) #set a k-fold to increase the robustness
feature_importances = [] # a container to save the importance of feature.
for train_index, _ in k_fold.split(X):
    rf_classifier.fit(X.iloc[train_index], y.iloc[train_index])
    feature_importances.append(rf_classifier.feature_importances_)
average_feature_importances = np.mean(feature_importances, axis=0)
sorted_indices = average_feature_importances.argsort()[::-1]
sorted_feature_importances = average_feature_importances[sorted_indices]
sorted_feature_names = X.columns[sorted_indices]
top_25_features_rf = sorted_feature_names[:25]
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names)
plt.xticks(rotation=90)
plt.title('Feature Importance (with Cross-Validation)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
print("Top 25 Features:")
print(top_25_features_rf)
# create a rfe which is using svm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=527)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model = SVC(kernel='linear')
num_features_to_select = 25
rfe_selector = RFE(estimator=svm_model, n_features_to_select=num_features_to_select)
rfe_selector.fit(X_train_scaled, y_train)
selected_feature_indices = rfe_selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
selected_model = rfe_selector.estimator_
feature_importance = selected_model.coef_[0]
sorted_indices = np.argsort(abs(feature_importance))[::]
sorted_feature_names = selected_feature_names[sorted_indices]
sorted_feature_importance = feature_importance[sorted_indices]
top_25_features_rfe = set(sorted_feature_names[:25])
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance of Selected Features')
plt.show()
#create a shap
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_scaled, y_train)
explainer = shap.Explainer(svm_model, X_train_scaled)
shap_values = explainer(X_train_scaled)
shap.summary_plot(shap_values, X_train_scaled, plot_type="bar", feature_names=X.columns, max_display=len(X.columns))
shap_values_matrix = shap_values.values
shap_values_sum = abs(shap_values_matrix)
top_25_features_shap = shap_values_sum.argsort()[::-1][:25]
# save the top 25
sorted_feature_names = selected_feature_names[sorted_indices]
top_25_features_shap = set(sorted_feature_names[:25])
print(top_25_features_shap)
print(top_25_features_rfe)
print(top_25_features_rf)
