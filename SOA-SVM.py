##SOA-SVM算法准备
import numpy as np
import pandas as pd
class_weight = {0: 1, 1: 1.4}
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import make_scorer, recall_score, f1_score, roc_auc_score
#initialize the parameters
population_size = 40; # the population of seagulls
max_iter =80; # the biggest number of iteration
c_range = (0,500) # the range of c in svm
gamma_range = (0,500); # the range of gamma in svm
#boundary detection
def boader_check(seagull):
    c,gamma = seagull;
    if(c<0):
        c=0.01
    if(c>500):
        c =500;
    if(gamma<0):
        gamma=0.0001
    if(gamma>500):
        gamma = 500;
    return c,gamma
def generate_individual():
    c = np.random.uniform(c_range[0],c_range[1]); #generate c from 0 to 500 in normal distribution
    gamma = np.random.uniform(gamma_range[0], gamma_range[1]); #ditto
    return c,gamma
def calculate_fitness(X_train,y_train,seagull):
    c,gamma = seagull;
    model = SVC(C=c,gamma = gamma,kernel='sigmoid',class_weight=class_weight);
    auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy');
    return np.mean(auc_scores)
def update_seagull(seagull, best_seagull,t,f_c,max_iter):
    c, gamma = seagull;
    u,v =1,1;
    A = f_c - (t*f_c/max_iter);
    c_new = A*c;
    gamma_new = A*gamma;
    best_c, best_gamma = best_seagull
    # MS
    B = np.random.uniform(0, 1)*2*(A**2);
    c_velocity = B*(best_c - c);
    gamma_velocity = B * (best_gamma - gamma);
    # DS
    c_new = abs(c_new+c_velocity)
    gamma_new = abs(gamma_new+gamma_velocity)
    #attack
    theta = np.random.random()
    r = u * np.exp(theta * v)
    x = r * np.cos(theta *2* np.pi)
    y = r * np.sin(theta *2* np.pi)
    z = r * theta
    c_new = c_new*x*y*z+best_c;
    gamma_new= gamma_new*x*y*z +best_gamma
    return c_new, gamma_new

# soa-svm
def seagull_optimize(X_train, y_train):
    seagulls = [generate_individual() for _ in range(population_size)]
    best_seagull = None;
    best_accuracy_list = [];
    cur_iter=0;
    f_c = 2
    for _ in tqdm(range(max_iter),desc='Optimizing SOA-SVM'):
        cur_iter+=1;
        for seagull in seagulls:
            seagull = boader_check(seagull)
            auc = calculate_fitness(X_train, y_train, seagull)
            if best_seagull is None or auc > calculate_fitness(X_train, y_train, best_seagull):
                best_seagull = seagull

        seagulls = [update_seagull(seagull, best_seagull,cur_iter,f_c,max_iter) for seagull in seagulls]
        best_accuracy_list.append(calculate_fitness(X_train, y_train, best_seagull))
    best_c, best_gamma = best_seagull
    return best_c, best_gamma, best_accuracy_list
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\12939\Desktop\毕业论文新\Scaled_Encoded_Final_dataset.csv')
X = data.drop(columns="Attrition")
y = data['Attrition']
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=427)
# use oversampling to balance the sample space
smote = SMOTE(random_state=427, sampling_strategy=0.3, k_neighbors=9)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
start_time = time.time()
best_c, best_gamma, best_accuracy_list = seagull_optimize(X_train_resampled, y_train_resampled)
end_time = time.time()
print("Time taken for grid search:", end_time - start_time, "seconds")
# test the result
model = SVC(C=best_c, gamma=best_gamma, kernel='sigmoid', class_weight=class_weight,probability=True)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
y_pred_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("best_c:", best_c)
print("best_gamma:", best_gamma)
print("AUC:", auc)
print("F1-score:", f1)
print("Accuracy:", accuracy)
plt.plot(best_accuracy_list)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Iteration vs. Accuracy')
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc, color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
