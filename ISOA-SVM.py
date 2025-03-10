import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
##initialize parameters
population_size = 40;
max_iter = 20;
c_range = (0, 500)
gamma_range = (0, 500);
class_weight = {0: 1, 1: 1.4};
mutation_rate = 0.1;
cross_rate = 0.5;


# boundary detection
def boader_check(seagull):
    c, gamma = seagull;
    if (c < 0):
        c = 0.1
    if (c > 500):
        c = 500
    if (gamma < 0):
        gamma = 0.0001
    if (gamma > 500):
        gamma = 500
    return c, gamma


# initialize the population
def generate_individual():
    c = np.random.uniform(c_range[0], c_range[1]);
    gamma = np.random.uniform(gamma_range[0], gamma_range[1]);
    return c, gamma


def calculate_fitness(X_train, y_train, seagull):
    seagull = boader_check(seagull);
    c, gamma = seagull;
    model = SVC(C=c, gamma=gamma, kernel='sigmoid', class_weight=class_weight);
    fitness = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy');

    return np.mean(fitness)



# Genetic selection using roulette
def gene_selection(seagulls, X_train, y_train):
    selected_seagulls = []
    fitness_values = [calculate_fitness(X_train, y_train, seagull) for seagull in seagulls]
    for _ in range(population_size):
        total_fitness = sum(fitness_values)
        pick = np.random.uniform(0, total_fitness)
        current = 0
        for i, seagull in enumerate(seagulls):
            current += fitness_values[i]
            if current > pick:
                selected_seagulls.append(seagull)
                break
    return selected_seagulls


# update the position of seagulls
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


# gene intersection
def gene_intersection(seagulls):
    new_gene = []
    for father in seagulls:
        child = father
        if np.random.rand() < cross_rate:
            mother = seagulls[np.random.randint(population_size)]
            a = np.random.rand()
            if a>0.5:
                child_2=(father[0],mother[1])
            else:
                child_2=(mother[0],father[1])
            new_gene.append(child_2)
        else:
            new_gene.append(child)
    return new_gene




# gene_mutation
def gene_mutation(seagulls):
    mutated_seagulls =[]
    for seagull in seagulls:
        if np.random.rand() < mutation_rate:
            mutated_c = seagull[0]*np.random.uniform(0.9, 1.1)
            mutated_gamma = seagull[1]*np.random.uniform(0.9, 1.1)
            mutated_seagulls.append((mutated_c,mutated_gamma))
        else:
            mutated_seagulls.append(seagull)
    return mutated_seagulls


# improved soa
def GA_SOA_SVM(X_train, y_train):
    best_accuracy_list = [];
    cur_iter = 0;
    f_c = 2;
    seagulls =[ generate_individual() for _ in range(population_size) ]
    best_seagull = [np.random.uniform(c_range[0], c_range[1]),
                    np.random.uniform(gamma_range[0], gamma_range[1])]
    for _ in tqdm(range(max_iter), desc="ISOA-SVM"):
        cur_iter += 1;
        selected_seagulls = gene_selection(seagulls, X_train, y_train);
        moved_seagulls = [update_seagull(selected_seagull, best_seagull, cur_iter, f_c, max_iter) for selected_seagull
                          in selected_seagulls];
        jiaohuan_seagulls = gene_intersection(moved_seagulls);
        mutated_seagulls = gene_mutation(jiaohuan_seagulls);
        seagulls = mutated_seagulls
        for final_seagull in seagulls:
            final_seagull = boader_check(final_seagull)
            fitness = calculate_fitness(X_train, y_train, final_seagull)
            if best_seagull is None or fitness > calculate_fitness(X_train, y_train, best_seagull):
                best_seagull = final_seagull
        best_accuracy_list.append(calculate_fitness(X_train, y_train, best_seagull))
    # Output the optimal solution and accuracy curve
    best_c, best_gamma = best_seagull
    return best_c, best_gamma, best_accuracy_list


##import data.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:Scaled_Encoded_Final_dataset.csv')
X = data.drop(columns="Attrition")
y = data['Attrition']
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=427)

#smote to balance the sample space
smote = SMOTE(random_state=427, sampling_strategy=0.3, k_neighbors=9)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
best_c, best_gamma, best_accuracy_list = GA_SOA_SVM(X_train_resampled, y_train_resampled)

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
y_pred_proba = model.predict_proba(X_test)[:,1]
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