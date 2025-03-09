import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#the path of file
path = r'C:Final dataset Attrition.csv'  #if you want to use this code, you should turn into your own path.
data_initial = pd.read_csv(path)
print(data_initial.dtypes);
# the description of numeric data and write it into an Excel
de_num = data_initial.describe();
excel_file_path = r'描述性统计.xlsx' # this also needs you to your own path
sheet_name = 'Sheet2'
with pd.ExcelWriter(excel_file_path) as writer:
    de_num.to_excel(writer, sheet_name=sheet_name)
# to make a histogram for qualitative data
columns = ['Attrition', 'BusinessTravel', 'Department', 'Gender',
           'JobRole', 'MaritalStatus', 'OverTime', 'Higher_Education',
           'Mode_of_work', 'Work_accident', 'Source_of_Hire', 'Job_mode']
fig, axs = plt.subplots(4, 3, figsize=(25, 10))
for i, column in enumerate(columns):
    row = i // 3
    col = i % 3
    sns.histplot(data_initial[column], ax=axs[row, col], color='skyblue')
    axs[row, col].set_title(column)
    axs[row, col].set_xlabel('')
    axs[row, col].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
# draw a heat  map to display the correlation between features.
plt.figure(figsize=(20,20))
sns.heatmap(data_initial.corr());

