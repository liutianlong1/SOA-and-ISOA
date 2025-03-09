# SOA and ISOA
 This code is aimed at realizing SOA algorithm to optimize parameters in svm. After that, we will compare Grid-search with SOA, in terms of time and accuracy.Finally, we find SOA algorithm still has some parameters to be optimized, so we combine genetic algorithm with SOA. This improved algorithm archieved better result.
## Chapter One   data processing 
### **1.1 data discription**
Here we found a public data set from kaggle.This dataset is based on IBM human resoures and adds more features to it,including the six parts of indexes in hr.<br>
Firstly, we show all the features to the author.As the following:
![image1](https://github.com/user-attachments/assets/1819694c-442f-443d-9d9e-1ca941a759f0)<br>
After that, 17 numerical data were described from the angles of mean value, standard deviation, minimum value, 25% quantile, 50% quantile, 75% quantile and maximum value respectively and the remaining 12 qualitative data were observed by drawing histograms.They are shown in the following.
![image2](https://github.com/user-attachments/assets/8b6060c8-3a51-44fc-a908-83fb44e6a871)
![image3](https://github.com/user-attachments/assets/bf77ca35-6c55-47dd-82a6-a54df43aad5e)<br>
### **1.2 data coding**
There are three kinds of conditions.Firstly,to binary categorical variable,such as attrition、gender we just use 0-1 encoding. Secondly,to some variables with obvious progressive relationship are encoded sequentially,such as higher-education.
And the rest are coded by one-hot. <br>
Next step is to standardize the data.with $$X_{s\tan\mathrm{dardized}}=\frac{X_i-mean\left( X_i \right)}{std\left( X_{\mathrm{i}}\right)}$$
### **1.3 feature selection**
Because the correlation between the data is low, we give up PCA to reduce dimension. Variance Threshold is also ineffective in this situation. Because variance threshold is based on the hypothese that the lower variance of feature is, the lower contributions the feature makes. After standardization,the variance of features tends to one, so we should give up this method.<br>
After filtering,we choose random forest、SHAP values and RFE. Our voting rule is that we run the three methods respectively and the top 30 each get one vote and the features which get 2 or 3 votes will be selected. Finally, we got a feature set of 25 features. To vertify the effectiveness, we run random forest and svm in both new feature set and old feature set. The two models are both improved.<br>
It's worth mentioning that this features selected by various feature selection methods can improve the stability and robustness of the model.
![image4](https://github.com/user-attachments/assets/b777893b-d53b-4221-8e19-4c593a081dcb)
![image5](https://github.com/user-attachments/assets/78948486-6ee4-49e3-8100-200026598cfd)
![image6](https://github.com/user-attachments/assets/a4bdb11c-4606-41be-966a-57d0c94ecda9)
![image7](https://github.com/user-attachments/assets/148c20fc-77f7-4ad9-a809-5db728419978)
![image8](https://github.com/user-attachments/assets/3eb4e765-6ef8-48eb-ac33-aaffc8f33aeb)
## Chapter Two   SOA-SVM
The full name of SOA alogrithm is seagull optimization alogrithm.Just as its name, the algorithm mimics a seagull.It includes two main part:migration and attack.The process of seagull migration refers to the process of individual seagull moving from the current position to other positions. In this process, individual collision and the process of individual seagull moving towards the optimal position in order to obtain more food should be considered. The attack process involves seagulls feeding in a spiral motion. The following will focus on the mathematical expression in these two processes.<br>
### **2.1migration**(global search)
The migration includes avoiding collision avoidance,finding the best direction and adjusting the position.
#### 2.1.1 collision avoidnace 
We introduce control factor and use it to calcuate the new position which guarantees adjacent seagulls don't collide. $$C_s\left(t \right) =A*P_s \left(t \right)$$ $$C_s\left(t \right)$$ denotes the new location of Seagull s at moment t.And $$P_s\left(t \right)$$ denotes the present location of Seagull s at moment t. And the formula for calculating A is $$A=f_c\left( 1-frac{t}{MAXGEN} \right)$$ and $$f_c$$ denotes the factor that controls A. Usually, it linearly decreasing to 0.
And then, in this new situation, the seagull will move in the optimal direction. $$M_s\left(t\right)=C_B *\left(P_{bs}\left(t\right)-P_s\left(t\right)\right)$$ And $$M_s\left(t\right)&& denotes the next direction a seagull will fly forward. $$C_B&& is a random number which balances the global search and local search. And it's usually calculated by $$C_B=2 * A^2 * rand$$ <br>
Finally, according to the above two results, we can calcualte the direction that moves to the best position. $$D_s\left(t \right)=C_s\left(t \right)+M_s\left(t \right)$$
### **2.1.2attack**
The seagull's attack takes a spiral flight.And the generation methods are as following.<br>
$$ \left\{ \begin{array}{l} x = R \cos(i) \\ y = R \sin(i) \\ z = R i \\ R = u e^{iv} \end{array} \right.$$
And the positon after an attack: $$P_s \left(t \right)=D_s\left(t \right) * x * y * z + P_{bs}\left(t \right)$$
### **2.2**SOA-SVM
![image9](https://github.com/user-attachments/assets/d2f03f9c-f12e-4767-aa1c-f8d2feea4ef9)
initialize parameters including c and g in svm, and A and B in soa. Let's start generation. The results are displayed. By the way, we choose 5-fold cross verification and then take the average, which will guarantee the robustness.
![image10](https://github.com/user-attachments/assets/386e9591-8ca2-4e1e-8c5a-ecca18eda3bb)
![image11](https://github.com/user-attachments/assets/e53300af-d3f0-453c-b3e6-cdbce2a9a3be)
We can conclude from the image10 and image11 that SOA-SVM archieve the accuracy of 0.871 and auc of 0.853,which means that the SOA-SVM model has strong distinguishing ability, prediction accuracy and stability in human resource risk prediction.
## Chapter Three  GS-svm and SOA-SVM
Then we compare GS-svm and SOA-svm, in terms of time and accuracy. We first set c and gamma from 0 to 500, and the step is 10.After running, we got the best c is 90 and gamma is 0.01.
![image12](https://github.com/user-attachments/assets/71a325ac-7973-46a8-8c61-512e7d5b9539)
Whatever the time consume and the accuracy,SOA-SVM achieve a complete victory. This shows the absolute advantage of heuristic algorithm in the process of parameter optimization.
## Chapter Final  ISOA-SVM
### 4.1 genetic algorithm
  Genetic algorithm is an adaptive algorithm based on Darwin's biological evolution theory and simulates the basis of biological evolution in nature. It has the advantages of strong global search ability and adaptability to high-dimensional problems.<br>
  The alogritm includes four parts:Gene coding and decoding, gene selection, gene exchange and gene mutation. Because the number of parameters is only two ,so we can use a two-tuples to indicate a chromosome.<br>
  The rules are as following. Exchange's rule is that if a rand >0.5, we will exchange $$c_1$$ and $$c_2$$ else we will exchange $$gamma_1$$ and $$gamma_2$$.And Mutations introduce random perturbations that change the original chromosome from 90 percent to 110 percent to produce new chromosomes in the sample.
### 4.2 ISOA-SVM
This combination takes into account that the genetic algorithm has a good global search ability and the Seagull algorithm has a good local search ability, so that the algorithm can better explore and use the search space ability.
The algorithm first generates the initial seagull group, calculates the fitness of each individual seagull in the seagull group, and uses the roulette method to eliminate the fittest seagulls, and the selected seagulls will continue to migrate to the optimal position, attack and update the new position. The seagulls that arrive at the new location become the parent seagulls, the parent seagulls nurture the children, and the children will undergo the process of gene exchange and gene mutation, which will increase the diversity of the seagulls.<br>
![image13](https://github.com/user-attachments/assets/b12c20b8-1b38-451c-9c55-f265dd89d9ad)
![image14](https://github.com/user-attachments/assets/4d2fda5a-fe6a-45ed-b460-331e3b1d63b4)
![image15](https://github.com/user-attachments/assets/422e6830-6da2-43ee-a207-f3f1065aa31f)
||SOA-SVM|SOA-SVM-全|ISOA-SVM|ISOA-SVM-全|
|---|---|---|---|---|
|迭代次数|80|80|20|20|
|AUC|0.853|0.811|0.856|0.821|
|准确率|0.871|0.832|0.906|0.875|
<br>
Compared with SOA-SVM, ISA-SVM shows some performance improvement. However, this performance boost comes at the cost of adding extra steps, which results in an increase in the execution time of the algorithm. It is worth noting that in order to strike a balance between time overhead and performance improvement, we chose to set the number of iterations of ISA-SVM to 20. This choice was based on consideration of the algorithm's runtime, ensuring that the ISA-SVM execution time at 20 iterations is comparable to the SOA-SVM execution time at 80 iterations.
