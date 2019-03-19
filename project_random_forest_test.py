# Display bands
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import time

from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

start = time.time()
debug = 0
test_samples = 50
num_folds = 10
num_trees = 5
if debug == 1:
    num_features = test_samples
else:
    num_features = 10
maxDepth = 20
maxFeatures = num_features
avg_indx = 0
std_indx = 1

#Load datasets
Pavia = np.load('PaviaHyperIm.npy')#103 bands
savePavia = np.load('PaviaHyperIm.npy')#103 bands
GT = np.load('gt_mat.npy')

if debug == 1:
    Pavia_Test = np.copy(Pavia)
    Pavia_Test = np.delete(Pavia_Test,range(test_samples,610),0)
    Pavia_Test = np.delete(Pavia_Test,range(test_samples,340),1)
    Pavia = Pavia_Test

    GT_Test = np.copy(GT)
    GT_Test = np.delete(GT_Test,range(test_samples,610),0)
    GT_Test = np.delete(GT_Test,range(test_samples,340),1)
    GT = GT_Test
else:
    Pavia = image_arr
    GT = image_class_arr

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Reshape HSI image to be Pixels by Bands
if debug == 1:
    paviaSpectra =  image_arr
    gtList =  image_class_arr
else:
    paviaSpectra = image_arr.reshape(720,30000)
    gtList = image_class_arr.reshape(720)

class foldClass:
    avg = 0
    std = 0

class dataClass:
    rawScores =  []
    foldScores = []
    def __init__(self):
        for i in range(num_folds):
            self.foldScores.append(foldClass)
    
class randForestClass():
    treeNum = [];
    treeDepth = [];
    dataFeatures = [];
    def __init__(self, numTrees=num_trees):
        for i in range(numTrees):
            self.treeNum.append(dataClass())
        for i in range(maxDepth):
            self.treeDepth.append(dataClass())
        for i in range(num_features+1):
            self.dataFeatures.append(dataClass())
    
treeScores = randForestClass();    #treeScores -> x0 = number of trees, x1 = test fold, x2 = runs
plot_x = np.zeros(num_trees)
plot_y = np.zeros((num_folds, 2, num_trees))

############################################## Varying Number of Trees #####################################################
for i in tqdm(range(num_trees)):
    #define classifier/estimator
    if i == 0:
        num_estimators = i+1
    else:
        num_estimators = (i)*10
        
    forest = RandomForestClassifier(criterion='entropy', n_estimators=num_estimators, n_jobs=-1)
    
    runs = [];
    #print("Number of Trees: ", num_estimators)
    for j in range(3):
        runs.append(cross_val_score(forest, paviaSpectra, y=gtList, cv=num_folds, n_jobs=-1))
    treeScores.treeNum[i].rawScores = np.copy((np.column_stack(runs))) #makes the test folds row mapped and run_number column mapped
                                             #so test_fold 1 -> runs[0][0], runs[0][1], runs[0][2] -> runs[0][:]
    #print(treeScores.treeNum[i].rawScores)
    
    for fold in range(num_folds):
        treeScores.treeNum[i].foldScores[fold].avg = np.copy(np.mean(treeScores.treeNum[i].rawScores[fold]))
        treeScores.treeNum[i].foldScores[fold].std = np.copy(np.std(treeScores.treeNum[i].rawScores[fold]))
        plot_y[fold][avg_indx][i] = np.copy(treeScores.treeNum[i].foldScores[fold].avg)
        plot_y[fold][std_indx][i] = np.copy(treeScores.treeNum[i].foldScores[fold].std)
        #print(plot_y[fold][avg_indx][i])
        

for i in range(num_trees):
    if i == 0:
        num_estimators = i+1
    else:
        num_estimators = (i)*10    
    plot_x[i] = np.copy(num_estimators)
    
plt.figure(figsize=(20,10))
plt.figure(1)
# multiple line plot
for fold_num in (range(num_folds)):
#     print(plot_y[fold_num][avg_indx][:])
    plt.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:], label='Test Fold {0}'.format(fold_num))    
plt.xlabel('Number of Trees',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title('Pavia Dataset Random Trees Classification',fontsize=20)
plt.legend()
plt.show()
############################################################################################################################

################################################ Varying Depth of Trees #####################################################
#plot_x = np.zeros(maxDepth)
#plot_y = np.zeros((num_folds, 2, maxDepth))
#
#for i in range(maxDepth):   
#    plot_x[i] = i+1
#
#for i in tqdm(range(maxDepth)):
#    #define classifier/estimator
#    forest = RandomForestClassifier(criterion='entropy', max_depth=(i+1), n_jobs=-1)
#    
#    runs = [];
#    #print("Number of Trees: ", num_estimators)
#    for j in range(3):
#        runs.append(cross_val_score(forest, paviaSpectra, gtList, cv=num_folds, n_jobs=-1))
#    treeScores.treeDepth[i].rawScores = (np.column_stack(runs)) #makes the test folds row mapped and run_number column mapped
#                                             #so test_fold 1 -> runs[0][0], runs[0][1], runs[0][2] -> runs[0][:]
#    
#    for fold in range(num_folds):
#        treeScores.treeDepth[i].foldScores[fold].avg = np.mean(treeScores.treeDepth[i].rawScores[fold])
#        treeScores.treeDepth[i].foldScores[fold].std = np.std(treeScores.treeDepth[i].rawScores[fold])
#        #print("Test Fold ", fold+1, " Avg Score & STD: ", treeScores.treeNum[i].foldScores[fold].avg, treeScores.treeNum[i].foldScores[fold].std)
#        plot_y[fold][avg_indx][i] = np.copy(treeScores.treeDepth[i].foldScores[fold].avg)
#        plot_y[fold][std_indx][i] = np.copy(treeScores.treeDepth[i].foldScores[fold].std)
#        #print(plot_y[fold][avg_indx][i])
#
#plt.figure(figsize=(20,10))
#plt.figure(1)
## multiple line plot
#for fold_num in (range(num_folds)):
#    plt.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:], label='Test Fold {0}'.format(fold_num))    
#plt.xlabel('Depth of Trees',fontsize=15)
#plt.ylabel('Accuracy',fontsize=15)
#plt.title('Pavia Dataset Random Trees Classification',fontsize=20)
#plt.legend()
#plt.show()
#############################################################################################################################
#   
############################################### Varying Number of Features #####################################################
#plot_x = np.zeros(maxFeatures)
#plot_y = np.zeros((num_folds, 2, maxFeatures))
#
#for i in range(maxFeatures):   
#    plot_x[i] = i+1
#
#for i in tqdm(range(maxFeatures)):
#    #define classifier/estimator
#    forest = RandomForestClassifier(criterion='entropy', max_features=(i+1), n_jobs=-1)
#    
#    runs = [];
#    #print("Number of Trees: ", num_estimators)
#    for j in range(3):
#        runs.append(cross_val_score(forest, paviaSpectra, gtList, cv=num_folds, n_jobs=-1))
#    treeScores.dataFeatures[i].rawScores = (np.column_stack(runs)) #makes the test folds row mapped and run_number column mapped
#                                             #so test_fold 1 -> runs[0][0], runs[0][1], runs[0][2] -> runs[0][:]
#    
#    for fold in range(num_folds):
#        treeScores.dataFeatures[i].foldScores[fold].avg = np.mean(treeScores.dataFeatures[i].rawScores[fold])
#        treeScores.dataFeatures[i].foldScores[fold].std = np.std(treeScores.dataFeatures[i].rawScores[fold])
#        #print("Test Fold ", fold+1, " Avg Score & STD: ", treeScores.treeNum[i].foldScores[fold].avg, treeScores.treeNum[i].foldScores[fold].std)
#        plot_y[fold][avg_indx][i] = np.copy(treeScores.dataFeatures[i].foldScores[fold].avg)
#        plot_y[fold][std_indx][i] = np.copy(treeScores.dataFeatures[i].foldScores[fold].std)
#        #print(plot_y[fold][avg_indx][i])
#
#plt2.rcParams['figure.figsize'] = (20.0, 10.0)
#plt2.figure(figsize=(20,10))
## multiple line plot
#for fold_num in (range(num_folds)):
#    plt2.errorbar(plot_x, plot_y[fold_num][avg_indx][:], yerr=plot_y[fold_num][std_indx][:], label='Test Fold {0}'.format(fold_num))    
#plt2.xlabel('Number of Features',fontsize=15)
#plt2.ylabel('Accuracy',fontsize=15)
#plt2.title('Pavia Dataset Random Trees Classification',fontsize=20)
#plt2.legend()
#plt2.show()
#############################################################################################################################
#
################################################ Feature Importances #####################################################
#forest = RandomForestClassifier(criterion='entropy', n_jobs=-1)
#forest.fit(paviaSpectra, gtList)
#importances = forest.feature_importances_
#
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#
## Print the feature ranking
#print("Feature ranking:")
#
#for f in range(paviaSpectra.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#plt.figure(figsize=(75,75))
#plt.bar(range(paviaSpectra.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(1, paviaSpectra.shape[1]+1, 10) )
#plt.xlabel("Features by Ranking")
#plt.xlim([-1, paviaSpectra.shape[1]])
#plt.title("Feature importances")
#plt.rc('font', size=150)          # controls default text sizes
#plt.show()
## ############################################################################################################################
#    
end = time.time()
print("Total Runtime: ", end - start)