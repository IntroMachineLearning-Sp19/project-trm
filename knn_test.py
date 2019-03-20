import numpy as np 
import time
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors, datasets
from sklearn import preprocessing as preprocess
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


os.system('proj_part1.py')

#Then, set up parameters for making figures
h = .02  # step size in the mesh

kmax = 10
uniform_scores = np.zeros(kmax+1, dtype=np.float64)
weighted_scores = np.zeros(kmax+1, dtype=np.float64)

#Next set up the classifiers we will run
#In this case, we are running K-NN with uniform distance (standard approach) and with weighted distances

for i in tqdm(range(kmax)): #exclusive of the 5
    
    X = image_arr.reshape(720,30000) #feature dataset
    Y = image_class_arr.reshape(720) #ground truth
        
    n_neighbors = i+1
    
    classifiers = []
    classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
    classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
    names = ['K-NN_Uniform', 'K-NN_Weighted']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=.1)
    
    
    # Iterate over classifiers
    for name, clf in zip(names, classifiers): 

        #Train the classifier
        clf.fit(X_train, y_train)

        #Test the classifier
        if name == 'K-NN_Uniform':
            uniform_scores[i+1] = (clf.score(X_test, y_test))
        else:
            weighted_scores[i+1] = (clf.score(X_test, y_test))
                    
figure = plt.figure()
plt.scatter(x=np.arange(len(uniform_scores)), y=uniform_scores)
plt.title("K from 1 to 100: Uniformed K-NN - Constant Dataset: Train/Test Split = 90%/10%")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")

figure = plt.figure()
plt.scatter(x=np.arange(len(weighted_scores)), y=weighted_scores)
plt.title("K from 1 to 100: Weighted K-NN - Constant Dataset: Train/Test Split = 90%/10%")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.show();