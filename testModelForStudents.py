# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:36:10 2019

@author: cmccurley
"""

#Paths to data (for reference)
# "/home/cmccurley/Desktop/MachineLearningFall2019/testData/Easy_Test_Images.npy"
# "/home/cmccurley/Desktop/MachineLearningFall2019/testData/Hard_Test_Images.npy"

"""
* Author:    Connor McCurley
* Date:        2019-04-19
* Desc:        This script provides the test accuracy for an easy or hard dataset 
*              as an aid for grading project01 of EEL4930: Machine Learning for 
*              the Spring 2019 semester at the University of Florida.
"""

#===============================================================================
#============================ Import Packages ==================================
#===============================================================================
import numpy as np

#===============================================================================
#============================ Function Definitions =============================
#===============================================================================

def load_true_labels(dataSet, easyLabelPath, hardLabelPath):
    """
    * Function:    load_true_labels()
    * Desc:        Loads the true label vector for the easy or hard dataset. Requires user 
    *              to specify paths to saved .txt list of labels  
    * Inputs:      dataSet - string specifying the dataset (e.g. "easy" or "hard")
    *              easyLabelPath - string specifying path to easy data labels
    *              hardLabelPath - string specifying path to hard data labels
    * Outputs:     trueLabels - list of labels
    """

    #Select correct path to labels
    if dataSet == "easy":
        trueLabelPath = easyLabelPath
    elif dataSet == "hard":
        trueLabelPath = hardLabelPath 
    else:
        print("Input correct dataset type")

    #load label list
    with open(trueLabelPath, "r") as file:
        trueLabels = eval(file.readline())

    return trueLabels

def load_estimated_labels(pathToEstLabels):
    """
    * Function:    load_estimated_labels()
    * Desc:        Loads the estimated label vector for the easy or hard dataset. Requires user 
    *              to specify paths to saved .txt list of labels  
    * Inputs:      pathToEstLabels - string specifying the path to the predicted labels
    * Outputs:     estLabels - list of labels
    """
    
    #Load predicted labels
    with open(pathToEstLabels, "r") as file:
        estLabels = eval(file.readline())


    return estLabels

def testAccuracy(dataSet, trueLabels, estLabels):
    """
    * Function:    testAccuracy()
    * Desc:        Calculates the percentage of correctly estimated labels and displays
    *              to the terminal. 
    * Inputs:      dataSet - string specifying the dataset (e.g. "easy" or "hard")
                trueLabels - list of groundtruth labels (e.g. 'a', 'f')
                estLabels - list of predicted labels (e.g. 'a', 'f')
    * Outputs:     none
    """
    
    if dataSet == "easy":
        letters = ['a','f']
    elif dataSet == "hard":
        letters = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    else:
        print("Input correct dataset type")
    
    actualCounts = np.zeros((len(letters),1))         #initialize the vector of actual counts for each letter
    correctCounts = np.zeros((len(letters),1))            #initialize the vector of estimated counts for each letter
    
    #Count the number of occurences for each letter in the label lists
    for ind in range(len(letters)):
        
        actualCounts[ind,0] = trueLabels.count(letters[ind]) #acutal number of letters in true labels
        
        #Get sample-wise comparison of labels
        estLetter = [i for i, x in enumerate(estLabels) if x == letters[ind]]
        actualLetter = [i for i, x in enumerate(trueLabels) if x == letters[ind]]
        
        #Get number of correctly predicted for a particular letter
        correctCounts[ind,0] = len(set(actualLetter).intersection(estLetter))
        
    #Compute accuracy for each letter in the dataset individually
    letterAccuracies = np.divide(correctCounts, actualCounts)
    
    #Find the average accuracy
    avgAccuracy = np.mean(letterAccuracies)
    
    #Compute the standard deviation of the accuracies
    letterSTD = np.std(letterAccuracies)
    
    #Obtain the final score
    score = avgAccuracy - (0.5*letterSTD)

    #display accuracy to the terminal
    print("*******************************")
    print("*******************************")

    if dataSet == "easy":
        print("Score on EASY test set: ")
    elif dataSet == "hard":
        print("Score on HARD test set: ")
    else:
        print("Input correct dataset type")
    
    print(str(round(score,2)))

    print("*******************************")
    print("*******************************")
    
#===============================================================================
#================================ Main Function  ===============================
#===============================================================================
if __name__ == '__main__':

    #===========================================================================
    #============================ User Defined Settings ========================
    #===========================================================================
    dataSet = "easy" #dataset ("easy" or "hard")
    pathToEstLabels = """estimatedEasyLabels.txt"""
    showLabels = 1 #flag to plot contents of label vectors and sizes

    easyLabelPath = """easyLabels.txt"""
    hardLabelPath = """hardLabels.txt"""

    #===========================================================================
    #=============================== Get Test Accuracy =========================
    #===========================================================================
    trueLabels = load_true_labels(dataSet, easyLabelPath, hardLabelPath) 
    estLabels = load_estimated_labels(pathToEstLabels)

    testAccuracy(dataSet, trueLabels, estLabels)
    
    #===========================================================================
    #============================ User Defined Settings ========================
    #===========================================================================
    dataSet = "hard" #dataset ("easy" or "hard")
    pathToEstLabels = """estimatedHardLabels.txt"""
    showLabels = 1 #flag to plot contents of label vectors and sizes

    easyLabelPath = """easyLabels.txt"""
    hardLabelPath = """hardLabels.txt"""

    #===========================================================================
    #=============================== Get Test Accuracy =========================
    #===========================================================================
    trueLabels = load_true_labels(dataSet, easyLabelPath, hardLabelPath) 
    estLabels = load_estimated_labels(pathToEstLabels)

    testAccuracy(dataSet, trueLabels, estLabels)