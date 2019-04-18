# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:49:21 2019

@author: tdtra
"""

# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir("C:/Users/tdtra/Google Drive/13 SPRING 2019/EEL4930/project-trm/TRM_Pics/2019_sp_ml_train_data/y/"): 
        dst = str(i) + ".jpg"
        src = "C:/Users/tdtra/Google Drive/13 SPRING 2019/EEL4930/project-trm/TRM_Pics/2019_sp_ml_train_data/y/" + filename
        dst = "C:/Users/tdtra/Google Drive/13 SPRING 2019/EEL4930/project-trm/TRM_Pics/2019_sp_ml_train_data/y/" + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            