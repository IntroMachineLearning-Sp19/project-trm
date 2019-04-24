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
#    if not os.path.exists(os.path.join('data/', 'testEasyOrdered')):
#        os.makedirs(os.path.join('data/', 'testEasyOrdered'))
        
    if not os.path.exists(os.path.join('data/', 'testHardOrdered')):
        os.makedirs(os.path.join('data/', 'testHardOrdered'))
      
#    i = 0    
#    for filename in os.listdir("data/testAF/"):
#        string = ""
#        if len(filename.strip(" .jpg ")) == 1:
#            string = "00"
#        elif len(filename.strip(" .jpg ")) == 2:
#            string = "0"        
#        elif len(filename.strip(" .jpg ")) == 3:
#            string = ""
#            
#        filename = filename.strip(" .jpg ")
#        dst = string + filename + ".jpg"
#        src = "data/testAF/" + filename + ".jpg"
#        dst = "data/testEasyOrdered/" + dst
#        os.rename(src, dst) 
#        i += 1
        
        
    i = 0    
    for filename in os.listdir("data/test/"):
        string = ""
        if len(filename.strip(" .jpg ")) == 1:
            string = "00"
        elif len(filename.strip(" .jpg ")) == 2:
            string = "0"        
        elif len(filename.strip(" .jpg ")) == 3:
            string = ""
            
        filename = filename.strip(" .jpg ")
        dst = string + filename + ".jpg"
        src = "data/test/" + filename + ".jpg"
        dst = "data/testHardOrdered/" + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            