# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:29:06 2019

@author: rosem
"""

import os 

if __name__ == "__main__":
    os.system('proj_part1.py')

    ########################## NO RED ###########################
    red_0 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            red_0[i][j][0] = 0
    plt.figure()
    plt.title('Red 0')
    plt.imshow(red_0)
    
    red_255 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            red_0[i][j][0] = 255
    plt.figure()
    plt.title('Red 255')
    plt.imshow(red_255)
    
    ########################## NO GREEN ###########################
    green_0 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            green_0[i][j][1] = 0
    plt.figure()
    plt.title('Green 0')
    plt.imshow(green_0)
    
    green_255 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            green_255[i][j][0] = 255
    plt.figure()
    plt.title('Green 255')
    plt.imshow(green_255)
    
    ########################## NO BLUE ###########################
    blue_0 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            blue_0[i][j][2] = 0
    plt.figure()
    plt.title('Blue 0')
    plt.imshow(blue_0)
    
    blue_255 = np.reshape(image_arr[0][0],(100,100,3))
    for i in range(100):
        for j in range(100):
            blue_255[i][j][2] = 255
    plt.figure()
    plt.title('Blue 255')
    plt.imshow(blue_255)