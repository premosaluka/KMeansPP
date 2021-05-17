#!/usr/bin/env python
# coding: utf-8

# Functions

# In[2]:


import numpy as np


# In[3]:


## Create binary mask with values 1 for regions of interest (0 or everything else)
## Mask_2 with original labels for regions of interest (1, 2 ,3, 10*)
## 10* if lesions are included in analysis

def binary_mask_function(mask,lesions):
    
    if lesions == False:

        binary_mask = mask
        #binary_mask = np.where(binary_mask == 10, 1, binary_mask)
        binary_mask = np.where(binary_mask >= 4, 0, binary_mask)
        mask_2 = binary_mask
        binary_mask = np.where(binary_mask >= 1, 1, binary_mask)
        
        return binary_mask,mask_2
    
    else:
        binary_mask = mask
        idx10 = binary_mask == 10
        #binary_mask = np.where(binary_mask ==10, 1, binary_mask)
        binary_mask = np.where(binary_mask >= 4, 0, binary_mask)
        binary_mask = np.where(idx10, 4, binary_mask)
        mask_2 = binary_mask
        binary_mask = np.where(binary_mask >= 1, 1, binary_mask)
        
        return binary_mask,mask_2


# In[13]:



## Combine all images into one array with dimension Nxd 
## each pixel now has 3 coordinates (t1,t2,pd)
## input template: (mask,t1,t2,pd)

def combine_images(mask,*args):

    
    combined = []
    coordinates = []
    group = []

    idx = 0;
    for i in range(181):
        for j in range(181):
            for k in range(216//5):
                #combined[idx,:]=([pd[i,j,k],t1[i,j,k],t2[i,j,k]])
                
                if mask[i,j,k] != 0: 
                    new_array = []
                    
                    for arg in args: 
                        
                        new_array.append(arg[i,j,k])
                        
                    combined.append(new_array)                 
                    coordinates.append([i,j,k])
                    group.append([mask[i,j,k]])
                    
                    #idx+=1
    return (combined, coordinates, group)                   


# In[5]:



import math
import random as rnd

# Find initial cluster centers 
# iY is input array of images size Nxd
# N number of all data points (pixels), d number of images

def kMeansInit( iY, iK ):
    N = len(iY)
    y0 = rnd.randint(0,N-1)
    y0 = [iY[y0]]
    for i in range(iK-1):
        sum1 = 0
        sum2 = 0
        cent = y0[-1]
        for pxl in iY:
            sum1 += ((np.linalg.norm(np.array(pxl) - np.array(cent)))**2) *np.array(pxl)
            sum2 += (np.linalg.norm(np.array(pxl) - np.array(cent)))**2    
        new_cent = sum1/sum2
        y0.append(new_cent) 
    
    ## 
    
        
    return y0


# In[15]:

# This function is not yet fully thought out yet
# It's role is to swap cluster center labels (1 for CSF, 2 for WM and 3 for GM)

def fix_labels(new_cents,labels,iK):
    indexes = new_cents[:,0].argsort()
    cents_sorted = new_cents[indexes]
    
    if iK == 4:
        labels = np.where(labels==indexes[0]+1,1,labels)
        labels = np.where(labels==indexes[1]+1,2,labels)
        labels = np.where(labels==indexes[2]+1,3,labels)
        labels = np.where(labels==indexes[3]+1,4,labels)
        return labels, cents_sorted
    
    elif iK >= 2:
        labels = np.where(labels==indexes[0]+1,1,labels)
        labels = np.where(labels==indexes[1]+1,2,labels)
        labels = np.where(labels==indexes[2]+1,3,labels)
        return labels, cents_sorted       

    
    else: 
        labels = np.where(labels==indexes[0]+1,1,labels)
        labels = np.where(labels==indexes[1]+1,2,labels)
        labels = np.where(labels==indexes[2]+1,3,labels)
        return labels, cents_sorted


# In[7]:

## Performs kmeans clustering 
## iY = input image size NxD
## iK number of clusters (3 if we want to segment 3 types of tissue, 4 if we include lesions)

def kMeansPP(iY,iK,iMaxIter):
    
    # initial centers 
    centers = np.array(kMeansInit(iY,iK))
    
    iY = np.array(iY)
    Iter = 0
    
    print(centers)

    while Iter < iMaxIter:
        
        # membership of pixels to clusters
        labels = np.array([])

        for pxl in iY:
            Si = np.array([])
            for cent in centers:
                Si = np.append(Si,np.array(np.linalg.norm(pxl-cent)))
            labels=np.append(labels,(np.argmin(Si) +1)) 
      
        new_cents = np.array([iY[labels == i+1].mean(axis=0)
                              for i in range(iK)])
        
        if Iter == 0:
            labels, new_cents = fix_labels(new_cents,labels,iK)
            
        print(new_cents)
        if np.all(new_cents == centers):
            break
        
        centers = new_cents
        Iter+=1 
        
    return centers,labels


# In[8]:

## Classifies membership of pixels to clusters 
## iMu are clusters after final iteration of kmeans


def nonparClassification( iY, iMu ):
    labels = np.array([])
    centers = iMu
    for pxl in iY:
        Si = np.array([])
        for cent in centers:
            Si = np.append(Si,np.array(np.linalg.norm(pxl-cent)))
        labels=np.append(labels,(np.argmin(Si) +1))
    return labels       


# In[9]:



def reconstruction(labels,coordinates):
    
    reconstructed_image = np.zeros((181,181,217//5))
    
    for i in range(len(labels)):
        reconstructed_image[tuple(coordinates[i])] = labels[i]
        
    return reconstructed_image   
    
    


# In[14]:

# Includes all previous functions 

def mrBrainSegmentation( mask,iMaxIter,iK,lesions,*args ):
    ## input template = (mask,imaxiter,ik,lesions=True/false,t1,t2,pd,)
    ## two images can be eliminated 
    
    binary_mask,mask_2 = binary_mask_function(mask,lesions)
     
    if lesions == True:
        iK = 4
    
    if len(args) == 3:  
        ## input template: (mask,t1,t2,pd)
        combined,coordinates, group = combine_images(mask_2,args[0],args[1],args[2])
    elif  len(args) == 2:
        combined,coordinates, group = combine_images(mask_2,args[0],args[1])
    else:
        combined,coordinates, group = combine_images(mask_2,args[0])
        
    combined = np.array(combined)
    coordinates = np.array(coordinates)
    trueLabels = np.array(group)
    trueLabels = trueLabels[:,0]

    
    segmentedLabels = nonparClassification(combined, kMeansPP(combined,iK,iMaxIter)[0])
    
    #segmentedLabels = kMeansPP(combined,iK,iMaxIter)[1]
    
    reconstructed_image = reconstruction(segmentedLabels,coordinates)
    
    return reconstructed_image,segmentedLabels, trueLabels


# In[11]:

## For quality of segmentation evaluation
## Computes overlap of true and predicted (segmented) labels 
## iS .. true labels
## iR ... predicted/segmented labels

def computeDiceCoeff( iS, iR):
    
    Coeff=[]

    for i in range(4):
        
        S=0
        R=0
        SR=0
        
        Si = np.where(iS == (i+1))[0]
        Ri = np.where(iR == (i+1))[0]
        
            
        S=Si.size
        R=Ri.size
        
        SR = np.intersect1d(Si,Ri).size
        if (S+R) == 0:
            Coeff.append(0)
        else:
            Coeff.append(2* SR / (S + R))
    
    return Coeff

