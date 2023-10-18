# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def incomplete(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    mh, mw = mask.shape[:2]
    num_completed = np.count_nonzero(mask)
    num_incomplete = (mh * mw) - num_completed
    return num_incomplete 
    
def load(path):
    sample = cv2.imread(path)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return sample

# divide sample into diferent templates zones
def sampleBreak(rGBsample, mask):
    sample = cv2.cvtColor(rGBsample, cv2.COLOR_BGR2GRAY)
    dilated_edge, zone0, zone1, fullmask = create_Masks(mask)
    sample_dilated_edge = sample * dilated_edge
    sample_reduced  = sample * zone0
    sample_inverted = sample * zone1
    return sample_dilated_edge, sample_reduced, sample_inverted

# Calculating Masks
def create_Masks(mask):
    # edge definition
    edges = cv2.Canny(mask,100,200)
    kernel = np.ones((11,11))
    dilated_edge = cv2.dilate(edges, kernel, iterations=1)
    inv_edge     = cv2.bitwise_not(dilated_edge)

    # Inverting the mask 
    mask_inverted = cv2.bitwise_not(mask)

    # Normalize to the range [0., 1.]
    mask = mask.astype(np.float64) / 255.
    mask_inverted = mask_inverted.astype(np.float64) / 255.
    dilated_edge = dilated_edge.astype(np.float64) / 255.
    inv_edge     = inv_edge.astype(np.float64) / 255.

    zone0 = mask * inv_edge
    zone1 = mask_inverted * inv_edge

    fullmask = zone0*1 + dilated_edge*2 +zone1*3

    return dilated_edge, zone0, zone1, fullmask

def show3Images(original, standart, probabilistic):
    plt.figure(figsize=(10,10))
    plt.subplot(131),plt.imshow(original,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.subplot(132),plt.imshow(standart,cmap = 'gray')
    plt.title('mask'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.subplot(133),plt.imshow(probabilistic,cmap = 'gray')
    plt.title('Probabilistic'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()

def show2Images(original, transformed):
    plt.figure(figsize=(10,10))
    plt.subplot(121),plt.imshow(original,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.subplot(122),plt.imshow(transformed,cmap = 'gray')
    plt.title('Transformed'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()    

# Display an array of images
def dispArrayImages(patches):
    n = len(patches)
    if n > 1:
        fig, axs = plt.subplots(1, n , figsize=(10, 10))
        for i in range(n):
            if (n<10): axs[i].set_title(patches[i][0])
            axs[i].imshow(cv2.cvtColor(patches[i][1], cv2.COLOR_BGR2RGB))
            axs[i].axis('off')
        plt.show()
    if n == 1:
        plt.figure(figsize=(10,10))
        plt.title(patches[0][0])
        plt.imshow(cv2.cvtColor(patches[0][1], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

#show all images in imagesList
def showImages(imagesList):
    n = len(imagesList)
    if n > 1:
        fig, axs = plt.subplots(1, n , figsize=(5, 5))
        for i in range(n):
            axs[i].imshow(cv2.cvtColor(imagesList[i], cv2.COLOR_BGR2RGB))
            axs[i].axis('off')
        plt.show()
    if n == 1:
        plt.imshow(cv2.cvtColor(imagesList[0], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

	
def sortingCoords(x1,x2):
    if x1 > x2:
        return x2, x1
    else:
        return x1, x2
    
def cropPatch(original, x1, x2, y1, y2):
    #get image dimensions
    plus = 15   
    height,width,c     = original.shape
    x1-=5
    x2+=5
    y1-=plus
    y2+=plus

    if y1 < 0:
        y1 = 0
    if y2 > height:
        y2 = height
    if x1 < 0:
        x1 = 0
    if x2 > width:
        x2 = width

    return original[y1:y2, x1:x2]

def probHough(mask, original, tresh=20, minPoints=30, maxGap=5):
    probabLines = original.copy()
    edges = cv2.Canny(mask, 100, 200)
    # Probabilistic Line Transform
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, tresh, None, minPoints, maxGap)
    dbPatches = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1, x2 = sortingCoords(l[0], l[2])
            y1, y2 = sortingCoords(l[1], l[3])
            angle = int(np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi)
            # Draw the probabilistic lines
            cv2.line(probabLines, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv2.LINE_AA)
            patch = cropPatch (original, x1, x2, y1, y2)
            dbPatches.append([angle ,patch])
    #show3Images(original,mask, probabLines)
    #dbPatches.sort(key=lambda x: x[0])
    return dbPatches,probabLines

#search nearest key in ordered patches
def searchNearestKey(patches, key):
    n = len(patches)
    if n > 1:
        for i in range(n):
            if key <= patches[i][0]:
                return patches[i][1]
        return patches[n-1][0],patches[n-1][1]
    if n == 1:
        return patches[0][0],patches[0][1]
    return None
     
#load a list of images from files
def loadImages(path, imageList):
    loadedImages = []
    for filename in imageList:
        img = cv2.imread(path + '\\' + filename+ '.png') 
        if img is not None:
            loadedImages.append(img)
    return loadedImages

def buildPatchesDB(masksList, imagesList):
    #iterate over imagesList
    patchesList = []
    for i in range(len(imagesList)):
        original = imagesList[i] 
        mask = masksList[i]
        patches, linesP = probHough(mask, original, tresh = 15, minPoints=15, maxGap=15)
        #dispArrayImages(patches)
        #append elements of patches to patchesList
        for p in patches:
            patchesList.append(p)
    patchesList.sort(key=lambda x: x[0])
    return patchesList