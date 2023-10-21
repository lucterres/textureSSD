# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadDataBase():
    #define localização dos diretórios de imagens
    inHouse=True
    
    if inHouse:
        #Desktop I3
        TRAIN_CSV = r'D:\_0Luciano\_0PHD\datasets\tgs-salt\train1090.csv'
        IMAGES_DIR = r'D:\_0Luciano\_0PHD\datasets\tgs-salt\train\images'
        MASK_DIR = r'D:\_0Luciano\_0PHD\datasets\tgs-salt\masks10-90'
    else:
        # ES00004605
        TRAIN_CSV = r'G:\_phd\dataset\tgs-salt\saltMaskOk.csv'
        IMAGES_DIR= r'G:\_phd\dataset\tgs-salt\train\images' 
        MASK_DIR = r'G:\_phd\dataset\tgs-salt\train\masks'

    df_train = pd.read_csv(TRAIN_CSV)
    fileNamesList = df_train.iloc[0:100,0]
    imagesList = loadImages(IMAGES_DIR, fileNamesList)
    masksList  = loadImages(MASK_DIR,  fileNamesList)
    patchesDB = buildPatchesDB(masksList, imagesList)
    return patchesDB

def makePatchMask(generat_mask, x1, x2):
    patchMask = generat_mask.copy()
    #set 0 to generat_mask columns from point x1 to x2
    if x1 < x2:
        patchMask[:,0:x1] = 0
        patchMask[:,x2:] = 0
    else:
        patchMask[:,0:x2] = 0
        patchMask[:,x1:] = 0
    return patchMask

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

# Display an array of images
def dispPatchesClass(patches):
    n = len(patches)
    if n > 1:
        fig, axs = plt.subplots(1, n , figsize=(10, 10))
        for i in range(n):
            patch = patches[i]
            patch.angle
            if (n<40): axs[i].set_title(patch.angle)
            axs[i].imshow(cv2.cvtColor(patch.image, cv2.COLOR_BGR2RGB))
            axs[i].axis('off')
        plt.show()
    if n == 1:
        patch = patches[0]
        plt.figure(figsize=(10,10))
        plt.title(patch.angle)
        plt.imshow(cv2.cvtColor(patch.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

#show all images in imagesList
def showImages(images, imagesTitle):
    n = len(images)
    if n > 1:
            fig, axs = plt.subplots(1, n , figsize=(5, 5))
            for i in range(n):
                axs[i].imshow(images[i],cmap = 'gray')
                axs[i].set_title(imagesTitle[i])
                axs[i].axis('off')
            plt.show()
    if n == 1:
        plt.figure(figsize=(5,5))
        plt.title(imagesTitle[0])
        plt.imshow(images[0],cmap = 'gray')
        plt.axis('off')
        plt.show()
	
   
def swap(a,b):
    if a > b:
        return b, a
    else:
        return a, b
    
def cropPatch(original, x1, x2, y1, y2):
    #get image dimensions
    height,width = original.shape[:2]
    plus = 15   

    if x1 > x2:
        x1, x2 = swap(x1,x2)
    if y1 > y2:
        y1, y2 = swap(y1,y2)

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

class Patch:
    def __init__(self,line,sample):
        self.line = line
        x1,y1,x2,y2 = line
        self.angle = int(-np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi)
        self.image = sample

def showLines(original, lines):
    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(original, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

# Probabilistic Hough Transform 
def probHough(mask, original, tresh=20, minPoints=30, maxGap=5, sort = False):
    probabLines = original.copy()
    edges = cv2.Canny(mask, 100, 200)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, tresh, None, minPoints, maxGap)
    dbPatches = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            line = linesP[i][0]
            # Draw the probabilistic 
            x,y,x2,y2 = line
            cv2.line(probabLines, (x, y), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            imgpatch = cropPatch (original, x, x2, y, y2)
            dbPatches.append(Patch(line,imgpatch))
        #sort dbPatches by angle
        if sort: dbPatches.sort(key=lambda x: x.angle)
        #show3Images(original,mask, probabLines)
        # if sort: dbPatches.sort(key=lambda x: x[0])
    return dbPatches,probabLines

#search nearest key in ordered patches
def searchNearestKey(patches, key):
    n = len(patches)
    if n > 1:
        for i in range(n):
            patch = patches[i]
            if key <= patch.angle:
                return patch.image
        return patches[n-1].angle,patches[n-1].image
    if n == 1:
        return patches[0].angle,patches[0].image
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
        imagePatches, linesP = probHough(mask, original, tresh = 15, minPoints=15, maxGap=15)
        #dispArrayImages(patches)
        #append elements of patches to patchesList
        for p in imagePatches:
            patchesList.append(p)
    #sort patches by angle
    patchesList.sort(key=lambda x: x.angle)
    return patchesList