# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import suport.locals as locals


# Rotaciona a imagem em torno do centro
def rotateImage(img, angle):
    # Calcula a matriz de rotação
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
    # Aplica a transformação afim à imagem
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # Exibe a imagem rotacionada
    return rotated

def loadDataBase(samples=1000, treshold=100, cache_path=None, rebuild=False):
    """Load (or build) the patches database.

    Parameters
    ----------
    samples : int
        Number of image/mask pairs to sample for building the DB.
    treshold : int
        Minimum area (pixels) for a patch to be included.
    cache_path : str or None
        If provided, an .npz file path used to load/save the cached patch list.
    rebuild : bool
        If True forces rebuilding the database even if cache exists.

    Returns
    -------
    list[Patch]
        List of Patch objects sorted by angle.
    """
    # Try loading from cache first
    if cache_path and (not rebuild) and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            patches_arr = data['patches']
            patches_list = []
            for item in patches_arr:
                # each item is a dict with keys: line, angle, image
                line = tuple(item['line'])
                angle = int(item['angle'])
                image = item['image']
                p = Patch(line, image)
                p.angle = angle  # override computed angle for fidelity
                patches_list.append(p)
            print(f"Patches DB loaded from cache: {cache_path} ({len(patches_list)} patches)")
            return patches_list
        except Exception as e:
            print(f"Falha ao carregar cache ({cache_path}), reconstruindo. Erro: {e}")

    # Define localização dos diretórios de imagens
    if locals.inHouseAMD:
        housepath=r'D:\\_PHD\datasets\\tgs-salt\\'
        TRAIN_CSV = housepath + r'saltMaskOk.csv'
        IMAGES_DIR= housepath + r'train\\images'
        MASK_DIR = housepath + r'train\\masks'
    elif locals.inNote:
        housepath=r'D:\\datasets\\tgs-salt\\'
        TRAIN_CSV = housepath + r'saltMaskOk.csv'
        IMAGES_DIR= housepath + r'train\\images'
        MASK_DIR = housepath + r'train\\masks'
    elif locals.in8700G:
        housepath=r'D:\\dataset\\tgs-salt\\'
        TRAIN_CSV = housepath + r'saltMaskOk.csv'
        IMAGES_DIR= housepath + r'train\\images'
        MASK_DIR = housepath + r'train\\masks'
    else:
        raise RuntimeError("Nenhum ambiente (locals) configurado para paths de dataset.")

    df_train = pd.read_csv(TRAIN_CSV)
    fileNamesList = df_train.iloc[0:samples,0]
    imagesList = loadImages(IMAGES_DIR, fileNamesList)
    masksList  = loadImages(MASK_DIR,  fileNamesList)
    patchesDB = buildPatchesDB(masksList, imagesList, treshold)

    # Save cache if path provided
    if cache_path:
        serializable = []
        for p in patchesDB:
            serializable.append({
                'line': np.array(p.line, dtype=np.int32),
                'angle': p.angle,
                'image': p.image
            })
        np.savez_compressed(cache_path, patches=np.array(serializable, dtype=object))
        print(f"Patches DB saved to cache: {cache_path} ({len(patchesDB)} patches)")

    return patchesDB

def makePatchMask(generat_mask, x1, x2):
    patchMask = generat_mask.copy()
    #set 0 to generat_mask columns from point x1 to x2 - the outside of the patch
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
    dilated_edge, zone0, zone1, fullmask = create_Masks(mask, dilatedEdge=True, size=3)
    
    sEdge = sample * dilated_edge.astype(np.uint8)
    sZ1   = sample * zone0.astype(np.uint8)
    sZ2   = sample * zone1.astype(np.uint8)
    #zone1Comp = cv2.bitwise_not(zone1)
    #blank = np.ones_like(sZ1)
    #zone1Comp= zone1Comp * blank
    #sZ2 = sZ2 + zone1Comp """
    return sEdge, sZ1, sZ2

# Calculating Masks
def create_Masks(mask, dilatedEdge=True, size=5):
    # edge definition
    edge = cv2.Canny(mask,100,200)
    kernel = np.ones((size,size))
    if dilatedEdge:
        edge = cv2.dilate(edge, kernel, iterations=1)
    inv_edge     = cv2.bitwise_not(edge)

    # Inverting the mask 
    mask_inverted = cv2.bitwise_not(mask)

    # Normalize to the range [0., 1.]
    mask = mask.astype(np.float64) / 255.
    mask_inverted = mask_inverted.astype(np.float64) / 255.
    edge = edge.astype(np.float64) / 255.
    inv_edge     = inv_edge.astype(np.float64) / 255.

    zone0 = mask * inv_edge
    zone1 = mask_inverted * inv_edge

    fullmask = zone0*1 + edge*2 +zone1*3

    return edge, zone0, zone1, fullmask

# Display an array of objects type image/patches
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
def showImages(images, imagesTitle,size=(5,5)):
    n = len(images)
    if n > 1:
            fig, axs = plt.subplots(1, n , figsize=size)
            for i in range(n):
                axs[i].imshow(images[i],cmap= 'gray')
                if len(imagesTitle) > 0: axs[i].set_title(imagesTitle[i])
                axs[i].axis('off')
            plt.show()
    if n == 1:
        plt.figure(figsize=size)
        if len(imagesTitle)==1 : plt.title(imagesTitle[0])
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


def searchAnglesinRange(patchesDB, angle, threshold):
    """
    Search for angles within a specified threshold in the patchesDB.

    Parameters:
    - patchesDB (list): List of patches from the database.
    - angle (float): The target angle to search for.
    - threshold (float): The threshold to determine the range of angles to search within.

    Returns:
    - list: List of patches with angles within the specified threshold.
    """
    result = []
    for patch in patchesDB:
        if abs(patch.angle - angle) <= threshold:
            result.append(patch)
    return result



#search nearest key in ordered patches
def searchNearestPatchByAngle(patches, angle):
    n = len(patches)
    if n > 1:
        for i in range(n):
            patch = patches[i]
            if angle <= patch.angle:
                return patch.image
        return patches[n-1].angle,patches[n-1].image
    if n == 1:
        return patches[0].angle,patches[0].image
    return None

def searchNearestPatchByAngleAndHistogram(samplesPatchesDB, angle, imgRGB):
    vecImages = searchAnglesinRange(samplesPatchesDB, angle, threshold=5)
    l = len(vecImages)
    if l > 5:
        p = identify_nearest_histogram(vecImages,imgRGB)
        return p.image
    else :
        print("Not enough samples")
        return None
    


# identify nearest patch from image using calcHist
def identify_nearest_histogram(patchList, image):
    if len(patchList) == 0:
        print("Empty list")
        return None
    h = (cv2.calcHist([image], [0], None, [256], [0,256]))
    original = cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    maxCorrelation = -1  # initial value
    nearestPatch = None  # result patch
    
    for p in patchList:
        i = p.image
        # Calcular os histogramas
        h = (cv2.calcHist([i], [0], None, [256], [0,256]))
        # Normalizar os histogramas
        h = cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # Comparar os histogramas de cada imagem com imagem original
        correlation = cv2.compareHist(original, h , cv2.HISTCMP_CORREL)
        if correlation > maxCorrelation:
            maxCorrelation = correlation
            nearestPatch = p

    return nearestPatch


#load a list of images from files
def loadImages(path, imageList):
    loadedImages = []
    for filename in imageList:
        img = cv2.imread(path + '\\' + filename+ '.png') 
        if img is not None: 
            loadedImages.append(img)
    return loadedImages

def buildPatchesDB(masksList, imagesList, treshold=100):
    #iterate over imagesList
    patchesList = []
    for i in range(len(imagesList)):
        original = imagesList[i] 
        mask = masksList[i]
        imagePatches, linesP = probHough(mask, original, tresh = 15, minPoints=15, maxGap=15)
        #dispArrayImages(patches)
        #append elements of patches to patchesList
        for p in imagePatches:
            x,y = p.image.shape[:2]
            if x*y > treshold:
                patchesList.append(p)
    #sort patches by angle
    patchesList.sort(key=lambda x: x.angle)
    return patchesList