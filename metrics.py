# This module provides functions to compute various image similarity metrics
# such as Mean Squared Error (MSE), Structural Similarity Index (SSIM),
# Peak Signal-to-Noise Ratio (PSNR), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
# and Local Binary Pattern (LBP) distance.
# Each function takes two images as input and returns the computed metric.
# The images should be in grayscale format for these calculations.
# The metrics can be used to evaluate the similarity between two images,
# which is useful in various applications such as image processing, computer vision, and machine learning.


# calcule o mse entre duas imagens
import cv2
import numpy as np  
from skimage.feature import local_binary_pattern

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
# Índice de Similaridade Estrutural (SSIM) entre duas imagens
def _ssim(imageA, imageB):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    muA = cv2.GaussianBlur(imageA.astype("float"), (11, 11), 1.5)
    muB = cv2.GaussianBlur(imageB.astype("float"), (11, 11), 1.5)

    sigmaA = cv2.GaussianBlur(imageA.astype("float") ** 2, (11, 11), 1.5) - muA ** 2
    sigmaB = cv2.GaussianBlur(imageB.astype("float") ** 2, (11, 11), 1.5) - muB ** 2
    sigmaAB = cv2.GaussianBlur(imageA.astype("float") * imageB.astype("float"), (11, 11), 1.5) - muA * muB

    ssim_map = ((2 * muA * muB + C1) * (2 * sigmaAB + C2)) / ((muA ** 2 + muB ** 2 + C1) * (sigmaA + sigmaB + C2))
    return np.mean(ssim_map)

def dssim(imageA, imageB):
    similarity = _ssim(imageA, imageB)
    dssim = (1 - similarity) / 2  # Convert SSIM to DSSIM
    return dssim  # Return DSSIM instead of SSIM for better interpretation

# calcula o PSNR entre duas imagens
def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # PSNR is infinite if images are identical
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

#calcula o MAE entre duas imagens
def mae(imageA, imageB):
    return np.mean(np.abs(imageA.astype("float") - imageB.astype("float")))

#Euclidean distance based on the Local Binary Pattern

#calcula o RMSE entre duas imagens
def rmse(imageA, imageB):
    return np.sqrt(mse(imageA, imageB))

#calcula o Euclidean distance based on the Local Binary Pattern (LBP) entre duas imagens
""" def lbp_distance(imageA, imageB):
    lbpA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    lbpB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    lbpA = cv2.normalize(lbpA, lbpA).flatten()
    lbpB = cv2.normalize(lbpB, lbpB).flatten()
    return np.linalg.norm(lbpA - lbpB) """

# compute LBP  using four neighbors, distance 1, and tile size of 64 pixels.
# Note: The LBP distance function assumes that the images are already in grayscale.
# If the images are not in grayscale, they should be converted before calling this function.
# This module provides a set of functions to compute various image similarity metrics.
def compute_lbp(image, P=4, R=1, tile_size=64):
    """
    Compute LBP for the image using four neighbors, distance 1, and tile size of 64 pixels.
    Returns a list of LBP histograms for each tile.
    """
    h, w = image.shape
    lbp_histograms = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                continue  # skip incomplete tiles
            lbp = local_binary_pattern(tile, P, R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # normalize
            lbp_histograms.append(hist)


# Calcula a distância euclidiana média entre os histogramas LBP de tiles 64x64, usando 4 vizinhos e raio 1
def lbp_tile_distance(imageA, imageB, P=4, R=1, tile_size=64):
    """
    Calcula a distância euclidiana média entre os histogramas LBP de tiles correspondentes
    das duas imagens, usando 4 vizinhos, raio 1 e tile size 64.
    As imagens devem estar em escala de cinza e ter o mesmo tamanho.
    """
    h, w = imageA.shape
    dist_total = 0
    count = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tileA = imageA[y:y+tile_size, x:x+tile_size]
            tileB = imageB[y:y+tile_size, x:x+tile_size]
            if tileA.shape != (tile_size, tile_size) or tileB.shape != (tile_size, tile_size):
                continue  # pula tiles incompletos
            lbpA = local_binary_pattern(tileA, P, R, method='uniform')
            lbpB = local_binary_pattern(tileB, P, R, method='uniform')
            # Histograma LBP
            histA, _ = np.histogram(lbpA.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            histB, _ = np.histogram(lbpB.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            # Normaliza
            histA = histA.astype('float') / (histA.sum() + 1e-6)
            histB = histB.astype('float') / (histB.sum() + 1e-6)
            dist_total += np.linalg.norm(histA - histB)
            count += 1
    return dist_total / count if count > 0 else None

# Exemplo de uso:
# imageA = cv2.imread('img1.png', 0)
# imageB = cv2.imread('img2.png', 0)
# dist = lbp_tile_distance(imageA, imageB)
# print(f"LBP tile distance: {dist}")
