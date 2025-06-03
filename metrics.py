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
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
# Índice de Similaridade Estrutural (SSIM) entre duas imagens
def ssim(imageA, imageB):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    muA = cv2.GaussianBlur(imageA.astype("float"), (11, 11), 1.5)
    muB = cv2.GaussianBlur(imageB.astype("float"), (11, 11), 1.5)

    sigmaA = cv2.GaussianBlur(imageA.astype("float") ** 2, (11, 11), 1.5) - muA ** 2
    sigmaB = cv2.GaussianBlur(imageB.astype("float") ** 2, (11, 11), 1.5) - muB ** 2
    sigmaAB = cv2.GaussianBlur(imageA.astype("float") * imageB.astype("float"), (11, 11), 1.5) - muA * muB

    ssim_map = ((2 * muA * muB + C1) * (2 * sigmaAB + C2)) / ((muA ** 2 + muB ** 2 + C1) * (sigmaA + sigmaB + C2))
    
    return np.mean(ssim_map)
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
def lbp_distance(imageA, imageB):
    lbpA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    lbpB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    lbpA = cv2.normalize(lbpA, lbpA).flatten()
    lbpB = cv2.normalize(lbpB, lbpB).flatten()
    return np.linalg.norm(lbpA - lbpB)



# one function to demonstrate the usage of the metrics



def main():
    # load the two input images
    imageA = cv2.imread("image1.png")
    imageB = cv2.imread("image2.png")

    # convert the images to grayscale
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the MSE between the two images
    m = mse(imageA, imageB)
    print(f"MSE: {m}")

    # compute the SSIM between the two images
    s = ssim(imageA, imageB)

    print(f"SSIM: {s}")

    # compute the PSNR between the two images
    p = psnr(imageA, imageB)
    print(f"PSNR: {p}")

    # compute the MAE between the two images
    a = mae(imageA, imageB)
    print(f"MAE: {a}")
    # compute the RMSE between the two images
    r = rmse(imageA, imageB)

    print(f"RMSE: {r}")

    # compute the LBP distance between the two images
    l = lbp_distance(imageA, imageB)
    print(f"LBP Distance: {l}")

    #percorre as imagens de um diretorio e calcule a média do diretório
    import os
    directory = "images"
    mse_list = []
    ssim_list = []
    psnr_list = []
    mae_list = []
    rmse_list = []
    lbp_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compute the metrics for each image
            m = mse(image_gray, imageA)
            s = ssim(image_gray, imageA)
            p = psnr(image_gray, imageA)
            a = mae(image_gray, imageA)
            r = rmse(image_gray, imageA)
            l = lbp_distance(image_gray, imageA)

            mse_list.append(m)
            ssim_list.append(s)
            psnr_list.append(p)
            mae_list.append(a)
            rmse_list.append(r)
            lbp_list.append(l)

            print(f"Metrics for {filename}:")
            print(f"MSE: {m}, SSIM: {s}, PSNR: {p}, MAE: {a}, RMSE: {r}, LBP Distance: {l}")

    if mse_list:
        print("\nAverage metrics for directory:")
        print(f"Mean MSE: {np.mean(mse_list)}")
        print(f"Mean SSIM: {np.mean(ssim_list)}")
        print(f"Mean PSNR: {np.mean(psnr_list)}")
        print(f"Mean MAE: {np.mean(mae_list)}")
        print(f"Mean RMSE: {np.mean(rmse_list)}")
        print(f"Mean LBP Distance: {np.mean(lbp_list)}")

if __name__ == "__main__":
    main()

