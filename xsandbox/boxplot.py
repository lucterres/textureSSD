import pandas as pd
import cv2
import numpy as np
from metrics import mse, dssim, lbp_distance, lbp_tile_distance
import matplotlib.pyplot as plt
import os

#desenhar um bloxplot com os valores de mse, ssim e lbp_distance
def boxplot(nomeOriginal, mse_values, ssim_values, lbp_distances):
    samples = len(mse_values)
    #titulo da figura
    #plt.suptitle(f'{nomeOriginal} - {samples} samples', fontsize=16)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.boxplot(mse_values)
    plt.title('MSE ')
    plt.subplot(1, 3, 2)
    plt.boxplot(lbp_distances)
    plt.title('LBP Distance ')
    plt.subplot(1, 3, 3)
    plt.boxplot(ssim_values)
    plt.title('DSSIM ')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta o layout para evitar sobreposição do título

#calcule valores mínimos, q1, q2, mediana, q3 e máximo
def statistics(listValues):
    min = np.min(listValues)
    q1 = np.percentile(listValues, 25)
    q2 = np.median(listValues)
    q3 = np.percentile(listValues, 75)
    max = np.max(listValues)
    print(f"   {min:.2f}, {q1:.2f}, {q2:.2f}, {q3:.2f}, {max:.2f}")
    #return min, q1, q2, q3, max

#criar função main para executar o código
def main():
    # Carregar o arquivo CSV com os caminhos das imagens
    df_imgs = pd.read_csv('result\\sintese.csv', sep=';' , header=None)

    imageA = 0
    imageB = 0
    mse_values = []
    dssim_values = []
    lbp_distances = []
    mse_total = []
    dssim_total = []
    lbp_total = []

    print(f"   Min , Q1 , Median , Q3 , Max , Mean ")
    print(f"      MSE")
    print(f"      LBP Distance")
    print(f"      DSSIM")

    for i, row in df_imgs.iterrows():
        for j, value in enumerate(row):
            if pd.notnull(value):
                if j == 0:
                    nomeOriginal = value
                    original = f"tgs_salt\\{value}.png"
                    # verifique se o arquivo existe amtes de tentar ler e avise de erro
                    if not os.path.exists(original):
                        print(f"File not found: {original}")
                        continue
                    imageA = cv2.imread(original,0)
                else:
                    nomeImage = value
                    sintese = f"result\\{value}.jpg"
                    # verifique se o arquivo existe amtes de tentar ler e avise de erro
                    if not os.path.exists(sintese):
                        print(f"File not found: {sintese}")
                        continue
                    imageB = cv2.imread(sintese,0)
                    mse_value = mse(imageA, imageB)
                    s = dssim(imageA, imageB)
                    lbp_td = lbp_tile_distance(imageA, imageB) #lbp_tile_dist(imageA, imageB)
                    mse_values.append(mse_value)
                    dssim_values.append(s)
                    lbp_distances.append(lbp_td)
        # stats do grupo
        if len(mse_values) == 0:
            print(f"No images found for group: {nomeOriginal}")
            continue
        # Salvar mse_values, ssim_values, lbp_distances em um arquivo CSV separado por ponto e vírgula.
        print(f"Group: {nomeOriginal} - Images: {len(mse_values)}")
        if len(mse_values) > 5:
            boxplot(nomeOriginal, mse_values, dssim_values, lbp_distances)
        statistics(mse_values)
        statistics(lbp_distances)
        statistics(dssim_values)
    
        mse_total = mse_total + mse_values
        lbp_total = lbp_total + lbp_distances
        dssim_total = dssim_total + dssim_values   
        
        mse_values = []
        dssim_values = []
        lbp_distances = []
    # stats do grupo
    print(f"Group: Total - Images: {len(mse_total)}")
    statistics(mse_total)
    statistics(lbp_total)
    statistics(dssim_total)
    boxplot('Total', mse_total, dssim_total, lbp_total)

    
if __name__ == "__main__":
    main()
    plt.show()  # Exibir o boxplot após a execução do código
    wait = input("Press Enter to exit...")  # Manter a janela aberta até o usuário pressionar Enter
    plt.close()  # Fechar a janela do boxplot após o usuário pressionar Enter

