import os
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def load_binary_mask(image_path):
    """
    Carrega uma imagem e converte para máscara binária.
    Assume que pixels não-pretos (> 0) são positivos.
    """
    img = Image.open(image_path).convert('L')  # Converte para escala de cinza
    mask = np.array(img)
    # Binariza: pixels > 0 são considerados como classe positiva (1)
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask


def calculate_metrics(ground_truth, prediction):
    """
    Calcula precision, recall e F1-score comparando duas máscaras binárias.
    
    Args:
        ground_truth: Máscara de referência (numpy array binário)
        prediction: Máscara predita (numpy array binário)
    
    Returns:
        dict com precision, recall e f1_score
    """
    # Flatten das máscaras para facilitar o cálculo
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()
    
    # Calcula True Positives, False Positives, False Negatives
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    
    # Calcula métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def main():
    parser = argparse.ArgumentParser(description="Calcula métricas de comparação de texturas.")
    parser.add_argument("--compare_dir", "-d", type=str, required=True, 
                        help="Diretório com as imagens sintetizadas.")
    parser.add_argument("--mask_path", "-m", type=str, default=None,
                        help="Caminho para a máscara de referência (default: compare_dir/Mask.png).")
    
    args = parser.parse_args()
    
    compare_dir = Path(args.compare_dir)
    
    if args.mask_path:
        mask_path = Path(args.compare_dir + '/' + args.mask_path)
    else:
        mask_path = compare_dir / "Mask.png"
    
    if not mask_path.exists():
        print(f"Erro: Arquivo {mask_path} não encontrado!")
        return
    
    if not compare_dir.exists():
        print(f"Erro: Diretório {compare_dir} não encontrado!")
        return
    
    # Carrega a máscara de referência
    print(f"Carregando máscara de referência: {mask_path}")
    ground_truth = load_binary_mask(mask_path)
    print(f"Dimensões da máscara: {ground_truth.shape}")
    print(f"Pixels positivos na máscara: {np.sum(ground_truth)}")
    print("-" * 80)
    
    # Lista todas as imagens PNG exceto Mask.png
    image_files = [f for f in compare_dir.glob("*.png") if f.name != "Mask.png"]
    
    if not image_files:
        print("Nenhuma imagem encontrada para comparação!")
        return
    
    print(f"Encontradas {len(image_files)} imagens para comparação\n")
    
    # Armazena os resultados
    results = []
    
    # Processa cada imagem
    for img_path in sorted(image_files):
        try:
            prediction = load_binary_mask(img_path)
            
            # Verifica se as dimensões são compatíveis
            if prediction.shape != ground_truth.shape:
                print(f"AVISO: {img_path.name} tem dimensões diferentes ({prediction.shape}). Pulando...")
                continue
            
            # Calcula métricas
            metrics = calculate_metrics(ground_truth, prediction)
            
            # Armazena resultado
            results.append({
                'filename': img_path.name,
                **metrics
            })
            
            # Exibe resultado individual
            print(f"{img_path.name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print()
            
        except Exception as e:
            print(f"Erro ao processar {img_path.name}: {e}")
            continue
    
    # Calcula estatísticas agregadas
    if results:
        print("=" * 80)
        print("ESTATÍSTICAS AGREGADAS")
        print("=" * 80)
        
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        
        print(f"\nPRECISION:")
        print(f"  Mínimo:  {min(precisions):.4f}")
        print(f"  Média:   {np.mean(precisions):.4f}")
        print(f"  Máximo:  {max(precisions):.4f}")
        
        print(f"\nRECALL:")
        print(f"  Mínimo:  {min(recalls):.4f}")
        print(f"  Média:   {np.mean(recalls):.4f}")
        print(f"  Máximo:  {max(recalls):.4f}")
        
        print(f"\nF1-SCORE:")
        print(f"  Mínimo:  {min(f1_scores):.4f}")
        print(f"  Média:   {np.mean(f1_scores):.4f}")
        print(f"  Máximo:  {max(f1_scores):.4f}")
        
        print(f"\nTotal de imagens processadas: {len(results)}")
    else:
        print("Nenhum resultado para exibir!")


if __name__ == "__main__":
    main()
