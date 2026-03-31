"""
Script para gerar sínteses de textura a partir do dataset NetherlandsF3
Executa synthesis_ablation_no_zones.py para as 10 primeiras imagens
Gera samples de 40x40 pixels para cada imagem

Author: Luciano Terres
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comparaBaseFerreira.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path(r"D:\dataset\NetherlandsF3\tiles_inlines")
PROJECT_ROOT = Path(__file__).parent
SCRIPT_TO_RUN = PROJECT_ROOT / "synthesis_ablation_no_zones.py"
RESULT_ROOT = PROJECT_ROOT / "result"

# Parâmetros de síntese
WINDOW_HEIGHT = 40
WINDOW_WIDTH = 40
KERNEL_SIZE = 11
ITERATIONS = 10
NUM_SAMPLES = 20


def get_image_files(dataset_path: Path, limit: int = 10) -> list:
    """
    Retorna lista das primeiras `limit` imagens da pasta
    Suporta formatos: jpg, jpeg, png, bmp
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if not dataset_path.exists():
        logger.error(f"Dataset path não existe: {dataset_path}")
        sys.exit(1)
    
    image_files = [
        f for f in sorted(dataset_path.iterdir())
        if f.is_file() and f.suffix.lower() in valid_extensions
    ][:limit]
    
    if not image_files:
        logger.error(f"Nenhuma imagem encontrada em: {dataset_path}")
        sys.exit(1)
    
    logger.info(f"Encontradas {len(image_files)} imagens no dataset")
    return image_files


def create_output_directories(result_base: Path) -> None:
    """Cria diretórios de saída se não existirem"""
    result_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretório de resultados: {result_base}")


def get_next_result_directory(result_root: Path, prefix: str = "compara") -> Path:
    """Retorna o próximo diretório sequencial no formato compara001."""
    seq = 1
    while True:
        candidate = result_root / f"{prefix}{seq:03d}"
        if not candidate.exists():
            return candidate
        seq += 1


def run_synthesis(image_path: Path, output_dir: Path, sample_index: int) -> bool:
    """
    Executa synthesis_ablation_no_zones.py para uma imagem
    
    Args:
        image_path: Caminho da imagem de entrada
        output_dir: Diretório para salvar resultados
        sample_index: Índice do sample (para nomeação)
    
    Returns:
        True se bem-sucedido, False caso contrário
    """
    try:
        output_path = output_dir / f"synthesis_{sample_index:02d}_{image_path.stem}.png"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processando: {image_path.name} (Sample {sample_index})")
        logger.info(f"Entrada: {image_path}")
        logger.info(f"Saída: {output_path}")
        logger.info(f"Tamanho: {WINDOW_HEIGHT}x{WINDOW_WIDTH}")
        logger.info(f"{'='*70}")
        
        # Construir comando
        cmd = [
            sys.executable,
            str(SCRIPT_TO_RUN),
            f"--sample_path={image_path}",
            f"--out_path={output_path}",
            f"--window_height={WINDOW_HEIGHT}",
            f"--window_width={WINDOW_WIDTH}",
            f"--kernel_size={KERNEL_SIZE}",
            f"--iterations={ITERATIONS}"
        ]
        
        logger.info(f"Executando: {' '.join(cmd)}")
        
        # Executar
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Sucesso: {image_path.name}")
            return True
        else:
            logger.error(f"✗ Falha ao processar {image_path.name}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout ao processar {image_path.name}")
        return False
    except Exception as e:
        logger.error(f"✗ Erro ao processar {image_path.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Função principal"""
    start_time = datetime.now()
    logger.info(f"\n{'#'*70}")
    logger.info(f"# COMPARABASEFERREIRA - Síntese de Textura")
    logger.info(f"# Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*70}\n")
    
    # Validar script
    if not SCRIPT_TO_RUN.exists():
        logger.error(f"Script não encontrado: {SCRIPT_TO_RUN}")
        sys.exit(1)

    result_base = get_next_result_directory(RESULT_ROOT)
    
    # Criar diretórios
    create_output_directories(result_base)
    
    # Obter imagens
    image_files = get_image_files(DATASET_PATH, limit=NUM_SAMPLES)
    logger.info(f"\nProcessando {len(image_files)} imagens...")
    
    # Executar síntese para cada imagem
    success_count = 0
    failed_count = 0
    
    for idx, image_path in enumerate(image_files, 1):
        if run_synthesis(image_path, result_base, idx):
            success_count += 1
        else:
            failed_count += 1
    
    # Resumo
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# RESUMO FINAL")
    logger.info(f"{'#'*70}")
    logger.info(f"Total de amostras: {NUM_SAMPLES}")
    logger.info(f"Sucesso: {success_count}")
    logger.info(f"Falhas: {failed_count}")
    logger.info(f"Duração total: {duration}")
    logger.info(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Resultados salvos em: {result_base}")
    logger.info(f"{'#'*70}\n")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
