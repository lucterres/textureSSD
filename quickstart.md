# Quick Start Guide

## Setup do Ambiente

1. **Criar e ativar o ambiente conda LIR:**
   ```bash
   conda env create -f environmentt.yml
   conda activate lir
   ```

2. **Verificar a instalação:**
   ```bash
   python -c "import cv2; import numpy; print('Dependências OK')"
   ```

## Uso Básico

### Síntese de Textura Simples

Execute o script principal com o caminho de uma imagem de entrada:

```bash
python synthesis.py --sample_path=<imagem.png>
```

### Com Parâmetros Customizados

```bash
python synthesis.py \
  --sample_path=tgs_salt/3ce657b8f7.png \
  --window_height=101 \
  --window_width=101 \
  --kernel_size=11 \
  --iterations=50 \
  --visualize
```

### Com Cache (Recomendado para Múltiplas Execuções)

```bash
python synthesis.py \
  --sample_path=tgs_salt/3ce657b8f7.png \
  --window_height=101 \
  --window_width=101 \
  --kernel_size=11 \
  --patches_cache_path=result/patches_db_cache.npz
```

## Parâmetros Principais

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--sample_path` | Caminho da imagem de entrada (obrigatório) | - |
| `--window_height` | Altura da textura sintetizada | 50 |
| `--window_width` | Largura da textura sintetizada | 50 |
| `--kernel_size` | Tamanho do kernel de síntese | 11 |
| `--iterations` | Número de repetições da síntese | 50 |
| `--visualize` | Visualizar progresso | desativado |
| `--out_path` | Caminho de saída (opcional) | automático |

## Exemplos de Entrada

Imagens de teste estão disponíveis em:
- `examples/161.jpg`
- `examples/D3.jpg`
- `examples/wood.jpg`
- `tgs_salt/` (dataset especializado)

### Teste Rápido

```bash
python synthesis.py --sample_path=examples/wood.jpg --visualize
```

## Estudo de Ablação

Para executar estudos de ablação sem zonas:

```bash
python synthesis_ablation_no_zones.py \
  --sample_path=tgs_salt/3ce657b8f7.png \
  --window_height=101 \
  --window_width=101 \
  --kernel_size=11 \
  --iterations=20 \
  --visualize
```

### Comparação em Lote com Dataset NetherlandsF3

Para gerar sínteses em lote a partir das imagens do dataset NetherlandsF3, use:

```bash
python comparaBaseFerreira.py
```

Esse script:
- lê imagens em `D:/dataset/NetherlandsF3/tiles_inlines`
- executa `synthesis_ablation_no_zones.py` para múltiplas amostras
- usa a configuração atual de ablação em lote com `selection_method=weighted` e `seed_mode=center`
- salva os resultados em um diretório sequencial dentro de `result/`, como `compara001/`

Se necessário, ajuste diretamente no script os parâmetros `WINDOW_HEIGHT`, `WINDOW_WIDTH`, `KERNEL_SIZE`, `ITERATIONS` e `NUM_SAMPLES`.

## Otimização com Cache

### Primeira Execução (Constrói o cache)
```bash
python synthesis.py \
  --sample_path=tgs_salt/0bdd44d530.png \
  --window_height=101 \
  --window_width=101 \
  --kernel_size=11 \
  --patches_cache_path=result/patches_db_cache.npz
```

### Execuções Subsequentes (Usa o cache)
```bash
python synthesis.py --sample_path=tgs_salt/0bdd44d530.png --window_height=101 --window_width=101 --kernel_size=11
```

### Forçar Reconstrução do Cache
```bash
python synthesis.py ... --rebuild_patches_db
```

## Outputs

Os resultados são salvos em:
- `result/` - Diretório principal de resultados
- `result/run_<id>/` - Métricas de cada execução
- `result/patches_db_cache.npz` - Cache de patches (se habilitado)

## Troubleshooting

**Erro ao importar OpenCV:**
```bash
conda install opencv
```

**Ambiente não encontrado:**
```bash
conda env list
conda activate lir
```

**Cache corrompido:**
```bash
rm result/patches_db_cache.npz
python synthesis.py ... --rebuild_patches_db
```

## Próximos Passos

- Veja [readme.md](readme.md) para documentação completa
- Verifique [ablation_study_usage.md](ablation_study_usage.md) para guias avançados
- Explore exemplos em `examples/` e `tgs_salt/`

## Algoritmo de Referência

Implementação baseada em:
**Efros & Leung - "Texture Synthesis by Non-parametric Sampling" (1999)**
