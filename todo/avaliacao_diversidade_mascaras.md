# Avaliação de Diversidade de Máscaras Geradas (VAE vs. Transformações Simples)

## Introdução

Este documento apresenta metodologias para avaliar quantitativamente a diversidade de máscaras de corpos salinos geradas por diferentes métodos, especificamente comparando:
- **VAE (Variational Autoencoder)**: Método proposto no artigo
- **Transformações Simples**: Transformações geométricas básicas (rotação, escala, deformação elástica)

---

## 1. Métricas para Avaliar Diversidade

### 1.1 Diversidade Intra-conjunto (Intra-set Diversity)

Mede quão diferentes são as máscaras geradas entre si dentro do mesmo conjunto.

```python
def intra_set_diversity(masks):
    """
    Calcula a diversidade média entre todas as máscaras do conjunto
    """
    n = len(masks)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            # Distância L1, L2 ou Hamming
            dist = np.mean(np.abs(masks[i] - masks[j]))
            distances.append(dist)
    return np.mean(distances), np.std(distances)
```

**Interpretação**: Valores maiores indicam maior diversidade.

---

### 1.2 Distância de Hausdorff

Mede a diferença entre os contornos das máscaras.

```python
from scipy.spatial.distance import directed_hausdorff

def hausdorff_diversity(masks):
    distances = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            contour_i = extract_contour(masks[i])
            contour_j = extract_contour(masks[j])
            dist = max(directed_hausdorff(contour_i, contour_j)[0],
                      directed_hausdorff(contour_j, contour_i)[0])
            distances.append(dist)
    return np.mean(distances)
```

---

### 1.3 Diversidade Geométrica

Analisa a variabilidade de propriedades geométricas das máscaras.

```python
def geometric_diversity(masks):
    features = []
    for mask in masks:
        features.append({
            'area': np.sum(mask),
            'perimeter': calculate_perimeter(mask),
            'compactness': 4*np.pi*area / (perimeter**2),
            'aspect_ratio': get_aspect_ratio(mask),
            'centroid_x': get_centroid(mask)[0],
            'centroid_y': get_centroid(mask)[1],
            'eccentricity': get_eccentricity(mask),
            'solidity': get_solidity(mask)
        })
    
    # Calcular variância/desvio padrão de cada característica
    df = pd.DataFrame(features)
    return df.std(), df.var()
```

**Características importantes**:
- **Área**: Porcentagem de sal na imagem
- **Perímetro**: Comprimento da fronteira
- **Compacidade**: Quão circular é a forma (1.0 = círculo perfeito)
- **Razão de aspecto**: Proporção largura/altura
- **Excentricidade**: Quão alongada é a forma (0 = círculo, 1 = linha)
- **Solidez**: Razão entre área e área do hull convexo

---

### 1.4 Métrica de Cobertura (Coverage)

Avalia se as máscaras geradas cobrem adequadamente o espaço de possibilidades das máscaras reais.

```python
def coverage_metric(generated_masks, real_masks, threshold=0.1):
    """
    Conta quantas máscaras reais têm pelo menos uma máscara 
    gerada similar (dentro do threshold)
    """
    covered = 0
    for real_mask in real_masks:
        min_dist = float('inf')
        for gen_mask in generated_masks:
            dist = np.mean(np.abs(real_mask - gen_mask))
            min_dist = min(min_dist, dist)
        if min_dist < threshold:
            covered += 1
    return covered / len(real_masks)
```

**Interpretação**: Valores próximos a 1.0 indicam boa cobertura do espaço real.

---

### 1.5 Fréchet Inception Distance (FID) Adaptado

Compara as distribuições de características entre máscaras reais e geradas.

```python
from scipy.linalg import sqrtm

def calculate_fid(real_features, generated_features):
    """
    FID entre máscaras reais e geradas
    Valores menores indicam maior similaridade com a distribuição real
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid
```

---

### 1.6 Detecção de Mode Collapse

Detecta se o modelo está gerando máscaras muito similares (colapso de modo).

```python
def detect_mode_collapse(masks, n_clusters=10):
    """
    Detecta se há colapso de modo usando clustering
    """
    from sklearn.cluster import KMeans
    
    masks_flat = masks.reshape(len(masks), -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(masks_flat)
    
    # Distribuição uniforme seria 1/n_clusters para cada cluster
    unique, counts = np.unique(labels, return_counts=True)
    distribution = counts / len(masks)
    
    # Entropia da distribuição (máxima = log(n_clusters))
    entropy = -np.sum(distribution * np.log(distribution + 1e-10))
    max_entropy = np.log(n_clusters)
    
    return {
        'entropy': entropy,
        'normalized_entropy': entropy / max_entropy,
        'cluster_distribution': distribution
    }
```

**Interpretação**: Entropia normalizada próxima a 1.0 indica boa diversidade (sem mode collapse).

---

### 1.7 Visualização com PCA

Visualiza a distribuição das máscaras no espaço de características reduzido.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_diversity_pca(vae_masks, simple_masks, real_masks):
    # Flatten masks
    vae_flat = vae_masks.reshape(len(vae_masks), -1)
    simple_flat = simple_masks.reshape(len(simple_masks), -1)
    real_flat = real_masks.reshape(len(real_masks), -1)
    
    # PCA
    all_data = np.vstack([vae_flat, simple_flat, real_flat])
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # Transform
    vae_pca = pca.transform(vae_flat)
    simple_pca = pca.transform(simple_flat)
    real_pca = pca.transform(real_flat)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], label='Real', alpha=0.5, s=30)
    plt.scatter(vae_pca[:, 0], vae_pca[:, 1], label='VAE', alpha=0.5, s=30)
    plt.scatter(simple_pca[:, 0], simple_pca[:, 1], label='Simple', alpha=0.5, s=30)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Diversity Comparison in PCA Space')
    plt.grid(True, alpha=0.3)
```

---

## 2. Quantidade de Máscaras para Comparação

### 2.1 Recomendações por Tipo de Análise

| Tipo de Análise | Quantidade Recomendada | Justificativa |
|-----------------|------------------------|---------------|
| **Básica** | 30-50 máscaras | Estatísticas descritivas, testes t simples |
| **Robusta** ⭐ | 100-200 máscaras | Métricas de diversidade confiáveis, análise PCA |
| **Abrangente** | 500-1000 máscaras | FID preciso, coverage metrics, validação cruzada |

### 2.2 Recomendação Específica para Dataset TGS Salt

Considerando o dataset TGS com ~4000 imagens de treino:

**Configuração Recomendada:**
- **VAE**: 200 máscaras geradas
- **Transformações simples**: 200 máscaras geradas
- **Máscaras reais (referência)**: 200 máscaras do dataset

### 2.3 Justificativa Estatística

Para comparação de médias com poder estatístico adequado:

```python
# Poder estatístico desejado: 0.80
# Nível de significância: 0.05
# Effect size esperado: médio (d = 0.5)

n_required = 64   # por grupo para detectar diferença média
n_recommended = 100  # com margem de segurança
n_ideal = 200     # para robustez e intervalos de confiança estreitos
```

### 2.4 Considerações Computacionais

Complexidade para métricas par-a-par:

| n máscaras | Comparações | Tempo estimado* |
|------------|-------------|-----------------|
| 100 | 4,950 | ~1-2 min |
| 200 | 19,900 | ~5-10 min |
| 500 | 124,750 | ~30-60 min |

*Tempo aproximado em hardware moderno para máscaras 101×101 pixels.

---

## 3. Protocolo de Avaliação Completo

### 3.1 Configuração Experimental

```python
# Configuração para ablation study
CONFIG = {
    'n_generated_vae': 200,
    'n_generated_simple': 200,
    'n_real_reference': 200,
    'n_bootstrap_samples': 1000,  # para intervalos de confiança
    'random_seed': 42,
    'image_size': (101, 101)
}
```

### 3.2 Pipeline de Avaliação

```python
# 1. Gerar conjuntos de máscaras
np.random.seed(CONFIG['random_seed'])
vae_masks = generate_masks_with_vae(n=CONFIG['n_generated_vae'])
simple_masks = generate_masks_with_transforms(n=CONFIG['n_generated_simple'])
real_masks = load_real_masks(n=CONFIG['n_real_reference'])

# 2. Avaliar diversidade intra-conjunto
vae_diversity_mean, vae_diversity_std = intra_set_diversity(vae_masks)
simple_diversity_mean, simple_diversity_std = intra_set_diversity(simple_masks)

# 3. Avaliar diversidade geométrica
vae_geo_std, vae_geo_var = geometric_diversity(vae_masks)
simple_geo_std, simple_geo_var = geometric_diversity(simple_masks)

# 4. Avaliar cobertura
vae_coverage = coverage_metric(vae_masks, real_masks)
simple_coverage = coverage_metric(simple_masks, real_masks)

# 5. Calcular FID
vae_features = extract_geometric_features(vae_masks)
simple_features = extract_geometric_features(simple_masks)
real_features = extract_geometric_features(real_masks)

vae_fid = calculate_fid(real_features, vae_features)
simple_fid = calculate_fid(real_features, simple_features)

# 6. Detectar mode collapse
vae_collapse = detect_mode_collapse(vae_masks)
simple_collapse = detect_mode_collapse(simple_masks)

# 7. Teste estatístico
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(
    compute_all_pairwise_distances(vae_masks),
    compute_all_pairwise_distances(simple_masks)
)
```

### 3.3 Bootstrap para Intervalos de Confiança

```python
def bootstrap_diversity_ci(masks, n_bootstrap=1000, confidence=0.95):
    """
    Calcula intervalo de confiança para métricas de diversidade
    usando bootstrap
    """
    diversities = []
    n = len(masks)
    
    for _ in range(n_bootstrap):
        # Reamostragem com reposição
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample_masks = masks[sample_idx]
        diversity, _ = intra_set_diversity(sample_masks)
        diversities.append(diversity)
    
    lower = np.percentile(diversities, (1-confidence)/2 * 100)
    upper = np.percentile(diversities, (1+confidence)/2 * 100)
    
    return np.mean(diversities), (lower, upper)

# Aplicar
vae_div, vae_ci = bootstrap_diversity_ci(vae_masks)
simple_div, simple_ci = bootstrap_diversity_ci(simple_masks)
```

---

## 4. Tabela de Resultados Esperados

| Métrica | VAE (Esperado) | Transformações Simples | Interpretação |
|---------|----------------|------------------------|---------------|
| **Diversidade Intra-conjunto** | Alta (>0.30) | Baixa (<0.20) | VAE explora espaço latente mais amplamente |
| **Variância Geométrica (área)** | Alta | Média | VAE gera formas mais variadas |
| **Variância Geométrica (forma)** | Alta | Baixa | Transformações preservam forma original |
| **Coverage** | Alta (>0.70) | Baixa (<0.40) | VAE cobre melhor o espaço real |
| **FID** | Baixo (<20) | Alto (>40) | VAE mais próximo da distribuição real |
| **Entropia Normalizada** | ~1.0 | <0.7 | VAE sem mode collapse |
| **Hausdorff Distance (média)** | Alta | Baixa | VAE gera contornos mais diversos |

---

## 5. Exemplo de Relatório de Resultados

```python
# Análise com 200 máscaras por conjunto
results = {
    'VAE': {
        'n_samples': 200,
        'diversity_mean': 0.342,
        'diversity_ci': (0.328, 0.356),
        'geometric_variance_area': 0.089,
        'geometric_variance_eccentricity': 0.156,
        'fid_score': 12.4,
        'coverage': 0.78,
        'normalized_entropy': 0.94
    },
    'Simple_Transforms': {
        'n_samples': 200,
        'diversity_mean': 0.187,
        'diversity_ci': (0.179, 0.195),
        'geometric_variance_area': 0.034,
        'geometric_variance_eccentricity': 0.045,
        'fid_score': 45.7,
        'coverage': 0.35,
        'normalized_entropy': 0.62
    },
    'Statistical_Test': {
        't_statistic': 8.45,
        'p_value': 1.2e-15,
        'conclusion': 'Diferença altamente significativa (p < 0.001)'
    }
}
```

### 5.1 Visualização de Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico comparativo
metrics = ['Diversity', 'Coverage', 'Norm. Entropy']
vae_scores = [0.342, 0.78, 0.94]
simple_scores = [0.187, 0.35, 0.62]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, vae_scores, width, label='VAE', alpha=0.8)
ax.bar(x + width/2, simple_scores, width, label='Simple Transforms', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Diversity Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
```

---

## 6. Conclusões e Recomendações

### 6.1 Tamanho de Amostra Recomendado

**Para ablation study robusto:**
- ✅ **200 máscaras por conjunto** (VAE, Simple, Real)
- ✅ Bootstrap com 1000 iterações para intervalos de confiança
- ✅ Múltiplas sementes aleatórias (3-5) para validação

### 6.2 Métricas Essenciais

**Mínimo necessário:**
1. Diversidade intra-conjunto
2. Variância de características geométricas (área, excentricidade)
3. FID score
4. Teste estatístico (t-test ou Mann-Whitney)

**Recomendado adicionar:**
5. Coverage metric
6. Detecção de mode collapse
7. Visualização PCA/t-SNE

### 6.3 Critérios de Validação

O VAE deve demonstrar:
- ✅ Diversidade intra-conjunto **significativamente maior** (p < 0.05)
- ✅ FID score **menor** (mais próximo da distribuição real)
- ✅ Coverage **maior** (>70% das máscaras reais cobertas)
- ✅ Entropia normalizada **próxima a 1.0** (sem mode collapse)
- ✅ Variância geométrica **maior** em múltiplas características

### 6.4 Apresentação em Paper

**Formato sugerido para seção de resultados:**

> "To evaluate mask diversity, we generated 200 masks using each method (VAE and simple geometric transformations) and compared them against 200 real masks from the TGS dataset. The VAE-generated masks exhibited significantly higher intra-set diversity (0.342 ± 0.014) compared to simple transformations (0.187 ± 0.008, p < 0.001). Furthermore, the VAE achieved superior coverage of the real mask distribution (78% vs. 35%) and lower FID score (12.4 vs. 45.7), indicating both greater diversity and closer alignment with the true geological distribution. Mode collapse analysis revealed normalized entropy of 0.94 for VAE versus 0.62 for simple transformations, confirming the VAE's ability to explore the full space of plausible salt geometries."

---

## Referências

- Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- Heusel et al. (2017) - GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)
- Zhou et al. (2018) - Non-stationary Texture Synthesis by Adversarial Expansion
- Powers (2011) - Evaluation: From Precision, Recall and F-measure to ROC

---

**Nota**: Este documento fornece um framework completo para avaliar objetivamente a contribuição do VAE na geração de máscaras diversas e geologicamente plausíveis, validando a escolha metodológica do artigo através de evidências quantitativas robustas.
