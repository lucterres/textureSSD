# Estudo de Ablação - Guia de Uso

## Visão Geral

Este documento descreve os experimentos de estudo de ablação projetados para avaliar a importância de cada componente na metodologia de síntese de texturas para geração de imagens sísmicas.

**Estudo de Ablação** é uma abordagem sistemática para validar escolhas de projeto removendo ou desabilitando componentes específicos e medindo o impacto nos resultados. Isso ajuda a identificar quais partes da metodologia são essenciais e quais podem ser redundantes.

## Experimentos Propostos

A metodologia inclui três componentes principais que podem ser avaliados através de ablação:

### 1. Síntese Orientada a Contexto com Separação de Zonas ✅ *Implementado*

**Questão:** Dividir a síntese em três zonas distintas (sal, rocha, fronteira) é importante?

**Método:** 
- **Completo:** Usa separação de zonas com síntese especializada para cada região
- **Ablado:** Síntese em passo único sem distinção de zonas

**Hipótese:** Remover a separação de zonas deve degradar a qualidade, especialmente nas fronteiras entre sal e sedimento, resultando em interfaces borradas ou irrealistas.

**Status:** ✅ Totalmente implementado
- Script: `ablation_compare_no_zones.py`
- Versão ablada: `synthesis_ablation_no_zones.py`

### 2. VAE para Geração de Máscaras ⚠️ *Proposto*

**Questão:** O VAE contribui significativamente para gerar geometrias de domos de sal realistas e diversas?

**Método:**
- **Completo:** Usa VAE para gerar máscaras plausíveis de domos de sal
- **Ablado:** Usa transformações geométricas simples (rotação, escala, deformação) em máscaras existentes

**Hipótese:** Transformações geométricas simples devem produzir máscaras menos diversas e geologicamente menos plausíveis.

**Status:** ⚠️ Não implementado ainda

### 3. Síntese de Fronteira Orientada por Ângulo ⚠️ *Proposto*

**Questão:** O tratamento especial para fronteiras (seleção de patches baseada em ângulo local) é necessário?

**Método:**
- **Completo:** Zona de fronteira usa seleção de patches orientada por ângulo
- **Ablado:** Fronteira sintetizada com o mesmo método genérico das regiões interiores

**Hipótese:** Síntese genérica nas fronteiras deve resultar em transições menos coerentes e perda de detalhes texturais alinhados com a curvatura do domo de sal.

**Status:** ⚠️ Não implementado ainda

---

## Experimento 1: Impacto da Separação de Zonas (Implementado)

### Início Rápido

#### Opção A: Executar Comparação Completa (Recomendado)

Executa ambos os métodos e gera análise comparativa:

```powershell
python ablation_compare_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 10
```

#### Opção B: Executar Apenas Método Ablado

Executa apenas a versão simplificada sem separação de zonas:

```powershell
python synthesis_ablation_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 50
```

Configuração alternativa recomendada para o método ablado quando o objetivo for melhorar o equilíbrio entre diversidade visual e estabilidade das métricas:

```powershell
python synthesis_ablation_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 50 `
  --selection_method weighted `
  --seed_mode center
```

- `--selection_method weighted`: introduz pequena aleatoriedade, mas ainda prioriza candidatos com baixo SSD.
- `--seed_mode center`: inicializa a síntese com um bloco 3x3 central da amostra, reduzindo a dispersão entre iterações.

Nos testes locais, esta combinação apresentou o melhor equilíbrio entre diversidade visual e estabilidade das métricas. A opção `best` torna o resultado mais determinístico e muito próximo da imagem original, enquanto `random` aumenta a diversidade, mas normalmente piora MSE e DSSIM.

#### Opção C: Executar Apenas Método Completo

Executa o método original com separação de zonas:

```powershell
python synthesis.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 50 `
  --patches_cache_path result/patches_db_cache.npz
```

### Parâmetros

| Parâmetro | Descrição | Padrão | Obrigatório |
|-----------|-----------|--------|-------------|
| `--sample_path` | Imagem de amostra de textura de entrada | - | Sim |
| `--sample_semantic_mask_path` | Máscara semântica (zonas: sal/rocha/fronteira) | - | Apenas completo |
| `--generat_mask_path` | Máscara binária de geração (onde sintetizar) | - | Sim |
| `--window_height` | Altura da janela de síntese (pixels) | 50 | Não |
| `--window_width` | Largura da janela de síntese (pixels) | 50 | Não |
| `--kernel_size` | Tamanho do kernel de síntese (deve ser ímpar) | 11 | Não |
| `--iterations` | Número de iterações de síntese | 50 (10 para comparar) | Não |
| `--selection_method` | Estratégia de seleção do pixel candidato (`uniform`, `weighted`, `best`) | `uniform` | Não |
| `--seed_mode` | Estratégia para escolher o seed inicial 3x3 (`random`, `center`) | `random` | Não |
| `--seed` | Seed aleatória para reprodutibilidade | `None` | Não |
| `--error_threshold` | Tolerância relativa ao menor SSD na seleção de candidatos | `0.1` | Não |
| `--visualize` | Mostrar processo de síntese (muito lento) | False | Não |
| `--output_dir` | Diretório de saída customizado | Auto-gerado | Não |

**Notas:**
- Tamanho da janela controla a área sintetizada de uma vez (maior = mais lento mas melhor coerência)
- Tamanho do kernel afeta precisão de correspondência (maior = mais rápido mas menos detalhado)
- Use números ímpares para kernel_size (11, 13, 15, etc.)
- A configuração original do método ablado usa os valores padrão do script, sem informar `--selection_method` e `--seed_mode`.
- `--selection_method weighted --seed_mode center` é uma alternativa recomendada quando você quiser reduzir a dispersão das métricas sem eliminar totalmente a aleatoriedade.
- Use `--selection_method best` se o objetivo for maximizar regularidade e minimizar MSE.
- Use `--seed_mode random` apenas quando quiser mais diversidade visual e aceitar piora nas métricas.

### Estrutura de Saída

Ao executar o script de comparação, os resultados são organizados da seguinte forma:

```
result/
├── ablation_comparison_[sample]_[timestamp]/
    ├── complete_method/
    │   ├── complete_001.jpg
    │   ├── complete_002.jpg
    │   ├── complete_003.jpg
    │   └── ...
    ├── ablated_method/
    │   ├── ablated_001.jpg
    │   ├── ablated_002.jpg
    │   ├── ablated_003.jpg
    │   └── ...
    ├── raw_results.csv              # Todos os resultados das iterações
    └── comparative_statistics.csv   # Estatísticas resumidas
```

### Entendendo as Métricas

A comparação avalia três métricas quantitativas:

#### 1. MSE (Erro Quadrático Médio)
- **O que é:** Diferenças quadráticas em nível de pixel entre original e sintetizado
- **Intervalo:** 0 a ∞ (menor é melhor)
- **Interpretação:** Mede similaridade direta pixel a pixel
- **Esperado:** Método ablado deve mostrar MSE maior (pior qualidade)

#### 2. DSSIM (Dissimilaridade Estrutural)
- **O que é:** 1 - SSIM (Índice de Similaridade Estrutural)
- **Intervalo:** 0 a 1 (menor é melhor)
- **Interpretação:** Mede diferenças estruturais perceptuais
- **Esperado:** Método ablado deve mostrar DSSIM maior (pior similaridade estrutural)

#### 3. Distância LBP (Local Binary Pattern Distance)
- **O que é:** Diferença em padrões de textura usando descritores LBP
- **Intervalo:** 0 a ∞ (menor é melhor)
- **Interpretação:** Mede similaridade de padrões de textura
- **Esperado:** Método ablado deve mostrar distância LBP maior (pior correspondência de textura)

#### 4. Tempo (Duração da Síntese)
- **O que é:** Tempo de execução em segundos
- **Esperado:** Método ablado pode ser mais rápido (algoritmo mais simples) ou similar

### Interpretação dos Resultados

#### Se a Hipótese estiver CORRETA (Separação de zonas é importante):
- ✅ Método ablado mostra métricas **significativamente piores** (MSE, DSSIM, LBP maiores)
- ✅ Inspeção visual revela **fronteiras borradas** entre sal e rocha
- ✅ Coerência geológica está **reduzida**
- ✅ Domos de sal parecem **menos realistas**

#### Se a Hipótese estiver INCORRETA (Separação de zonas não é crítica):
- ❌ Métricas são **similares** ou método ablado é melhor
- ❌ Qualidade visual é **comparável**
- ❌ Separação de zonas pode ser **complexidade desnecessária**

### Fluxo de Trabalho

Siga esta abordagem sistemática para o estudo de ablação:

#### Passo 1: Preparação
```powershell
# Verificar se os arquivos de entrada existem
Test-Path tgs_salt/0bdd44d530.png
Test-Path tgs_salt/0bdd44d530Mask.png

# Assegurar que o banco de dados de patches está em cache (para método completo mais rápido)
# Isso é feito automaticamente durante a primeira execução
```

#### Passo 2: Executar Teste Rápido
```powershell
# Comece com 3 iterações para verificar que tudo funciona
python ablation_compare_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 3
```

#### Passo 3: Executar Experimento Completo
```powershell
# Executar com 10-50 iterações para robustez estatística
python ablation_compare_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 10
```

#### Passo 4: Revisar Resultados Quantitativos

Abrir os arquivos CSV gerados:
```powershell
# Ver estatísticas resumidas
Get-Content result/ablation_comparison_*/comparative_statistics.csv

# Ou abrir no Excel/LibreOffice
start result/ablation_comparison_*/comparative_statistics.csv
```

Procure por:
- Diferenças médias entre métodos completo e ablado
- Desvios padrão (consistência entre iterações)
- Mudanças percentuais nas métricas

#### Passo 5: Inspeção Visual

Comparar imagens sintetizadas lado a lado:

```powershell
# Abrir pastas de imagens
explorer result/ablation_comparison_*/complete_method
explorer result/ablation_comparison_*/ablated_method
```

**O que procurar:**
- Nitidez e coerência das fronteiras
- Continuidade textural dentro das zonas de sal e rocha
- Artefatos ou padrões irrealistas
- Plausibilidade geológica geral

#### Passo 6: Documentar Resultados

Criar um documento de resultados (copiar template `todo/ablation_no_zones_results.md`):

```markdown
# Ablation Study Results: Zone Separation

## Experiment Details
- Date: [date]
- Sample: [sample ID]
- Iterations: [N]
- Parameters: window=[HxW], kernel=[K]

## Quantitative Results

| Metric | Complete (mean±std) | Ablated (mean±std) | Δ% |
|--------|---------------------|--------------------|----|
| MSE    | ...                 | ...                | ... |
| DSSIM  | ...                 | ...                | ... |
| LBP    | ...                 | ...                | ... |
| Time   | ...                 | ...                | ... |

## Visual Analysis
[Include screenshots or image references]

## Conclusion
[Validate or reject hypothesis]
```

#### Passo 7: Análise Estatística (Opcional)

Usar notebooks para análise mais profunda:
```powershell
# Abrir notebook Boxplots
jupyter notebook Boxplots.ipynb
```

Gerar boxplots para visualizar distribuições de métricas entre iterações.

### Exemplo: Fluxo de Trabalho Completo

```powershell
# 1. Executar experimento na amostra 0bdd44d530
python ablation_compare_no_zones.py `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png `
  --iterations 10

# 2. Verificar diretório de resultados (copiar o caminho da saída)
$resultsDir = "result/ablation_comparison_0bdd44d530_20260223_143022"

# 3. Ver estatísticas
Get-Content "$resultsDir/comparative_statistics.csv" | Select-Object -First 30

# 4. Abrir imagens para comparação visual
explorer "$resultsDir/complete_method"
explorer "$resultsDir/ablated_method"

# 5. Copiar melhores exemplos para documentação
Copy-Item "$resultsDir/complete_method/complete_005.jpg" -Destination "documentation/ablation_complete_best.jpg"
Copy-Item "$resultsDir/ablated_method/ablated_005.jpg" -Destination "documentation/ablation_ablated_best.jpg"
```

### Dicas e Melhores Práticas

#### Otimização de Desempenho
- **Comece pequeno:** Use `--iterations 3` para testes iniciais
- **Reduza tamanho da janela:** Janelas menores sintetizam mais rápido (mas podem reduzir qualidade)
- **Aumente tamanho do kernel:** Kernels maiores (15, 17) são mais rápidos mas menos precisos
- **Use cache de patches:** Carregado automaticamente para método completo (acelera significativamente)

#### Resultados Confiáveis
- **Múltiplas iterações:** Use pelo menos 10 iterações para variação estatística
- **Múltiplas amostras:** Teste em diferentes amostras sísmicas para garantir consistência
- **Documente condições:** Registre todos os parâmetros para reprodutibilidade
- **Verificação visual:** Sempre complemente métricas quantitativas com análise visual

#### Problemas Comuns

**Desempenho está muito lento:**
```powershell
# Reduzir carga computacional
python ablation_compare_no_zones.py `
  --window_height 50 `
  --window_width 50 `
  --kernel_size 15 `
  --iterations 5 `
  --sample_path tgs_salt/0bdd44d530.png `
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png `
  --generat_mask_path tgs_salt/0bdd44d530Mask.png
```

**Sem memória:**
- Fechar aplicações desnecessárias
- Reduzir tamanho da janela
- Processar menos iterações de uma vez

**Imagens parecem erradas:**
- Verificar se os caminhos de entrada estão corretos
- Verificar se as máscaras são binárias (0 e 255)
- Assegurar que amostra e máscaras têm as mesmas dimensões

**Erro ao carregar banco de dados de patches:**
- Banco de dados é construído automaticamente na primeira execução
- Verificar `suport/locals.py` para configuração do dataset
- Construir manualmente se necessário (ver documentação synthesis.py)

### Solução de Problemas

#### Erro: "Unable to read image from sample_path"
```powershell
# Verificar se arquivo existe
Test-Path tgs_salt/0bdd44d530.png

# Usar caminho absoluto se necessário
python ablation_compare_no_zones.py `
  --sample_path "D:\0Code\_phdSeismic\textureSSD\tgs_salt\0bdd44d530.png" `
  --sample_semantic_mask_path "D:\0Code\_phdSeismic\textureSSD\tgs_salt\0bdd44d530Mask.png" `
  --generat_mask_path "D:\0Code\_phdSeismic\textureSSD\tgs_salt\0bdd44d530Mask.png" `
  --window_height 101 `
  --window_width 101 `
  --kernel_size 11 `
  --iterations 10
```

#### Erro: "Patches DB loading failed"
- O banco de dados de patches é opcional mas melhora o desempenho
- É construído automaticamente do dataset na primeira execução
- Verificar `suport/locals.py` para caminhos corretos do dataset

#### Erro: "Kernel size must be odd"
```powershell
# Usar apenas números ímpares: 7, 9, 11, 13, 15, etc.
--kernel_size 11  # ✅ Correto
--kernel_size 12  # ❌ Errado
```

#### Síntese demora muito tempo
- **Normal:** 3-15 minutos por iteração dependendo do tamanho da janela
- **Muito lento (>20 min/iteração):**
  - Reduzir `--window_height` e `--window_width`
  - Aumentar `--kernel_size`
  - Não usar flag `--visualize` (extremamente lento)

---

## Experimentos Futuros (Ainda Não Implementados)

### Experimento 2: Contribuição do VAE

**Para implementar:**
1. Criar `synthesis_ablation_no_vae.py`
2. Substituir geração de máscara VAE por transformações geométricas
3. Criar script de comparação similar a `ablation_compare_no_zones.py`

**Transformações sugeridas:**
- Rotação aleatória (-30° a +30°)
- Escala aleatória (0.8x a 1.2x)
- Deformações elásticas
- Recorte e translação aleatórios

### Experimento 3: Síntese de Fronteira

**Para implementar:**
1. Modificar `synthesis.py` para adicionar flag `--disable_angle_boundary`
2. Comparar resultados com fronteira orientada por ângulo vs síntese genérica
3. Focar métricas especificamente em regiões de fronteira

**Abordagem de avaliação:**
- Extrair zona de fronteira das imagens sintetizadas
- Calcular métricas apenas para aquela região
- Comparação visual de nitidez e coerência da fronteira

---

## Organização de Dados

### Estrutura de Arquivos Recomendada

```
textureSSD/
├── tgs_salt/                       # Amostras de entrada
│   ├── 0bdd44d530.png
│   ├── 0bdd44d530Mask.png
│   └── ...
├── result/                         # Todos os resultados
│   ├── patches_db_cache.npz       # Cache do banco de dados de patches
│   ├── ablation_comparison_*/     # Resultados de comparação
│   └── ablation_no_zones_*/       # Resultados de método único
├── documentation/                  # Artefatos de documentação
│   ├── ablation_study_results.md
│   ├── figures/
│   └── ...
└── [scripts e código]
```

### Arquivamento de Resultados

Após completar os experimentos:

```powershell
# Criar arquivo para resultados importantes
$timestamp = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path "result/ablation_comparison_*" `
                 -DestinationPath "result/ablation_archive_$timestamp.zip"

# Mover para documentação
Move-Item "result/ablation_comparison_0bdd44d530_best" `
          "documentation/ablation_results_zone_separation"
```

---

## Integração com o Artigo

### Onde Incluir os Resultados

Estes resultados do estudo de ablação devem ser incluídos em:

1. **Seção 4:** Resultados e Avaliação
   - Subseção 4.X: Estudo de Ablação
   - Apresentar comparações quantitativas (tabelas com métricas)
   - Incluir comparações visuais (figura com 2-3 exemplos)

2. **Tabelas:**
   - Média ± Desvio Padrão para cada métrica
   - Diferenças percentuais
   - Significância estatística (p-values se executar testes t)

3. **Figuras:**
   - Comparação lado a lado: Completo vs Ablado
   - Detalhes ampliados nas fronteiras
   - Boxplots mostrando distribuições de métricas

### Template LaTeX

```latex
\subsection{Estudo de Ablação: Separação de Zonas}

Para validar a importância da separação de zonas orientada a contexto,
implementamos uma versão ablada do nosso método que realiza
síntese sem distinguir entre zonas de sal, rocha e fronteira.

\begin{table}[h]
\centering
\caption{Comparação de Métodos Completo vs Ablado (N=10 iterações)}
\begin{tabular}{lcccc}
\toprule
\textbf{Método} & \textbf{MSE} & \textbf{DSSIM} & \textbf{LBP} & \textbf{Tempo (s)} \\
\midrule
Completo        & $X \pm Y$    & $X \pm Y$      & $X \pm Y$    & $X \pm Y$ \\
Ablado          & $X \pm Y$    & $X \pm Y$      & $X \pm Y$    & $X \pm Y$ \\
\midrule
$\Delta\%$      & $+Z\%$       & $+Z\%$         & $+Z\%$       & $\pm Z\%$ \\
\bottomrule
\end{tabular}
\label{tab:ablation_zones}
\end{table}

Resultados mostram que remover a separação de zonas leva a
degradação significativa em todas as métricas de qualidade...
```

---

## Referências

- **Efros & Leung (1999):** Texture Synthesis by Non-parametric Sampling
- **Método original:** [synthesis.py](synthesis.py)
- **Método ablado:** [synthesis_ablation_no_zones.py](synthesis_ablation_no_zones.py)
- **Ferramenta de comparação:** [ablation_compare_no_zones.py](ablation_compare_no_zones.py)
- **Explicação (PT-BR):** [todo/ablation_study_explicacao.md](todo/ablation_study_explicacao.md)

---

## Suporte

Para problemas ou questões:
1. Verificar resultados existentes nos diretórios `result/ablation_*`
2. Revisar mensagens de erro cuidadosamente
3. Verificar se todos os arquivos de entrada são imagens válidas
4. Assegurar que o ambiente Python tem todas as dependências (ver `environmentt.yml`)

**Dependências comuns:**
```powershell
pip install opencv-python numpy pandas scipy scikit-image
```

---

*Última atualização: 23 de Fevereiro de 2026*
