import os
import sys
import pandas as pd

# Permite passar o caminho do arquivo como argumento, senão usa o hardcode abaixo
if len(sys.argv) > 1:
    meuarquivo = sys.argv[1]
else:
    meuarquivo = r'C:\Dados\0Code\textureSSD\result\run_20250913_155213_a3ad2df0\run_metrics_8ac7ff10.csv'

if not os.path.exists(meuarquivo):
    raise FileNotFoundError(f"O arquivo {meuarquivo} não foi encontrado.")
print(f"metrics CSV encontrado: {meuarquivo}")

# Lê ignorando linhas de comentário (começam com #) e usando separador ponto-e-vírgula
df = pd.read_csv(meuarquivo, sep=';', comment='#')

expected_cols = ['iteration', 'output_file', 'time_sec', 'mse', 'dssim', 'lbp_distance']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print('Colunas presentes no arquivo:', list(df.columns))
    raise ValueError(f'Colunas esperadas ausentes: {missing}')

# Garante tipos numéricos onde necessário
numeric_cols = ['time_sec', 'mse', 'dssim', 'lbp_distance']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Calcula estatísticas
stats_df = pd.DataFrame({
    'min': df[numeric_cols].min(),
    'Q1': df[numeric_cols].quantile(0.25),
    'median': df[numeric_cols].median(),
    'Q3': df[numeric_cols].quantile(0.75),
    'max': df[numeric_cols].max(),
    'mean': df[numeric_cols].mean(),
    'std': df[numeric_cols].std(),
    'count': df[numeric_cols].count()
})

print('\nEstatísticas descritivas por métrica:')
print(stats_df)

# Salva ao lado do CSV original
out_dir = os.path.dirname(meuarquivo)
out_path = os.path.join(out_dir, 'summary_stats.csv')
stats_df.to_csv(out_path, sep=';', float_format='%.6f')
print(f'Arquivo de estatísticas salvo em: {out_path}')

