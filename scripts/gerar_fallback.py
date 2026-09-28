"""
Gera o CSV de fallback (data/dataset_fallback.csv) usado pela dashboard quando a
API do Banco Central esta indisponivel.

Os valores abaixo foram coletados da API publica do BCB (SGS) em 28/09/2026,
a partir das seguintes series MENSAIS:

  - 4189 : Taxa de juros Selic acumulada no mes anualizada base 252 (% a.a.)
           https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados
  - 3698 : Taxa de cambio livre - Dolar americano (venda) - media do periodo (R$/US$)
           https://api.bcb.gov.br/dados/serie/bcdata.sgs.3698/dados
  - 433  : IPCA - variacao percentual mensal (%)
           https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados

Observacao de transparencia: os Checkpoints 1 e 2 usavam as series DIARIAS
432 (meta Selic) e 1 (PTAX venda), agregadas para media mensal. Aqui usamos as
series mensais equivalentes publicadas pelo proprio BCB, que sao numericamente
muito proximas e evitam baixar ~7.000 pontos diarios. A dashboard, quando tem
internet, usa as series originais 432/1/433 exatamente como no Checkpoint 1.
"""
import pandas as pd

SELIC = [
    13.17, 12.82, 12.15, 11.59, 11.15, 10.15, 10.01, 9.15, 8.35, 8.01, 7.40, 7.00,
    6.90, 6.72, 6.58, 6.40, 6.40, 6.40, 6.40, 6.40, 6.40, 6.40, 6.40, 6.40,
    6.40, 6.40, 6.40, 6.40, 6.40, 6.40, 6.40, 5.90, 5.71, 5.38, 4.90, 4.59,
    4.40, 4.19, 3.95, 3.65, 3.01, 2.58, 2.15, 1.94, 1.90, 1.90, 1.90, 1.90,
    1.90, 1.90, 2.23, 2.65, 3.29, 3.76, 4.15, 5.01, 5.43, 6.30, 7.65, 8.76,
    9.15, 10.49, 11.15, 11.65, 12.51, 12.89, 13.15, 13.58, 13.65, 13.65, 13.65, 13.65,
    13.65, 13.65, 13.65, 13.65, 13.65, 13.65, 13.65, 13.19, 12.97, 12.65, 12.17, 11.87,
    11.65, 11.15, 11.00, 10.65, 10.46, 10.40, 10.40, 10.40, 10.50, 10.65, 11.04, 11.77,
    12.24, 13.15, 13.57, 14.15, 14.55, 14.74, 14.90, 14.90, 14.90, 14.90, 14.90, 14.90,
    14.90, 14.90, 14.80, 14.64, 14.40, 14.29, 14.15, 13.94,
]

CAMBIO = [
    3.1966, 3.1042, 3.1279, 3.1362, 3.2095, 3.2954, 3.2061, 3.1509, 3.1348, 3.1912, 3.2594, 3.2919,
    3.2106, 3.2415, 3.2792, 3.4075, 3.6361, 3.7732, 3.8288, 3.9298, 4.1165, 3.7584, 3.7867, 3.8851,
    3.7417, 3.7236, 3.8465, 3.8962, 4.0015, 3.8588, 3.7793, 4.0200, 4.1215, 4.0870, 4.1553, 4.1096,
    4.1495, 4.3410, 4.8839, 5.3256, 5.6434, 5.1966, 5.2802, 5.4612, 5.3995, 5.6258, 5.4178, 5.1456,
    5.3562, 5.4165, 5.6461, 5.5621, 5.2911, 5.0319, 5.1567, 5.2517, 5.2797, 5.5400, 5.5569, 5.6514,
    5.5341, 5.1966, 4.9684, 4.7580, 4.9551, 5.0492, 5.3681, 5.1433, 5.2370, 5.2503, 5.2747, 5.2424,
    5.2007, 5.1717, 5.2115, 5.0197, 4.9828, 4.8516, 4.8008, 4.9035, 4.9370, 5.0648, 4.8983, 4.8972,
    4.9144, 4.9644, 4.9801, 5.1291, 5.1330, 5.3890, 5.5420, 5.5526, 5.5416, 5.6241, 5.8071, 6.0970,
    6.0218, 5.7656, 5.7468, 5.7837, 5.6674, 5.5471, 5.5285, 5.4469, 5.3674, 5.3855, 5.3409, 5.4531,
    5.3380, 5.2006, 5.2316, 5.0331, 4.9837, 5.1276, 5.1139, 5.1532,
]

IPCA = [
    0.38, 0.33, 0.25, 0.14, 0.31, -0.23, 0.24, 0.19, 0.16, 0.42, 0.28, 0.44,
    0.29, 0.32, 0.09, 0.22, 0.40, 1.26, 0.33, -0.09, 0.48, 0.45, -0.21, 0.15,
    0.32, 0.43, 0.75, 0.57, 0.13, 0.01, 0.19, 0.11, -0.04, 0.10, 0.51, 1.15,
    0.21, 0.25, 0.07, -0.31, -0.38, 0.26, 0.36, 0.24, 0.64, 0.86, 0.89, 1.35,
    0.25, 0.86, 0.93, 0.31, 0.83, 0.53, 0.96, 0.87, 1.16, 1.25, 0.95, 0.73,
    0.54, 1.01, 1.62, 1.06, 0.47, 0.67, -0.68, -0.36, -0.29, 0.59, 0.41, 0.62,
    0.53, 0.84, 0.71, 0.61, 0.23, -0.08, 0.12, 0.23, 0.26, 0.24, 0.28, 0.56,
    0.42, 0.83, 0.16, 0.38, 0.46, 0.21, 0.38, -0.02, 0.44, 0.56, 0.39, 0.52,
    0.16, 1.31, 0.56, 0.43, 0.26, 0.24, 0.26, -0.11, 0.48, 0.09, 0.18, 0.33,
    0.33, 0.70, 0.88, 0.67, 0.58, 0.16, 0.07, -0.32,
]

assert len(SELIC) == len(CAMBIO) == len(IPCA) == 116, (len(SELIC), len(CAMBIO), len(IPCA))

datas = pd.date_range('2017-01-01', periods=116, freq='MS')

df = pd.DataFrame({
    'data': datas,
    'selic': SELIC,
    'cambio': CAMBIO,
    'ipca_mensal': IPCA,
})
df['ano_mes'] = df['data'].dt.strftime('%Y-%m')
df = df[['ano_mes', 'selic', 'cambio', 'ipca_mensal', 'data']]
df.to_csv('data/dataset_fallback.csv', index=False)

print('CSV de fallback gerado:', df.shape)
print(df.head(3).to_string(index=False))
print('...')
print(df.tail(3).to_string(index=False))
