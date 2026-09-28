# Dashboard — Política Monetária, Câmbio e Inflação no Brasil

**Checkpoint 3 — Desenvolvimento da Dashboard**
Disciplina: Data Science & Analytics (DSA) | Prazo de entrega: 18/10/2026

Aplicação analítica interativa em **Streamlit** que reúne os resultados das Partes 1 e 2 do
projeto: os indicadores (KPIs) construídos no Checkpoint 1 e a modelagem preditiva do
Checkpoint 2, permitindo explorar os dados de forma organizada e intuitiva.

**URL da aplicação hospedada:** _(preencher após o deploy — ver seção "Deploy" abaixo)_
**URL do repositório:** https://github.com/LeonardoAugustoBacelar/Checkpoint03-Data-Science-Analytics

---

## O que a dashboard contém

| Requisito do enunciado | Onde está na aplicação |
|---|---|
| Fonte dos dados | Aba **"Fonte dos dados"** (tabela com séries, códigos SGS e URLs) + indicador de origem na barra lateral |
| KPIs / indicadores calculados | Aba **"Visão geral"** (6 métricas + medidor de Selic) |
| Gráficos estáticos | Aba **"Exploração dos dados"** (painel matplotlib com as 3 séries) |
| Gráficos interativos | Abas **"Visão geral"**, **"Exploração"** e **"Análise preditiva"** (Plotly: linhas, medidor, matriz de correlação, dispersão com reta de regressão, barras, real × previsto) |
| Filtros para exploração | Barra lateral: intervalo de período e seleção de séries; aba preditiva: formulação do alvo |

### Indicadores (KPIs) — do Checkpoint 1

- Selic atual e Selic média do período (com variação em pontos percentuais)
- Câmbio no fim do período e variação percentual no intervalo
- IPCA acumulado no período
- Correlação de Pearson Selic × Câmbio e Selic × IPCA

Todos os KPIs são **recalculados dinamicamente** conforme o período selecionado no filtro.

### Análise preditiva — do Checkpoint 2

Problema de **regressão**: prever o câmbio USD/BRL do mês seguinte a partir de Selic, IPCA e
câmbio do mês corrente mais defasagens (lags) de 1 a 3 meses. O split treino/teste é
**cronológico** (80% primeiros meses), nunca aleatório, por se tratar de série temporal.

Quatro modelos são treinados e comparados por **RMSE, MAE e MAPE**:

| Modelo | Biblioteca |
|---|---|
| Regressão Linear | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |
| Rede Neural (MLP) | scikit-learn |

A aba inclui um controle para alternar a **formulação do alvo** entre *nível* (prever o câmbio em
R$) e *variação* (prever a variação percentual). Essa alternância evidencia um ponto metodológico
relevante: modelos de árvore e redes neurais **não extrapolam** além da faixa de valores vista no
treino, então têm desempenho ruim ao prever o nível de uma série com tendência de alta — problema
que desaparece ao prever a variação, que é aproximadamente estacionária.

---

## Fonte dos dados

API pública do **Banco Central do Brasil** — Sistema Gerenciador de Séries Temporais (SGS).
Acesso livre, sem cadastro ou chave de API.

| Série | Código SGS | URL |
|---|---|---|
| Meta Selic definida pelo Copom (% a.a.) | 432 | https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados |
| Taxa de câmbio USD/BRL — venda (PTAX) | 1 | https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados |
| IPCA — variação percentual mensal (%) | 433 | https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados |

Documentação: https://dadosabertos.bcb.gov.br/dataset/?res_format=API

**Integração:** Selic e câmbio são publicados em frequência diária; o IPCA é mensal. As séries
diárias são agregadas para média mensal e unidas ao IPCA pelo par ano-mês.

**Janela consultada:** sempre os últimos 9 anos e 11 meses, porque a API do BCB limita cada
consulta a um intervalo máximo de 10 anos.

### Fallback offline

Se a API do BCB estiver indisponível, a aplicação carrega automaticamente
`data/dataset_fallback.csv` — dados **reais** coletados em 28/09/2026, e exibe um aviso indicando
que está usando a cópia local. Assim a dashboard continua funcional durante a avaliação mesmo se o
servidor do BCB estiver fora do ar.

O CSV de fallback foi gerado a partir das séries mensais equivalentes do próprio BCB
(4189 — Selic anualizada, 3698 — câmbio médio mensal, 433 — IPCA), evitando baixar ~7.000 pontos
diários. O script que o gera, com a documentação da proveniência, está em
`scripts/gerar_fallback.py`.

---

## Estrutura do projeto

```
.
├── app.py                      # Aplicação Streamlit (dashboard)
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
├── .gitignore
├── .streamlit/
│   └── config.toml             # Configuração do servidor
├── data/
│   └── dataset_fallback.csv    # Dados reais (fallback offline)
└── scripts/
    └── gerar_fallback.py       # Script que gera o CSV de fallback
```

---

## Execução local

Requer **Python 3.9 ou superior**.

```bash
# 1. Clonar o repositório
git clone <URL-DO-SEU-REPOSITORIO>
cd <NOME-DA-PASTA>

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Rodar a aplicação
streamlit run app.py
```

A aplicação abre automaticamente no navegador em `http://localhost:8501`.
Se não abrir sozinha, acesse esse endereço manualmente.

**Sem internet?** A aplicação funciona mesmo assim: detecta que a API do BCB não respondeu e usa o
CSV de fallback incluído no repositório, exibindo um aviso.

---

## Deploy no Streamlit Community Cloud

1. Suba este projeto para um repositório **público** no GitHub.
2. Acesse https://share.streamlit.io e entre com a conta do GitHub.
3. Clique em **"New app"** e selecione o repositório, o branch (`main`) e o arquivo `app.py`.
4. Clique em **"Deploy"**. O primeiro build leva alguns minutos (instala as dependências).
5. Copie a URL pública gerada e preencha no topo deste README.

A aplicação fica acessível publicamente pela internet, sem que o avaliador precise instalar nada.

---

## Notas técnicas

**Rede neural:** no notebook do Checkpoint 2 a rede neural é implementada em **Keras/TensorFlow**.
Nesta dashboard ela foi substituída pelo `MLPRegressor` do scikit-learn — mesma família de modelo
(perceptron multicamadas), porém muito mais leve. O TensorFlow sozinho ultrapassaria o limite de
memória do plano gratuito do Streamlit Community Cloud. Os demais modelos são idênticos aos do
Checkpoint 2.

**Cache:** o download dos dados e o treinamento dos modelos usam `@st.cache_data` com validade de
6 horas, para que a navegação entre abas e filtros seja instantânea sem refazer requisições à API
nem retreinar modelos desnecessariamente.

**Versões testadas em 28/09/2026:** streamlit 1.64.0, pandas 3.0.2, numpy 2.4.4, plotly 7.1.0,
scikit-learn 1.8.0, xgboost 3.2.0, matplotlib 3.10.9, requests 2.33.1, statsmodels 0.15.0.
O `requirements.txt` declara versões mínimas compatíveis em vez de fixas, para não conflitar com a
versão de Python do serviço de hospedagem.

---

## Contexto do projeto

Esta dashboard é a **Parte 3** de um trabalho em três etapas:

1. **Checkpoint 1** — coleta, integração, análise exploratória e construção dos indicadores;
2. **Checkpoint 2** — modelagem preditiva comparando scikit-learn, XGBoost e rede neural;
3. **Checkpoint 3** (esta aplicação) — dashboard interativa reunindo os resultados das duas etapas.
