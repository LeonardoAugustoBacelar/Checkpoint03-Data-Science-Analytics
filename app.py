"""
Checkpoint 3 - Dashboard Analitica
Politica Monetaria, Cambio e Inflacao no Brasil

Reune os resultados do Checkpoint 1 (indicadores/KPIs) e do Checkpoint 2
(modelagem preditiva) em uma aplicacao interativa em Streamlit.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
# Configuracao da pagina
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Selic, Cambio e Inflacao | Dashboard DSA",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

CAMINHO_FALLBACK = os.path.join("data", "dataset_fallback.csv")

FONTES = [
    {
        "Serie": "Meta Selic definida pelo Copom (% a.a.)",
        "Codigo SGS": "432",
        "URL": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados",
    },
    {
        "Serie": "Taxa de cambio USD/BRL - venda (PTAX)",
        "Codigo SGS": "1",
        "URL": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados",
    },
    {
        "Serie": "IPCA - variacao percentual mensal (%)",
        "Codigo SGS": "433",
        "URL": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados",
    },
]


# ----------------------------------------------------------------------------
# Coleta e preparacao dos dados
# ----------------------------------------------------------------------------
def obter_serie_bcb(codigo, data_inicial, data_final, nome_coluna, tentativas=3, timeout=30):
    """Baixa uma serie temporal do SGS/BCB. Levanta RuntimeError se falhar."""
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
        f"?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resposta.raise_for_status()
            serie = pd.DataFrame(resposta.json())
            serie["data"] = pd.to_datetime(serie["data"], dayfirst=True)
            serie[nome_coluna] = serie["valor"].astype(float)
            return serie[["data", nome_coluna]]
        except requests.exceptions.RequestException as exc:
            ultimo_erro = exc
            if tentativa < tentativas:
                time.sleep(2 * tentativa)
    raise RuntimeError(f"Falha ao baixar serie {codigo}: {ultimo_erro}")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def carregar_dados():
    """
    Carrega o dataset integrado mensal.

    Tenta primeiro a API do Banco Central (mesmas series do Checkpoint 1).
    Se a API estiver indisponivel, usa o CSV de fallback versionado no repositorio.

    Retorna: (DataFrame, origem, detalhe)
    """
    hoje = datetime.today()
    data_final = hoje.strftime("%d/%m/%Y")
    data_inicial = (hoje - relativedelta(years=9, months=11)).strftime("%d/%m/%Y")

    try:
        df_selic = obter_serie_bcb(432, data_inicial, data_final, "selic")
        df_cambio = obter_serie_bcb(1, data_inicial, data_final, "cambio")
        df_ipca = obter_serie_bcb(433, data_inicial, data_final, "ipca_mensal")

        for frame in (df_selic, df_cambio, df_ipca):
            frame["ano_mes"] = frame["data"].dt.to_period("M")

        selic_m = df_selic.groupby("ano_mes")["selic"].mean().round(2)
        cambio_m = df_cambio.groupby("ano_mes")["cambio"].mean().round(4)
        ipca_m = df_ipca.groupby("ano_mes")["ipca_mensal"].mean().round(2)

        df = pd.concat([selic_m, cambio_m, ipca_m], axis=1).reset_index()
        df["ano_mes"] = df["ano_mes"].astype(str)
        df["data"] = pd.to_datetime(df["ano_mes"])
        df = df.dropna().sort_values("data").reset_index(drop=True)

        return df, "api", f"API do Banco Central, consultada em {hoje.strftime('%d/%m/%Y as %H:%M')}"

    except Exception as exc:  # noqa: BLE001 - qualquer falha cai para o fallback
        if not os.path.exists(CAMINHO_FALLBACK):
            raise
        df = pd.read_csv(CAMINHO_FALLBACK, parse_dates=["data"])
        df = df.sort_values("data").reset_index(drop=True)
        return df, "fallback", f"CSV local (API indisponivel: {type(exc).__name__})"


# ----------------------------------------------------------------------------
# Indicadores (Checkpoint 1)
# ----------------------------------------------------------------------------
def calcular_kpis(df):
    """Calcula os KPIs definidos no Checkpoint 1 para o recorte recebido."""
    return {
        "selic_media": round(float(df["selic"].mean()), 2),
        "selic_atual": float(df["selic"].iloc[-1]),
        "cambio_medio": round(float(df["cambio"].mean()), 2),
        "cambio_inicial": float(df["cambio"].iloc[0]),
        "cambio_final": float(df["cambio"].iloc[-1]),
        "cambio_var_pct": round(
            float((df["cambio"].iloc[-1] - df["cambio"].iloc[0]) / df["cambio"].iloc[0] * 100), 2
        ),
        "corr_selic_cambio": round(float(df["selic"].corr(df["cambio"])), 3),
        "corr_selic_ipca": round(float(df["selic"].corr(df["ipca_mensal"])), 3),
        "ipca_acumulado": round(float(((1 + df["ipca_mensal"] / 100).prod() - 1) * 100), 2),
    }


# ----------------------------------------------------------------------------
# Modelagem preditiva (Checkpoint 2)
# ----------------------------------------------------------------------------
FEATURES = [
    "selic", "ipca_mensal", "cambio",
    "selic_lag1", "selic_lag2", "selic_lag3",
    "ipca_mensal_lag1", "ipca_mensal_lag2", "ipca_mensal_lag3",
    "cambio_lag1", "cambio_lag2",
]
TARGET = "cambio_prox_mes"


def preparar_dados_ml(df):
    """Cria lags e a variavel alvo (cambio do mes seguinte), como no Checkpoint 2."""
    base = df.copy()
    for col in ["selic", "ipca_mensal", "cambio"]:
        for lag in [1, 2, 3]:
            if col == "cambio" and lag == 3:
                continue
            base[f"{col}_lag{lag}"] = base[col].shift(lag)
    base[TARGET] = base["cambio"].shift(-1)
    return base.dropna().reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def treinar_modelos(df, modo_alvo="nivel"):
    """
    Treina os modelos do Checkpoint 2 sobre o recorte recebido.

    Split cronologico 80/20 (nunca aleatorio: e serie temporal).

    modo_alvo:
      "nivel"    -> os modelos preveem diretamente o cambio em R$ do mes seguinte.
      "variacao" -> os modelos preveem a variacao percentual do cambio, e a previsao
                    e reconvertida para R$ depois. Remove a tendencia da serie e
                    permite que modelos que nao extrapolam (arvores e redes neurais)
                    funcionem fora da faixa de valores vista no treino.

    Em ambos os casos as metricas sao calculadas sobre o cambio em R$, para que os
    dois modos sejam comparaveis entre si.

    Retorna (tabela de metricas, DataFrame de previsoes, n_treino, n_teste).
    """
    dados = preparar_dados_ml(df)
    if len(dados) < 30:
        return None, None, 0, 0

    X = dados[FEATURES]
    y_nivel = dados[TARGET]
    # alvo em variacao percentual em relacao ao cambio do mes corrente
    y_variacao = (dados[TARGET] / dados["cambio"] - 1) * 100
    y = y_variacao if modo_alvo == "variacao" else y_nivel

    corte = int(len(dados) * 0.8)

    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train = y.iloc[:corte]
    y_test = y_nivel.iloc[corte:]          # avaliacao sempre no nivel (R$)
    cambio_base_test = dados["cambio"].iloc[corte:].values
    datas_test = dados["data"].iloc[corte:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelos = {}

    lin = LinearRegression().fit(X_train_s, y_train)
    modelos["Regressao Linear (sklearn)"] = lin.predict(X_test_s)

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42).fit(X_train, y_train)
    modelos["Random Forest (sklearn)"] = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0
    ).fit(X_train, y_train)
    modelos["XGBoost"] = xgb.predict(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        max_iter=2000, random_state=42,
    ).fit(X_train_s, y_train)
    modelos["Rede Neural (MLP)"] = mlp.predict(X_test_s)

    if modo_alvo == "variacao":
        modelos = {
            nome: cambio_base_test * (1 + pred / 100)
            for nome, pred in modelos.items()
        }

    linhas = []
    for nome, pred in modelos.items():
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
        linhas.append({
            "Modelo": nome,
            "RMSE (R$)": round(rmse, 4),
            "MAE (R$)": round(mae, 4),
            "MAPE (%)": round(mape, 2),
        })

    metricas = pd.DataFrame(linhas).sort_values("RMSE (R$)").reset_index(drop=True)

    previsoes = pd.DataFrame({"data": datas_test.values, "Real": y_test.values})
    for nome, pred in modelos.items():
        previsoes[nome] = pred

    return metricas, previsoes, len(X_train), len(X_test)


# ----------------------------------------------------------------------------
# Carregamento + barra lateral (filtros)
# ----------------------------------------------------------------------------
with st.spinner("Carregando dados do Banco Central..."):
    df_completo, origem, detalhe_origem = carregar_dados()

st.sidebar.header("Filtros")

data_min = df_completo["data"].min().date()
data_max = df_completo["data"].max().date()

periodo = st.sidebar.slider(
    "Periodo de analise",
    min_value=data_min,
    max_value=data_max,
    value=(data_min, data_max),
    format="MM/YYYY",
)

df = df_completo[
    (df_completo["data"].dt.date >= periodo[0]) & (df_completo["data"].dt.date <= periodo[1])
].reset_index(drop=True)

series_disponiveis = {
    "Selic (% a.a.)": "selic",
    "Cambio USD/BRL (R$)": "cambio",
    "IPCA mensal (%)": "ipca_mensal",
}
series_escolhidas = st.sidebar.multiselect(
    "Series nos graficos",
    options=list(series_disponiveis.keys()),
    default=list(series_disponiveis.keys()),
)

st.sidebar.divider()
if origem == "api":
    st.sidebar.success("Dados ao vivo da API do BCB")
else:
    st.sidebar.warning("Usando CSV local (API fora do ar)")
st.sidebar.caption(detalhe_origem)
st.sidebar.caption(f"{len(df)} meses no recorte atual (de {len(df_completo)} disponiveis)")

# ----------------------------------------------------------------------------
# Cabecalho
# ----------------------------------------------------------------------------
st.title("Politica Monetaria, Cambio e Inflacao no Brasil")
st.caption(
    "Dashboard analitica - Checkpoint 3 | Data Science & Analytics | "
    "Reune os indicadores do Checkpoint 1 e a modelagem preditiva do Checkpoint 2"
)

if origem == "fallback":
    st.warning(
        "A API do Banco Central nao respondeu agora, entao a aplicacao esta usando o "
        "CSV de fallback versionado no repositorio (dados reais coletados em 28/09/2026). "
        "Todos os graficos e modelos continuam funcionando normalmente."
    )

if len(df) < 12:
    st.error("Selecione um periodo com pelo menos 12 meses para que os indicadores facam sentido.")
    st.stop()

kpis = calcular_kpis(df)

aba_visao, aba_exploracao, aba_predicao, aba_fontes = st.tabs(
    ["Visao geral", "Exploracao dos dados", "Analise preditiva", "Fonte dos dados"]
)

# ----------------------------------------------------------------------------
# Aba 1 - Visao geral (KPIs)
# ----------------------------------------------------------------------------
with aba_visao:
    st.subheader("Indicadores do periodo selecionado")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Selic atual",
        f"{kpis['selic_atual']:.2f}%",
        delta=f"{kpis['selic_atual'] - kpis['selic_media']:+.2f} p.p. vs media",
    )
    c2.metric(
        "Cambio no fim do periodo",
        f"R$ {kpis['cambio_final']:.2f}",
        delta=f"{kpis['cambio_var_pct']:+.2f}% no periodo",
    )
    c3.metric("IPCA acumulado", f"{kpis['ipca_acumulado']:.2f}%")
    c4.metric("Selic media do periodo", f"{kpis['selic_media']:.2f}%")

    c5, c6 = st.columns(2)
    c5.metric(
        "Correlacao Selic x Cambio",
        f"{kpis['corr_selic_cambio']:.3f}",
        help="Pearson, de -1 a 1. Testa a hipotese de que juros mais altos valorizam o real.",
    )
    c6.metric(
        "Correlacao Selic x IPCA",
        f"{kpis['corr_selic_ipca']:.3f}",
        help="Pearson, de -1 a 1. Mede a associacao entre politica monetaria e inflacao.",
    )

    st.divider()
    st.subheader("Selic atual x media historica do recorte")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=kpis["selic_atual"],
            delta={"reference": kpis["selic_media"]},
            title={"text": "Selic (% a.a.)"},
            gauge={
                "axis": {"range": [0, 16], "dtick": 2},
                "steps": [
                    {"range": [0, 5], "color": "#d4efdf"},
                    {"range": [5, 10], "color": "#fdebd0"},
                    {"range": [10, 16], "color": "#fadbd8"},
                ],
                "threshold": {
                    "value": kpis["selic_media"],
                    "line": {"color": "red", "width": 3},
                },
            },
        )
    )
    gauge.update_layout(height=320, margin=dict(t=60, b=10))
    st.plotly_chart(gauge, width='stretch')

    st.subheader("Evolucao das series (interativo)")
    cols_plot = [series_disponiveis[s] for s in series_escolhidas] or ["selic"]
    fig_linhas = px.line(
        df, x="data", y=cols_plot,
        labels={"value": "Valor", "data": "Data", "variable": "Serie"},
        title="Series ao longo do tempo",
    )
    fig_linhas.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_linhas, width='stretch')

# ----------------------------------------------------------------------------
# Aba 2 - Exploracao
# ----------------------------------------------------------------------------
with aba_exploracao:
    st.subheader("Graficos estaticos (matplotlib)")
    st.caption("Visao separada de cada serie, na escala propria de cada uma.")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(df["data"], df["selic"], color="#c0392b")
    axes[0].set_title("Taxa Selic (% a.a.) - media mensal")
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["data"], df["cambio"], color="#2980b9")
    axes[1].set_title("Cambio USD/BRL - media mensal")
    axes[1].grid(alpha=0.3)
    axes[2].plot(df["data"], df["ipca_mensal"], color="#27ae60")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_title("IPCA - variacao mensal (%)")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matriz de correlacao")
        corr = df[["selic", "cambio", "ipca_mensal"]].corr().round(3)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
            labels=dict(color="Correlacao"),
        )
        fig_corr.update_layout(height=380)
        st.plotly_chart(fig_corr, width='stretch')

    with col_b:
        st.subheader("Dispersao Selic x Cambio")
        fig_disp = px.scatter(
            df, x="selic", y="cambio", trendline="ols",
            labels={"selic": "Selic (% a.a.)", "cambio": "Cambio USD/BRL (R$)"},
            hover_data={"ano_mes": True},
        )
        fig_disp.update_layout(height=380)
        st.plotly_chart(fig_disp, width='stretch')

    st.divider()
    st.subheader("Estatisticas descritivas")
    st.dataframe(
        df[["selic", "cambio", "ipca_mensal"]].describe().round(2),
        width='stretch',
    )

    st.subheader("Dados do recorte")
    st.dataframe(
        df[["ano_mes", "selic", "cambio", "ipca_mensal"]],
        width='stretch', height=280,
    )
    st.download_button(
        "Baixar dados filtrados (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_filtrado.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# Aba 3 - Analise preditiva
# ----------------------------------------------------------------------------
with aba_predicao:
    st.subheader("Previsao do cambio do mes seguinte")
    st.markdown(
        "**Problema:** regressao. **Alvo:** cambio USD/BRL do mes seguinte. "
        "**Features:** Selic, IPCA e cambio do mes corrente mais defasagens (lags) de 1 a 3 meses. "
        "O split treino/teste e **cronologico** (80% primeiros meses para treino), "
        "nunca aleatorio, porque se trata de serie temporal."
    )

    modo_rotulo = st.radio(
        "Formulacao do alvo",
        ["Nivel - prever o cambio em R$", "Variacao - prever a variacao % do cambio"],
        horizontal=True,
        help=(
            "O cambio tem tendencia de alta. Modelos baseados em arvores e redes neurais nao "
            "conseguem extrapolar alem da faixa de valores vista no treino; prever a variacao "
            "percentual remove essa tendencia e costuma melhorar muito o resultado deles."
        ),
    )
    modo_alvo = "variacao" if modo_rotulo.startswith("Variacao") else "nivel"

    with st.spinner("Treinando modelos..."):
        metricas, previsoes, n_treino, n_teste = treinar_modelos(df, modo_alvo)

    if metricas is None:
        st.error(
            "O recorte selecionado tem poucos meses para treinar os modelos. "
            "Amplie o periodo na barra lateral (minimo recomendado: 4 anos)."
        )
    else:
        st.caption(f"Treino: {n_treino} meses | Teste: {n_teste} meses")

        melhor = metricas.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Melhor modelo (menor RMSE)", melhor["Modelo"])
        m2.metric("RMSE", f"R$ {melhor['RMSE (R$)']:.4f}")
        m3.metric("Erro percentual medio (MAPE)", f"{melhor['MAPE (%)']:.2f}%")

        st.subheader("Comparacao entre os modelos")
        st.dataframe(metricas, width='stretch')

        fig_erro = go.Figure()
        fig_erro.add_trace(go.Bar(x=metricas["Modelo"], y=metricas["RMSE (R$)"], name="RMSE (R$)"))
        fig_erro.add_trace(go.Bar(x=metricas["Modelo"], y=metricas["MAE (R$)"], name="MAE (R$)"))
        fig_erro.update_layout(
            barmode="group", height=400,
            yaxis_title="Erro (R$)",
            title="Erro por modelo (quanto menor, melhor)",
        )
        st.plotly_chart(fig_erro, width='stretch')

        st.subheader("Cambio real x previsto (periodo de teste)")
        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=previsoes["data"], y=previsoes["Real"], name="Real",
                mode="lines+markers", line=dict(color="black", width=3),
            )
        )
        for nome in metricas["Modelo"]:
            fig_pred.add_trace(
                go.Scatter(x=previsoes["data"], y=previsoes[nome], name=nome, mode="lines")
            )
        fig_pred.update_layout(
            height=450, hovermode="x unified",
            xaxis_title="Data", yaxis_title="Cambio USD/BRL (R$)",
        )
        st.plotly_chart(fig_pred, width='stretch')

        with st.expander("Por que a formulacao do alvo muda tanto o resultado?"):
            dados_ml = preparar_dados_ml(df)
            corte_ml = int(len(dados_ml) * 0.8)
            faixa_tr = dados_ml[TARGET].iloc[:corte_ml]
            faixa_te = dados_ml[TARGET].iloc[corte_ml:]
            extrapola = float(faixa_te.max()) > float(faixa_tr.max())

            st.markdown(
                f"""
No recorte atual, o cambio no periodo de **treino** varia entre
**R$ {faixa_tr.min():.2f}** e **R$ {faixa_tr.max():.2f}**, enquanto no periodo de
**teste** varia entre **R$ {faixa_te.min():.2f}** e **R$ {faixa_te.max():.2f}**.

{"Ou seja, o teste contem valores **acima de tudo o que o modelo viu no treino**." if extrapola
 else "Neste recorte o teste fica dentro da faixa vista no treino."}

Isso importa porque **arvores de decisao (Random Forest, XGBoost) e redes neurais nao extrapolam**:
suas previsoes ficam limitadas a faixa de valores do treino. A regressao linear, por construcao,
consegue projetar para fora dessa faixa - e por isso costuma vencer quando o alvo e o **nivel** de
uma serie com tendencia.

Ao trocar o alvo para a **variacao percentual**, a serie fica aproximadamente estacionaria (sem
tendencia), o problema de extrapolacao desaparece e os modelos nao-lineares passam a competir em
igualdade. Alterne o controle acima para ver o efeito nas metricas.

Este e tambem o motivo de, em previsao cambial, modelos sofisticados frequentemente nao superarem
alternativas simples - um resultado classico da literatura economica (o chamado *puzzle* de
Meese-Rogoff).
"""
            )

        st.divider()
        st.subheader("Como isso complementa os indicadores do Checkpoint 1")
        st.markdown(
            f"""
No Checkpoint 1, a correlacao Selic x Cambio (**{kpis['corr_selic_cambio']:.3f}** neste recorte)
e um numero **estatico**: diz se existe associacao linear, mas nao antecipa valores futuros.

Os modelos acima transformam essa relacao em uma ferramenta **prospectiva**: usam Selic, IPCA e o
historico recente do cambio em conjunto para estimar o proximo mes, capturando tambem relacoes
nao-lineares que a correlacao simples nao enxerga (caso do Random Forest e do XGBoost).

O erro percentual medio do melhor modelo neste recorte e **{melhor['MAPE (%)']:.2f}%**. Quanto menor,
mais a informacao de politica monetaria e do proprio historico cambial basta para antecipar o
cambio; quanto maior, mais peso tem fatores externos ao dataset (fluxo de capital internacional,
commodities, cenario fiscal) - conclusao igualmente valida e coerente com a literatura economica.
"""
        )

# ----------------------------------------------------------------------------
# Aba 4 - Fonte dos dados
# ----------------------------------------------------------------------------
with aba_fontes:
    st.subheader("Fonte dos dados")
    st.markdown(
        "Todos os dados vem da **API publica do Banco Central do Brasil** - "
        "Sistema Gerenciador de Series Temporais (SGS). Acesso livre, sem cadastro ou chave."
    )
    st.dataframe(pd.DataFrame(FONTES), width='stretch')

    st.markdown(
        f"""
**Origem dos dados nesta sessao:** {detalhe_origem}

**Janela consultada:** a aplicacao pede sempre os ultimos 9 anos e 11 meses, porque a API do BCB
limita cada consulta a um intervalo maximo de 10 anos.

**Integracao das series:** Selic (432) e cambio (1) sao publicados em frequencia diaria; o IPCA (433)
e mensal. Para integrar as tres, as series diarias sao agregadas para **media mensal** e unidas ao
IPCA pelo par ano-mes.

**Fallback:** se a API estiver indisponivel, a aplicacao usa o arquivo `data/dataset_fallback.csv`,
versionado no repositorio, com dados reais coletados em 28/09/2026 a partir das series mensais
equivalentes do proprio BCB (4189, 3698 e 433). Isso garante que a dashboard continue funcionando
durante a avaliacao mesmo se o servidor do BCB estiver fora do ar.

**Documentacao oficial:** https://dadosabertos.bcb.gov.br/dataset/?res_format=API
"""
    )

    st.divider()
    st.subheader("Sobre o projeto")
    st.markdown(
        """
Esta dashboard e a **Parte 3** de um trabalho em tres etapas:

1. **Checkpoint 1** - coleta, integracao, analise exploratoria e construcao dos indicadores (KPIs);
2. **Checkpoint 2** - modelagem preditiva (regressao) para prever o cambio do mes seguinte,
   comparando modelos do scikit-learn, XGBoost e rede neural;
3. **Checkpoint 3** (esta aplicacao) - dashboard interativa reunindo os resultados das duas etapas.

**Nota tecnica:** no notebook do Checkpoint 2, a rede neural e implementada em Keras/TensorFlow.
Nesta dashboard ela foi trocada pelo `MLPRegressor` do scikit-learn, que e a mesma familia de modelo
(perceptron multicamadas) porem muito mais leve - o TensorFlow sozinho ultrapassaria o limite de
memoria do plano gratuito do Streamlit Community Cloud. Os demais modelos sao identicos aos do
Checkpoint 2.
"""
    )
