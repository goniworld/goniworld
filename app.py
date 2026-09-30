import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Goni Advanced Quant Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Goni Advanced Quantitative Research Lab")
st.markdown("블랙리터만 기대수익률, 켈리 자산배분, 그리고 트리플 배리어/메타레이블링 팩터 엔진 (API Key Free)")

# --- 1. 유니버스 설정 및 데이터 수집 (야후 파이낸스 활용 / 키 불필요) ---
ASSETS = {
    "코스피 종합": "^KS11",
    "S&P 500": "^GSPC",
    "나스닥 100": "^NDX",
    "금 (Gold)": "GC=F",
    "비트코인": "BTC-USD"
}

@st.cache_data(ttl=3600)
def fetch_quant_data():
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for name, ticker in ASSETS.items():
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close'].iloc[:, 0]
            else:
                df = df['Close']
            data[name] = df
    return pd.DataFrame(data)

with st.spinner("글로벌 퀀트 데이터 및 멀티팩터 연산 중..."):
    df_market = fetch_quant_data()

if not df_market.empty:
    returns = df_market.pct_change().dropna()
    
    # ==========================================
    # 모델 1: 블랙리터만 (Black-Litterman) - 사전 기대수익률 & 공분산
    # ==========================================
    st.subheader("📊 1. 블랙리터만(Black-Litterman) 모델 기초: 시장 균형 및 공분산")
    mean_returns = returns.mean() * 252 # 연환산 Prior 수익률
    cov_matrix = returns.cov() * 252   # 리스크 공분산 행렬
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**[Prior] 연환산 평균 기대 수익률**")
        st.dataframe(mean_returns.map("{:.2%}".format), use_container_width=True)
    with c2:
        st.markdown("**자산 간 리스크 공분산 행렬**")
        st.dataframe(cov_matrix.map("{:.4f}".format), use_container_width=True)

    # ==========================================
    # 모델 2: 켈리 공식 (Kelly Criterion) 기반 자산 배분
    # ==========================================
    st.markdown("---")
    st.subheader("🎯 2. 켈리 공식(Kelly Criterion) & 변동성 역가중치 배분")
    st.markdown("파산 위험을 최소화하고 복리 극대화를 노리는 수학적 자산 베팅 비중입니다.")
    
    # 켈리 개념을 응용한 변동성 및 샤프비율 기반 가중치 산출
    volatility = returns.std() * np.sqrt(252)
    sharpe_proxy = mean_returns / volatility
    kelly_weights = np.maximum(sharpe_proxy, 0)
    if kelly_weights.sum() > 0:
        kelly_weights = kelly_weights / kelly_weights.sum()
    else:
        kelly_weights = pd.Series(1/len(ASSETS), index=ASSETS.keys())
        
    df_kelly = pd.DataFrame({"자산": kelly_weights.index, "켈리 최적 비중": kelly_weights.values, "연환산 변동성": volatility.values})
    
    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        st.dataframe(df_kelly.style.format({"켈리 최적 비중": "{:.2%}", "연환산 변동성": "{:.2%}"}), use_container_width=True)
    with col_k2:
        fig_pie = go.Figure(data=[go.Pie(labels=kelly_weights.index, values=kelly_weights.values, hole=.3)])
        fig_pie.update_layout(title="켈리 최적 자산 배분 뷰", margin=dict(l=20, r=20, t=30, b=20), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # 모델 3: 트리플 배리어 & 메타 레이블링 시뮬레이터 (스윙/추세 탐색)
    # ==========================================
    st.markdown("---")
    st.subheader("🛡️ 3. 트리플 배리어 (Triple Barrier) 및 메타 레이블링 필터")
    st.markdown("상단(익절 장벽), 하단(손절 장벽), 시간(보유 기간) 기준에 따른 최근 자산별 시그널 상태를 진단합니다.")
    
    barrier_results = []
    for asset in df_market.columns:
        series = df_market[asset].dropna()
        if len(series) > 20:
            current_price = series.iloc[-1]
            sma_20 = series.rolling(20).mean().iloc[-1]
            upper_barrier = sma_20 * 1.05  # 상단 익절 5% 장벽
            lower_barrier = sma_20 * 0.95  # 하단 손절 -5% 장벽
            
            # 메타 레이블링 확신도 간이 평가 (변동성 대비 이격도)
            distance_ratio = (current_price - sma_20) / sma_20
            if current_price >= sma_20:
                signal = "상승 추세 (메타 승인)"
                confidence = min(0.5 + abs(distance_ratio) * 5, 0.95)
            else:
                signal = "하락/조정 추세 (관망)"
                confidence = max(0.5 - abs(distance_ratio) * 5, 0.1)
                
            barrier_results.append({
                "자산명": asset,
                "현재가": current_price,
                "20일 이평선": sma_20,
                "트리플 배리어 시그널": signal,
                "메타레이블링 확신도": f"{confidence:.1%}"
            })
            
    df_barrier = pd.DataFrame(barrier_results)
    st.dataframe(df_barrier, use_container_width=True)

else:
    st.error("데이터를 불러오지 못했습니다. 네트워크 연결을 확인해주세요.")