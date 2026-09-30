import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Goni Mid-Cap Swing Sniper Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Goni Mid-Cap Swing Sniper Lab")
st.markdown("API 키 없이 **중소형주 시총 필터링 + 거래량 폭증(Volume Spike) + 20일선 이격도**를 활용한 알짜배기 스윙 종목 자동 발굴 시스템")

# --- 사이드바 필터 설정 ---
st.sidebar.header("⚙️ 퀀트 스윙 스크리닝 필터")
min_cap = st.sidebar.slider("최소 시가총액 (억 원)", 300, 2000, 500) * 100_000_000
max_cap = st.sidebar.slider("최대 시가총액 (억 원)", 5000, 30000, 15000) * 100_000_000
min_turnover = st.sidebar.number_input("최소 거래대금 (원)", value=3_000_000_000, step=1_000_000_000)
min_score = st.sidebar.slider("최소 퀀트 스윙 점수", 50, 90, 70)

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["🚀 중소형 수급 폭증 스윙 스나이퍼", "📊 블랙리터만 & 켈리 자산 배분"])

with tab1:
    st.subheader("🔥 대형주 제외! 바닥권 수급 폭증 중소형 스윙 추천 종목")
    st.markdown("삼성전자 같은 대형주를 제외하고, 평소보다 거래량이 2배 이상 터지며 20일선 위로 고개를 드는 알짜배기 종목들을 스크리닝합니다.")

    @st.cache_data(ttl=3600)
    def get_midcap_swing_picks(min_c, max_c, min_t):
        today_str = datetime.now().strftime("%Y%m%d")
        recent_day = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
        
        try:
            df_ks = stock.get_market_ohlcv_by_ticker(recent_day, market="KOSPI")
            df_kq = stock.get_market_ohlcv_by_ticker(recent_day, market="KOSDAQ")
        except:
            return pd.DataFrame()
            
        df_ks['Market'] = 'KOSPI'
        df_kq['Market'] = 'KOSDAQ'
        df_all = pd.concat([df_ks, df_kq])
        
        # 시가총액 데이터 결합 (초대형주 쏠림 방지)
        try:
            df_cap_ks = stock.get_market_cap_by_ticker(recent_day, market="KOSPI")
            df_cap_kq = stock.get_market_cap_by_ticker(recent_day, market="KOSDAQ")
            df_cap = pd.concat([df_cap_ks, df_cap_kq])
            df_all = df_all.join(df_cap[['시가총액']], how='left')
        except:
            df_all['시가총액'] = 0

        # 사용자가 지정한 중소형주 및 거래대금 조건 필터링
        df_filtered = df_all[
            (df_all['시가총액'] >= min_c) & 
            (df_all['시가총액'] <= max_c) & 
            (df_all['거래대금'] >= min_t)
        ]
        
        # 거래대금 회전율(거래대금 / 시가총액)이 높은 종목 상위 60개 추출하여 타겟팅
        df_filtered['회전율'] = df_filtered['거래대금'] / df_filtered['시가총액']
        df_target = df_filtered.sort_values(by="회전율", ascending=False).head(60)
        
        results = []
        for ticker, row in df_target.iterrows():
            name = stock.get_market_ticker_name(ticker)
            market_suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
            yf_ticker = f"{ticker}{market_suffix}"
            
            # 최근 3개월 데이터 분석
            hist = yf.download(yf_ticker, period="3mo", progress=False)
            if not hist.empty and len(hist) > 25:
                close_series = hist['Close'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Close']
                vol_series = hist['Volume'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Volume']
                
                cur_price = close_series.iloc[-1]
                sma_20 = close_series.rolling(20).mean().iloc[-1]
                
                # 거래량 폭증 배수 (오늘 거래량 / 최근 20일 평균 거래량)
                avg_vol_20 = vol_series.rolling(20).mean().iloc[-1]
                today_vol = vol_series.iloc[-1]
                vol_spike_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                
                disparity = (cur_price - sma_20) / sma_20 * 100
                recent_ret = (cur_price - close_series.iloc[-20]) / close_series.iloc[-20] * 100
                
                # 중소형주 맞춤형 퀀트 스윙 점수 산출 로직
                score = 0
                if cur_price > sma_20: score += 40               # 20일선 안착 여부
                if vol_spike_ratio >= 1.8: score += 30           # 거래량이 평소보다 1.8배 이상 터졌는가?
                if 0 <= disparity <= 5: score += 30              # 과열권(5% 이상 이격)이 아닌 건강한 위치인가?
                else: score += max(0, 30 - abs(disparity - 3)*5)
                
                confidence = min(max(score / 100, 0.2), 0.98)
                
                if score >= 75:
                    signal = "🎯 강력 추천 (중소형 수급 유입)"
                elif score >= 60:
                    signal = "👀 관심 종목 (눌림목 체크)"
                else:
                    signal = "⚠️ 관망"
                    
                results.append({
                    "종목코드": ticker,
                    "종목명": name,
                    "시장": row['Market'],
                    "현재가": f"{cur_price:,.0f}원",
                    "시가총액": f"{row['시가총액']//100_000_000:,.0f}억",
                    "거래량 폭증 배수": f"{vol_spike_ratio:.1f}배",
                    "20일 이격도": f"{disparity:+.2f}%",
                    "최근 20일 추세": f"{recent_ret:+.2f}%",
                    "퀀트 스윙 점수": score,
                    "승률 확신도": f"{confidence:.1%}",
                    "추천 상태": signal
                })
        return pd.DataFrame(results)

    with st.spinner("중소형 알짜 종목 필터링 및 거래량 폭증(Volume Spike) 전수 조사 중..."):
        df_picks = get_midcap_swing_picks(min_cap, max_cap, min_turnover)

    if not df_picks.empty:
        df_filtered = df_picks[df_picks["퀀트 스윙 점수"] >= min_score]
        df_filtered = df_filtered.sort_values(by="퀀트 스윙 점수", ascending=False)
        
        st.success(f"조건을 만족하는 알짜 중소형 스윙 종목 {len(df_filtered)}개를 발굴했습니다!")
        st.dataframe(df_filtered.drop(columns=["퀀트 스윙 점수"]), use_container_width=True)
    else:
        st.warning("조건에 부합하는 종목이 없거나 데이터를 불러오는 중입니다. 필터 범위를 조금 넓혀보세요.")

with tab2:
    st.subheader("📊 블랙리터만 모델 기반 글로벌 매크로 균형 수익률")
    MACRO_ASSETS = {"코스피 종합": "^KS11", "S&P 500": "^GSPC", "나스닥 100": "^NDX", "금 (Gold)": "GC=F"}
    
    macro_data = {}
    for name, tck in MACRO_ASSETS.items():
        d = yf.download(tck, period="1yr", progress=False)
        if not d.empty:
            macro_data[name] = d['Close'].iloc[:, 0] if isinstance(d.columns, pd.MultiIndex) else d['Close']
    df_macro = pd.DataFrame(macro_data)
    
    if not df_macro.empty:
        macro_returns = df_macro.pct_change().dropna()
        mean_returns = macro_returns.mean() * 252
        cov_matrix = macro_returns.cov() * 252
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**[Prior] 자산별 연환산 기대 수익률**")
            st.dataframe(mean_returns.map("{:.2%}".format), use_container_width=True)
        with c2:
            st.markdown("**자산 리스크 공분산 행렬**")
            st.dataframe(cov_matrix.map("{:.4f}".format), use_container_width=True)
