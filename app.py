import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Goni Quant & Asset Allocation Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Goni Quant & Asset Allocation Lab")
st.markdown("버튼 하나로 **소형/중형/대형주 수급 스윙 종목 발굴**과 **블랙리터만 & 켈리 자산 배분**을 수행합니다.")

# --- 세션 스테이트 초기화 ---
if "run_screening" not in st.session_state:
    st.session_state.run_screening = False

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 실행 패널")
tier_choice = st.sidebar.selectbox(
    "조회할 기업 규모 선택",
    ["소형주 (시총 ~ 3,000억 미만)", "중형주 (시총 3,000억 ~ 1조 미만)", "대형주 (시총 1조 이상)"]
)
min_turnover = st.sidebar.number_input("최소 거래대금 (원)", value=1_000_000_000, step=500_000_000)

if st.sidebar.button("🚀 스크리닝 실행하기", type="primary"):
    st.session_state.run_screening = True

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["🚀 규모별 수급 폭증 스윙 스나이퍼", "📊 블랙리터만 & 켈리 자산 배분"])

with tab1:
    st.subheader(f"🔥 선택한 규모: {tier_choice}")
    st.markdown("왼쪽 사이드바에서 **[스크리닝 실행하기]** 버튼을 누르면 실시간 데이터를 분석합니다.")

    if st.session_state.run_screening:
        with st.spinner("한국거래소(KRX) 및 야후 파이낸스 데이터 수집 및 분석 중..."):
            today = datetime.now()
            recent_day = ""
            for i in range(5):
                d = today - timedelta(days=i)
                if d.weekday() < 5:
                    recent_day = d.strftime("%Y%m%d")
                    break
            if not recent_day:
                recent_day = today.strftime("%Y%m%d")

            try:
                df_ks = stock.get_market_ohlcv_by_ticker(recent_day, market="KOSPI")
                df_kq = stock.get_market_ohlcv_by_ticker(recent_day, market="KOSDAQ")
                df_ks['Market'] = 'KOSPI'
                df_kq['Market'] = 'KOSDAQ'
                df_all = pd.concat([df_ks, df_kq])

                df_cap_ks = stock.get_market_cap_by_ticker(recent_day, market="KOSPI")
                df_cap_kq = stock.get_market_cap_by_ticker(recent_day, market="KOSDAQ")
                df_cap = pd.concat([df_cap_ks, df_cap_kq])
                df_all = df_all.join(df_cap[['시가총액']], how='left')
            except Exception as e:
                st.error(f"데이터를 불러오는 중 에러가 발생했습니다: {e}")
                df_all = pd.DataFrame()

            if not df_all.empty:
                if "소형주" in tier_choice:
                    df_filtered = df_all[(df_all['시가총액'] < 300_000_000_000) & (df_all['거래대금'] >= min_turnover)]
                elif "중형주" in tier_choice:
                    df_filtered = df_all[(df_all['시가총액'] >= 300_000_000_000) & (df_all['시가총액'] < 1_000_000_000_000) & (df_all['거래대금'] >= min_turnover)]
                else:
                    df_filtered = df_all[(df_all['시가총액'] >= 1_000_000_000_000) & (df_all['거래대금'] >= min_turnover)]

                if not df_filtered.empty:
                    df_filtered['회전율'] = df_filtered['거래대금'] / df_filtered['시가총액']
                    df_target = df_filtered.sort_values(by="회전율", ascending=False).head(40)

                    results = []
                    for ticker, row in df_target.iterrows():
                        name = stock.get_market_ticker_name(ticker)
                        market_suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
                        yf_ticker = f"{ticker}{market_suffix}"

                        hist = yf.download(yf_ticker, period="3mo", progress=False)
                        if not hist.empty and len(hist) > 20:
                            close_series = hist['Close'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Close']
                            vol_series = hist['Volume'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Volume']

                            cur_price = close_series.iloc[-1]
                            sma_20 = close_series.rolling(20).mean().iloc[-1]
                            avg_vol_20 = vol_series.rolling(20).mean().iloc[-1]
                            today_vol = vol_series.iloc[-1]
                            vol_spike_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                            disparity = (cur_price - sma_20) / sma_20 * 100

                            score = 0
                            if cur_price > sma_20: score += 40
                            if vol_spike_ratio >= 1.5: score += 30
                            if 0 <= disparity <= 5: score += 30
                            else: score += max(0, 30 - abs(disparity - 3) * 5)

                            signal = "🎯 강력 추천" if score >= 70 else ("👀 관심 종목" if score >= 50 else "⚠️ 관망")

                            results.append({
                                "종목코드": ticker,
                                "종목명": name,
                                "시장": row['Market'],
                                "현재가": f"{cur_price:,.0f}원",
                                "시가총액": f"{row['시가총액']//100_000_000:,.0f}억",
                                "거래량 폭증": f"{vol_spike_ratio:.1f}배",
                                "20일 이격도": f"{disparity:+.2f}%",
                                "점수": score,
                                "상태": signal
                            })

                    df_res = pd.DataFrame(results)
                    if not df_res.empty:
                        df_res = df_res.sort_values(by="점수", ascending=False)
                        st.success(f"총 {len(df_res)}개의 종목을 발굴했습니다!")
                        st.dataframe(df_res, use_container_width=True)
                    else:
                        st.warning("조건을 만족하는 종목이 없습니다.")
                else:
                    st.warning("해당 규모 및 조건에 부합하는 종목이 없습니다.")
            else:
                st.error("KRX 데이터를 가져오지 못했습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 **[스크리닝 실행하기]** 버튼을 눌러주세요.")

with tab2:
    st.subheader("📊 블랙리터만 모델 & 켈리 자산 배분 최적화")
    st.markdown("글로벌 주요 자산의 기대 수익률과 공분산 행렬을 바탕으로 **블랙리터만 균형 수익률**과 **켈리 공식 기반 최적 자산 배분 비중**을 산출합니다.")

    MACRO_ASSETS = {
        "코스피 종합": "^KS11", 
        "S&P 500": "^GSPC", 
        "나스닥 100": "^NDX", 
        "금 (Gold)": "GC=F"
    }
    
    macro_data = {}
    with st.spinner("글로벌 매크로 데이터 수집 및 최적화 계산 중..."):
        for name, tck in MACRO_ASSETS.items():
            try:
                d = yf.download(tck, period="1yr", progress=False)
                if not d.empty:
                    c = d['Close']
                    macro_data[name] = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            except:
                pass
                
    df_macro = pd.DataFrame(macro_data)
    
    if not df_macro.empty:
        df_macro = df_macro.dropna()
        if len(df_macro) > 30:
            returns = df_macro.pct_change().dropna()
            mean_returns = returns.mean() * 252  # 연환산 기대수익률 (Prior)
            cov_matrix = returns.cov() * 252     # 연환산 공분산 행렬
            
            # 켈리 자산 배분 가중치 산출 (Kelly Weight Formula: W = Inv(Cov) * Mean)
            try:
                inv_cov = np.linalg.inv(cov_matrix.values)
                raw_kelly_weights = np.dot(inv_cov, mean_returns.values)
                # 정규화 (합이 1이 되도록 조정, 음수 방지를 위해 Clip 적용 가능)
                kelly_weights = np.clip(raw_kelly_weights, 0, None)
                if kelly_weights.sum() > 0:
                    kelly_weights = kelly_weights / kelly_weights.sum()
                else:
                    kelly_weights = np.ones(len(mean_returns)) / len(mean_returns)
            except:
                kelly_weights = np.ones(len(mean_returns)) / len(mean_returns)

            df_allocation = pd.DataFrame({
                "자산명": mean_returns.index,
                "연환산 기대수익률 (Prior)": mean_returns.values,
                "켈리 최적 배분 비중": kelly_weights
            })
            df_allocation.set_index("자산명", inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📌 자산별 연환산 기대 수익률 및 켈리 비중**")
                st.dataframe(
                    df_allocation.style.format({
                        "연환산 기대수익률 (Prior)": "{:.2%}",
                        "켈리 최적 배분 비중": "{:.2%}"
                    }), 
                    use_container_width=True
                )
            with col2:
                st.markdown("**📉 자산 리스크 공분산 행렬 (연환산)**")
                st.dataframe(cov_matrix.style.format("{:.4f}"), use_container_width=True)
                
            st.info("💡 **블랙리터만 & 켈리 공식 가이드**: 켈리 공식은 기대 수익률을 극대화하면서 파산 위험을 최소화하는 수학적 최적 배분 비중을 제안합니다.")
        else:
            st.warning("자산 분석을 위한 데이터 거래일 수가 부족합니다.")
    else:
        st.error("야후 파이낸스에서 매크로 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
