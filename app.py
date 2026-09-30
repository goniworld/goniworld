import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Goni Cap-Tier Swing Sniper Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Goni Cap-Tier Swing Sniper Lab")
st.markdown("버튼 하나로 **소형주 / 중형주 / 대형주** 규모별 수급 폭증 및 스윙 종목을 자동으로 발굴합니다.")

# --- 세션 스테이트 초기화 (버튼형 실행 제어) ---
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
tab1, tab2 = st.tabs(["🚀 규모별 수급 폭증 스윙 스나이퍼", "📊 글로벌 매크로 분석"])

with tab1:
    st.subheader(f"🔥 선택한 규모: {tier_choice}")
    st.markdown("왼쪽 사이드바에서 **[스크리닝 실행하기]** 버튼을 누르면 실시간 데이터를 분석합니다.")

    if st.session_state.run_screening:
        with st.spinner("한국거래소(KRX) 및 야후 파이낸스 데이터 수집 및 분석 중..."):
            # 최근 영업일 설정 로직
            today = datetime.now()
            recent_day = ""
            for i in range(5):
                d = today - timedelta(days=i)
                if d.weekday() < 5:  # 주말 제외
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
                # 시가총액 규모별 필터 조건 설정
                if "소형주" in tier_choice:
                    df_filtered = df_all[(df_all['시가총액'] < 300_000_000_000) & (df_all['거래대금'] >= min_turnover)]
                elif "중형주" in tier_choice:
                    df_filtered = df_all[(df_all['시가총액'] >= 300_000_000_000) & (df_all['시가총액'] < 1_000_000_000_000) & (df_all['거래대금'] >= min_turnover)]
                else:  # 대형주
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

                            if score >= 70:
                                signal = "🎯 강력 추천"
                            elif score >= 50:
                                signal = "👀 관심 종목"
                            else:
                                signal = "⚠️ 관망"

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
                        st.warning("조건을 만족하는 종목이 없습니다. 거래대금 조건을 낮춰보세요.")
                else:
                    st.warning("해당 규모 및 조건에 부합하는 종목이 없습니다.")
            else:
                st.error("KRX 데이터를 가져오지 못했습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 **[스크리닝 실행하기]** 버튼을 눌러주세요.")

with tab2:
    st.subheader("📊 글로벌 매크로 자산 분석")
    MACRO_ASSETS = {"코스피": "^KS11", "S&P 500": "^GSPC", "나스닥": "^NDX", "금": "GC=F"}
    
    macro_data = {}
    for name, tck in MACRO_ASSETS.items():
        try:
            d = yf.download(tck, period="6mo", progress=False)
            if not d.empty:
                c = d['Close']
                macro_data[name] = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        except:
            pass
            
    df_macro = pd.DataFrame(macro_data)
    if not df_macro.empty:
        df_macro = df_macro.dropna()
        returns = df_macro.pct_change().dropna()
        st.markdown("**자산별 연환산 기대 수익률**")
        st.dataframe((returns.mean() * 252).apply("{:.2%}".format), use_container_width=True)
    else:
        st.info("매크로 데이터를 일시적으로 불러오지 못했습니다.")
