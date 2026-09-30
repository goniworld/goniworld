import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Goni Cap-Tier Swing Sniper Lab", page_icon="🎯", layout="wide")

st.title("🎯 Goni Cap-Tier Swing Sniper Lab")
st.markdown("버튼 하나로 **소형/중형/대형주 규모별 장중 실시간 수급 폭증 종목**을 자동 발굴합니다.")

if "run_screening" not in st.session_state:
    st.session_state.run_screening = False

st.sidebar.header("⚙️ 실행 패널")
tier_choice = st.sidebar.selectbox(
    "조회할 기업 규모 선택",
    ["소형주 (시총 ~ 3,000억 미만)", "중형주 (시총 3,000억 ~ 1조 미만)", "대형주 (시총 1조 이상)"]
)
min_score = st.sidebar.slider("최소 퀀트 스윙 점수", 50, 90, 60)

if st.sidebar.button("🚀 실시간 스크리닝 실행", type="primary"):
    st.session_state.run_screening = True

tab1, tab2 = st.tabs(["🚀 장중 실시간 수급 스나이퍼", "📊 블랙리터만 & 켈리 자산 배분"])

with tab1:
    st.subheader(f"🔥 선택한 규모: {tier_choice} (장중 실시간 분석)")
    
    if st.session_state.run_screening:
        with st.spinner("KRX 시총 데이터 및 야후 파이낸스 실시간 주가 연동 중... (약 30~60초 소요)"):
            # 1. pykrx 에러를 피하기 위해 무조건 전일(가장 최근 마감일) 시총 데이터만 가져옴
            today = datetime.now()
            valid_day = ""
            for i in range(1, 7): # 오늘 제외, 어제부터 탐색
                d = today - timedelta(days=i)
                if d.weekday() < 5:
                    test_day = d.strftime("%Y%m%d")
                    try:
                        temp_ks = stock.get_market_cap_by_ticker(test_day, market="KOSPI")
                        if not temp_ks.empty:
                            valid_day = test_day
                            break
                    except:
                        pass
            
            if valid_day:
                df_cap_ks = stock.get_market_cap_by_ticker(valid_day, market="KOSPI")
                df_cap_kq = stock.get_market_cap_by_ticker(valid_day, market="KOSDAQ")
                df_cap_ks['Market'] = 'KOSPI'
                df_cap_kq['Market'] = 'KOSDAQ'
                df_cap = pd.concat([df_cap_ks, df_cap_kq])
                
                # 2. 규모별 시가총액 필터링 (불필요한 연산 축소)
                if "소형주" in tier_choice:
                    df_filtered = df_cap[(df_cap['시가총액'] < 300_000_000_000) & (df_cap['시가총액'] > 50_000_000_000)]
                elif "중형주" in tier_choice:
                    df_filtered = df_cap[(df_cap['시가총액'] >= 300_000_000_000) & (df_cap['시가총액'] < 1_000_000_000_000)]
                else:
                    df_filtered = df_cap[df_cap['시가총액'] >= 1_000_000_000_000]

                # 시총 대비 거래대금(어제 기준 활성도) 상위 60개만 추려서 실시간 조회 속도 최적화
                df_filtered['회전율'] = df_filtered['거래대금'] / df_filtered['시가총액']
                df_target = df_filtered.sort_values(by="회전율", ascending=False).head(60)

                results = []
                progress_bar = st.progress(0)
                
                # 3. yfinance 실시간 데이터 수집 (오늘 장중 데이터)
                tickers_yf = []
                for t, r in df_target.iterrows():
                    suffix = ".KS" if r['Market'] == 'KOSPI' else ".KQ"
                    tickers_yf.append(f"{t}{suffix}")
                
                # yf.download를 한 번에 호출하여 속도 대폭 개선
                live_data = yf.download(tickers_yf, period="3mo", progress=False)
                
                for idx, (ticker, row) in enumerate(df_target.iterrows()):
                    progress_bar.progress((idx + 1) / len(df_target))
                    name = stock.get_market_ticker_name(ticker)
                    yf_t = f"{ticker}.KS" if row['Market'] == 'KOSPI' else f"{ticker}.KQ"
                    
                    try:
                        # MultiIndex 컬럼 처리
                        if isinstance(live_data.columns, pd.MultiIndex):
                            c_series = live_data['Close'][yf_t].dropna()
                            v_series = live_data['Volume'][yf_t].dropna()
                        else:
                            c_series = live_data['Close'].dropna()
                            v_series = live_data['Volume'].dropna()
                            
                        if len(c_series) < 25:
                            continue
                            
                        # iloc[-1]은 현재 장중 실시간 가격과 거래량을 의미함!
                        cur_price = c_series.iloc[-1]
                        today_vol = v_series.iloc[-1]
                        
                        sma_20 = c_series.rolling(20).mean().iloc[-1]
                        avg_vol_20 = v_series.rolling(20).mean().iloc[-2] # 어제까지의 20일 평균
                        
                        vol_spike_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                        disparity = (cur_price - sma_20) / sma_20 * 100
                        
                        score = 0
                        if cur_price > sma_20: score += 40
                        if vol_spike_ratio >= 1.5: score += 30
                        if 0 <= disparity <= 5: score += 30
                        else: score += max(0, 30 - abs(disparity - 3) * 5)
                        
                        if score >= min_score:
                            signal = "🎯 장중 수급 유입" if score >= 70 else "👀 눌림목 체크"
                            results.append({
                                "종목명": name,
                                "현재 주가 (실시간)": f"{cur_price:,.0f}원",
                                "시가총액": f"{row['시가총액']//100_000_000:,.0f}억",
                                "실시간 거래량 폭증": f"{vol_spike_ratio:.1f}배",
                                "20일선 이격도": f"{disparity:+.2f}%",
                                "점수": score,
                                "상태": signal
                            })
                    except:
                        continue
                
                progress_bar.empty()
                
                df_res = pd.DataFrame(results)
                if not df_res.empty:
                    df_res = df_res.sort_values(by="점수", ascending=False).reset_index(drop=True)
                    st.success(f"현재 시간 기준 실시간 수급 포착 종목 {len(df_res)}개 발굴!")
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.warning("현재 장중에 수급이 강하게 들어오는 조건 부합 종목이 없습니다.")
            else:
                st.error("기본 데이터를 불러올 수 없습니다.")
    else:
        st.info("👈 왼쪽 사이드바에서 **[실시간 스크리닝 실행]**을 눌러 지금 움직이는 종목을 찾아보세요.")

with tab2:
    st.subheader("📊 블랙리터만 모델 & 켈리 자산 배분 최적화")
    MACRO_ASSETS = {"코스피 종합": "^KS11", "S&P 500": "^GSPC", "나스닥 100": "^NDX", "금 (Gold)": "GC=F"}
    
    macro_data = {}
    with st.spinner("매크로 데이터 연산 중..."):
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
            mean_returns = returns.mean() * 252 
            cov_matrix = returns.cov() * 252     
            
            try:
                inv_cov = np.linalg.inv(cov_matrix.values)
                raw_kelly_weights = np.dot(inv_cov, mean_returns.values)
                kelly_weights = np.clip(raw_kelly_weights, 0, None)
                if kelly_weights.sum() > 0:
                    kelly_weights = kelly_weights / kelly_weights.sum()
                else:
                    kelly_weights = np.ones(len(mean_returns)) / len(mean_returns)
            except:
                kelly_weights = np.ones(len(mean_returns)) / len(mean_returns)

            df_allocation = pd.DataFrame({
                "자산명": mean_returns.index,
                "기대수익률": mean_returns.values,
                "최적 배분비중": kelly_weights
            }).set_index("자산명")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_allocation.style.format("{:.2%}"), use_container_width=True)
            with col2:
                st.dataframe(cov_matrix.style.format("{:.4f}"), use_container_width=True)
