import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="Goni Quant & Asset Lab", page_icon="🎯", layout="wide")
st.title("🎯 Goni Quant & Asset Allocation Lab")
st.markdown("실시간 장중 수급 스윙 종목 발굴과 매크로 자산 배분 비중을 계산합니다.")

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["🚀 장중 수급 스나이퍼", "📊 블랙리터만 & 켈리 자산 배분"])

# ==========================================
# TAB 1 : 수급 폭증 스윙 스나이퍼
# ==========================================
with tab1:
    st.subheader("🔥 실시간 중소형/대형주 수급 폭증 스크리닝")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        tier_choice = st.selectbox("기업 규모 선택", ["소형주 (500억~3,000억 미만)", "중형주 (3,000억~1조 미만)", "대형주 (1조 이상)"])
    with c2:
        min_score = st.slider("최소 퀀트 점수", 50, 90, 60)
    with c3:
        st.write("") # 버튼 위치 정렬용
        st.write("")
        run_btn1 = st.button("🚀 스크리닝 실행", type="primary", use_container_width=True)

    if run_btn1:
        with st.spinner("KRX 시총 데이터 및 실시간 주가 분석 중... (약 20~40초 소요)"):
            today = datetime.now()
            valid_day = None
            df_cap = pd.DataFrame()

            # 1. 안전한 최근 영업일 찾기 (최대 10일 전까지 역순 탐색)
            for i in range(1, 10):
                d = today - timedelta(days=i)
                if d.weekday() < 5:
                    test_day = d.strftime("%Y%m%d")
                    try:
                        # 최신 pykrx API 호환
                        tmp = stock.get_market_cap(test_day, market="KOSPI")
                        if not tmp.empty and '거래대금' in tmp.columns:
                            valid_day = test_day
                            break
                    except:
                        pass

            if valid_day:
                try:
                    df_ks = stock.get_market_cap(valid_day, market="KOSPI")
                    df_kq = stock.get_market_cap(valid_day, market="KOSDAQ")
                    df_ks['Market'] = 'KOSPI'
                    df_kq['Market'] = 'KOSDAQ'
                    df_cap = pd.concat([df_ks, df_kq])
                except Exception as e:
                    st.error(f"KRX 데이터 연동 에러: {e}")

            if not df_cap.empty:
                # 2. 규모 필터링
                if "소형주" in tier_choice:
                    df_f = df_cap[(df_cap['시가총액'] >= 50_000_000_000) & (df_cap['시가총액'] < 300_000_000_000)]
                elif "중형주" in tier_choice:
                    df_f = df_cap[(df_cap['시가총액'] >= 300_000_000_000) & (df_cap['시가총액'] < 1_000_000_000_000)]
                else:
                    df_f = df_cap[df_cap['시가총액'] >= 1_000_000_000_000]

                # 시총 대비 회전율 상위 50개 종목 추출
                df_f['회전율'] = df_f['거래대금'] / df_f['시가총액']
                df_target = df_f.sort_values(by="회전율", ascending=False).head(50)

                # 3. 야후 파이낸스 실시간 데이터 병렬 다운로드
                tickers_yf = [f"{t}.KS" if r['Market'] == 'KOSPI' else f"{t}.KQ" for t, r in df_target.iterrows()]
                live_data = yf.download(tickers_yf, period="3mo", progress=False)

                results = []
                pb = st.progress(0)
                
                for idx, (ticker, row) in enumerate(df_target.iterrows()):
                    pb.progress((idx + 1) / len(df_target))
                    yf_t = f"{ticker}.KS" if row['Market'] == 'KOSPI' else f"{ticker}.KQ"
                    
                    try:
                        # yfinance 최신 버전 MultiIndex 처리
                        if isinstance(live_data.columns, pd.MultiIndex):
                            c_series = live_data['Close'][yf_t].dropna()
                            v_series = live_data['Volume'][yf_t].dropna()
                        else:
                            c_series = live_data['Close'].dropna()
                            v_series = live_data['Volume'].dropna()

                        if len(c_series) < 25: 
                            continue

                        cur_price = float(c_series.iloc[-1])
                        today_vol = float(v_series.iloc[-1])
                        sma_20 = float(c_series.rolling(20).mean().iloc[-1])
                        avg_vol_20 = float(v_series.rolling(20).mean().iloc[-2])

                        vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                        disp = (cur_price - sma_20) / sma_20 * 100

                        # 스윙 점수 산출
                        score = 0
                        if cur_price > sma_20: score += 40
                        if vol_ratio >= 1.5: score += 30
                        if 0 <= disp <= 5: score += 30
                        else: score += max(0, 30 - abs(disp - 3) * 5)

                        if score >= min_score:
                            results.append({
                                "종목명": stock.get_market_ticker_name(ticker),
                                "현재가": f"{cur_price:,.0f}원",
                                "시가총액": f"{row['시가총액']//100_000_000:,.0f}억",
                                "거래량 배수": f"{vol_ratio:.1f}배",
                                "20일선 이격도": f"{disp:+.2f}%",
                                "점수": round(score),
                                "상태": "🎯 강력" if score >= 70 else "👀 관심"
                            })
                    except:
                        continue
                
                pb.empty()
                
                if results:
                    st.success(f"조건 부합 종목 {len(results)}개 발견!")
                    df_res = pd.DataFrame(results).sort_values("점수", ascending=False).reset_index(drop=True)
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.warning("현재 기준 조건에 맞는 강한 수급 종목이 없습니다.")
            else:
                st.error("KRX 데이터를 불러오지 못했습니다. 장 마감 직후이거나 점검 중일 수 있습니다.")

# ==========================================
# TAB 2 : 매크로 자산 블랙리터만 & 켈리
# ==========================================
with tab2:
    st.subheader("📊 블랙리터만 모델 & 켈리 최적화")
    st.markdown("버튼을 누르면 야후 파이낸스에서 주요 글로벌 자산 데이터를 가져와 최적 비중을 계산합니다.")
    
    run_macro = st.button("📈 자산 배분 비중 계산하기", type="primary", use_container_width=True)
    
    if run_macro:
        # yfinance에서 안정적으로 받아와지는 핵심 지수들로 세팅
        assets = {"KOSPI": "^KS11", "S&P500": "^GSPC", "NASDAQ": "^IXIC", "GOLD": "GC=F"}
        
        with st.spinner("글로벌 매크로 데이터 수집 및 공분산 연산 중..."):
            macro_dict = {}
            for name, tk in assets.items():
                try:
                    d = yf.download(tk, period="1yr", progress=False)
                    if not d.empty:
                        c = d['Close']
                        macro_dict[name] = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
                except:
                    pass
            
            df_m = pd.DataFrame(macro_dict).dropna()
            
            if len(df_m) > 30:
                ret = df_m.pct_change().dropna()
                mean_ret = ret.mean() * 252  # 연환산 기대수익률
                cov = ret.cov() * 252        # 연환산 공분산
                
                # 켈리 최적 가중치 공식 (Kelly Criterion)
                try:
                    inv_cov = np.linalg.inv(cov.values)
                    raw_w = np.dot(inv_cov, mean_ret.values)
                    # 공매도 방지 (비중이 마이너스인 것 0으로 클리핑)
                    w = np.clip(raw_w, 0, None)
                    # 총합 100%로 정규화
                    if w.sum() > 0:
                        w = w / w.sum()
                    else:
                        w = np.ones(len(w)) / len(w)
                except:
                    w = np.ones(len(mean_ret)) / len(mean_ret)
                    
                res_df = pd.DataFrame({
                    "자산명": mean_ret.index,
                    "연환산 기대수익률": mean_ret.values,
                    "켈리 최적비중": w
                }).set_index("자산명")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📌 자산별 기대수익률 및 배분 비중**")
                    st.dataframe(res_df.style.format("{:.2%}"), use_container_width=True)
                with col2:
                    st.markdown("**📉 자산 리스크 공분산 행렬**")
                    st.dataframe(cov.style.format("{:.4f}"), use_container_width=True)
            else:
                st.error("야후 파이낸스에서 연산에 필요한 충분한 데이터를 가져오지 못했습니다. 네트워크 상태를 확인하세요.")
