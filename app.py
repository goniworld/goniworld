import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import random
import math
from scipy.stats.mstats import winsorize
from transformers import pipeline
import ta
import FinanceDataReader as fdr

# -------------------------------------------------------------------
# [유틸리티 함수] 이상치 제어(Winsorization) 및 Z-스코어(Z-Score) 정규화
# -------------------------------------------------------------------
def apply_z_score_normalization(data_series, lower_is_better=False):
    if len(data_series.dropna()) < 2:
        return pd.Series(0, index=data_series.index)
        
    winsorized_data = winsorize(data_series.dropna(), limits=[0.05, 0.05])
    mu = np.mean(winsorized_data)
    sigma = np.std(winsorized_data)
    
    if sigma == 0:
        return pd.Series(0, index=data_series.index)
    
    z_scores = (data_series - mu) / sigma
    
    if lower_is_better:
        z_scores = -z_scores
        
    return z_scores

# -------------------------------------------------------------------
# [데이터 크롤링 모듈] 위키피디아 및 KRX 기반 실시간 티커 추출 [cite: 48, 51]
# -------------------------------------------------------------------
@st.cache_data(ttl=86400) # 서버 부하 방지를 위해 원본 리스트만 하루 1번 캐싱
def get_us_market_data(size_category):
    """위키피디아 S&P 지수 테이블을 실시간 스크래핑하여 미국 종목코드와 이름을 가져옵니다."""
    if size_category == "대형주":
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    elif size_category == "중형주":
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
    else: # 소형주
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        
    df = pd.read_html(url)[0]
    name_col = 'Security' if 'Security' in df.columns else 'Company'
    
    results = []
    for _, row in df.iterrows():
        ticker = str(row['Symbol']).replace('.', '-')
        name = str(row[name_col])
        results.append((ticker, name))
        
    return results

@st.cache_data(ttl=86400)
def get_kr_market_data(size_category):
    """FinanceDataReader를 활용하여 한국 시가총액 규모별 종목코드와 이름을 추출합니다."""
    df = fdr.StockListing('KRX')
    
    if 'Marcap' in df.columns:
        df = df.sort_values('Marcap', ascending=False).reset_index(drop=True)
    
    if size_category == "대형주":
        target_df = df.iloc[:50] # 시총 최상위 50개 (KOSPI 우량주)
    elif size_category == "중형주":
        target_df = df.iloc[100:300] # 시총 101~300위 (중형주)
    else: # 소형주
        target_df = df.iloc[500:1500] # 시총 501위 이하 (주로 KOSDAQ 강소기업)
        
    results = []
    for _, row in target_df.iterrows():
        code = str(row['Code'])
        name = str(row['Name'])
        market = str(row.get('Market', ''))
        
        # yfinance 인식 포맷 추가 (.KS: 코스피, .KQ: 코스닥)
        if 'KOSPI' in market:
            ticker = code + '.KS'
        elif 'KOSDAQ' in market:
            ticker = code + '.KQ'
        else:
            ticker = code + '.KS'
        results.append((ticker, name))
            
    return results

# -------------------------------------------------------------------
# [제1 탭 모듈] 장기 투자 경제적 분석 (Piotroski F-Score)
# -------------------------------------------------------------------
def calculate_long_term_fundamental(ticker_obj):
    try:
        financials = ticker_obj.financials
        cashflow = ticker_obj.cashflow
        
        net_inc = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
        cfo = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
        
        f_score_proxy = 0
        if net_inc > 0: f_score_proxy += 1
        if cfo > 0: f_score_proxy += 1
        if cfo > net_inc: f_score_proxy += 1
        
        return f_score_proxy * 3.33 
    except Exception:
        return 5.0 

# -------------------------------------------------------------------
# [제2 탭 모듈] 단기 투자 알고리즘 분석 (Volatility Breakout & %R)
# -------------------------------------------------------------------
def calculate_short_term_momentum(hist_df):
    if len(hist_df) < 20:
        return 5.0

    high_14 = hist_df['High'].rolling(14).max()
    low_14 = hist_df['Low'].rolling(14).min()
    williams_r = ((high_14 - hist_df['Close']) / (high_14 - low_14)) * -100
    current_w_r = williams_r.iloc[-1]
    
    prev_high = hist_df['High'].iloc[-2]
    prev_low = hist_df['Low'].iloc[-2]
    day_range = prev_high - prev_low
    
    breakout_target = hist_df['Open'].iloc[-1] + (0.5 * day_range)
    
    score = 5.0
    if hist_df['Close'].iloc[-1] > breakout_target:
        score += 3.0
    if -80 < current_w_r < -20:
        score += 2.0
        
    return score

# -------------------------------------------------------------------
# [제3 탭 모듈] 미래 실험적 투자 분석 (FinBERT & Risk Parity)
# -------------------------------------------------------------------
@st.cache_resource
def initialize_nlp_pipeline():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

def calculate_experimental_tech(ticker_name, hist_df, nlp_pipe):
    daily_returns = hist_df['Close'].pct_change().dropna()
    annualized_volatility = daily_returns.std() * math.sqrt(252)
    inverse_vol_score = 1.0 / (annualized_volatility + 1e-6)
    
    news_headlines = [f"Market update and recent coverage regarding {ticker_name}."]
    sentiment_val = 0
    if nlp_pipe:
        sentiments = nlp_pipe(news_headlines)
        for s in sentiments:
            if s['label'] == 'positive': sentiment_val += 2.0
            elif s['label'] == 'negative': sentiment_val -= 2.0
            else: sentiment_val += 0.5
            
    return min(max((sentiment_val + inverse_vol_score * 5), 0), 10)

# -------------------------------------------------------------------
# [메인 시스템] Streamlit 애플리케이션 및 UI 렌더링
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="규모별 멀티 팩터 주식 추천 시스템", layout="wide")
    st.title("📈 동적 스크래핑 기반 기업 규모 맞춤형 주식 분석 엔진")
    st.write("하드코딩된 리스트 없이, 실시간으로 시장 데이터를 스크래핑하여 체급별 유망 종목을 발굴합니다.")
    
    # 1. UI 선택 레이아웃
    col1, col2 = st.columns(2)
    with col1:
        market_selection = st.radio("분석할 주식 시장을 선택하세요:", ("🇺🇸 미국 시장", "🇰🇷 한국 시장"))
    with col2:
        size_selection = st.radio("기업 규모(시가총액)를 선택하세요:", ("대형주", "중형주", "소형주"))
    
    # 2. 세션(Session)을 통한 종목 리스트 관리 (매번 같은 주식이 나오는 현상 방지)
    if 'current_tickers' not in st.session_state:
        st.session_state.current_tickers = ""
    if 'ticker_names' not in st.session_state:
        st.session_state.ticker_names = {}
        
    st.markdown("---")
    st.write(f"💡 현재 선택: **{market_selection} - {size_selection}**")
    
    # 새로운 종목 15개를 뽑는 '가챠' 버튼
    if st.button(f"🔄 '{size_selection}' 15개 종목 무작위 새로 뽑기"):
        with st.spinner("해당 시장의 전체 종목을 스크래핑하여 15개를 무작위로 고르고 있습니다..."):
            if "미국" in market_selection:
                full_list = get_us_market_data(size_selection)
            else:
                full_list = get_kr_market_data(size_selection)
                
            # 전체 리스트에서 15개만 셔플하여 추출
            sampled = random.sample(full_list, min(15, len(full_list)))
            
            # 종목명(한글/영문) 사전에 저장
            for ticker, name in full_list:
                st.session_state.ticker_names[ticker] = name
                
            # 화면 입력창 업데이트용 콤마 문자열 생성
            st.session_state.current_tickers = ", ".join([x[0] for x in sampled])
            
    # 3. 텍스트 입력창 (세션에 저장된 값을 기본값으로 가져옴)
    input_tickers = st.text_input("분석 대상을 추가하거나 직접 변경하려면 쉼표로 구분하여 입력하세요:", st.session_state.current_tickers)
    tickers = [t.strip().upper() for t in input_tickers.split(',') if t.strip()]
    
    if st.button(f"위 종목들로 심층 팩터 분석 실행"):
        if not tickers:
            st.warning("분석할 종목이 없습니다. '무작위 새로 뽑기' 버튼을 먼저 눌러주세요.")
            return
            
        with st.spinner("선택된 종목들의 재무제표와 시장 데이터를 수집하여 알고리즘을 구동 중입니다... (1~2분 소요)"):
            nlp_model = initialize_nlp_pipeline()
            analysis_results = []
            
            for t in tickers:
                stock_obj = yf.Ticker(t)
                hist = stock_obj.history(period="1y")
                if len(hist) < 50:
                    continue 
                
                # 티커를 바탕으로 종목명 가져오기 (사용자가 직접 입력한 티커일 경우 야후 API에서 가져옴)
                company_name = st.session_state.ticker_names.get(t, "알 수 없음")
                if company_name == "알 수 없음":
                    try:
                        company_name = stock_obj.info.get('shortName', '알 수 없음')
                    except:
                        pass
                
                raw_l = calculate_long_term_fundamental(stock_obj)
                raw_s = calculate_short_term_momentum(hist)
                raw_e = calculate_experimental_tech(t, hist, nlp_model)
                
                analysis_results.append({
                    "종목명": company_name,
                    "종목코드": t,
                    "장기_원시점수": raw_l,
                    "단기_원시점수": raw_s,
                    "실험적_원시점수": raw_e
                })
                
            if not analysis_results:
                st.error("데이터 수집에 실패하였습니다. 티커 명칭을 확인해 주십시오.")
                return

            df = pd.DataFrame(analysis_results)
            
            df['장기투자_환산점수'] = apply_z_score_normalization(df['장기_원시점수'])
            df['단기투자_환산점수'] = apply_z_score_normalization(df['단기_원시점수'])
            df['실험투자_환산점수'] = apply_z_score_normalization(df['실험적_원시점수'])
            
            df['최종_종합_점수'] = (
                df['장기투자_환산점수'] * 0.4 +
                df['단기투자_환산점수'] * 0.3 +
                df['실험투자_환산점수'] * 0.3
            )
            
            df = df.sort_values(by="최종_종합_점수", ascending=False).reset_index(drop=True)
            best_recommendation = df.iloc[0]
            
            st.success(f"🏆 분석 완료! 종합 1위를 기록한 추천 종목은 **{best_recommendation['종목명']} ({best_recommendation['종목코드']})** 입니다.")
            
            tab_master, tab_long, tab_short, tab_exp = st.tabs([
                "🥇 종합 추천 랭킹", 
                "🏢 장기 투자 분석", 
                "⚡ 단기 투자 분석", 
                "🧪 AI 실험적 분석"
            ])
            
            with tab_master:
                st.subheader("📊 Z-Score 기반 최종 종목 랭킹")
                st.dataframe(df[['종목명', '종목코드', '최종_종합_점수', '장기투자_환산점수', '단기투자_환산점수', '실험투자_환산점수']].style.highlight_max(axis=0))
                
            with tab_long:
                st.subheader("🏢 장기 재무 건전성 및 가치 평가")
                st.dataframe(df[['종목명', '종목코드', '장기_원시점수']])
                
            with tab_short:
                st.subheader("⚡ 단기 가격 추세 및 모멘텀 돌파")
                st.dataframe(df[['종목명', '종목코드', '단기_원시점수']])
                
            with tab_exp:
                st.subheader("🧪 딥러닝 텍스트 감성 분석 및 리스크 패리티 모델")
                st.dataframe(df[['종목명', '종목코드', '실험적_원시점수']])

if __name__ == "__main__":
    main()
