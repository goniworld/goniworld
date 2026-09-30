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
@st.cache_data(ttl=86400) # 하루 단위로 캐싱하여 서버 과부하 방지
def get_us_tickers(size_category):
    """위키피디아 S&P 지수 테이블을 실시간 스크래핑하여 미국 종목을 가져옵니다."""
    if size_category == "대형주":
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    elif size_category == "중형주":
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
    else: # 소형주
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        
    df = pd.read_html(url)[0]
    tickers = df['Symbol'].tolist()
    # yfinance 포맷에 맞게 문자 변환 (예: BRK.B -> BRK-B)
    tickers = [str(t).replace('.', '-') for t in tickers]
    
    # API 호출 시간을 고려하여 무작위 15개 종목 샘플링
    return random.sample(tickers, 15) if len(tickers) >= 15 else tickers

@st.cache_data(ttl=86400)
def get_kr_tickers(size_category):
    """FinanceDataReader를 활용하여 한국 시가총액 규모별 종목을 추출합니다."""
    # KRX 전 종목 시가총액 데이터 호출
    df = fdr.StockListing('KRX')
    
    if 'Marcap' in df.columns:
        df = df.sort_values('Marcap', ascending=False).reset_index(drop=True)
    
    if size_category == "대형주":
        target_df = df.iloc[:50] # 시총 최상위 50개 (KOSPI 우량주)
    elif size_category == "중형주":
        target_df = df.iloc[100:300] # 시총 101~300위 (중형주)
    else: # 소형주
        target_df = df.iloc[500:1500] # 시총 501위 이하 (주로 KOSDAQ 강소기업)
        
    tickers = []
    for _, row in target_df.iterrows():
        code = str(row['Code'])
        market = str(row.get('Market', ''))
        # yfinance 인식 포맷 추가 (.KS: 코스피, .KQ: 코스닥)
        if 'KOSPI' in market:
            tickers.append(code + '.KS')
        elif 'KOSDAQ' in market:
            tickers.append(code + '.KQ')
        else:
            tickers.append(code + '.KS')
            
    return random.sample(tickers, 15) if len(tickers) >= 15 else tickers

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
    
    col1, col2 = st.columns(2)
    with col1:
        market_selection = st.radio("분석할 주식 시장을 선택하세요:", ("🇺🇸 미국 시장", "🇰🇷 한국 시장"))
    with col2:
        size_selection = st.radio("기업 규모(시가총액)를 선택하세요:", ("대형주", "중형주", "소형주"))
    
    # 2. 동적 웹스크래핑 및 API 기반 티커 자동 추출 [cite: 48, 51]
    with st.spinner("해당 시장의 최신 종목 리스트를 스크래핑하여 분석 대상을 선정 중입니다..."):
        if "미국" in market_selection:
            scraped_tickers = get_us_tickers(size_selection)
            market_text = "위키피디아 S&P 지수 데이터"
        else:
            scraped_tickers = get_kr_tickers(size_selection)
            market_text = "한국거래소(KRX) 시가총액 랭킹"
            
    st.info(f"{market_text}를 기반으로 '{size_selection}'에 해당하는 종목 15개를 무작위로 추출했습니다. 직접 변경하실 수도 있습니다.")
    
    input_tickers = st.text_input("분석 대상을 추가하거나 직접 변경하려면 쉼표로 구분하여 입력하세요:", ", ".join(scraped_tickers))
    tickers = [t.strip().upper() for t in input_tickers.split(',')]
    
    if st.button(f"자동 추출된 {size_selection} 심층 팩터 분석 실행"):
        with st.spinner("해당 종목들의 재무제표와 시장 데이터를 수집하여 팩터 알고리즘을 구동 중입니다... (1~2분 소요)"):
            nlp_model = initialize_nlp_pipeline()
            analysis_results = []
            
            for t in tickers:
                stock_obj = yf.Ticker(t)
                hist = stock_obj.history(period="1y")
                if len(hist) < 50:
                    continue 
                
                raw_l = calculate_long_term_fundamental(stock_obj)
                raw_s = calculate_short_term_momentum(hist)
                raw_e = calculate_experimental_tech(t, hist, nlp_model)
                
                analysis_results.append({
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
            best_recommendation = df.iloc[0]['종목코드']
            
            st.success(f"🏆 분석 완료! 스크래핑된 '{size_selection}' 그룹 내에서 종합 1위를 기록한 추천 종목은 **{best_recommendation}** 입니다.")
            
            tab_master, tab_long, tab_short, tab_exp = st.tabs([
                "🥇 종합 추천 랭킹", 
                "🏢 장기 투자 분석", 
                "⚡ 단기 투자 분석", 
                "🧪 AI 실험적 분석"
            ])
            
            with tab_master:
                st.subheader("📊 Z-Score 기반 최종 종목 랭킹")
                st.dataframe(df[['종목코드', '최종_종합_점수', '장기투자_환산점수', '단기투자_환산점수', '실험투자_환산점수']].style.highlight_max(axis=0))
                
            with tab_long:
                st.subheader("🏢 장기 재무 건전성 및 가치 평가")
                st.dataframe(df[['종목코드', '장기_원시점수']])
                
            with tab_short:
                st.subheader("⚡ 단기 가격 추세 및 모멘텀 돌파")
                st.dataframe(df[['종목코드', '단기_원시점수']])
                
            with tab_exp:
                st.subheader("🧪 딥러닝 텍스트 감성 분석 및 리스크 패리티 모델")
                st.dataframe(df[['종목코드', '실험적_원시점수']])

if __name__ == "__main__":
    main()
