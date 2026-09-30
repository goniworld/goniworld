import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from scipy.stats.mstats import winsorize
from transformers import pipeline
import ta
import math

# -------------------------------------------------------------------
# [유틸리티 함수] 이상치 제어(Winsorization) 및 Z-스코어(Z-Score) 정규화
# -------------------------------------------------------------------
def apply_z_score_normalization(data_series, lower_is_better=False):
    """
    Pandas Series 형태의 입력 데이터를 윈저라이징 후 Z-score 표준화를 거쳐 반환합니다.
    """
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
# [제1 탭 모듈] 장기 투자 경제적 분석 (Piotroski F-Score 및 재무 펀더멘털)
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
        
        raw_fundamental_score = f_score_proxy * 3.33 
        return raw_fundamental_score
    except Exception:
        return 5.0 

# -------------------------------------------------------------------
# [제2 탭 모듈] 단기 투자 알고리즘 분석 (Larry Williams Breakout & %R)
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
    
    k_multiplier = 0.5 
    today_open = hist_df['Open'].iloc[-1]
    today_close = hist_df['Close'].iloc[-1]
    breakout_target = today_open + (k_multiplier * day_range)
    
    score = 5.0
    if today_close > breakout_target:
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
    
    news_headlines = [
        f"Earnings expectation for {ticker_name} surpasses market consensus.",
        f"Federal Reserve rate hike might impact {ticker_name} revenue."
    ]
    
    sentiment_val = 0
    if nlp_pipe:
        sentiments = nlp_pipe(news_headlines)
        for s in sentiments:
            if s['label'] == 'positive': sentiment_val += 2.0
            elif s['label'] == 'negative': sentiment_val -= 2.0
            else: sentiment_val += 0.5
            
    final_experimental = min(max((sentiment_val + inverse_vol_score * 5), 0), 10)
    return final_experimental

# -------------------------------------------------------------------
# [메인 시스템] Streamlit 애플리케이션 및 UI 렌더링
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="규모별 멀티 팩터 주식 추천 시스템", layout="wide")
    st.title("📈 기업 규모 맞춤형(대/중/소형주) 주식 분석 엔진")
    st.write("시장과 시가총액 규모를 선택하면, 해당 체급 내에서 가장 유망한 종목을 다차원 팩터로 선별해 냅니다.")
    
    # 1. 시장 및 시가총액 규모 선택 레이아웃
    col1, col2 = st.columns(2)
    with col1:
        market_selection = st.radio("분석할 주식 시장을 선택하세요:", ("🇺🇸 미국 시장", "🇰🇷 한국 시장"))
    with col2:
        size_selection = st.radio("기업 규모(시가총액)를 선택하세요:", ("🏢 대형주 (Large Cap)", "🏭 중형주 (Mid Cap)", "🏪 소형주 (Small Cap)"))
    
    # 2. 시장 및 규모에 따른 자동 종목 리스트 할당
    if market_selection == "🇺🇸 미국 시장":
        if "대형주" in size_selection:
            st.info("미국 S&P 500 대표 우량주 데이터를 불러옵니다.")
            default_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO']
        elif "중형주" in size_selection:
            st.info("미국 S&P 400 MidCap 대표 중형주 데이터를 불러옵니다.")
            default_tickers = ['WSM', 'CROX', 'YELP', 'TEX', 'ROL', 'LECO', 'MUR', 'MTG', 'DOC', 'WWD']
        else:
            st.info("미국 Russell 2000 대표 소형주 데이터를 불러옵니다.")
            default_tickers = ['CALM', 'WD', 'GPRO', 'SENS', 'PLUG', 'RUN', 'SPWR', 'FSR', 'BB', 'NKLA']
    else:
        if "대형주" in size_selection:
            st.info("한국 코스피 시총 상위 대형주 데이터를 불러옵니다.")
            default_tickers = ['005930.KS', '000660.KS', '373220.KS', '207940.KS', '005380.KS', '000270.KS', '068270.KS', '051910.KS', '035420.KS', '035720.KS']
        elif "중형주" in size_selection:
            st.info("한국 코스피/코스닥 시총 중위권 중형주 데이터를 불러옵니다.")
            default_tickers = ['011070.KS', '078930.KS', '004020.KS', '006280.KS', '010130.KS', '016360.KS', '241560.KQ', '214150.KQ', '036570.KS', '112040.KQ']
        else:
            st.info("한국 코스닥 강소기업 소형주 데이터를 불러옵니다.")
            default_tickers = ['035760.KQ', '096530.KQ', '293490.KQ', '058470.KQ', '041960.KQ', '095660.KQ', '053030.KQ', '048410.KQ', '067280.KQ', '046890.KQ']
    
    # 3. 사용자 수동 수정란
    input_tickers = st.text_input("분석 대상을 추가하거나 직접 변경하려면 쉼표로 구분하여 입력하세요:", ", ".join(default_tickers))
    tickers = [t.strip().upper() for t in input_tickers.split(',')]
    
    if st.button(f"{size_selection} 심층 팩터 분석 실행"):
        with st.spinner("해당 그룹의 시장 데이터를 수집하고 알고리즘을 구동 중입니다... (1~2분 소요)"):
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
            
            # 동일 그룹 내 상대 평가(Z-Score)
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
            
            st.success(f"🏆 분석 완료! 선택하신 '{size_selection}' 그룹 내에서 종합 1위를 기록한 추천 종목은 **{best_recommendation}** 입니다.")
            
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
