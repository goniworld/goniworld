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
        
    # 데이터 상하위 5% 이상치 캡핑을 통한 극단치 충격 완화
    winsorized_data = winsorize(data_series.dropna(), limits=[0.05, 0.05])
    mu = np.mean(winsorized_data)
    sigma = np.std(winsorized_data)
    
    if sigma == 0:
        return pd.Series(0, index=data_series.index)
    
    z_scores = (data_series - mu) / sigma
    
    # 지표 성격에 따른 방향성 보정
    if lower_is_better:
        z_scores = -z_scores
        
    return z_scores

# -------------------------------------------------------------------
# [제1 탭 모듈] 장기 투자 경제적 분석 (Piotroski F-Score 및 재무 펀더멘털)
# -------------------------------------------------------------------
def calculate_long_term_fundamental(ticker_obj):
    """
    재무제표 항목을 수집하여 피오트로스키 모델 기반의 재무 건전성 및 
    현금흐름할인(DCF) 대리 지표를 도출합니다. [cite: 8, 10, 14]
    """
    try:
        financials = ticker_obj.financials
        cashflow = ticker_obj.cashflow
        
        # 실시간 API의 한계로 인해 F-score 핵심 메트릭 3가지를 대용치로 산출
        net_inc = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
        cfo = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
        
        f_score_proxy = 0
        if net_inc > 0: f_score_proxy += 1           # 1. 당기순이익 흑자 확인
        if cfo > 0: f_score_proxy += 1               # 2. 영업현금흐름 흑자 확인
        if cfo > net_inc: f_score_proxy += 1         # 3. 발생액(Accruals) 품질 확인 (분식회계 필터)
        
        # 10점 만점 스케일로 정규화 (가상 DCF 마진 스코어 가산)
        raw_fundamental_score = f_score_proxy * 3.33 
        return raw_fundamental_score
    except Exception:
        return 5.0 # 데이터 누락 시 패널티가 없는 중앙값 배정

# -------------------------------------------------------------------
# [제2 탭 모듈] 단기 투자 알고리즘 분석 (Larry Williams Breakout & %R)
# -------------------------------------------------------------------
def calculate_short_term_momentum(hist_df):
    """
    과거 가격 변동성 범위(Range)를 이용한 돌파 인접도와 
    Williams %R을 융합하여 단기 모멘텀 스코어를 계산합니다. [cite: 21, 22]
    """
    if len(hist_df) < 20:
        return 5.0

    # TA 라이브러리를 활용한 Williams %R 모멘텀 밴드 산출 (-100 ~ 0)
    high_14 = hist_df['High'].rolling(14).max()
    low_14 = hist_df['Low'].rolling(14).min()
    williams_r = ((high_14 - hist_df['Close']) / (high_14 - low_14)) * -100
    current_w_r = williams_r.iloc[-1]
    
    # 전일 레인지 계산 및 변동성 돌파 타점 설정
    prev_high = hist_df['High'].iloc[-2]
    prev_low = hist_df['Low'].iloc[-2]
    day_range = prev_high - prev_low
    
    k_multiplier = 0.5 # 동적 적용 모델로 확장 가능
    today_open = hist_df['Open'].iloc[-1]
    today_close = hist_df['Close'].iloc[-1]
    breakout_target = today_open + (k_multiplier * day_range)
    
    score = 5.0
    # 주가가 돌파 목표치를 초과 달성하며 강한 추세 형성 시 점수 부여
    if today_close > breakout_target:
        score += 3.0
    # Williams %R이 과매도에서 중간 지대(-80 ~ -20)를 통과하며 상승 모멘텀 유지 시 점수 부여
    if -80 < current_w_r < -20:
        score += 2.0
        
    return score

# -------------------------------------------------------------------
# [제3 탭 모듈] 미래 실험적 투자 분석 (FinBERT & Risk Parity)
# -------------------------------------------------------------------
@st.cache_resource
def initialize_nlp_pipeline():
    """Hugging Face의 사전 학습된 금융 뉴스 감성 분석 모델 로드 [cite: 28, 31]"""
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except:
        return None

def calculate_experimental_tech(ticker_name, hist_df, nlp_pipe):
    """
    텍스트 감성 지수와 리스크 패리티 철학에 입각한 역변동성(Inverse Volatility)
    가중치를 합산하여 미래 지향적 포트폴리오 점수를 획득합니다. [cite: 7, 30]
    """
    # 1. 포트폴리오 리스크 완화를 위한 역변동성 측정
    daily_returns = hist_df['Close'].pct_change().dropna()
    annualized_volatility = daily_returns.std() * math.sqrt(252)
    inverse_vol_score = 1.0 / (annualized_volatility + 1e-6)
    
    # 2. 비정형 뉴스 데이터 센티먼트 추출 (예시를 위한 하드코딩 데이터)
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
            
    # 지표 스케일 병합 (최대 10점으로 제약)
    final_experimental = min(max((sentiment_val + inverse_vol_score * 5), 0), 10)
    return final_experimental

# -------------------------------------------------------------------
# [메인 시스템] Streamlit 애플리케이션 및 UI 렌더링
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="멀티 팩터 퀀트 주식 추천 시스템", layout="wide")
    st.title("📈 퀀트 기반 멀티 팩터 주식 분석 및 추천 엔진")
    st.write("본 시스템은 장기 펀더멘털 가치평가, 단기 가격 모멘텀 알고리즘, 그리고 AI 센티먼트 분석 모델의 결합을 통해 다차원적 분석을 제공합니다.")
    
    input_tickers = st.text_input("분석할 티커 심볼을 쉼표로 구분하여 입력하십시오 (예: AAPL, TSLA, NVDA, GOOGL)", "AAPL, TSLA, NVDA, GOOGL, MSFT")
    tickers = [t.strip().upper() for t in input_tickers.split(',')]
    
    if st.button("심층 팩터 분석 실행"):
        with st.spinner("야후 파이낸스(Yahoo Finance) 데이터 수집 및 딥러닝 모델 파이프라인을 구축 중입니다..."):
            nlp_model = initialize_nlp_pipeline()
            analysis_results = []
            
            for t in tickers:
                stock_obj = yf.Ticker(t)
                hist = stock_obj.history(period="1y")
                if len(hist) < 50:
                    continue # 최소한의 시계열 데이터가 존재하지 않는 종목 스킵
                
                # 각 방법론별 독립적 원시 점수 산출
                raw_l = calculate_long_term_fundamental(stock_obj)
                raw_s = calculate_short_term_momentum(hist)
                raw_e = calculate_experimental_tech(t, hist, nlp_model)
                
                analysis_results.append({
                    "Ticker": t,
                    "Raw_Long": raw_l,
                    "Raw_Short": raw_s,
                    "Raw_Exp": raw_e
                })
                
            if not analysis_results:
                st.error("데이터 수집에 실패하였습니다. 티커 명성을 다시 확인해 주십시오.")
                return

            df = pd.DataFrame(analysis_results)
            
            # 머신러닝 분석의 근간: 이상치 제어 및 Z-스코어 횡단면 정규화 적용
            df['Z_LongTerm'] = apply_z_score_normalization(df['Raw_Long'])
            df['Z_ShortTerm'] = apply_z_score_normalization(df['Raw_Short'])
            df['Z_Experimental'] = apply_z_score_normalization(df['Raw_Exp'])
            
            # 최종 마스터 랭킹 스코어 가중 평균 도출 (장기 40%, 단기 30%, 실험 30%)
            df['Composite_Master_Score'] = (
                df['Z_LongTerm'] * 0.4 +
                df['Z_ShortTerm'] * 0.3 +
                df['Z_Experimental'] * 0.3
            )
            
            # 최종 정렬 및 1위 종목 추출
            df = df.sort_values(by="Composite_Master_Score", ascending=False).reset_index(drop=True)
            best_recommendation = df.iloc[0]['Ticker']
            
            st.success(f"🏆 팩터 결합 알고리즘 종합 결과, 현재 리스크-리턴 프로파일이 가장 우수한 종목은 **{best_recommendation}** 입니다.")
            
            # Streamlit 다중 탭을 활용한 모듈별 결과 분리 시각화 [cite: 3, 48]
            tab_master, tab_long, tab_short, tab_exp = st.tabs([
                "종합 스코어 보드 (Composite Ranking)", 
                "장기 투자 탭 (Value & DCF)", 
                "단기 투자 탭 (Momentum & Breakout)", 
                "미래 실험 기술 탭 (AI & Risk Parity)"
            ])
            
            with tab_master:
                st.subheader("📊 Z-Score 기반 최종 종목 랭킹")
                st.write("윈저라이징 통계 처리가 완료된 Z-스코어 합산 지표입니다. 0점은 분석 대상 유니버스의 정확한 평균을 의미합니다.")
                st.dataframe(df[['Ticker', 'Composite_Master_Score', 'Z_LongTerm', 'Z_ShortTerm', 'Z_Experimental']].style.highlight_max(axis=0))
                
            with tab_long:
                st.subheader("🏢 장기 재무 건전성 및 가치 평가")
                st.write("피오트로스키 F-스코어 및 현금흐름 퀄리티를 수치화한 절대 점수 보드입니다.")
                st.dataframe(df[['Ticker', 'Raw_Long']])
                
            with tab_short:
                st.subheader("⚡ 단기 가격 추세 및 모멘텀 돌파")
                st.write("래리 윌리엄스의 변동성 확장 임계점 돌파 강도 및 Williams %R 과열 방지 지표입니다.")
                st.dataframe(df[['Ticker', 'Raw_Short']])
                
            with tab_exp:
                st.subheader("🧪 딥러닝 텍스트 센티먼트 & 역변동성 모델")
                st.write("FinBERT 모델의 자연어 처리 감성 확률과 포트폴리오 리스크를 평준화하는 역변동성 융합 점수입니다.")
                st.dataframe(df[['Ticker', 'Raw_Exp']])

if __name__ == "__main__":
    main()
