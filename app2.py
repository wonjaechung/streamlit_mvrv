import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

st.set_page_config(layout="wide")

# ----------------------
# 초기 데이터 준비
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('mvrv_mock.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    return df

df = load_data()

# ----------------------
# 구간 설정
# ----------------------
zones = [
    (0.0, 0.7), (0.7, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 1.8),
    (1.8, 2.2), (2.2, 2.6), (2.6, 3.4), (3.4, 4.0),
    (4.0, 5.5), (5.5, 7.0), (7.0, 10.0)
]

# ----------------------
# Accuracy 계산 함수
# ----------------------
def calculate_accuracy(df, zone, holding_days, return_thresh, entries=None):
    if entries is None:
        entries = df[(df['MVRV'] >= zone[0]) & (df['MVRV'] < zone[1])].index

    success = 0
    total = 0

    for entry_date in entries:
        exit_date = entry_date + timedelta(days=holding_days)
        if exit_date in df.index:
            entry_price = df.loc[entry_date, 'BTC_Price']
            exit_price = df.loc[exit_date, 'BTC_Price']
            ret = (exit_price - entry_price) / entry_price
            if ret > return_thresh:
                success += 1
            total += 1

    accuracy = success / total if total > 0 else None
    return accuracy, total

def normalize_score_linear(acc):
    return round(acc, 0) if acc is not None else None

# ----------------------
# Streamlit UI
# ----------------------
st.title("📊 구간별 accuracy 계산 자동화")

col1, col2 = st.columns(2)
with col1:
    holding_days = st.slider("보유 기간 (일 단위)", 30, 720, 360, step=30)
with col2:
    return_thresh = st.slider("수익률 임계값 (%)", -20.0, 50.0, 10.0, step=1.0) / 100

# ----------------------
# 첫 번째 테이블: 기본 구간 분석
# ----------------------
st.header("1️⃣ MVRV Range별 Accuracy 분석")
with st.spinner("계산 중..."):
    summary_results = []
    for zone in zones:
        acc, count = calculate_accuracy(df, zone, holding_days=holding_days, return_thresh=return_thresh)
        summary_results.append({
            'MVRV Range': f"{zone[0]} ~ {zone[1]}",
            'Accuracy': round(acc * 100, 2) if acc is not None else None,
            'Count': count
        })
    summary_df = pd.DataFrame(summary_results)
    summary_df['Score'] = summary_df['Accuracy'].apply(normalize_score_linear)
    st.dataframe(summary_df, use_container_width=True)

# ----------------------
# 2 Column layout for 히트맵
# ----------------------
col3, col4 = st.columns(2)

with col3:
    st.header("📆 연도별 MVRV Accuracy 히트맵")
    yearly_results = []
    for year in sorted(df['Year'].unique()):
        entries_in_year = df[df.index.year == year]

        for zone in zones:
            filtered_entries = entries_in_year[(entries_in_year['MVRV'] >= zone[0]) & (entries_in_year['MVRV'] < zone[1])].index
            acc, count = calculate_accuracy(df, zone, holding_days, return_thresh, entries=filtered_entries)
            yearly_results.append({
                'Year': year,
                'MVRV Range': f"{zone[0]} ~ {zone[1]}",
                'Accuracy (%)': round(acc * 100, 2) if acc is not None else None,
                'Count': count
            })

    yearly_accuracy_df = pd.DataFrame(yearly_results)
    pivot_table = yearly_accuracy_df.pivot(index='Year', columns='MVRV Range', values='Accuracy (%)')
    pivot_table = pivot_table.apply(pd.to_numeric, errors='coerce')

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, linecolor='gray', ax=ax1)
    ax1.set_title("MVRV Accuracy by Year (%)")
    st.pyplot(fig1)

with col4:
    st.header("📉 시장 사이클별 MVRV Accuracy 히트맵")
    cycle_periods = {
        "① 2015–2017 Bull'": ("2015-01-01", "2017-12-31"),
        "② 2018 Bear": ("2018-01-01", "2018-12-31"),
        "③ 2019–2021 Bull'": ("2019-01-01", "2021-12-31"),
        "④ 2022 Bear'": ("2022-01-01", "2022-12-31"),
        "⑤ 2023–2024 Bull'": ("2023-01-01", "2024-12-31")
    }

    cycle_results = []
    for label, (start, end) in cycle_periods.items():
        df_cycle = df.loc[start:end]
        for zone in zones:
            acc, count = calculate_accuracy(df_cycle, zone, holding_days, return_thresh)
            cycle_results.append({
                'Cycle': label,
                'MVRV Range': f"{zone[0]} ~ {zone[1]}",
                'Accuracy (%)': round(acc * 100, 2) if acc is not None else None,
                'Count': count
            })

    cycle_df = pd.DataFrame(cycle_results)
    pivot_cycle = cycle_df.pivot(index='Cycle', columns='MVRV Range', values='Accuracy (%)')
    pivot_cycle = pivot_cycle.apply(pd.to_numeric, errors='coerce')

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_cycle, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, linecolor='gray', ax=ax2)
    ax2.set_title("MVRV Accuracy by Cycle (%)")
    st.pyplot(fig2)

