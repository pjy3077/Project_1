import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. Streamlit 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="PORT-MIS 대기율 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 2. 데이터 연산 및 캐싱 (캐싱을 적용해야 매번 재연산하지 않아 빠릅니다)
# ---------------------------------------------------------
@st.cache_data
def process_data():
  # 1. 파일 읽기 (merged_df.csv)
  df = pd.read_csv('./processed_data/merged_df.csv', low_memory=False)

  # Key 컬럼 표준화
  if '입항횟수_년도' in df.columns:
    df.rename(columns={'입항횟수_년도': '입항횟수_연도'}, inplace=True)

  key_cols = ['호출부호', '선박명', '입항횟수_연도', '입항횟수_횟수']
  for k in key_cols:
    if k in df.columns:
      df[k] = df[k].astype(str).str.replace('.0', '', regex=False).str.strip()

  # 날짜 변환
  datetime_cols = ['지정일시_FROM', '지정일시_TO', '신청일시', '입항일시', '출항일시']
  for col in datetime_cols:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')

  # 2. 접안/출항대기 구분 및 TW 연산
  df['접안대기_여부'] = (
      df['사용목적명'].astype(str).str.contains('접안대기', na=False)
  )
  df['출항대기_여부'] = (
      df['사용목적명'].astype(str).str.contains('출항대기', na=False)
  )

  df['대기시간_TW'] = np.where(
      df['접안대기_여부'],
      (df['지정일시_TO'] - df['지정일시_FROM']).dt.total_seconds() / 3600.0,
      0.0,
  )
  df['대기시간_TW'] = df['대기시간_TW'].apply(lambda x: max(x, 0.0))

  # 3. 입항 건별 요약 (summary_df) 생성
  berth_times = (
      df[df['접안대기_여부']]
      .groupby(key_cols)['지정일시_TO']
      .max()
      .reset_index()
      .rename(columns={'지정일시_TO': '최종접안대기시점'})
  )
  dept_times = (
      df[df['출항대기_여부']]
      .groupby(key_cols)['지정일시_FROM']
      .min()
      .reset_index()
      .rename(columns={'지정일시_FROM': '최초출항대기시점'})
  )

  summary_df = (
      df.groupby(key_cols)
      .agg({'입항일시': 'min', '출항일시': 'max', '대기시간_TW': 'sum'})
      .reset_index()
  )

  summary_df = pd.merge(summary_df, berth_times, on=key_cols, how='left')
  summary_df = pd.merge(summary_df, dept_times, on=key_cols, how='left')

  # 4. TS (서비스시간) 및 WR (대기율) 산출
  cond1 = (
      summary_df['최종접안대기시점'].isna() & summary_df['최초출항대기시점'].isna()
  )
  cond2 = (
      summary_df['최종접안대기시점'].isna() & summary_df['최초출항대기시점'].notna()
  )
  cond3 = (
      summary_df['최종접안대기시점'].notna() & summary_df['최초출항대기시점'].isna()
  )
  cond4 = (
      summary_df['최종접안대기시점'].notna() & summary_df['최초출항대기시점'].notna()
  )

  summary_df['서비스시간_TS'] = np.select(
      [cond1, cond2, cond3, cond4],
      [
          (summary_df['출항일시'] - summary_df['입항일시']).dt.total_seconds()
          / 3600.0,
          (
              summary_df['최초출항대기시점'] - summary_df['입항일시']
          ).dt.total_seconds()
          / 3600.0,
          (
              summary_df['출항일시'] - summary_df['최종접안대기시점']
          ).dt.total_seconds()
          / 3600.0,
          (
              summary_df['최초출항대기시점'] - summary_df['최종접안대기시점']
          ).dt.total_seconds()
          / 3600.0,
      ],
      default=np.nan,
  )

  summary_df['대기율_WR'] = np.where(
      (summary_df['서비스시간_TS'].notna())
      & (summary_df['서비스시간_TS'] > 0),
      summary_df['대기시간_TW'] / summary_df['서비스시간_TS'],
      0.0,
  )

  return summary_df


# ---------------------------------------------------------
# 3. Streamlit 화면 그리기
# ---------------------------------------------------------
st.title('⚓ 항만 운영 대시보드')
st.caption('PORT-MIS 대기시간, 서비스시간 및 대기율(WR) 모니터링')

# 데이터 로딩 표시
with st.spinner('데이터 연산 중입니다...'):
  try:
    summary_df = process_data()
  except Exception as e:
    st.error(f'데이터 처리 중 오류가 발생했습니다: {e}')
    st.stop()

# 주요 평균 지표 계산
avg_tw_all = summary_df['대기시간_TW'].mean()
avg_tw_waited = summary_df[summary_df['대기시간_TW'] > 0]['대기시간_TW'].mean()
avg_ts = summary_df['서비스시간_TS'].mean()
avg_wr_all = summary_df['대기율_WR'].mean()
avg_wr_waited = summary_df[summary_df['대기시간_TW'] > 0]['대기율_WR'].mean()
total_ships = len(summary_df)

# KPI 카드 출력
st.subheader('📌 주요 지표 요약')
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(label='총 입항 건수', value=f'{total_ships:,} 건')
with col2:
  st.metric(label='평균 대기시간 (TW)', value=f'{avg_tw_all:.2f} 시간')
  st.caption(f'대기발생건: {avg_tw_waited:.2f} 시간')
with col3:
  st.metric(label='평균 서비스시간 (TS)', value=f'{avg_ts:.2f} 시간')
with col4:
  st.metric(label='평균 대기율 (WR)', value=f'{avg_wr_all * 100:.2f} %')
  st.caption(f'대기발생건: {avg_wr_waited * 100:.2f} %')

st.divider()

# 차트 및 테이블
col_left, col_right = st.columns([1, 1])

with col_left:
  st.subheader('📊 대기시간(TW) 분포 (대기 발생 건)')
  tw_waited = summary_df[summary_df['대기시간_TW'] > 0]['대기시간_TW']
  if len(tw_waited) > 0:
    st.bar_chart(tw_waited.value_counts(bins=10))
  else:
    st.info('대기 발생 건이 없습니다.')

with col_right:
  st.subheader('📈 대기시간 상위 선박 Top 5')
  top_waiting = (
      summary_df.sort_values(by='대기시간_TW', ascending=False)
      .head(5)[['선박명', '대기시간_TW', '서비스시간_TS', '대기율_WR']]
      .reset_index(drop=True)
  )
  st.dataframe(top_waiting, use_container_width=True)

st.subheader('📑 요약 데이터 전체 보기')
st.dataframe(summary_df, use_container_width=True)