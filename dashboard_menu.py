import numpy as np
import pandas as pd

# 1. 파일 읽기 (merged_df.csv)
df = pd.read_csv(
    './processed_data/merged_df.csv', low_memory=False
)  # DtypeWarning 방지

# Key 컬럼 표준화
if '입항횟수_년도' in df.columns:
  df.rename(columns={'입항횟수_년도': '입항횟수_연도'}, inplace=True)

key_cols = ['호출부호', '선박명', '입항횟수_연도', '입항횟수_횟수']
for k in key_cols:
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
        (summary_df['출항일시'] - summary_df['최종접안대기시점']).dt.total_seconds()
        / 3600.0,
        (
            summary_df['최초출항대기시점'] - summary_df['최종접안대기시점']
        ).dt.total_seconds()
        / 3600.0,
    ],
    default=np.nan,
)

summary_df['대기율_WR'] = np.where(
    (summary_df['서비스시간_TS'].notna()) & (summary_df['서비스시간_TS'] > 0),
    summary_df['대기시간_TW'] / summary_df['서비스시간_TS'],
    0.0,
)

# Streamlit용 결과 데이터 파일 저장
summary_df.to_csv(
    './processed_data/port_waiting_summary.csv',
    index=False,
    encoding='utf-8-sig',
)

# 5. 주요 지표 평균 출력
print('=' * 50)
print('📊 전체 입항 건 기준 주요 지표 평균')
print('=' * 50)

avg_tw_all = summary_df['대기시간_TW'].mean()
avg_tw_waited = summary_df[summary_df['대기시간_TW'] > 0]['대기시간_TW'].mean()
avg_ts = summary_df['서비스시간_TS'].mean()
avg_wr_all = summary_df['대기율_WR'].mean()
avg_wr_waited = summary_df[summary_df['대기시간_TW'] > 0]['대기율_WR'].mean()

print(f'1. 평균 대기시간(TW): {avg_tw_all:.2f} 시간 (전체 입항 건 기준)')
print(f'   └ 대기 발생 건만 기준: {avg_tw_waited:.2f} 시간')
print(f'2. 평균 서비스시간(TS): {avg_ts:.2f} 시간')
print(
    f'3. 평균 대기율(WR): {avg_wr_all * 100:.2f}% ({avg_wr_all:.4f}) (전체'
    ' 기준)'
)
print(f'   └ 대기 발생 건만 기준: {avg_wr_waited * 100:.2f}% ({avg_wr_waited:.4f})')
print('=' * 50)
