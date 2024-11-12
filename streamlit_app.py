import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pytrends.request import TrendReq


# 제목 표시
st.title("Data Visualization GUI")

# 각 헤더 배경색 (연두색)만 적용
st.markdown("""
    <style>
        /* 연구 현황 헤더 배경색 (연두색) */
        .header-container {
            background-color: #90EE90; /* 연두색 */
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        /* 한-미 협력 헤더 배경색 (연두색) */
        .header-container-usa {
            background-color: #90EE90; /* 연두색 */
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        /* 우수 연구자, 저널 헤더 배경색 (연두색) */
        .header-container-excellence {
            background-color: #90EE90; /* 연두색 */
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        /* 구글 트렌드 헤더 배경색 (연두색) */
        .header-container-trend {
            background-color: #90EE90; /* 연두색 */
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# 페이지 선택
page = st.sidebar.radio("페이지 선택", ("연구 현황", "한-미 협력", "우수 연구자, 저널", "구글 트렌드"))

if page == "연구 현황":

  # 연구 현황 헤더에 배경색 추가 (연두색)
  st.markdown('<div class="header-container"><h2>연구 현황</h2></div>', unsafe_allow_html=True)



  # 첫 번째 기능: 전체 논문 수
  st.header("Total Publications by Year")

  uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (EX : ToppublicationyearsinPublications)")
  if uploaded_file is not None:
      try:
          df = pd.read_csv(uploaded_file, skiprows=12, skipfooter=3, engine='python')
          df = df[['Publication years', 'Publications']]
          df['Publication years'] = df['Publication years'].astype(int)
          df['Publications'] = df['Publications'].astype(int)

          # Publication years를 오름차순으로 정렬
          df = df.sort_values('Publication years', ascending=True)

          min_val = df['Publications'].min() * 0.9
          max_val = df['Publications'].max() * 1.1

          fig, ax = plt.subplots(figsize=(12, 6))
          ax.plot(df['Publication years'], df['Publications'], color='blue', marker='o', linestyle='-', label='Total Publications')
          ax.set_ylim(min_val, max_val)
          ax.set_title('Total Publications by Year', fontsize=16)
          ax.set_xlabel('Publication Year', fontsize=14)
          ax.set_ylabel('Number of Publications', fontsize=14)
          ax.grid(axis='x')
          ax.legend()
          plt.xticks(df['Publication years'], rotation=0)

          st.pyplot(fig)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")


  # 첫 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EC%97%B0%EB%8F%84%EB%B3%84%20%EB%85%BC%EB%AC%B8%20%EC%88%98%20%EC%B6%94%EC%9D%B4.webm")


  # 두 번째 기능: 국가별 Scholarly Output 시각화
  st.header("Top N Countries by Scholarly Output and FWCI")

  uploaded_file_scholarly = st.file_uploader("CSV 파일을 업로드하세요 (EX : Countries_Regions_-_Heterogeneous_integration_-_Table-view-selected-metrics)")
  if uploaded_file_scholarly is not None:
      try:
          df = pd.read_csv(uploaded_file_scholarly, skiprows=19, skipfooter=3, engine='python')
          df = df[['Country/Region', 'Scholarly Output', 'Field-Weighted Citation Impact']]

          # 평균 논문 수 계산
          average_output = df['Scholarly Output'].mean()

          # 평균보다 큰 논문 수를 가진 국가들만 필터링
          df_filtered = df[df['Scholarly Output'] > average_output]

          # 한국의 Scholarly Output 및 FWCI 찾기
          korea_row = df_filtered[df_filtered['Country/Region'] == 'South Korea']
          korea_output = korea_row['Scholarly Output'].values[0] if not korea_row.empty else None
          korea_fwci = korea_row['Field-Weighted Citation Impact'].values[0] if not korea_row.empty else None

          # Scholarly Output 상위 N개국 선택 (한국 포함)
          n = 10
          while n <= len(df_filtered):
              top_scholarly_output = df_filtered.nlargest(n, 'Scholarly Output')
              if korea_output in top_scholarly_output['Scholarly Output'].values:
                  break
              n += 1

          # Field-Weighted Citation Impact 상위 N개국 선택 (한국 포함)
          m = 10
          while m <= len(df_filtered):
              top_fwci = df_filtered.nlargest(m, 'Field-Weighted Citation Impact')
              if korea_fwci in top_fwci['Field-Weighted Citation Impact'].values:
                  break
              m += 1

          # 첫 번째 그래프 설정: Scholarly Output 상위 N개국
          fig, ax1 = plt.subplots(figsize=(10, 6))
          for index, country in enumerate(top_scholarly_output['Country/Region']):
              if country == 'South Korea':
                  bar = ax1.barh(country, top_scholarly_output['Scholarly Output'].iloc[index],
                                color='lightyellow', hatch='//', edgecolor='black', linewidth=1.5)
              else:
                  ax1.barh(country, top_scholarly_output['Scholarly Output'].iloc[index], color='gray')

          ax1.set_xlabel('Scholarly Output', fontsize=14)
          ax1.set_title(f'Top {n} Countries by Scholarly Output', fontsize=16)
          ax1.invert_yaxis()
          ax1.grid(axis='x')
          st.pyplot(fig)

          # 두 번째 그래프 설정: Field-Weighted Citation Impact 상위 N개국
          fig, ax2 = plt.subplots(figsize=(10, 6))
          for index, country in enumerate(top_fwci['Country/Region']):
              if country == 'South Korea':
                  bar = ax2.barh(country, top_fwci['Field-Weighted Citation Impact'].iloc[index],
                                color='lightyellow', hatch='//', edgecolor='black', linewidth=1.5)
              else:
                  ax2.barh(country, top_fwci['Field-Weighted Citation Impact'].iloc[index], color='gray')

          ax2.set_xlabel('Field-Weighted Citation Impact', fontsize=14)
          ax2.set_title(f'Top {m} Countries by Field-Weighted Citation Impact', fontsize=16)
          ax2.invert_yaxis()
          ax2.grid(axis='x')
          st.pyplot(fig)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 두 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EA%B5%AD%EA%B0%80%EB%B3%84%20%EB%85%BC%EB%AC%B8%20%EC%88%98,%20FWCI%20%EC%83%81%EC%9C%84%EA%B5%AD%EA%B0%80.webm")



  # 세 번째 기능: 상위 N개 기관 시각화
  st.header("Top N Institutions by Scholarly Output and FWCI (KOREA)")

  uploaded_file_institution = st.file_uploader("CSV 파일을 업로드하세요 (EX : Institutions_-_Heterogeneous_integration_-_Table-view-selected-metrics)")

  # 사용자가 표시할 상위 N개 기관 수를 입력
  num_top_institutions = st.number_input("상위 몇 개의 기관을 표시할까요?", min_value=1, value=10)

  if uploaded_file_institution is not None:
      try:
          df = pd.read_csv(uploaded_file_institution, skiprows=20, skipfooter=3, engine='python')
          df = df[['Institution', 'Scholarly Output', 'Field-Weighted Citation Impact']]
          df['Scholarly Output'] = df['Scholarly Output'].astype(float)
          df['Field-Weighted Citation Impact'] = df['Field-Weighted Citation Impact'].astype(float)

          # 사용자가 입력한 수만큼 상위 기관을 선택
          top_institutions = df.sort_values(['Scholarly Output', 'Field-Weighted Citation Impact'], ascending=[False, False]).head(num_top_institutions)

          fig, ax1 = plt.subplots(figsize=(12, 6))
          ax1.set_ylabel('Institution')
          ax1.set_xlabel('Scholarly Output', fontsize=14)
          ax1.barh(top_institutions['Institution'], top_institutions['Scholarly Output'], color='tab:blue', label='Scholarly Output')
          ax1.invert_yaxis()

          ax2 = ax1.twiny()
          ax2.set_xlabel('Field-Weighted Citation Impact', color='tab:orange', fontsize=14)
          ax2.plot(top_institutions['Field-Weighted Citation Impact'], top_institutions['Institution'], color='tab:orange', marker='o', linewidth=2)

          plt.title(f'Top {num_top_institutions} Institutions by Scholarly Output and FWCI', fontsize=16)
          ax1.grid(axis='x')
          st.pyplot(fig)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 세 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%ED%95%9C%EA%B5%AD%20%EC%83%81%EC%9C%84%2010%EA%B8%B0%EA%B4%80%20%EB%85%BC%EB%AC%B8%20%EC%88%98,%20FWCI%20+%20%ED%95%9C%EA%B5%AD%20%EA%B8%B0%EA%B4%80%20%EA%B5%AD%EC%A0%9C%20%ED%98%91%EB%A0%A5.webm")


elif page == "한-미 협력":

  # 한-미 협력 헤더에 배경색 추가 (연두색)
  st.markdown('<div class="header-container-usa"><h2>한-미 협력</h2></div>', unsafe_allow_html=True)

  # 네 번째 기능: 미국, 한국, 공동 출판물 수 시각화
  st.header("US, Korean, and Joint US-KR Publications")
  uploaded_us = st.file_uploader("미국 논문 수 CSV 파일을 업로드하세요, EX : ToppublicationyearsinPublicationsintheUnitedStates")
  uploaded_kr = st.file_uploader("한국 논문 수 CSV 파일을 업로드하세요, EX : ToppublicationyearsinPublicationsinSouthKorea")
  uploaded_joint = st.file_uploader("미국-한국 공동 출판물 수 CSV 파일을 업로드하세요, EX : ToppublicationyearsinPublicationsinSouthKorea (1)")

  if uploaded_us is not None and uploaded_kr is not None and uploaded_joint is not None:
      try:
          df_us = pd.read_csv(uploaded_us, skiprows=12, skipfooter=3, engine='python')
          df_kr = pd.read_csv(uploaded_kr, skiprows=12, skipfooter=3, engine='python')
          df_joint = pd.read_csv(uploaded_joint, skiprows=12, skipfooter=3, engine='python')

          df_us = df_us[['Publication years', 'Publications']]
          df_kr = df_kr[['Publication years', 'Publications']]
          df_joint = df_joint[['Publication years', 'Publications']]

          for df in [df_us, df_kr, df_joint]:
              df['Publication years'] = df['Publication years'].astype(int)
              df['Publications'] = df['Publications'].astype(int)

          df_merged_us = pd.merge(df_us, df_joint, on='Publication years', how='outer', suffixes=('_US', '_Joint'))
          df_merged = pd.merge(df_merged_us, df_kr, on='Publication years', how='outer', suffixes=('', '_KR'))

          df_merged.fillna(0, inplace=True)

          # Publication years 기준으로 오름차순 정렬
          df_merged = df_merged.sort_values('Publication years', ascending=True)

          max_value = max(df_merged['Publications_US'].max(), df_merged['Publications'].max(), df_merged['Publications_Joint'].max())
          y_max = max_value * 1.3

          # 그래프 설정
          fig, ax = plt.subplots(figsize=(12, 6))

          # x축 위치 설정
          x = np.arange(len(df_merged['Publication years']))
          width = 0.2  # 막대 너비

          # 각 논문 수에 대해 막대 설정
          bars_us = ax.bar(x - width, df_merged['Publications_US'], width, color='lightblue', label='US Publications')
          bars_kr = ax.bar(x, df_merged['Publications'], width, color='lightgreen', label='Korean Publications')
          bars_joint = ax.bar(x + width, df_merged['Publications_Joint'], width, color='lightcoral', label='Joint US-KR Publications')

          # 막대 위에 논문 건수 표시
          for i in range(len(df_merged)):
              ax.text(x[i] - width, df_merged['Publications_US'][i] + 2,
                      f"{int(df_merged['Publications_US'][i])}", ha='center', va='bottom', color='black', fontsize=14)
              ax.text(x[i], df_merged['Publications'][i] + 2,
                      f"{int(df_merged['Publications'][i])}", ha='center', va='bottom', color='black', fontsize=14)
              ax.text(x[i] + width, df_merged['Publications_Joint'][i] + 2,
                      f"{int(df_merged['Publications_Joint'][i])}", ha='center', va='bottom', color='black', fontsize=14)

          # y축 최대값 설정
          ax.set_ylim(0, y_max)

          # x축 연도 레이블 설정
          ax.set_xticks(x)
          ax.set_xticklabels(df_merged['Publication years'], rotation=0)
          ax.set_xlabel('Publication Year')
          ax.set_ylabel('Number of Publications')
          ax.set_title('US, Korean, and Joint US-KR Publications by Year')
          ax.legend()

          plt.tight_layout()

          # 그래프를 보여줍니다.
          st.pyplot(fig)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 네 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/(%EB%AF%B8%EA%B5%AD,%20%ED%95%9C%EA%B5%AD,%20%EA%B3%B5%EB%8F%99)%20%EB%85%BC%EB%AC%B8%20%EC%88%98.webm")



  # 다섯 번째 기능
  st.header("Proportion of International Collaboration in US Publications")

  # 파일 업로드 (미국 연도별 논문 수)
  uploaded_us = st.file_uploader("미국 연도별 논문 수 CSV 파일을 업로드하세요, EX : ToppublicationyearsinPublicationsintheUnitedStates")
  # 파일 업로드 (미국과 한국의 공동 연도별 논문 수)
  uploaded_joint = st.file_uploader("미국과 한국의 공동 연도별 논문 수 CSV 파일을 업로드하세요, EX : ToppublicationyearsinPublicationsinSouthKorea (1)")
  # 파일 업로드 (국제 협력 건수)
  uploaded_collab = st.file_uploader("미국의 국제 협력 건수 CSV 파일을 업로드하세요, EX : Heterogeneous+integration_-_International+Collaboration_by_year_for_selected_Countries_Regions")

  if uploaded_us is not None and uploaded_joint is not None and uploaded_collab is not None:
      try:
          # 데이터 읽기 (미국 연도별 논문 수)
          df_us = pd.read_csv(uploaded_us, skiprows=12, skipfooter=3, engine='python')
          df_us = df_us[['Publication years', 'Publications']]

          # 데이터 읽기 (미국과 한국의 공동 연도별 논문 수)
          df_joint = pd.read_csv(uploaded_joint, skiprows=12, skipfooter=3, engine='python')
          df_joint = df_joint[['Publication years', 'Publications']]

          # 데이터 읽기 (국제 협력 건수)
          df_collab = pd.read_csv(uploaded_collab, skiprows=16, skipfooter=1, engine='python')

          # 데이터 변환 (행과 열을 전환)
          df_collab_transposed = df_collab.set_index('Country/Region').T
          df_collab_transposed = df_collab_transposed[['United States']]  # 미국 데이터만 선택

          # 인덱스를 열로 변환하고 'Publication years'로 설정
          df_collab_transposed.reset_index(inplace=True)
          df_collab_transposed.rename(columns={'index': 'Publication years', 'United States': 'International Collaboration Publications'}, inplace=True)

          # 데이터 타입 변환
          df_us['Publication years'] = df_us['Publication years'].astype(int)
          df_us['Publications'] = df_us['Publications'].astype(int)

          df_joint['Publication years'] = df_joint['Publication years'].astype(int)
          df_joint['Publications'] = df_joint['Publications'].astype(int)

          df_collab_transposed['Publication years'] = df_collab_transposed['Publication years'].astype(int)
          df_collab_transposed['International Collaboration Publications'] = df_collab_transposed['International Collaboration Publications'].astype(int)

          # 데이터 병합
          df_merged = pd.merge(df_us, df_joint, on='Publication years', how='outer', suffixes=('_US', '_Joint'))
          df_merged = pd.merge(df_merged, df_collab_transposed, on='Publication years', how='left')

          # 결측치를 0으로 대체
          df_merged.fillna(0, inplace=True)

          # Publication years 기준으로 오름차순 정렬
          df_merged = df_merged.sort_values('Publication years', ascending=True)

          # 첫 번째 그래프: 미국 전체 논문 중 국제 협력 논문이 차지하는 비율
          plt.figure(figsize=(12, 6))
          bar_width = 0.4  # 바 너비
          x = df_merged['Publication years']

          plt.bar(x, df_merged['Publications_US'], width=bar_width, color='gray', edgecolor='black', label='US Publications')
          plt.bar(x, df_merged['International Collaboration Publications'], width=bar_width, color='white', edgecolor='black', label='International Collaboration Publications', hatch='//', alpha=0.7)

          # 비율을 계산하고 텍스트로 추가
          for i in range(len(df_merged)):
              total_pub = df_merged['Publications_US'][i]
              collab_pub = df_merged['International Collaboration Publications'][i]
              if total_pub > 0:
                  ratio = (collab_pub / total_pub) * 100
                  plt.text(x[i], collab_pub + (total_pub * 0.01), f'{ratio:.1f}%', ha='center', va='bottom', color='black', fontsize=11)

          plt.title('Proportion of International Collaboration in US Publications')
          plt.xlabel('Publication Year')
          plt.ylabel('Number of Publications')
          plt.legend()
          plt.grid(axis='y')
          plt.tight_layout()
          st.pyplot(plt)

          # 두 번째 그래프: 국제 협력 중에서 한국과의 협력 논문이 차지하는 비율
          plt.figure(figsize=(12, 6))
          plt.bar(x, df_merged['International Collaboration Publications'], width=bar_width, color='gray', edgecolor='black', label='International Collaboration Publications')
          plt.bar(x, df_merged['Publications_Joint'], width=bar_width, color='white', edgecolor='black', label='US-Korea Joint Publications', hatch='//')

          # 비율을 계산하고 텍스트로 추가
          for i in range(len(df_merged)):
              total_collab = df_merged['International Collaboration Publications'][i]
              joint_pub = df_merged['Publications_Joint'][i]
              if total_collab > 0:
                  ratio_joint = (joint_pub / total_collab) * 100
                  plt.text(x[i], joint_pub + (total_collab * 0.01), f'{ratio_joint:.1f}%', ha='center', va='bottom', color='black', fontsize=11)

          plt.title('Proportion of US-Korea Joint Publications in International Collaborations')
          plt.xlabel('Publication Year')
          plt.ylabel('Number of Publications')
          plt.legend()
          plt.grid(axis='y')
          plt.tight_layout()
          st.pyplot(plt)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 다섯 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EB%AF%B8%EA%B5%AD%20%EA%B5%AD%EC%A0%9C%20%ED%98%91%EB%A0%A5%20%ED%98%84%ED%99%A9.webm")



  # 여섯 번째 기능: 미국과 협력한 국가들의 상위 N개 논문 수 시각화

  import seaborn as sns

  st.header("Top N Countries by Number of Publications in Collaboration with the United States")
  uploaded_file_countries = st.file_uploader("CSV 파일을 업로드하세요 (EX : TopcountriesregionsinPublicationsintheUnitedStates)")

  # 사용자가 표시할 상위 N개 국가 수를 입력
  num_top_countries = st.number_input("상위 몇 개의 국가를 표시할까요?", min_value=1, value=10)

  if uploaded_file_countries is not None:
      try:
          df = pd.read_csv(uploaded_file_countries, skiprows=12, skipfooter=3, engine='python')
          df = df[['Countries/Regions', 'Publications']]
          df['Publications'] = pd.to_numeric(df['Publications'], errors='coerce')
          df = df[df['Countries/Regions'] != 'United States']  # 미국 제외

          top_n = df.nlargest(num_top_countries, 'Publications')  # 상위 N개 국가 선택

          # Seaborn 스타일 적용
          plt.figure(figsize=(14, 8))
          barplot = sns.barplot(
              x='Publications',
              y='Countries/Regions',
              data=top_n,
              palette='viridis',  # 색상 팔레트 변경
              edgecolor='black'
          )

          # 제목 및 축 레이블 설정
          barplot.set_title(f'Top {num_top_countries} Countries by Number of Publications in Collaboration with the United States', fontsize=18, fontweight='bold')
          barplot.set_xlabel('Number of Publications', fontsize=14)
          barplot.set_ylabel('Country/Region', fontsize=14)

          # 막대 위에 값 표시
          for i in barplot.containers:
              barplot.bar_label(i, fmt='%.0f', label_type='edge', fontsize=12)

          # y축 내부 그리드 선 제거 (y축 그리드 선을 명시적으로 제거)
          barplot.yaxis.grid(False)  # y축 그리드 제거
          plt.grid(axis='x')  # x축 그리드 유지

          # 레이아웃 최적화
          plt.tight_layout()
          st.pyplot(plt)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 여섯 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EB%AF%B8%EA%B5%AD%EA%B3%BC%EC%9D%98%20%EC%83%81%EC%9C%84%20%ED%98%91%EB%A0%A5%20%EA%B5%AD%EA%B0%80.webm")



  # 일곱 번째 기능: 상위 N개 기관의 학술 출력 및 국제 협력 비율 시각화
  st.header("Top N Institutions by Scholarly Output with International Collaboration")
  uploaded_file_institutions = st.file_uploader("CSV 파일을 업로드하세요 (EX : Institutions_-_Heterogeneous_integration_-_Table-view-selected-metrics (1))")

  # 사용자가 표시할 상위 N개 기관 수를 입력
  num_top_institutions = st.number_input("상위 몇 개의 기관을 표시할까요?", min_value=1, value=10)

  if uploaded_file_institutions is not None:
      try:
          # 데이터 읽기 (21행부터 시작하고 마지막 3행을 건너뜁니다.)
          df_institutions = pd.read_csv(uploaded_file_institutions, skiprows=20, skipfooter=3, engine='python')

          # 필요한 열만 선택
          df_institutions = df_institutions[['Institution', 'Scholarly Output', 'International Collaboration', 'Field-Weighted Citation Impact']]

          # 데이터 타입 변환
          df_institutions['Scholarly Output'] = df_institutions['Scholarly Output'].astype(float)
          df_institutions['International Collaboration'] = df_institutions['International Collaboration'].astype(float)
          df_institutions['Field-Weighted Citation Impact'] = df_institutions['Field-Weighted Citation Impact'].astype(float)

          # Scholarly Output으로 내림차순 정렬, 값이 같다면 FWCI로 내림차순 정렬
          top_n_institutions = df_institutions.sort_values(['Scholarly Output', 'Field-Weighted Citation Impact'],
                                                            ascending=[False, False]).head(num_top_institutions)

          # International Collaboration 비율 계산
          top_n_institutions['International Collaboration Ratio (%)'] = (
              top_n_institutions['International Collaboration'] / top_n_institutions['Scholarly Output'] * 100
          )

          # 그래프 그리기
          plt.figure(figsize=(14, 8))

          # 전체 논문 수 막대 그래프
          plt.barh(top_n_institutions['Institution'], top_n_institutions['Scholarly Output'],
                  color='lightgray', label='Total Scholarly Output')

          # 국제 협력 논문 수 덧칠
          bars = plt.barh(top_n_institutions['Institution'], top_n_institutions['International Collaboration'],
                          color='skyblue', label='International Collaboration')

          # 각 막대의 오른쪽에 국제 협력 비율 텍스트로 표시
          for i in range(len(top_n_institutions)):
              international_collab = top_n_institutions['International Collaboration'].iloc[i]
              # 바의 끝 바로 오른쪽에 위치
              plt.text(international_collab, bars[i].get_y() + bars[i].get_height() / 2,
                      f"{top_n_institutions['International Collaboration Ratio (%)'].iloc[i]:.1f}%",  # 비율 텍스트
                      va='center', ha='left', color='black', fontsize=15)

          # 그래프 설정
          plt.xlabel('Number of Publications', fontsize=14)
          plt.ylabel('Institution', fontsize=14)
          plt.title(f'Top {num_top_institutions} Institutions by Scholarly Output with International Collaboration', fontsize=16)
          plt.legend(fontsize=12)
          plt.gca().invert_yaxis()  # y축 반전하여 높은 값이 위로 오도록 설정
          plt.grid(axis='x')

          # 그래프를 보여줍니다.
          plt.tight_layout()
          st.pyplot(plt)

      except Exception as e:
          st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

  # 일곱 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EB%AF%B8%EA%B5%AD%20%EA%B8%B0%EA%B4%80%20%EA%B5%AD%EC%A0%9C%20%ED%98%91%EB%A0%A5.webm")


elif page == "우수 연구자, 저널":

  # 우수 연구자, 저널 헤더에 배경색 추가 (연두색)
  st.markdown('<div class="header-container-excellence"><h2>우수 연구자, 저널</h2></div>', unsafe_allow_html=True)

  # 여덟 번째 기능: 우수 연구자
  st.header("Top N Authors")
  uploaded_file_authors = st.file_uploader("CSV 파일을 업로드하세요 (EX : Authors_-_Heterogeneous_integration_-_Table-view-selected-metrics)", key="author_uploader")

  # 사용자에게 표시할 상위 N개 저자 수 입력
  num_top_authors = st.number_input("상위 몇 명의 저자를 표시할까요?", min_value=1, value=10)

  if uploaded_file_authors is not None:
      # 데이터 읽기 (21행부터 시작하고 마지막 3행을 건너뜁니다.)
      df = pd.read_csv(uploaded_file_authors, skiprows=20, skipfooter=3, engine='python')

      # 필요한 열 선택
      df = df[['Author', 'Affiliation', 'Scholarly Output', 'Field-Weighted Citation Impact']]

      # Scholarly Output을 기준으로 내림차순 정렬하고, 같을 경우 Field-Weighted Citation Impact로 정렬
      df_sorted = df.sort_values(by=['Scholarly Output', 'Field-Weighted Citation Impact'], ascending=[False, False])

      # 상위 N개 선택
      top_n_authors = df_sorted.head(num_top_authors)

      # 데이터프레임을 웹 페이지에 표시
      st.write("Top Sorted Authors Data")
      st.dataframe(top_n_authors)

      # 전체 데이터프레임을 사용하여 각 Affiliation별 저자 수 계산
      affiliation_counts = df['Affiliation'].value_counts()

      # 기관별 연구자 수를 데이터프레임으로 변환
      affiliation_df = affiliation_counts.reset_index()
      affiliation_df.columns = ['Affiliation', 'Number of Authors']

      # 전체 기관 수를 표로 표시
      st.write("Number of Authors by Affiliation")
      st.dataframe(affiliation_df)

      # 상위 5개 기관의 연구자 수만 선택
      top_5_affiliations = affiliation_counts.head(5)

      # 시각화: 상위 5개 기관의 연구자 수를 수평 바 차트로 표시
      st.write("Top 5 Affiliations by Number of Authors")
      plt.figure(figsize=(10, 5))
      top_5_affiliations.plot(kind='barh', color='skyblue')  # 수평 바 차트
      plt.title('Top 5 Affiliations by Number of Authors')
      plt.xlabel('Number of Authors')
      plt.ylabel('Affiliation')
      plt.gca().invert_yaxis()  # y축의 값이 높은 것부터 표시
      st.pyplot(plt)

  # 여덟 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EB%AF%B8%EA%B5%AD,%20%ED%95%9C%EA%B5%AD%20%EC%9A%B0%EC%88%98%20%EC%97%B0%EA%B5%AC%EC%9E%90.webm")



  # 아홉 번째 기능: 주요 저널
  st.header("Top N Scopus Sources")

  # 파일 업로드
  uploaded_file_journals = st.file_uploader("CSV 파일을 업로드하세요 (EX : Scopus+Sources_-_Heterogeneous_integration_-_Table-view-selected-metrics)", key="journal_uploader")

  # 사용자가 표시할 상위 N개 저널 수를 입력
  num_top_journals = st.number_input("상위 몇 개의 저널을 표시할까요?", min_value=1, value=5)

  if uploaded_file_journals is not None:
      # 데이터 읽기 (21행부터 시작하고 마지막 3행을 건너뜁니다.)
      df = pd.read_csv(uploaded_file_journals, skiprows=20, skipfooter=3, engine='python')

      # 필요한 열 선택
      df = df[['Scopus Source', 'Scholarly Output', 'Field-Weighted Citation Impact']]

      # Scholarly Output을 기준으로 내림차순 정렬하고, 같을 경우 Field-Weighted Citation Impact로 정렬
      df_sorted = df.sort_values(by=['Scholarly Output', 'Field-Weighted Citation Impact'], ascending=[False, False])

      # 상위 N개 선택
      top_n_journals = df_sorted.head(num_top_journals)

      # 데이터프레임을 웹 페이지에 표시
      st.write("Top Sorted Journals Data")
      st.dataframe(top_n_journals)


  # 아홉 번째 기능에 대한 동영상 삽입
  st.video("https://github.com/dohy-choi/SciVal/raw/refs/heads/main/%EB%AF%B8%EA%B5%AD,%20%ED%95%9C%EA%B5%AD%20%EC%A3%BC%EC%9A%94%20%EC%A0%80%EB%84%90.webm")


elif page == "구글 트렌드":

  # 구글 트렌드 헤더에 배경색 추가 (연두색)
  st.markdown('<div class="header-container-trend"><h2>구글 트렌드</h2></div>', unsafe_allow_html=True)

  # 사용자 입력: 키워드
  keyword = st.text_input("구글 트렌드 키워드를 입력하세요 (예: Semiconductor)")

  if keyword:
      # 구글 트렌드 데이터 수집 (pytrends 사용)
      pytrends = TrendReq(hl='en-US', tz=360)
      pytrends.build_payload([keyword], cat=0, timeframe='today 5-y', geo='', gprop='')
      trends_data = pytrends.interest_over_time()

      # 'isPartial' 컬럼 제거
      trends_data = trends_data.drop(columns=['isPartial'])

      # 데이터프레임 확인
      st.write("전체 데이터:")
      st.dataframe(trends_data)

      # 시간에 따른 관심도 시각화
      st.write(f"{keyword}에 대한 구글 트렌드 관심도 변화 (5년간)")
      plt.figure(figsize=(12, 6))
      plt.plot(trends_data.index, trends_data[keyword], label=f'{keyword} Interest', color='b')
      plt.title(f"Google Trends Interest Over Time for '{keyword}'")
      plt.xlabel("Date")
      plt.ylabel("Interest Level")
      plt.legend()
      plt.grid(True)
      st.pyplot(plt)

      # 연도별 관심도 비교
      trends_data['Year'] = trends_data.index.year
      yearly_avg = trends_data.groupby('Year')[keyword].mean()
      st.write("연도별 평균 관심도 변화")
      plt.figure(figsize=(12, 6))
      plt.bar(yearly_avg.index, yearly_avg, color='skyblue')
      plt.title(f"Yearly Average Interest Level for '{keyword}'")
      plt.xlabel("Year")
      plt.ylabel("Average Interest Level")
      st.pyplot(plt)


      # 국가별 관심도 시각화 (Top 10 국가만)
      st.write("국가별 관심도 비교 (Top 10)")
      pytrends.build_payload([keyword], cat=0, timeframe='today 5-y', geo='', gprop='')
      geo_data = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
      geo_data = geo_data[geo_data[keyword] > 0]  # 0 관심도 제외
      top_10_countries = geo_data.sort_values(by=keyword, ascending=False).head(10)
      plt.figure(figsize=(14, 8))
      sns.heatmap(top_10_countries, annot=True, cmap="YlGnBu", cbar=True)
      plt.title(f"Top 10 Countries for '{keyword}' Interest")
      plt.xlabel("Interest Level")
      st.pyplot(plt)


      # 키워드 비교 기능
      keyword2 = st.text_input("비교할 키워드를 입력하세요 (예: Electronics)")
      if keyword2:
          pytrends.build_payload([keyword, keyword2], cat=0, timeframe='today 5-y', geo='', gprop='')
          compare_data = pytrends.interest_over_time()
          compare_data = compare_data.drop(columns=['isPartial'])
          st.write(f"'{keyword}'와 '{keyword2}' 관심도 비교 (5년간)")
          plt.figure(figsize=(12, 6))
          plt.plot(compare_data.index, compare_data[keyword], label=keyword, color='b')
          plt.plot(compare_data.index, compare_data[keyword2], label=keyword2, color='r')
          plt.title("Google Trends Interest Comparison")
          plt.xlabel("Date")
          plt.ylabel("Interest Level")
          plt.legend()
          st.pyplot(plt)
          

