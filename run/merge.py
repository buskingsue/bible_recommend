import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('bible.csv', encoding='utf-8')

# 'reference' 컬럼 생성: 예) "창세기 1:1"
df['reference'] = df['book'] + ' ' + df['chapter'].astype(str) + ':' + df['verse'].astype(str)

# 필요한 열만 선택
df_merged = df[['reference', 'content']].rename(columns={'content': 'verse'})

# 새로운 CSV 파일로 저장
df_merged.to_csv('merged_bible.csv', index=False, encoding='utf-8-sig')

# 딕셔너리 리스트로 변환 (원래 목적에 사용하려면 유지)
bible_verses = df_merged.to_dict(orient='records')
