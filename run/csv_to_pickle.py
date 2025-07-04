# 처음 한 번만 실행하여 pickle 파일로 저장

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# CSV에서 데이터 읽기
df = pd.read_csv("C:/test/bible_recommend/run/merged_bible.csv")
bible_verses = df.to_dict(orient='records')
verse_texts = [item["verse"] for item in bible_verses]

# 모델 로딩 및 임베딩
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
verse_embeddings = model.encode(verse_texts, convert_to_tensor=True)

# pickle 파일로 저장
with open("C:/test/bible_recommend/run/bible_verses.pkl", "wb") as f:
    pickle.dump(bible_verses, f)

with open("C:/test/bible_recommend/run/verse_embeddings.pkl", "wb") as f:
    pickle.dump(verse_embeddings, f)
