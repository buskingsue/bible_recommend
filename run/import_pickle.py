# 필요한 라이브러리를 불러옵니다.
import streamlit as st  # Streamlit: 웹 앱 개발용
from sentence_transformers import SentenceTransformer, util  # 문장 임베딩 및 유사도 계산
import pickle  # pickle: 파이썬 객체를 파일로 저장하거나 불러올 수 있게 해줌

# pickle 파일에서 성경 구절 리스트를 불러옵니다.
with open("C:/test/bible_recommend/run/bible_verses.pkl", "rb") as f:
    bible_verses = pickle.load(f)  # [{'reference': ..., 'verse': ...}, ...] 형식의 리스트

# verse 필드만 뽑아서 리스트로 만듭니다.
verse_texts = [item["verse"] for item in bible_verses]  # 문장만 추출

# pickle 파일에서 미리 계산된 임베딩을 불러옵니다.
with open("C:/test/bible_recommend/run/verse_embeddings.pkl", "rb") as f:
    verse_embeddings = pickle.load(f)  # 텐서 형식의 문장 임베딩

# SentenceTransformer 모델을 로드합니다.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Streamlit 앱 제목
st.title("문장에 맞는 성경구절 추천 앱")

# 사용자 입력
input_sentence = st.text_input("문장을 입력하세요:")

# 문장이 입력되었을 때
if input_sentence:
    # 입력 문장을 임베딩
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)

    # 코사인 유사도 계산
    cos_scores = util.cos_sim(input_embedding, verse_embeddings)[0]

    # 상위 3개 유사한 구절 추출
    top_results = cos_scores.topk(3)

    # 결과 출력
    st.write("입력 문장과 유사한 성경구절:")
    for score, idx in zip(top_results[0], top_results[1]):
        verse_info = bible_verses[idx]
        st.write(f"**{verse_info['reference']}**: {verse_info['verse']} (유사도: {score:.2f})")
