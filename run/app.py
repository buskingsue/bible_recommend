# streamlit, sentence_transformers의 SentenceTransformer 및 util 모듈을 불러옵니다.
import streamlit as st  # Streamlit: 웹 앱 인터페이스를 쉽게 구축할 수 있는 라이브러리입니다.
from sentence_transformers import SentenceTransformer, util  # SentenceTransformer와 util: 문장 임베딩과 코사인 유사도 계산에 사용됩니다.

# 사전 학습된 문장 임베딩 모델을 로드합니다.
# 여기서는 다국어 지원 모델인 'all-MiniLM-L6-v2'를 사용합니다.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 예시로 사용할 성경구절 데이터를 리스트로 정의합니다.
# 각 항목은 'reference'(성경구절 위치)와 'verse'(구절 내용)를 포함합니다.
bible_verses = [
    {"reference": "요한복음 3:16", "verse": "하나님이 세상을 이처럼 사랑하사 독생자를 주셨으니, 이는 그를 믿는 자마다 멸망치 않고 영생을 얻게 하려 하심이라."},
    {"reference": "시편 23:1", "verse": "여호와는 나의 목자시니 내게 부족함이 없으리로다."},
    {"reference": "빌립보서 4:13", "verse": "내게 능력 주시는 자 안에서 내가 모든 것을 할 수 있느니라."},
    {"reference": "로마서 8:28", "verse": "우리가 알거니와 하나님을 사랑하는 자 곧 그의 뜻대로 부르심을 입은 자들에게는 모든 것이 합력하여 선을 이루느니라."},
    {"reference": "잠언 3:5-6", "verse": "너는 마음을 다하여 여호와를 신뢰하고 네 명철을 의지하지 말라. 네 길을 여호와께 맡기라 그리하면 그가 너를 인도하시리라."},
    {"reference": "이사야 41:10",
     "verse": "두려워하지 말라 내가 너와 함께 함이라 놀라지 말라 나는 네 하나님이 됨이라 내가 너를 굳세게 하리라 참으로 너를 도와주리라 참으로 나의 의로운 오른손으로 너를 붙들리라."},
    {"reference": "마태복음 11:28", "verse": "수고하고 무거운 짐 진 자들아 다 내게로 오라 내가 너희를 쉬게 하리라."},
    {"reference": "로마서 12:2", "verse": "이 세대를 본받지 말고 오직 마음을 새롭게 함으로 변화를 받아 하나님의 뜻이 무엇인지 분별하도록 하라."},
    {"reference": "시편 46:1", "verse": "하나님은 우리의 피난처시요 힘이시니 환난 중에 만날 큰 도움이시라."},
    {"reference": "요한일서 4:18", "verse": "사랑 안에는 두려움이 없고 온전한 사랑이 두려움을 내쫓나니 두려움에는 형벌이 있음이라."}
]

# 각 성경구절의 'verse' 텍스트만 추출하여 리스트를 만듭니다.
verse_texts = [item["verse"] for item in bible_verses]

# 모든 성경구절에 대해 문장 임베딩을 계산합니다.
# convert_to_tensor=True 옵션으로 텐서 형식으로 반환받습니다.
verse_embeddings = model.encode(verse_texts, convert_to_tensor=True)

# Streamlit 앱 제목을 설정합니다.
st.title("문장에 맞는 성경구절 추천 앱")

# 사용자로부터 입력 문장을 받습니다.
input_sentence = st.text_input("문장을 입력하세요:")

# 사용자가 문장을 입력한 경우 아래 코드를 실행합니다.
if input_sentence:
    # 입력 문장의 임베딩을 계산합니다.
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)

    # 입력 문장과 모든 성경구절 간의 코사인 유사도를 계산합니다.
    cos_scores = util.cos_sim(input_embedding, verse_embeddings)[0]

    # 가장 높은 유사도를 가진 상위 3개의 구절 인덱스를 구합니다.
    top_results = cos_scores.topk(3)

    # 결과를 화면에 출력합니다.
    st.write("입력 문장과 유사한 성경구절:")
    # top_results[0]: 유사도 점수, top_results[1]: 해당 인덱스
    for score, idx in zip(top_results[0], top_results[1]):
        verse_info = bible_verses[idx]
        st.write(f"**{verse_info['reference']}**: {verse_info['verse']} (유사도: {score:.2f})")
