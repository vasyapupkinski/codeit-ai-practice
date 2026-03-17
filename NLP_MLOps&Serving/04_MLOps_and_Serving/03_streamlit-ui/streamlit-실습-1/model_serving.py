import streamlit as st
from transformers import pipeline

# 1. 앱 제목 설정: 웹페이지 상단에 큰 글씨로 제목을 표시합니다.
st.title("AI Emotion Detector(Model Caching Practice)")

# 2. @st.cache_resource 데코레이터:
#    - 이 함수(load_model)가 실행된 결과를 메모리(캐시)에 저장해둡니다.
#    - 처음에만 모델을 다운로드/로딩하고, 두 번째부터는 저장된 모델을 바로 꺼내 씁니다. (속도 향상 핵심!)
#    - 만약 이 데코레이터가 없으면, 버튼을 누를 때마다 모델을 다시 로딩해서 엄청 느려집니다.
@st.cache_resource
def load_model():
    # Hugging Face의 transformers 라이브러리를 사용해 'sentiment-analysis'(감성 분석) 파이프라인을 생성합니다.
    # 'matthewburke/korean_sentiment' 모델 사용 (안정적이고 성능 좋음)
    return pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

# 3. st.spinner: 모델 로딩 중에 사용자가 심심하지 않게 'AI model loading...'이라는 뱅글뱅글 도는 UI를 보여줍니다.
#    with 블록 안의 코드가 실행되는 동안만 스피너가 돌아갑니다.
with st.spinner("AI model loading..."):
    # load_model() 호출:
    # - 첫 실행: 모델을 로딩하고 캐시에 저장 (시간 걸림)
    # - 재실행: 캐시에서 즉시 가져옴 (순식간)
    classifier = load_model()

# 4. 안내 문구 출력
st.write("Write a Comment and get an emotion")

# 5. 텍스트 입력창 생성 (Text Area):
#    - 사용자로부터 긴 글을 입력받을 수 있는 상자를 만듭니다.
#    - 기본값으로 "나는 AI 엔지니어링과정이 재밌습니다."가 미리 채워져 있습니다.
user_input = st.text_area("Enter a comment", "AI is amazing")

# 6. '분석하기' 버튼 클릭 이벤트 처리:
#    - 사용자가 버튼을 '클릭한 순간'에만 아래 if문 내부가 실행됩니다.
#    - Streamlit은 버튼 클릭 등 상호작용이 있을 때마다 코드 전체를 맨 위부터 다시 실행하는 특징이 있습니다.
if st.button("Analyze"):
    # 사용자가 입력한 텍스트가 비어있지 않은지 확인합니다.
    if user_input:
        # 7. 모델 추론 (Inference):
        #    - classifier(user_input)을 호출하면 AI가 문장을 분석합니다.
        #    - 결과는 리스트 형태([{'label': 'POSITIVE', 'score': 0.99...}])로 나옵니다.
        #    - [0]을 붙여서 리스트 껍질을 벗기고, 내부의 딕셔너리만 변수 'result'에 담습니다.
        result = classifier(user_input)[0] 
        
        # 8. 결과 파싱: 딕셔너리에서 감정 라벨('label')과 확신 점수('score')를 각각 꺼냅니다.
        label = result['label']
        score = result['score']

        # 모델마다 라벨 이름이 다를 수 있어(LABEL_1 등), POSITIVE/NEGATIVE로 통일해줍니다.
        if label == "LABEL_1":
            label = "POSITIVE"
        elif label == "LABEL_0":
            label = "NEGATIVE"

        # 9. 조건부 로직 (Conditional Logic):
        #    - 점수(확신도)가 0.7 미만이면 '중립(Neutral)'으로 판단 (AI가 확신하지 못함)
        if score < 0.7:
            label_text = "Neutral"
            st.info(f"Neutral Comment (AI is unsure, Score: {score:.2%})")

        #    - 점수가 0.7 이상이고, 라벨이 'POSITIVE'면 긍정
        elif label == "POSITIVE":
            label_text = "POSITIVE"
            st.success(f"Positive Comment (Score: {score:.2%})")

        #    - 점수가 0.7 이상이고, 라벨이 'NEGATIVE'면 부정
        else:
            label_text = "NEGATIVE"
            st.error(f"Negative Comment (Score: {score:.2%})")

        # 10. 결과 시각화 (Layout & Metrics):
        #     - 화면을 좌우 2개 컬럼으로 나눕니다.
        col1, col2 = st.columns(2)
        
        # 왼쪽 컬럼(col1): 감정 라벨 표시
        with col1:
            st.metric("Emotion", label_text)
            
        # 오른쪽 컬럼(col2): 점수와 게이지 바 표시
        with col2:
            st.metric("Score", f"{score:.2%}") # 점수를 퍼센트(%) 형태로 변환하여 표시 (예: 0.99 -> 99.00%)
            st.progress(score) # 점수(0.0~1.0)에 맞춰 파란색 진행 막대를 채웁니다.

    else:
        # 입력된 텍스트가 없을 경우 경고 메시지를 띄웁니다.
        st.warning("Please enter a comment")