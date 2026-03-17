import streamlit as st

from model_4 import MyModel

st.set_page_config(page_title="Structured AI App")
st.title("AI Emotion Detector")
st.info("Seperating UI Code and Model Logic")

@st.cache_resource
# ↑ '@'는 데코레이터(Decorator)입니다. 함수에 "캐싱(저장)" 능력을 부여하는 스티커 같은 존재입니다.
# 작동 원리:
# 1. 이 함수가 호출되면 먼저 메모리(캐시)를 확인합니다.
# 2. 이미 만들어둔 모델이 있다면? -> 함수를 실행하지 않고 저장된 모델을 바로 반환합니다. (Model init 출력 X)
# 3. 없다면? -> 함수를 실행해서 모델을 만들고, 캐시에 저장한 뒤 반환합니다. (Model init 출력 O - 최초 1회만)
def get_model_instance():
    return MyModel()

model = get_model_instance()
text = st.text_input("Enter text to see if it's Spam")
keyword_str = st.sidebar.text_input("Enter keywords to filter", value="광고,무료")
keywords = keyword_str.split(",")

if st.button("Scan"):
    result = model.predict(text, keywords)
    
    st.json(result)

    if result['is_spam']:
        st.warning("High probability of Spam")
        st.info(f"Reason: {result['reason']}")
    else:
        st.success("Low probability of Spam")
        st.info(f"Reason: {result['reason']}")