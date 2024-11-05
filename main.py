import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 파일 경로 설정
file_path = '/my/path/training_data.csv'

# 파일에서 데이터를 로드합니다.
# 데이터는 분류, 제목, 키워드, url 컬럼으로 구성되어 있다고 가정합니다.
data = pd.read_csv(file_path)

# 키워드 열에서 쉼표로 구분된 키워드를 공백으로 변환하여 모델 입력 형식으로 변환
data['키워드'] = data['키워드'].apply(lambda x: ' '.join(x.split(',')))

# 데이터 분할 (분류와 키워드 열 사용)
X_train, X_test, y_train, y_test = train_test_split(data['키워드'], data['분류'], test_size=0.3, random_state=42)

# 모델 파이프라인 생성 (TF-IDF 벡터화 및 나이브 베이즈 분류기)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("테스트 데이터에서의 정확도:", accuracy)
print("분류 보고서:\n", report)

# 새로운 키워드 리스트를 입력하면 해당 분류를 예측하는 함수
def predict_category(keyword_list):
    # 키워드 리스트를 공백으로 결합하여 모델 입력 형식에 맞게 변환
    input_text = ' '.join(keyword_list)
    prediction = model.predict([input_text])
    return prediction[0]

# 예시 키워드 리스트로 분류 예측
# test_keywords = ["괴산", "체험", "캠프", "농촌", "문화", "지역", "경제", "활성"]
test_keywords = ["고창군","라오스","계절","근로자","유치","총력","아시아투데이","신동준","전북특별자치","고창군","부족","농촌","일손","부족","해결","고창군","심덕섭","고창","군수","라오스","비엔티안","노동","사회","복지부","국장","아누손","캄싱사왓","면담","군수","라오스","계절","근로자","유치시","고창","농촌","환경","안정적","적응","지원","약속","라오스","정부","근로자들","무단이탈","방지","방법","조율","관리","체계","방안","협의","고창군","농촌","인력","라오스","계절","근로자","여부","유치","가능","확인"]
predicted_category = predict_category(test_keywords)
print("예측된 분류:", predicted_category)
