import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# 파일 경로 설정
training_file_path = 'third_party_data.csv'
test_file_path = 'test_data.csv'
output_file_path = 'prediction_results.csv'

# 파일에서 학습 데이터를 로드합니다.
training_data = pd.read_csv(training_file_path)

# 키워드 열에서 쉼표로 구분된 키워드를 공백으로 변환하여 모델 입력 형식으로 변환
print(training_data['분류'].value_counts())  # 각 클래스별 데이터 수 확인
training_data['키워드'] = training_data['키워드'].apply(lambda x: ' '.join(x.split(',')))

# 데이터 분할 (분류와 키워드 열 사용)
X_train, X_test, y_train, y_test = train_test_split(training_data['키워드'], training_data['분류'], test_size=0.3, random_state=42)

# 학습 및 테스트 데이터 수 출력
print(f"학습 데이터 수: {len(X_train)}")
print(f"테스트 데이터 수: {len(X_test)}")

# 모델 파이프라인 생성 (TF-IDF 벡터화 및 나이브 베이즈 분류기)
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터 로드
test_data = pd.read_csv(test_file_path)

# 키워드 열에서 쉼표로 구분된 키워드를 공백으로 변환하여 모델 입력 형식으로 변환
test_data['키워드'] = test_data['키워드'].apply(lambda x: ' '.join(x.split(',')) if isinstance(x, str) else '')

# 예측 수행
test_data['예측값'] = test_data['키워드'].apply(lambda x: model.predict([x])[0])

# 결과 데이터프레임 생성
result_df = test_data[['키워드', '예측값', '분류']].rename(columns={'분류': '실제값'})

# 정확도 계산
accuracy = accuracy_score(result_df['실제값'], result_df['예측값'])
print("최종 정답률:", accuracy)

# 결과를 CSV 파일로 저장
result_df.to_csv(output_file_path, index=False)
print(f"결과가 {output_file_path}에 저장되었습니다.")
