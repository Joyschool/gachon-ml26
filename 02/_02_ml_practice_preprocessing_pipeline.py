# [실습] 학생 성적 데이터 전처리 파이프라인 — Raw → 정제 → 분리 → 전처리 → ML  (2차시 학습용 데이터와 전처리)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. 문제가 있는 학습 데이터 만들기 (일부러 NaN, 100시간, None 을 넣음)
data = {
    "study_hours": [1, 2, 3, 4, np.nan, 6, 7, 8, 2, 3, 4, 5, 6, 7, 100, 4],
    "attendance":  [55, 60, 65, 70, 75, 80, 85, 95, 58, np.nan, 72, 76, 82, 88, 90, 70],
    "assignment":  [40, 50, 55, 60, 65, 75, 80, 95, 45, 55, 68, 72, 82, 87, 90, 60],
    "major": ["CS", "CS", "AI", "SW", "AI", "CS", "SW", "AI", "CS", None, "SW", "AI", "CS", "SW", "AI", "SW"],
    "pass":  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
}
df = pd.DataFrame(data)

# 2. 데이터 상태 확인 (EDA)
print(df.head()); print("\n크기:", df.shape)
print("\n결측치\n", df.isna().sum())
print("\n기초 통계\n", df.describe())          # study_hours 최대값 100 → 이상!

# 3. 잘못된 값 처리 — 도메인 지식: 하루 공부시간은 0~24시간
df.loc[(df["study_hours"] < 0) | (df["study_hours"] > 24), "study_hours"] = np.nan

# 4. 중복 제거
print("\n중복 개수:", df.duplicated().sum())
df = df.drop_duplicates()

# 5. X(문제) / y(정답) 분리
X = df.drop(columns=["pass"]); y = df["pass"]

# 6. Train / Test 분리 — 전처리 fit 전에 먼저 나눈다 (Data Leakage 방지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("\nTrain:", X_train.shape, " Test:", X_test.shape)

# 7. 수치형: 결측 → 중앙값, 그 다음 표준화
numeric_features = ["study_hours", "attendance", "assignment"]
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])

# 8. 범주형: 결측 → 최빈값, 그 다음 One-Hot
categorical_features = ["major"]
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

# 9. ColumnTransformer — 컬럼 종류별로 다른 전처리
preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_features),
                                  ("cat", categorical_pipeline, categorical_features)])

# 10. 전처리 + 모델을 하나의 Pipeline 으로
model = Pipeline([("preprocessing", preprocessor),
                  ("classifier", LogisticRegression(max_iter=1000))])

# 11. 학습: fit 은 Train 에만 — 전처리 통계(중앙값·평균·표준편차)도 Train 에서만 계산됨
model.fit(X_train, y_train)

# 12. 평가: Test 는 transform 만 거친다
print("\nAccuracy:", model.score(X_test, y_test))

# 13. 전처리된 데이터를 직접 보기 — 모델이 보는 것은 결국 숫자 행렬
processor = model.named_steps["preprocessing"]
processed_df = pd.DataFrame(processor.transform(X_train), columns=processor.get_feature_names_out())
print("\n전처리 결과\n", processed_df.head().round(2))
