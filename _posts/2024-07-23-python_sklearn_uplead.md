---
layout: single
title:  "파이썬으로 만드는 간단한 "
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 싸이키런의 파이프라인을 구성하는 예제이다.
단계별로 설명하고 있으며, 싸이키런의 구조를 쉽게 이해할 수 있다.  
코드에는 약간의 오류가 있지만, 이해하는데에 문제는 없다.


```python
# 전체 코드

# 단계 1: 데이터 준비
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 단계 2: 파이프라인 구성 요소 정의
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 단계 3: 파이프라인 훈련
pipeline.fit(X_train, y_train)

# 단계 4: 예측 및 평가
y_pred = pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')

# 단계 5: 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Best Test Accuracy: {best_accuracy:.2f}')

# 단계 6: 모델 저장 및 로드
import joblib
joblib.dump(best_model, 'best_model.pkl')
loaded_model = joblib.load('best_model.pkl')
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f'Loaded Model Test Accuracy: {loaded_accuracy:.2f}')

# 단계 7: Flask로 웹 서비스 배포
from flask import Flask, request, jsonify
app = Flask(__name__)
model = joblib.load('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = [data['feature1'], data['feature2'], data['feature3'], data['feature4']]
    prediction = model.predict([X_new])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```