"""
Использование обученной модели: даём новый текст — получаем предсказание.

Запуск:
    python predict.py "Отличный сервис, всё понравилось"
"""

import sys
import joblib

from preprocess import clean_text

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def predict_sentiment(text: str) -> str:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return prediction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python predict.py "твой текст здесь"')
        sys.exit(1)

    input_text = sys.argv[1]
    result = predict_sentiment(input_text)
    print(f"Текст: {input_text}")
    print(f"Тональность: {result}")
