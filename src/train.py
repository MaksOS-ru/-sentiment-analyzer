"""
Обучение модели классификации тональности отзывов.

Что происходит на каждом шаге — объясняю прямо в коде,
чтобы было понятно, а не просто "запусти и жди".
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocess import clean_texts

DATA_PATH = "../data/reviews.csv"
MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def main():
    # 1. Загружаем данные
    df = pd.read_csv(DATA_PATH)
    print(f"Загружено отзывов: {len(df)}")

    # 2. Чистим тексты (см. preprocess.py)
    df["text_clean"] = clean_texts(df["text"].tolist())

    # 3. Делим на обучающую и тестовую выборки.
    #    Модель учим на train, а на test потом проверяем,
    #    насколько хорошо она угадывает то, чего не видела.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"],
        test_size=0.25, random_state=42, stratify=df["label"]
    )

    # 4. Векторизация: превращаем текст в числа методом TF-IDF.
    #    Простыми словами: каждому слову присваивается вес —
    #    чем чаще слово встречается в конкретном отзыве и реже
    #    во всех остальных, тем больше у него "вес" (значимость).
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Обучаем модель. LogisticRegression — простой и надёжный
    #    выбор для текстовой классификации на старте.
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # 6. Проверяем качество на тестовых данных
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nТочность (accuracy): {accuracy:.2f}")
    print("\nПодробный отчёт:")
    print(classification_report(y_test, predictions))

    # 7. Сохраняем модель и векторизатор, чтобы потом
    #    использовать их в predict.py без повторного обучения
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nМодель сохранена в {MODEL_PATH}")
    print(f"Векторизатор сохранён в {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
