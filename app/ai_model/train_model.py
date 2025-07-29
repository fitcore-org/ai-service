import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Iniciando o treinamento do modelo de sentimento...")

# 1. Carregar dados rotulados
df = pd.read_csv('training_feedback.csv')

# Remove linhas com dados ausentes, se houver
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Garante que o texto é string
df['text'] = df['text'].astype(str)

print(f"Total de {len(df)} feedbacks carregados para treinamento.")

# 2. Definir features (X) e labels (y)
X = df['text']
y = df['sentiment']

# 3. Dividir dados para treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Criar o pipeline de Machine Learning
#    Etapa 1: Vetoriza o texto. Usamos bigramas (ngram_range) para capturar expressões como "muito bom".
#    Etapa 2: Treina um classificador de Regressão Logística, que é leve e eficaz.
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=2)),
    ('clf', LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr'))
])

# 5. Treinar o modelo com os dados de treino
print("Treinando o pipeline...")
sentiment_pipeline.fit(X_train, y_train)

# 6. Avaliar a performance com os dados de teste
print("\nAvaliação do modelo no conjunto de teste:")
y_pred = sentiment_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Salvar o pipeline treinado em um arquivo
#    Este arquivo é tudo o que você precisa para fazer predições no seu microserviço.
model_filename = 'sentiment_model.joblib'
joblib.dump(sentiment_pipeline, model_filename)

print(f"\nTreinamento concluído! Modelo salvo como '{model_filename}'")