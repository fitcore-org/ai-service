import pandas as pd
import joblib
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Iniciando o treinamento do modelo de sentimento...")

# 1. Carregar dados rotulados
df = pd.read_csv('training_feedback.csv')

# Remove linhas com dados ausentes, se houver
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Garante que o texto é string
df['text'] = df['text'].astype(str)

def normalize_text(text):
    """
    Normaliza texto para melhorar a acurácia do modelo de sentimento.
    
    Aplica:
    - Conversão para minúsculas
    - Remoção de acentos
    - Normalização de pontuação
    - Remoção de caracteres especiais
    - Normalização de espaços
    - Padronização de gírias comuns
    """
    # Converte para minúsculas
    text = text.lower()
    
    # Remove acentos mantendo caracteres especiais do português
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Padroniza gírias e expressões comuns do dataset
    girias_map = {
        r'\bmassa\b': 'bom',
        r'\btop\b': 'bom', 
        r'\bshow\b': 'bom',
        r'\barretado\b': 'bom',
        r'\baff\b': 'ruim',
        r'\bvish\b': 'ruim',
        r'\brs\b': '',  # Remove risos
        r'\bkkk+\b': '',  # Remove risos
        r'\bhaha+\b': '',  # Remove risos
        r'\bvc\b': 'voce',
        r'\bpra\b': 'para',
        r'\bpro\b': 'para',
        r'\bne\b': 'nao e',
        r'\bta\b': 'esta'
    }
    
    for padrao, substituicao in girias_map.items():
        text = re.sub(padrao, substituicao, text)
    
    # Remove pontuação excessiva (mantém apenas . , ! ?)
    text = re.sub(r'[^\w\s.!?,-]', ' ', text)
    
    # Normaliza pontuação repetida
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Remove números isolados (mantém em contexto como "24h")
    text = re.sub(r'\b\d+\b(?!\w)', '', text)
    
    # Normaliza espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    # Remove espaços no início e fim
    text = text.strip()
    
    return text

# Aplica normalização nos textos
logger.info("Normalizando textos...")
df['text_original'] = df['text'].copy()  # Salva originais para comparação
df['text'] = df['text'].apply(normalize_text)
logger.info("Normalização concluída.")

# Log alguns exemplos da normalização
logger.info("Exemplos de normalização:")
for i in range(min(3, len(df))):
    logger.info(f"  Original: '{df.iloc[i]['text_original'][:50]}...'")
    logger.info(f"  Normalizado: '{df.iloc[i]['text'][:50]}...'")

logger.info(f"Total de {len(df)} feedbacks carregados para treinamento.")

# 2. Definir features (X) e labels (y)
X = df['text']
y = df['sentiment']

# 3. Dividir dados para treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Criar o pipeline de Machine Learning
#    Etapa 1: Vetoriza o texto. Usamos bigramas (ngram_range) para capturar expressões como "muito bom".
#    Etapa 2: Treina um classificador de Regressão Logística, que é leve e eficaz.
#    Nota: Com textos normalizados, podemos usar configurações mais refinadas no TfidfVectorizer
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),          # Unigramas e bigramas
        max_df=0.85,                 # Ignora termos muito frequentes (mais restritivo)
        min_df=2,                    # Palavra deve aparecer pelo menos 2 vezes
        max_features=5000,           # Limita vocabulário para evitar overfitting
        strip_accents=None,          # Já fizemos normalização manual
        lowercase=False,             # Já convertemos para minúsculas
        stop_words=None,             # Não remove stopwords automaticamente (algumas podem ser úteis para sentimento)
        token_pattern=r'\b[a-z]{2,}\b'  # Tokens de pelo menos 2 letras
    )),
    ('clf', LogisticRegression(
        random_state=42, 
        solver='liblinear', 
        multi_class='ovr',
        C=1.0,                       # Regularização padrão
        max_iter=1000                # Mais iterações para convergência
    ))
])

# 5. Treinar o modelo com os dados de treino
logger.info("Treinando o pipeline...")
sentiment_pipeline.fit(X_train, y_train)

# 6. Avaliar a performance com os dados de teste
logger.info("\nAvaliação do modelo no conjunto de teste:")
y_pred = sentiment_pipeline.predict(X_test)

# Calcular métricas detalhadas
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Acurácia geral: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Relatório de classificação
logger.info("\nDesempenho detalhado:")
logger.info(classification_report(y_test, y_pred))

# Métricas básicas
classification_rep = classification_report(y_test, y_pred, output_dict=True)
logger.info(f"\nResumo de Métricas:")
logger.info(f"  Acurácia geral: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
logger.info(f"\nMatriz de Confusão:")
logger.info(f"Classes: {sentiment_pipeline.classes_}")
logger.info(f"Matriz:\n{cm}")

# Análise de probabilidades para verificar confiança
y_proba = sentiment_pipeline.predict_proba(X_test)
max_probabilities = np.max(y_proba, axis=1)
avg_confidence = np.mean(max_probabilities)
min_confidence = np.min(max_probabilities)
max_confidence = np.max(max_probabilities)

logger.info(f"\nAnálise de Confiança das Predições:")
logger.info(f"  Confiança média: {avg_confidence:.4f}")
logger.info(f"  Confiança mínima: {min_confidence:.4f}")
logger.info(f"  Confiança máxima: {max_confidence:.4f}")

# Mostrar exemplos de predições com baixa confiança
low_confidence_indices = np.where(max_probabilities < 0.6)[0]
if len(low_confidence_indices) > 0:
    logger.warning(f"\n{len(low_confidence_indices)} predições com baixa confiança (<60%):")
    for i in low_confidence_indices[:5]:  # Mostra apenas os primeiros 5
        idx = i
        logger.warning(f"  Texto: '{X_test.iloc[idx][:50]}...'")
        logger.warning(f"  Real: {y_test.iloc[idx]}, Predito: {y_pred[idx]}, Confiança: {max_probabilities[idx]:.3f}")

print(classification_report(y_test, y_pred))

# 7. Salvar o pipeline treinado em um arquivo
#    Este arquivo é tudo o que você precisa para fazer predições no seu microserviço.
model_filename = 'sentiment_model.joblib'
joblib.dump(sentiment_pipeline, model_filename)

logger.info(f"\nTreinamento concluído! Modelo salvo como '{model_filename}'")
logger.info(f"Acurácia final: {accuracy:.4f} ({accuracy*100:.2f}%)")
logger.info(f"Confiança média das predições: {avg_confidence:.4f}")

# Salvar métricas em arquivo para análise posterior
metrics_summary = {
    'accuracy': accuracy,
    'avg_confidence': avg_confidence,
    'min_confidence': min_confidence,
    'max_confidence': max_confidence,
    'total_samples': len(df),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open('training_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

logger.info("Métricas salvas em 'training_metrics.json'")
logger.info("=== TREINAMENTO FINALIZADO ===")

print(f"\nTreinamento concluído! Modelo salvo como '{model_filename}'")