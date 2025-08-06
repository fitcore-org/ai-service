# FitCore AI Service

Microserviço de análise de feedbacks e previsão financeira para plataformas fitness.

## Funcionalidades

### Análise de Sentimentos
- Classificação automática de feedbacks de usuários
- Modelo de machine learning baseado em scikit-learn
- Processamento de texto com NLTK
- Análise de frequência de palavras e sentimentos

### Previsão Financeira
- Previsões de lucro mensal usando Facebook Prophet
- Sazonalidade específica para negócio fitness
- Tratamento inteligente de outliers
- Intervalos de confiança para previsões

### Agendamento Automático
- Análise de feedbacks executada de hora em hora
- Previsões atualizadas semanalmente
- Geração mensal de novas previsões
- Limpeza automática de logs antigos

## Tecnologias

- **FastAPI**: API REST moderna e eficiente
- **Prophet**: Modelo de previsão temporal do Facebook
- **scikit-learn**: Machine learning para análise de sentimentos
- **NLTK**: Processamento de linguagem natural
- **SQLModel**: ORM baseado em SQLAlchemy
- **APScheduler**: Agendamento de tarefas
- **PostgreSQL**: Banco de dados principal

## Instalação

### Usando Docker

```bash
docker build -t fitcore-ai .
docker run -p 8000:8000 fitcore-ai
```

### Desenvolvimento Local

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --reload
```

## Configuração

Configure as variáveis de ambiente:

```env
DATABASE_URL=postgresql://user:password@localhost/fitcore
RUN_VALIDATE_ON_STARTUP=false
```

## API Endpoints

### Feedbacks
- `POST /feedbacks/` - Criar novo feedback
- `GET /feedbacks/` - Listar feedbacks com filtros
- `GET /feedbacks/sentiment-analysis` - Análise de sentimentos

## Estrutura do Projeto

```
app/
├── main.py                 # Aplicação FastAPI
├── routes/                 # Endpoints da API
├── services/               # Lógica de negócio
├── model/                  # Modelos de dados
├── database/               # Conexão e configuração do BD
├── ai_model/               # Modelo de machine learning
└── tests/                  # Testes e validações
```

## Dados de Teste

Para popular o banco com dados de exemplo:

```bash
cd app/database
python populate_sample_data.py
```

## Monitoramento

O serviço inclui logging detalhado e métricas para:
- Performance dos modelos de ML
- Qualidade das previsões
- Status dos agendamentos
- Análise de dados históricos
