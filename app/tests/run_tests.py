from sqlmodel import Session, select
from database.db_connect import engine
from model.models import Feedback, Word_Frequency, SentimentEnum
from services.analyzer import populating_word_frequency_service
from database.seed_data import create_test_feedbacks, clear_test_data

def validate_sentiment_analysis():
    """Valida se a análise de sentimentos está funcionando corretamente"""
    
    print("INICIANDO VALIDAÇÃO DO SISTEMA")
    print("=" * 50)
    
    # 1. Limpa e popula dados de teste
    clear_test_data()
    create_test_feedbacks()
    
    with Session(engine) as session:
        # 2. Executa o processamento completo
        print("\nExecutando processamento completo...")
        populating_word_frequency_service(session)
        
        # 3. Valida sentiment analysis
        print("\nVALIDANDO SENTIMENT ANALYSIS:")
        feedbacks_by_sentiment = {}
        for sentiment in [SentimentEnum.positive, SentimentEnum.negative, SentimentEnum.neutral]:
            count = session.exec(
                select(Feedback).where(Feedback.sentiment == sentiment)
            ).all()
            feedbacks_by_sentiment[sentiment.value] = len(count)
            print(f"  {sentiment.value.upper()}: {len(count)} feedbacks")
        
        # 4. Valida word frequency
        print("\nVALIDANDO WORD FREQUENCY:")
        for sentiment in [SentimentEnum.positive, SentimentEnum.negative, SentimentEnum.neutral]:
            words = session.exec(
                select(Word_Frequency).where(Word_Frequency.sentiment == sentiment)
            ).all()
            
            print(f"\n  {sentiment.value.upper()} ({len(words)} palavras):")
            for word in words[:5]:  # Top 5
                print(f"    • {word.word}: {word.frequency}")
        
        # 5. Validações 
        print("\nVALIDAÇÕES DE DADOS:")
        
        # Verifica se há feedbacks não processados
        unprocessed = session.exec(
            select(Feedback).where(Feedback.sentiment == SentimentEnum.no_analyzed)
        ).all()
        print(f"  • Feedbacks não processados: {len(unprocessed)} (deve ser 0)")
        
        # Verifica se há palavras salvas
        total_words = session.exec(select(Word_Frequency)).all()
        print(f" • Total de palavras salvas: {len(total_words)} (deve ser ~30)")
        
        # Verifica distribuição equilibrada
        word_counts = {}
        for sentiment in [SentimentEnum.positive, SentimentEnum.negative, SentimentEnum.neutral]:
            count = len([w for w in total_words if w.sentiment == sentiment])
            word_counts[sentiment.value] = count
            print(f" • Palavras {sentiment.value}: {count} (deve ser ~10)")
        
        print("\n" + "=" * 50)
        print("VALIDAÇÃO CONCLUÍDA!")
        
        return {
            "feedbacks_by_sentiment": feedbacks_by_sentiment,
            "total_words": len(total_words),
            "word_counts": word_counts,
            "unprocessed_count": len(unprocessed)
        }

if __name__ == "__main__":
    validate_sentiment_analysis()