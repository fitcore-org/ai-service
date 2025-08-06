"""
Agendador para execução automática de previsões financeiras

Este módulo configura tarefas agendadas para gerar previsões 
automaticamente em intervalos regulares.
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from services.prophet_forecaster import ProfitForecaster

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastScheduler:
    """
    Classe responsável pelo agendamento de previsões automáticas
    """
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.forecaster = ProfitForecaster()
        
    def setup_scheduled_tasks(self):
        """
        Configura as tarefas agendadas para previsões
        """
        try:
            # Gerar previsões todo dia 1º de cada mês às 06:00
            self.scheduler.add_job(
                func=self._monthly_forecast_job,
                trigger=CronTrigger(day=1, hour=6, minute=0),
                id='monthly_forecast',
                name='Geração mensal de previsões',
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            
            # Atualizar previsões toda segunda-feira às 08:00
            self.scheduler.add_job(
                func=self._weekly_forecast_update,
                trigger=CronTrigger(day_of_week='mon', hour=8, minute=0),
                id='weekly_forecast_update',
                name='Atualização semanal de previsões',
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            
            # Limpeza de logs antigos - todo domingo às 02:00
            self.scheduler.add_job(
                func=self._cleanup_old_logs,
                trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
                id='log_cleanup',
                name='Limpeza de logs antigos',
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            
            logger.info("Tarefas agendadas configuradas com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao configurar tarefas agendadas: {e}")
            raise
    
    def _monthly_forecast_job(self):
        """
        Job executado mensalmente para gerar novas previsões
        """
        try:
            logger.info("Iniciando geração mensal de previsões...")
            
            # Gera previsões para 6 meses
            summary = self.forecaster.generate_and_save_forecasts(periods=6)
            
            logger.info(f"Previsões mensais geradas: {summary.total_forecasts} registros")
            logger.info(f"Lucro médio previsto: ${summary.avg_predicted_profit:,.2f}")
            logger.info(f"Período: {summary.forecast_period_start} até {summary.forecast_period_end}")
            
        except Exception as e:
            logger.error(f"Erro na geração mensal de previsões: {e}")
    
    def _weekly_forecast_update(self):
        """
        Job executado semanalmente para atualizar previsões se houver novos dados
        """
        try:
            logger.info("Verificando necessidade de atualização semanal...")
            
            # Verifica se há previsões atuais
            current_forecasts = self.forecaster.get_current_forecasts()
            
            if not current_forecasts:
                logger.info("[PREVISAO] Nenhuma previsão atual encontrada. Gerando novas...")
                self._monthly_forecast_job()
                return
            
            # Verifica se as previsões são muito antigas (mais de 7 dias)
            latest_forecast = max(current_forecasts, key=lambda x: x.forecast_date)
            days_old = (datetime.now().date() - latest_forecast.forecast_date).days
            
            # Verifica se a última previsão é para uma data no passado
            if latest_forecast.forecast_date <= datetime.now().date():
                logger.info("Previsões desatualizadas (data no passado). Atualizando...")
                self._monthly_forecast_job()
            elif days_old > 7:
                logger.info(f"Previsões antigas detectadas ({days_old} dias). Atualizando...")
                self._monthly_forecast_job()
            else:
                logger.info(f"Previsões atuais ainda válidas ({days_old} dias)")
            
        except Exception as e:
            logger.error(f"Erro na atualização semanal: {e}")
    
    def _cleanup_old_logs(self):
        """
        Job para limpeza de logs antigos (placeholder para implementação futura)
        """
        try:
            logger.info("Executando limpeza de logs antigos...")
            # Implementar limpeza de logs se necessário
            logger.info("Limpeza de logs concluída")
            
        except Exception as e:
            logger.error(f"Erro na limpeza de logs: {e}")
    
    def _initial_forecast_generation(self):
        """
        Gera previsões iniciais ao iniciar a aplicação
        """
        try:
            logger.info("Iniciando geração de previsões ao startup da aplicação...")
            
            # Verifica se há dados históricos suficientes
            from sqlmodel import Session, select
            from database.db_connect import engine
            from model.models import Profit
            
            with Session(engine) as session:
                profit_count = len(session.exec(select(Profit)).all())
                
                if profit_count < 3:
                    logger.warning(f"Dados históricos insuficientes ({profit_count} registros). Mínimo: 3, Recomendado: 6+")
                    logger.info("Execute o script populate_sample_data.py para gerar dados de exemplo")
                    return
                elif profit_count < 6:
                    logger.warning(f"Dados limitados ({profit_count} registros). Previsões básicas serão geradas. Recomendado: 6+ meses para melhor qualidade.")
                
                # Log detalhado dos dados históricos para diagnóstico
                profits = session.exec(select(Profit).order_by("period_start")).all()
                logger.info(f"[DADOS] DADOS HISTÓRICOS DISPONÍVEIS ({len(profits)} registros):")
                for profit in profits:
                    logger.info(f"   {profit.period_start.strftime('%Y-%m')}: ${profit.net_profit:,.2f}")
            
            # Verifica se já existem previsões válidas
            current_forecasts = self.forecaster.get_current_forecasts()
            
            # Se não há previsões ou elas estão desatualizadas, gera novas
            should_generate = True
            
            if current_forecasts:
                # Verifica se as previsões são atuais (pelo menos uma previsão futura)
                future_forecasts = [f for f in current_forecasts if f.forecast_date > datetime.now().date()]
                
                if len(future_forecasts) >= 3:  # Pelo menos 3 meses de previsões futuras
                    logger.info(f"Previsões válidas encontradas ({len(future_forecasts)} meses futuros). Pulando geração inicial.")
                    should_generate = False
            
            if should_generate:
                logger.info("[GERANDO] GERANDO NOVAS PREVISÕES...")
                # Gera previsões para 6 meses
                summary = self.forecaster.generate_and_save_forecasts(periods=6)
                
                logger.info(f"[SUCESSO] Previsões iniciais geradas com sucesso!")
                logger.info(f"   Total de previsões: {summary.total_forecasts}")
                logger.info(f"   Lucro médio previsto: ${summary.avg_predicted_profit:,.2f}")
                logger.info(f"   Período: {summary.forecast_period_start} até {summary.forecast_period_end}")
                
                # Alertas baseados nos resultados
                if summary.avg_predicted_profit < 0:
                    logger.warning("[ALERTA] ALERTA: Previsões indicam lucro médio negativo")
                    logger.info("[SUGESTAO] SUGESTÕES:")
                    logger.info("   • Revisar dados do último período (podem conter anomalias)")
                    logger.info("   • Verificar se houve eventos excepcionais recentes")
                    logger.info("   • Considerar adicionar mais dados históricos")
                elif summary.avg_predicted_profit < 1000:
                    logger.warning("[ALERTA] Previsões indicam lucros baixos")
                
                # Log básico do modelo já treinado
                logger.info("🤖 Modelo Prophet treinado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na geração inicial de previsões: {e}")
            # Não propaga o erro para não impedir o startup da aplicação
    
    def start_scheduler(self):
        """
        Inicia o agendador e executa geração inicial de previsões
        """
        try:
            if not self.scheduler.running:
                self.setup_scheduled_tasks()
                self.scheduler.start()
                logger.info("Agendador de previsões iniciado")
                
                # Executa geração inicial de previsões ao iniciar
                logger.info("Executando geração inicial de previsões...")
                try:
                    self._initial_forecast_generation()
                except Exception as e:
                    logger.error(f"Erro na geração inicial de previsões: {e}")
                
                # Log das próximas execuções
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    next_run = job.next_run_time
                    if next_run:
                        logger.info(f"Próxima execução de '{job.name}': {next_run}")
            else:
                logger.warning("Agendador já está em execução")
                
        except Exception as e:
            logger.error(f"Erro ao iniciar agendador: {e}")
            raise
    
    def stop_scheduler(self):
        """
        Para o agendador
        """
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                logger.info("Agendador de previsões parado")
            else:
                logger.warning("Agendador já estava parado")
                
        except Exception as e:
            logger.error(f"Erro ao parar agendador: {e}")
    
# Instância global do agendador
forecast_scheduler = ForecastScheduler()

def start_forecast_scheduler():
    """
    Função para iniciar o agendador (chamada no startup da aplicação)
    """
    forecast_scheduler.start_scheduler()

def stop_forecast_scheduler():
    """
    Função para parar o agendador (chamada no shutdown da aplicação)
    """
    forecast_scheduler.stop_scheduler()
