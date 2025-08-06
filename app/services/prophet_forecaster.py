"""
Serviço de previsão financeira usando Facebook Prophet

Este módulo implementa previsões de lucro líquido mensal usando Prophet.
Mantém apenas 6 meses de previsões futuras no banco de dados.
Usado apenas pelo scheduler automático.
"""

import pandas as pd
from datetime import datetime
from typing import List
import logging

from prophet import Prophet
from sqlmodel import Session, select

from model.models import Profit, ProfitForecast, ForecastResponse, ForecastSummary
from database.db_connect import engine

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfitForecaster:
    """
    Classe responsável por gerar previsões de lucro usando Prophet
    Usada apenas pelo scheduler automático
    """
    
    def __init__(self, model_version: str = "v2.0"):
        self.model_version = model_version
        self.prophet_model = None
        
    def _fetch_historical_data(self, session: Session) -> pd.DataFrame:
        """
        Busca dados históricos de lucro do banco de dados
        
        Returns:
            DataFrame com colunas 'ds' (data) e 'y' (lucro líquido)
        """
        try:
            # Query para buscar dados ordenados por período
            query = select(Profit).order_by("period_start")
            result = session.exec(query).all()
            
            if not result:
                raise ValueError("Nenhum dado histórico encontrado na tabela profits")
                
            # Converte para DataFrame do Prophet
            df = pd.DataFrame([
                {"ds": row.period_start, "y": float(row.net_profit)} 
                for row in result
            ])
            
            # Converte coluna de data
            df['ds'] = pd.to_datetime(df['ds'])
            
            logger.info(f"Dados históricos carregados: {len(df)} registros")
            logger.info(f"Período: {df['ds'].min()} até {df['ds'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados históricos: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida e limpa os dados para garantir qualidade das previsões
        Inclui tratamento simples de outliers
        
        Args:
            df: DataFrame com dados históricos
            
        Returns:
            DataFrame limpo e validado
        """
        if len(df) < 2:
            raise ValueError(f"Dados insuficientes. Necessário pelo menos 2 pontos, encontrados {len(df)}")
        
        # Verifica se há valores nulos
        if df['y'].isnull().any():
            logger.warning("Valores nulos encontrados nos dados. Removendo...")
            df = df.dropna()
        
        # Tratamento simples de outliers apenas se temos dados suficientes
        if len(df) >= 6:
            # Calcula quartis e IQR
            q1 = df['y'].quantile(0.25)
            q3 = df['y'].quantile(0.75)
            iqr = q3 - q1
            
            # Limites conservadores para outliers (3x IQR em vez de 1.5x)
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Identifica outliers
            outliers_mask = (df['y'] < lower_bound) | (df['y'] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                logger.warning(f"[OUTLIERS] Detectados {outliers_count} outliers extremos")
                
                # Suaviza outliers em vez de removê-los (mantém sazonalidade)
                df_clean = df.copy()
                
                # Para outliers negativos, substitui pela mediana dos valores baixos
                negative_outliers = df['y'] < lower_bound
                if negative_outliers.any():
                    median_low = df[df['y'] >= lower_bound]['y'].quantile(0.25)
                    df_clean.loc[negative_outliers, 'y'] = median_low
                    logger.info(f"   {negative_outliers.sum()} outliers negativos suavizados para ${median_low:,.2f}")
                
                # Para outliers positivos, substitui pela mediana dos valores altos
                positive_outliers = df['y'] > upper_bound
                if positive_outliers.any():
                    median_high = df[df['y'] <= upper_bound]['y'].quantile(0.75)
                    df_clean.loc[positive_outliers, 'y'] = median_high
                    logger.info(f"   {positive_outliers.sum()} outliers positivos suavizados para ${median_high:,.2f}")
                
                return df_clean
            else:
                logger.info("[OK] Nenhum outlier extremo detectado")
        
        return df
    
    def _train_model(self, df: pd.DataFrame) -> Prophet:
        """
        Treina o modelo Prophet com os dados históricos
        Inclui sazonalidade específica para negócio fitness
        
        Args:
            df: DataFrame com dados históricos formatados para Prophet
            
        Returns:
            Modelo Prophet treinado
        """
        try:
            data_points = len(df)
            logger.info(f"[TREINAMENTO] Treinando modelo fitness com {data_points} pontos de dados")
            
            # Configuração otimizada para negócio fitness
            if data_points <= 6:
                logger.info(f"[DADOS] Poucos dados ({data_points} pontos) - configuração conservadora")
                model = Prophet(
                    changepoint_prior_scale=0.01,    # Menos sensível a mudanças
                    seasonality_prior_scale=0.8,     # Sazonalidade moderada para fitness
                    interval_width=0.95,             # Intervalo de confiança maior
                    n_changepoints=min(2, data_points - 1),  # Máximo 2 pontos de mudança
                    yearly_seasonality='auto',       # Deixa automático para poucos dados
                    weekly_seasonality='auto',       # Não aplicável para dados mensais
                    daily_seasonality='auto'         # Não aplicável para dados mensais
                )
            else:
                logger.info(f"[DADOS] Dados suficientes ({data_points} pontos) - configuração fitness otimizada")
                model = Prophet(
                    changepoint_prior_scale=0.05,    # Moderadamente sensível a mudanças
                    seasonality_prior_scale=1.0,     # Sazonalidade normal para captar padrões fitness
                    interval_width=0.90,             # Intervalo de confiança padrão
                    yearly_seasonality='auto',       # Deixa automático, vamos customizar depois
                    weekly_seasonality='auto',       # Não aplicável
                    daily_seasonality='auto'         # Não aplicável
                )
            
            # Adiciona sazonalidade anual customizada para fitness (12 meses)
            # Período de 12 porque os dados são mensais e queremos captar sazonalidade anual
            model.add_seasonality(
                name='fitness_yearly',
                period=12,                    # 12 meses no ano
                fourier_order=4,              # Complexidade moderada (captura até 4 harmônicos)
                prior_scale=1.2               # Um pouco mais forte que o padrão
            )
            
            # Se temos dados suficientes, adiciona sazonalidade semestral para captar
            # os dois picos principais do fitness (Janeiro e Outubro)
            if data_points >= 12:
                model.add_seasonality(
                    name='fitness_semester',
                    period=6,                 # Semestral (2 picos por ano)
                    fourier_order=2,          # Complexidade baixa
                    prior_scale=0.8           # Menos intenso que a anual
                )
                logger.info("[SAZONALIDADE] Sazonalidade semestral adicionada (captura picos de Janeiro e Outubro)")
            
            # Treina o modelo
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)
            
            logger.info("[SUCESSO] Modelo fitness treinado com sazonalidade customizada!")
            logger.info("   [ANUAL] Sazonalidade anual: Ativa (período 12 meses)")
            if data_points >= 12:
                logger.info("   [SEMESTRAL] Sazonalidade semestral: Ativa (captura picos fitness)")
            
            return model
            
        except Exception as e:
            logger.error(f"Erro no treinamento do modelo: {e}")
            raise
    
    def _generate_forecast(self, model: Prophet, periods: int = 6) -> pd.DataFrame:
        """
        Gera previsões futuras usando o modelo treinado
        
        Args:
            model: Modelo Prophet treinado
            periods: Número de períodos futuros (meses)
            
        Returns:
            DataFrame com previsões
        """
        try:
            # Gera dataframe para períodos futuros
            future = model.make_future_dataframe(periods=periods, freq='MS')  # MS = Month Start
            
            # Faz a previsão
            forecast = model.predict(future)
            
            # Filtra apenas os períodos futuros
            forecast_future = forecast.tail(periods).copy()
            
            logger.info(f"Previsões geradas para {periods} meses")
            
            return forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Erro na geração de previsões: {e}")
            raise
    
    def _clear_old_forecasts(self, session: Session):
        """
        Remove previsões antigas do banco de dados
        
        Args:
            session: Sessão do banco de dados
        """
        try:
            # Remove todas as previsões existentes
            forecasts = session.exec(select(ProfitForecast)).all()
            for forecast in forecasts:
                session.delete(forecast)
            session.commit()
            
            logger.info(f"Removidas {len(forecasts)} previsões antigas")
            
        except Exception as e:
            logger.error(f"Erro ao limpar previsões antigas: {e}")
            session.rollback()
            raise
    
    def _save_forecasts(self, session: Session, forecasts_df: pd.DataFrame):
        """
        Salva previsões no banco de dados
        
        Args:
            session: Sessão do banco de dados
            forecasts_df: DataFrame com previsões
        """
        try:
            forecast_objects = []
            
            for _, row in forecasts_df.iterrows():
                forecast = ProfitForecast(
                    forecast_date=row['ds'].date(),
                    predicted_net_profit=float(row['yhat']),
                    lower_bound=float(row['yhat_lower']),
                    upper_bound=float(row['yhat_upper']),
                    model_version=self.model_version
                )
                forecast_objects.append(forecast)
            
            # Adiciona todos os objetos à sessão
            for forecast in forecast_objects:
                session.add(forecast)
            
            session.commit()
            logger.info(f"Salvadas {len(forecast_objects)} previsões no banco")
            
        except Exception as e:
            logger.error(f"Erro ao salvar previsões: {e}")
            session.rollback()
            raise
    
    def _analyze_forecast_quality(self, historical_df: pd.DataFrame, forecasts_df: pd.DataFrame):
        """
        Analisa qualidade das previsões geradas incluindo padrões sazonais
        
        Args:
            historical_df: DataFrame com dados históricos
            forecasts_df: DataFrame com previsões
        """
        try:
            # Estatísticas básicas
            hist_mean = historical_df['y'].mean()
            hist_std = historical_df['y'].std()
            pred_mean = forecasts_df['yhat'].mean()
            pred_std = forecasts_df['yhat'].std()
            
            # Análise de sazonalidade fitness
            hist_min = historical_df['y'].min()
            hist_max = historical_df['y'].max()
            pred_min = forecasts_df['yhat'].min()
            pred_max = forecasts_df['yhat'].max()
            
            logger.info("[ANALISE] Análise completa das previsões fitness:")
            logger.info(f"   [HISTORICO] Histórico - Média: ${hist_mean:,.2f} (±${hist_std:,.2f})")
            logger.info(f"   [PREVISAO] Previsões - Média: ${pred_mean:,.2f} (±${pred_std:,.2f})")
            logger.info(f"   [VARIACAO_HIST] Variação histórica: ${hist_min:,.2f} a ${hist_max:,.2f}")
            logger.info(f"   [VARIACAO_PREV] Variação prevista: ${pred_min:,.2f} a ${pred_max:,.2f}")
            
            # Calcula diferença percentual
            change_percent = ((pred_mean - hist_mean) / hist_mean) * 100 if hist_mean != 0 else 0
            if abs(change_percent) > 50:
                logger.warning(f"[ALERTA] Grande mudança prevista: {change_percent:+.1f}%")
            else:
                logger.info(f"   [TENDENCIA] Tendência: {change_percent:+.1f}% em relação ao histórico")
            
            # Análise específica dos meses previstos (sazonalidade fitness)
            forecast_months = []
            for _, row in forecasts_df.iterrows():
                month = row['ds'].month
                month_names = {
                    1: 'Jan [PICO]', 2: 'Fev [ALTA]', 3: 'Mar [NORMAL]', 4: 'Abr [NORMAL]', 
                    5: 'Mai [NORMAL]', 6: 'Jun [BAIXA]', 7: 'Jul [ALTA]', 8: 'Ago [NORMAL]',
                    9: 'Set [ALTA]', 10: 'Out [PICO]', 11: 'Nov [ALTA]', 12: 'Dez [BAIXA]'
                }
                month_name = month_names.get(month, f'M{month}')
                profit = row['yhat']
                forecast_months.append(f"{month_name}: ${profit:,.0f}")
            
            logger.info("   [MENSAL] Previsões por mês (com sazonalidade fitness):")
            for forecast_month in forecast_months:
                logger.info(f"      {forecast_month}")
            
            # Alerta para qualidade dos dados
            if len(historical_df) < 12:
                logger.warning(f"⚠️  Dados limitados ({len(historical_df)} pontos). Recomendado: 12+ para sazonalidade completa")
            elif len(historical_df) >= 24:
                logger.info("✅ Dados suficientes para capturar padrões sazonais robustos")
            else:
                logger.info("📊 Dados adequados para previsão com sazonalidade básica")
            
        except Exception as e:
            logger.warning(f"Erro na análise de qualidade: {e}")

    def generate_and_save_forecasts(self, periods: int = 6, force_update: bool = False) -> ForecastSummary:
        """
        Método principal para gerar e salvar previsões
        
        Args:
            periods: Número de meses para prever (padrão: 6)
            force_update: Se True, força a regeneração mesmo com previsões existentes
            
        Returns:
            Resumo das previsões geradas
        """
        with Session(engine) as session:
            try:
                logger.info(f"🔄 Iniciando geração de previsões fitness (modelo {self.model_version})")
                
                # 1. Busca dados históricos
                df = self._fetch_historical_data(session)
                
                # 2. Valida e limpa dados (inclui tratamento de outliers)
                df = self._validate_data(df)
                
                # 3. Treina modelo com sazonalidade fitness
                model = self._train_model(df)
                self.prophet_model = model
                
                # 4. Gera previsões
                forecasts_df = self._generate_forecast(model, periods)
                
                # 5. Limpa previsões antigas
                self._clear_old_forecasts(session)
                
                # 6. Salva novas previsões
                self._save_forecasts(session, forecasts_df)
                
                # 7. Analisa e reporta qualidade das previsões
                self._analyze_forecast_quality(df, forecasts_df)
                
                # 8. Retorna resumo
                avg_profit = float(forecasts_df['yhat'].mean())
                start_date = forecasts_df['ds'].min().date()
                end_date = forecasts_df['ds'].max().date()
                
                return ForecastSummary(
                    total_forecasts=len(forecasts_df),
                    forecast_period_start=start_date,
                    forecast_period_end=end_date,
                    avg_predicted_profit=avg_profit,
                    model_version=self.model_version,
                    created_at=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Erro no processo de previsão: {e}")
                raise

    def get_current_forecasts(self) -> List[ForecastResponse]:
        """
        Retorna as previsões atuais do banco de dados
        Usado apenas pelo scheduler
        """
        with Session(engine) as session:
            try:
                results = session.exec(select(ProfitForecast)).all()
                results = sorted(results, key=lambda x: x.forecast_date)
                
                forecasts = []
                for forecast in results:
                    response = ForecastResponse(
                        forecast_date=forecast.forecast_date,
                        predicted_net_profit=forecast.predicted_net_profit,
                        lower_bound=forecast.lower_bound,
                        upper_bound=forecast.upper_bound,
                        confidence_interval=forecast.upper_bound - forecast.lower_bound
                    )
                    forecasts.append(response)
                
                return forecasts
                
            except Exception as e:
                logger.error(f"Erro ao buscar previsões: {e}")
                raise
