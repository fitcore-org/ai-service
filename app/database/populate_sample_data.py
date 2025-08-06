"""
Script para popular o banco com dados realísticos de negócio fitness

Este script cria dados sintéticos de lucros mensais seguindo a sazonalidade
específica de academias e aplicações fitness, incluindo:
- Picos em Janeiro (resoluções de ano novo) e Outubro (verão)
- Quedas em Junho (frio) e Dezembro (festas)
- Padrões realísticos de receita e despesas
"""

import uuid
import sys
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import random

# Adiciona o diretório pai ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session
from database.db_connect import engine
from model.models import Profit

def generate_sample_profit_data(months_back: int = 24) -> None:
    """
    Gera dados sintéticos de lucros mensais para negócio fitness
    
    Args:
        months_back: Número de meses históricos para gerar (incluindo o mês atual)
    """
    
    with Session(engine) as session:
        print(f"🔄 Gerando {months_back} meses de dados fitness...")
        
        # Data atual (agosto 2025)
        current_month = date(2025, 8, 1)
        
        # Data de início (meses atrás a partir do mês atual)
        start_date = current_month - relativedelta(months=months_back-1)
        
        # Parâmetros para simulação realística de negócio fitness
        base_revenue = 8500   # Receita base mensal (mais conservador para fitness)
        base_expenses = 6200  # Despesas base mensal
        
        # Tendência de crescimento anual mais realista para fitness
        growth_rate = 0.08  # 8% ao ano (crescimento orgânico)
        monthly_growth = growth_rate / 12
        
        # Sazonalidade específica para negócio fitness
        seasonal_factors = {
            1: 1.4,   # Janeiro - Alta de matrículas (resoluções de ano novo)
            2: 1.2,   # Fevereiro - Continuação do impulso de janeiro
            3: 1.1,   # Março - Estabilização
            4: 1.0,   # Abril - Normal
            5: 1.0,   # Maio - Normal
            6: 0.8,   # Junho - Queda por causa do frio
            7: 1.1,   # Julho - Férias escolares, leve alta de matrículas
            8: 1.0,   # Agosto - Volta gradual das atividades
            9: 1.1,   # Setembro - Volta às atividades normais
            10: 1.3,  # Outubro - Início da busca por forma para o verão
            11: 1.2,  # Novembro - Black Friday (promoções), mantém alta
            12: 0.9   # Dezembro - Redução geral (festas e férias)
        }
        
        profits_data = []
        
        for i in range(months_back):
            # Data do período
            current_date = start_date + relativedelta(months=i)
            period_start = current_date.replace(day=1)
            
            # Último dia do mês
            if current_date.month == 12:
                period_end = date(current_date.year + 1, 1, 1) - relativedelta(days=1)
            else:
                period_end = date(current_date.year, current_date.month + 1, 1) - relativedelta(days=1)
            
            # Aplicar tendência de crescimento gradual
            growth_factor = 1 + (monthly_growth * i)
            
            # Aplicar sazonalidade específica do fitness
            seasonal_factor = seasonal_factors.get(current_date.month, 1.0)
            
            # Variabilidade mais controlada para negócio fitness (±15%)
            random_factor = random.uniform(0.85, 1.15)
            
            # Calcular receitas com padrões de academia/fitness
            monthly_revenue = base_revenue * growth_factor * seasonal_factor * random_factor
            
            # Despesas variam menos que receitas (custos mais fixos)
            expense_variability = random.uniform(0.92, 1.08)  # ±8% de variação
            monthly_expenses = base_expenses * growth_factor * expense_variability
            
            # Ajuste específico para meses de alta sazonalidade (mais custos operacionais)
            if seasonal_factor > 1.2:  # Janeiro e Outubro
                monthly_expenses *= 1.1  # 10% mais custos operacionais
            
            # Garantir margem mínima positiva
            if monthly_revenue <= monthly_expenses:
                monthly_expenses = monthly_revenue * 0.85  # Margem mínima de 15%
            
            net_profit = monthly_revenue - monthly_expenses
            profit_margin = (net_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
            
            # Criar registro
            profit = Profit(
                id=str(uuid.uuid4()),
                period_start=period_start,
                period_end=period_end,
                total_revenue=round(monthly_revenue, 2),
                total_expenses=round(monthly_expenses, 2),
                net_profit=round(net_profit, 2),
                profit_margin=round(profit_margin, 2),
                created_at=datetime.now()
            )
            
            profits_data.append(profit)
            
            # Adicionar emoji indicativo da sazonalidade
            season_emoji = "🔥" if seasonal_factor >= 1.3 else "📈" if seasonal_factor >= 1.1 else "📉" if seasonal_factor <= 0.9 else "📊"
            
            print(f"📅 {period_start.strftime('%Y-%m')} {season_emoji}: Receita: ${monthly_revenue:,.2f}, "
                  f"Despesas: ${monthly_expenses:,.2f}, Lucro: ${net_profit:,.2f} (Margem: {profit_margin:.1f}%)")
        
        # Salvar no banco
        try:
            for profit in profits_data:
                session.add(profit)
            
            session.commit()
            print(f"✅ {len(profits_data)} registros de lucros salvos com sucesso!")
            
            # Estatísticas detalhadas
            avg_revenue = sum(p.total_revenue for p in profits_data) / len(profits_data)
            avg_profit = sum(p.net_profit for p in profits_data) / len(profits_data)
            avg_margin = sum(p.profit_margin for p in profits_data) / len(profits_data)
            
            # Encontrar melhores e piores meses
            best_month = max(profits_data, key=lambda x: x.net_profit)
            worst_month = min(profits_data, key=lambda x: x.net_profit)
            
            print(f"\n📊 Estatísticas dos dados fitness gerados:")
            print(f"   • Receita média mensal: ${avg_revenue:,.2f}")
            print(f"   • Lucro médio mensal: ${avg_profit:,.2f}")
            print(f"   • Margem de lucro média: {avg_margin:.1f}%")
            print(f"   • Melhor mês: {best_month.period_start.strftime('%Y-%m')} (${best_month.net_profit:,.2f})")
            print(f"   • Pior mês: {worst_month.period_start.strftime('%Y-%m')} (${worst_month.net_profit:,.2f})")
            print(f"   • Período: {profits_data[0].period_start} até {profits_data[-1].period_end}")
            print(f"\n🏋️  Sazonalidade fitness aplicada:")
            print(f"   🔥 Picos: Janeiro (Resoluções), Outubro (Verão)")
            print(f"   📉 Baixas: Junho (Frio), Dezembro (Festas)")
            print(f"   🎯 Inclui mês atual: Agosto 2025")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Erro ao salvar dados: {e}")
            raise

def clear_existing_data():
    """
    Limpa dados existentes de lucros
    """
    with Session(engine) as session:
        try:
            # Remove todos os registros existentes
            session.query(Profit).delete()
            session.commit()
            print("🗑️  Dados existentes removidos")
        except Exception as e:
            session.rollback()
            print(f"❌ Erro ao limpar dados: {e}")

if __name__ == "__main__":
    print("🏋️  Iniciando população do banco com dados fitness realísticos...")
    print("📅 Incluindo período até Agosto 2025 (mês atual)")
    
    # Pergunta se deve limpar dados existentes
    response = input("Deseja limpar dados existentes? (s/n): ").lower().strip()
    if response in ['s', 'sim', 'y', 'yes']:
        clear_existing_data()
    
    # Gera novos dados
    months = int(input("Quantos meses de histórico gerar? (padrão: 24): ") or "24")
    generate_sample_profit_data(months_back=months)
    
    print("\n✨ População do banco concluída com sazonalidade fitness!")
