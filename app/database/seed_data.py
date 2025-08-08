from sqlmodel import Session, delete
from database.db_connect import engine
from model.models import Feedback, Word_Frequency

def create_test_feedbacks():
    """Cria feedbacks de teste realistas para validar o sistema"""
    
    test_feedbacks = [
        # === FEEDBACKS POSITIVOS (15 feedbacks) ===
        # Palavras-chave: instrutor, professor, equipamento, aparelho, esteira, halter, musculacao, funcional, spinning, treino, academia, limpeza, atendimento
        "Instrutor massa! Professor atencioso, treino motivador, equipamentos funcionando. Academia top!",
        "Equipamentos novos, esteira show, aparelhos limpos. Limpeza impecavel, estrutura excelente!",
        "Aula spinning incrivel! Professor energetico, musica animada. Funcional divertido, recomendo!",
        "Melhor academia! Professores capacitados, instrutores motivadores, ambiente agradavel. Treino eficiente!",
        "Custo beneficio otimo! Aparelhos qualidade, halteres variados, preco justo. Estrutura completa!",
        "Manutencao perfeita! Equipamentos funcionando, aparelhos conservados. Treino prazeroso, academia organizada!",
        "Atendimento excelente! Recepcionista simpatica, instrutores prestativos. Ajuda profissional, duvidas esclarecidas!",
        "Vestiario limpo! Banheiro organizado, chuveiro funcionando. Limpeza detalhada, higiene impecavel!",
        "Aulas coletivas fantasticas! Professores animados, funcional dinamico, spinning energico. Treinos variados!",
        "Resultados visiveis! Acompanhamento instrutor, treino personalizado, evolucao corporal. Metodologia eficaz!",
        "Som ambiente perfeito! Playlist motivadora, musica energetica. Treino pesado, concentracao total!",
        "Academia organizada! Horario pico controlado, equipamentos disponiveis. Investimento válido!",
        "Equipamentos modernos! Halteres completos, aparelhos atualizados. Variedade exercicios, tecnologia avancada!",
        "Academia referencia! Limpeza padrao, atendimento profissional, estrutura diferenciada. Experiencia superior!",
        "Ambiente acolhedor! Treinos eficazes, professores qualificados. Satisfacao total, academia recomendada!",

        # === FEEDBACKS NEGATIVOS (15 feedbacks) ===
        # Palavras-chave: quebrado, sujo, lotado, caro, ruim, defeito, problema, lento, barulhento
        "Experiencia pessima! Esteira quebrada, ar condicionado defeituoso. Equipamentos ruins, manutencao inexistente!",
        "Academia lotada! Aparelhos ocupados, equipamentos insuficientes. Treino impossivel, superlotacao terrivel!",
        "Professor despreparado! Instrutor desatencioso, acompanhamento ruim. Decepcionante, orientacao inadequada!",
        "Vestiario sujo! Banheiro fedorento, chao molhado. Limpeza pessima, higiene deploravel!",
        "Mensalidade cara! Aparelhos velhos, equipamentos obsoletos. Manutencao deficiente, investimento desperdicado!",
        "Instrutor grosseiro! Atendimento terrivel, ajuda negada. Exercicio mal orientado, experiencia horrivel!",
        "Musica barulhenta! Som alto, playlist ruim. Concentracao impossivel, ambiente estressante!",
        "Academia superlotada! Equipamentos sempre ocupados, aparelhos disputados. Matricula cancelada, experiencia frustrante!",
        "Banheiro nojento! Papel inexistente, chao sujo. Qualidade pessima, higiene abandonada!",
        "Instrutores ausentes! Musculacao desassistida, orientacao negada. Professores conversando, alunos ignorados!",
        "Ar condicionado quebrado! Ambiente abafado, treino insuportavel. Manutencao atrasada, problema cronico!",
        "Promessas falsas! Avaliacao fisica cancelada, acompanhamento inexistente. Insatisfacao total, servico deficiente!",
        "Leg press quebrado! Aparelho defeituoso, manutencao atrasada. Descaso administracao, equipamento abandonado!",
        "Aula desanimada! Professora desmotivada, zumba sem energia. Experiencia decepcionante, qualidade baixa!",
        "Preco exorbitante! Estrutura fraca, atendimento deficiente. Academia superestimada, investimento ruim!",

        # === FEEDBACKS NEUTROS (10 feedbacks) ===
        # Palavras-chave: regular, medio, comum, basico, padrao, razoavel
        "Academia regular. Equipamentos antigos funcionando, estrutura basica. Experiencia comum, nada excepcional.",
        "Treino padrao. Movimento normal, fluxo regular. Equipamentos funcionais, experiencia mediana.",
        "Esteiras razoaveis. Anilhas suficientes, aparelhos funcionando. Quantidade adequada, variedade basica.",
        "Atendimento basico. Recepcao funcional, orientacao minima. Preco medio, servico padrao.",
        "Funcional regular. Sala adequada, movimento controlado. Aula comum, intensidade media.",
        "Estrutura mediana. Equipamentos funcionais, manutencao regular. Diferenciais limitados, academia comum.",
        "Servico padrao. Promessas cumpridas, experiencia previsivel. Academia regular, surpresas inexistentes.",
        "Ar condicionado funcionando. Area cardio adequada, temperatura controlada. Condicoes normais, ambiente regular.",
        "Mensalidade compativel. Preco medio, custo regional. Academias similares, valor adequado.",
        "Horario adequado. Funcionamento regular, fluxo variavel. Frequencia normal, movimento controlado."
    ]
    
    with Session(engine) as session:
        print("Populando banco com dados de teste...")
        
        for text in test_feedbacks:
            feedback = Feedback(text=text) # sentiment fica None
            session.add(feedback)
        
        session.commit()
        print(f"{len(test_feedbacks)} feedbacks de teste criados!")

def clear_test_data():
    """Remove todos os dados de teste"""
    
    with Session(engine) as session:
        # Remove feedbacks e word_frequency
        session.execute(delete(Feedback))
        session.execute(delete(Word_Frequency))
        session.commit()
        print("Dados de teste removidos.")

if __name__ == "__main__":
    clear_test_data()
    create_test_feedbacks()
    print("Dados de teste criados com sucesso!")