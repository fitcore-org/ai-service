from sqlmodel import Session, delete
from database.db_connect import engine
from model.models import Feedback, Word_Frequency

def create_test_feedbacks():
    """Cria feedbacks de teste realistas para validar o sistema"""
    
    test_feedbacks = [
        # === FEEDBACKS POSITIVOS (15 feedbacks) ===
        # Foco em instrutores, limpeza, equipamentos e ambiente
        "Adorei o treino hoje! O instrutor foi super atencioso e motivador, fez toda a diferença.",
        "A academia está sempre limpa e os equipamentos são novos. Excelente estrutura!",
        "A aula de spinning foi incrível, muito divertida e com uma energia contagiante. Recomendo muito!",
        "Melhor decisão que tomei foi treinar aqui. Ambiente agradável e professores excelentes.",
        "O custo-benefício é ótimo. Aparelhos de qualidade e um preço justo pela estrutura que oferecem.",
        "Parabéns pela manutenção, todos os aparelhos funcionando perfeitamente. Dá gosto de treinar assim.",
        "A recepcionista foi muito simpática e me ajudou com minhas dúvidas. Ótimo atendimento desde a entrada.",
        "Vestiário sempre limpo e organizado. É um detalhe que faz uma grande diferença na experiência.",
        "Amo as aulas coletivas! São sempre animadas e os professores são fantásticos.",
        "Estou vendo resultados incríveis no meu corpo. O acompanhamento dos instrutores é perfeito.",
        "A playlist da academia hoje estava perfeita para treinar pesado! Som ambiente muito bom.",
        "Mesmo em horário de pico, a academia é bem organizada. Vale a pena cada centavo.",
        "Equipamentos modernos e bem cuidados. A variedade de halteres é excelente.",
        "Uma das melhores academias que já frequentei. Limpeza, atendimento e estrutura impecáveis.",
        "Muito satisfeito! O ambiente é acolhedor e os treinos são ótimos.",

        # === FEEDBACKS NEGATIVOS (15 feedbacks) ===
        # Foco em equipamentos quebrados, superlotação, limpeza e atendimento ruim
        "Péssima experiência. A esteira estava quebrada de novo e o ar condicionado não funciona direito.",
        "Horrível treinar à noite, impossível usar os aparelhos de tão lotado. Faltam equipamentos.",
        "O professor de hoje parecia despreparado e mal deu atenção para os alunos. Fiquei decepcionado.",
        "Vestiário muito sujo, com cheiro ruim. A limpeza deixa muito a desejar.",
        "A mensalidade é muito cara para o que oferecem. Aparelhos velhos e precisa de manutenção urgente.",
        "O instrutor foi grosso quando pedi ajuda com o exercício. Atendimento terrível.",
        "Música ambiente muito alta e de mau gosto. Impossível se concentrar no treino.",
        "Cancelei minha matrícula. A academia está sempre cheia e os equipamentos vivem ocupados.",
        "O banheiro estava sem papel e com o chão molhado. Uma nojeira, qualidade péssima.",
        "Ninguém para ajudar na área de musculação. Os instrutores ficam conversando e ignoram os alunos.",
        "O ar condicionado está quebrado há semanas, treinar aqui se tornou insuportável e abafado.",
        "Muito insatisfeito. Prometeram uma avaliação física que nunca aconteceu.",
        "Aparelho de leg press quebrado há mais de um mês. Um descaso com os alunos.",
        "A aula de Zumba foi desanimada, a professora parecia sem energia. Experiência ruim.",
        "Preço alto para uma estrutura tão fraca e um atendimento tão ruim. Não recomendo.",

        # === FEEDBACKS NEUTROS (10 feedbacks) ===
        # Foco em observações factuais, sem forte carga emocional
        "A academia é ok. Os equipamentos são um pouco antigos, mas a maioria funciona.",
        "O treino hoje foi padrão, sem nada de excepcional. O movimento estava normal.",
        "A quantidade de esteiras é razoável, mas poderia ter mais anilhas de 5kg.",
        "O atendimento na recepção foi básico, apenas o necessário. Preço na média do bairro.",
        "A aula de funcional estava bem puxada hoje. A sala estava com bastante gente.",
        "A estrutura é mediana. Não é ruim, mas também não tem grandes diferenciais.",
        "A academia cumpre o que promete, mas sem surpresas. É uma experiência comum.",
        "O ar condicionado estava funcionando bem na área de cárdio. O restante estava normal.",
        "O valor da mensalidade é compatível com academias do mesmo porte na região.",
        "O horário de funcionamento é bom. O fluxo de pessoas varia bastante durante o dia."
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