import os
import holmes_extractor as holmes

if __name__ in ('__main__', 'example_chatbot_DE_insurance'):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    ontology = holmes.Ontology(os.sep.join((
        script_directory, 'example_chatbot_DE_insurance_ontology.owl')))
    holmes_manager = holmes.Manager(model='de_core_news_lg', ontology=ontology, number_of_workers=2)
    holmes_manager.register_search_phrase('Jemand benötigt eine Versicherung')
    holmes_manager.register_search_phrase('Ein ENTITYPER schließt eine Versicherung ab')
    holmes_manager.register_search_phrase('ENTITYPER benötigt eine Versicherung')
    holmes_manager.register_search_phrase('Eine Versicherung für einen Zeitraum')
    holmes_manager.register_search_phrase('Eine Versicherung fängt an')
    holmes_manager.register_search_phrase('Jemand zahlt voraus')

    holmes_manager.start_chatbot_mode_console()
    # e.g. 'Richard Hudson und Max Mustermann brauchen eine Krankenversicherung für die nächsten fünf Jahre'
