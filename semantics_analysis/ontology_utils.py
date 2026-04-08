import json

predicates_by_class_pair = {}

relations_metadata_by_class_pair = {}

loaded_relation_ids = set()

attribute_classes = {'Date', 'Lang', 'Value'}

CLASS_ALIASES = {
    'модель', 'метод', 'задача', 'объект', 'язык', 'организация', 'приложение', 'метрика',
    'персона', 'набор данных', 'проект', 'исследование', 'библиотека'
}

with open('metadata/new_term_classes.json', 'r', encoding='utf-8') as f:
    term_metadata_by_class = json.load(f)

loaded_classes = term_metadata_by_class.keys()

with open('metadata/relations.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

for key, predicates in metadata.items():
    class1, class2 = key.split('_')

    predicates_by_class_pair[(class1, class2)] = list(predicates.keys())

    relations_metadata_by_class_pair[(class1, class2)] = predicates

    for predicate, metadata in predicates.items():
        rel_id = f'{class1}_{predicate}_{class2}'

        loaded_relation_ids.add(rel_id)

for class_ in loaded_classes:
    if class_ in attribute_classes:
        continue

    loaded_relation_ids.add(f'{class_}_isAlternativeNameFor_{class_}')
