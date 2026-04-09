from typing import List, Union

from pyvis.network import Network

from semantics_analysis.entities import Relation, Term, TermMention
from semantics_analysis.constants import TERM_CLASSES as LABEL_LIST

colors = [
    '#f44336', '#e81e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688',
    '#4caf50', '#8bc34a', '#cddc39', '#ff5722', '#ffc107', '#ff9800', '#ffeb3b'
]


def get_color(class_: str):
    if class_ in LABEL_LIST:
        return colors[LABEL_LIST.index(class_)]
    return '#000000'


def image_by_class(class_: str):
    return f'icons/{class_.lower()}.png'


def preprocess_is_alternative_name(relations: List[Relation]) -> List[Relation]:
    alt_rel = None

    for rel in relations:
        if rel.predicate == 'isAlternativeNameFor':
            alt_rel = rel
            break

    if alt_rel is None:
        return relations

    new_relations = [Relation(alt_rel.term1, 'alternativeNameFor', alt_rel.term2)]
    alt_term = alt_rel.term1

    for rel in relations:
        if rel.term1 != alt_term and rel.term2 != alt_term:
            new_relations.append(rel)

    return preprocess_is_alternative_name(new_relations)


def get_title(term: Union[Term, TermMention]) -> str:
    window = 50

    if isinstance(term, Term) and not term.mentions:
        return (f'Class: {term.class_}\n'
                f'Value: {term.value}')

    if isinstance(term, TermMention):
        mention = term
    else:
        mention = term.mentions[0]

    context = mention.text[:mention.start_pos][-window:] + f'[{mention.value}]' + mention.text[mention.end_pos:][:window]
    context = '...' + context + '...'

    return (f'Class: {term.class_}\n'
            f'Value: {term.value}\n'
            f'Context: {context}')


def get_node(term: Union[Term, TermMention]) -> str:
    if isinstance(term, TermMention):
        return f'{term.norm_value}, {term.class_}'

    return f'{term.value}, {term.class_}'


def display_relation_graph(
        terms: List[Term],
        relations: List[Relation],
        output_file: str = 'relations.html'
):
    if not relations:
        return

    relations = list(set(relations))

    nt = Network(
        width='100%',
        height='800px',
        notebook=False,
        directed=True,
        font_color='#000000',
        neighborhood_highlight=True
    )

    mass = 5

    size = 20

    considered_terms = set()

    # display relations
    for rel in relations:
        color1 = get_color(rel.term1.class_)
        color2 = get_color(rel.term2.class_)

        node1 = get_node(rel.term1)
        node2 = get_node(rel.term2)

        considered_terms.add(rel.term1)
        considered_terms.add(rel.term2)

        nt.add_node(
            node1,
            title=get_title(rel.term1),
            mass=mass,
            size=size,
            label=rel.term1.value,
            color=color1,
            shape='image',
            image=image_by_class(rel.term1.class_)
        )
        nt.add_node(
            node2,
            title=get_title(rel.term2),
            mass=mass,
            size=size,
            label=rel.term2.value,
            color=color2,
            shape='image',
            image=image_by_class(rel.term2.class_)
        )

        if color1 in ['#ffeb3b', '#ffc107', '#ff9800']:
            opacity = 'AA'
        else:
            opacity = '66'

        edge_color = '#' + color1[1:] + opacity

        nt.add_edge(
            node1,
            node2,
            label=rel.predicate,
            title=rel.predicate,
            color=edge_color,
            arrowStrikethrough=False,
            font={
                'size': 10,
                'align': 'top'
            }
        )

    for term_id, term in enumerate(terms):
        if term not in considered_terms:
            continue

        first_mention = term.mentions[0]
        mentions = [first_mention] + [m for m in term.mentions[1:] if m.norm_value != first_mention.norm_value]

        for mention in mentions:
            nt.add_node(
                get_node(mention),
                title=get_title(mention),
                mass=mass,
                size=size,
                label=mention.norm_value,
                color=get_color(term.class_),
                shape='image',
                image=image_by_class(term.class_),
                group=term_id
            )

        pairs = [p for p in zip(mentions[1:], mentions)]

        if len(mentions) > 2:
            pairs.append((mentions[0], mentions[-1]))

        for term1, term2 in pairs:
            color = get_color(term1.class_)

            if color in ['#ffeb3b', '#ffc107', '#ff9800']:
                opacity = 'AA'
            else:
                opacity = '66'

            edge_color = '#' + color[1:] + opacity

            nt.add_edge(
                get_node(term1),
                get_node(term2),
                label='isAlternativeNameFor',
                title='isAlternativeNameFor',
                color=edge_color,
                arrowStrikethrough=False,
                font={
                    'size': 10,
                    'align': 'top'
                }
            )

    # nt.repulsion(node_distance=300)
    # nt.barnes_hut(gravity=-200)
    nt.force_atlas_2based(gravity=-70, central_gravity=0.02, spring_strength=0.09)

    nt.show_buttons(filter_=['physics'])
    nt.show(output_file, notebook=False)


def main():
    # sentences = read_sentences('tests/sentences.json')
    #
    # for sent in sentences:
    #     if 13 < len(sent.relations) < 100:
    #         display_relation_graph([], preprocess_is_alternative_name(sent.relations))
    #         sleep(5)

    display_relation_graph([], relations=[
        Relation(
            term1=Term(value='Mistral 8x7b', ontology_class='Model', mentions=[]),
            predicate='является примером',
            term2=Term(value='LLM', ontology_class='Model', mentions=[]),
        ),
        Relation(
            term1=Term(value='Mistral 8x7b', ontology_class='Model', mentions=[]),
            predicate='создана',
            term2=Term(value='декабрь 2023г.', ontology_class='Date', mentions=[]),
        ),
        Relation(
            term1=Term(value='Mistral 8x7b', ontology_class='Model', mentions=[]),
            predicate='имеет автора',
            term2=Term(value='Mistral AI', ontology_class='Organization', mentions=[]),
        ),
        Relation(
            term1=Term(value='LLM', ontology_class='Model', mentions=[]),
            predicate='решает задачу',
            term2=Term(value='извлечение отношений', ontology_class='Task', mentions=[]),
        ),
    ])


if __name__ == '__main__':
    main()
