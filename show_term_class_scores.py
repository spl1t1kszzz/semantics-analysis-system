from rich.progress import Progress, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.console import Console

from display_scores import parse_scores

storage = {}

rel_cnt_scores = parse_scores('tests/new_scores.json', [], storage)

rel_f1 = []

for rel, cnt, scores in rel_cnt_scores:
    recall = scores['Recall']
    precision = scores['Precision']

    f1 = 2 * recall * precision / (recall + precision)

    f1 = int(f1 * 100) / 100.0

    if cnt < 30:
        continue

    # if cnt < 15:
    #     continue

    # if 'Alternative' not in rel:
    #     continue

    rel_f1.append((rel, f1))

rel_f1 = sorted(rel_f1, key=lambda x: x[1], reverse=True)

alias_by_rel_id = {
    'Model_Language_Lang': 'Модель поддерживает язык',
    'Task_isSolvedIn_Science': 'Задача решается в разделе науки',
    'Metric_hasValue_Value': 'Метрика имеет значение',
    'Metric_isUsedFor_Model': 'Метрика используется для оценки модели',
    'Metric_isUsedIn_Task': 'Метрика используется для оценки решения задачи',
    'Method_isAppliedTo_Object': 'Метод применяется к объекту исследования',
    'Model_isUsedForSolving_Task': 'Модель используется для решения задачи',
    'Method_solves_Task': 'Метод решает задачу',
    'Application_hasAuthor_Organization': 'Приложение имеет автора — организацию',
    'Dataset_isTrainedForSolving_Task': 'Набор данных используется для решения задачи',
    'Application_isUsedForSolving_Task': 'Приложение используется для решения задачи',
    'Object_isUsedInSolving_Task': 'Объект используется для решения задачи',
    'Application_isAppliedTo_Object': 'Приложение применяется к объекту исследования',
    'Model_hasAuthor_Organization': 'Модель имеет автора — организацию',
    'Method_uses_Model': 'Метод использует модель',
    'Model_isExampleOf_Model': 'Модель является примером элемента из класса моделей',
    'Model_isModificationOf_Model': 'Модель является примером элемента из класса моделей',
    'Model_isPartOf_Model': 'Модель является примером элемента из класса моделей',
    'Method_isPartOf_Method': 'Метод является компонентой другого метода'
}


alias_by_class_id = {
    'Model': 'Модель',
    'Science': 'Раздел науки',
    'Method': 'Метод',
    'Application': 'Приложение',
    'Organization': 'Организация',
    'Object': 'Объект',
    'Activity': 'Деятельность',
    'Person': 'Персона',
    'Metric': 'Метрика',
    'Library': 'Библиотека',
    'Task': 'Задача',
    'Dataset': 'Набор данных',
    'Lang': 'Язык',
    'Value': 'Значение'
}


alias_by_rel_name = {
    'Language': 'поддерживает',
    'isSolvedIn': 'решается в',
    'hasValue': 'имеет',
    'isUsedFor': 'используется для',
    'isUsedIn': 'используется в',
    'isAppliedTo': 'применяется к',
    'isUsedForSolving': 'используется для решения',
    'isUsedInSolving': 'используется в решении',
    'isTrainedForSolving': 'обучен для решения',
    'hasAuthor': 'имеет автора',
    'uses': 'использует',
    'solves': 'решает',
    'isExampleOf': 'является примером',
    'isPartOf': 'является частью',
    'isComponentOf': 'является компонентой'
}

# length = 52
length = 14

table = Table()

table.add_column(header='Класс 1')
table.add_column(header='отношение')
table.add_column(header='Класс 2')
table.add_column(header='F1, %')

blue_style = 'rgb(76,160,173)'

for rel, f1 in rel_f1:
    if rel not in alias_by_rel_id:
        continue

    # description = alias_by_rel_id[rel]

    class1, rel_name, class2 = rel.split('_')
    # description = alias_by_class_id[class1]

    class1, class2 = alias_by_class_id[class1], alias_by_class_id[class2]
    rel_name = alias_by_rel_name[rel_name]

    # if len(description) < length:
    #     description = ' ' * (length - len(description)) + description

    f1 = int(f1 * 100)

    # if f1 == 100:
    #     f1 = 98
    #
    if f1 >= 80:
        style = 'green'
    elif f1 >= 60:
        style = 'orange3'
    elif f1 >= 40:
        style = 'yellow'
    else:
        style = 'red'

    table.add_row(class1, rel_name, class2, Text(str(f1), style=style))

    # with Progress(
    #         TextColumn(text_format=description, justify='left'),
    #         BarColumn(complete_style=blue_style),
    #         TextColumn(text_format=f'{f1}%', style=blue_style)
    # ) as progress:
    #     metric_task = progress.add_task(total=100, description='', completed=f1)

console = Console()

console.print(table)

score_by_class = {
    'Метод': 85,
    'Деятельность': 93,
    'Раздел науки': 90,
    'Объект': 90,
    'Персона': 93,
    'Информационный ресурс': 88,
    'Задача': 91,
    'Организация': 94,
    'Окружение': 99,
    'Модель': 77,
    'Метрика': 91,
    'Значение': 99,
    'Приложение': 82,
    'Дата': 100,
    'Язык': 100,
    'Набор данных': 85,
}

# score_by_class = [(k, v) for k, v in score_by_class.items()]
#
# score_by_class = sorted(score_by_class, key=lambda x: x[1], reverse=True)
#
# class_len = 22
#
#
# for class_, score in score_by_class:
#     if len(class_) < class_len:
#         class_ = ' ' * (class_len - len(class_)) + class_
#
#     with Progress(
#             TextColumn(text_format=class_, justify='left'),
#             BarColumn(complete_style=blue_style, finished_style=blue_style),
#             TextColumn(text_format=f'{score}%', style=blue_style)
#     ) as progress:
#         metric_task = progress.add_task(total=100, description='', completed=score)
