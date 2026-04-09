TERM_CLASSES = [
    'Method',
    'Activity',
    'Science',
    'Object',
    'Person',
    'InfoResource',
    'Task',
    'Organization',
    'Environment',
    'Model',
    'Metric',
    'Value',
    'Application',
    'Date',
    'Lang',
    'Dataset',
]

TERM_CLASS_TO_ID = {cls: i for i, cls in enumerate(TERM_CLASSES)}
TERM_ID_TO_CLASS = {i: cls for i, cls in enumerate(TERM_CLASSES)}
