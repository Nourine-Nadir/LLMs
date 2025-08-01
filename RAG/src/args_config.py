PARSER_CONFIG = {
    'model_name':{
    'flags' : ('-mn', '--model_name'),
    'help': 'The LLM model name',
    'type': str,
    'default':'mistral:instruct'
    },
    'embedding_model_name':{
    'flags' : ('-emn', '--embedding_model_name'),
    'help': 'The embedding model name',
    'type': str,
    'default':'deepseek-r1:14b'
    },

    'data_filepath':{
            'flags' : ('-dfp', '--data_filepath'),
            'help': 'JSON data file path for context extraction',
            'type':str,
            'default': '../../Data/JSON_data/All_laws.json'
        },

    'tf_idf':{
        'flags' : ('-tfidf', '--tf_idf'),
        'help': 'Enable use of TF-IDF',
        'type':bool,
        'default': True
        },

    'tf_idf_topk':{
        'flags' : ('-topk', '--tf_idf_topk'),
        'help': 'Number of Top documents to extract for the context ',
        'type':int,
        'default': 5
        },





}