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
    'default':'nomic-embed-text'
    },

    'data_filepath':{
            'flags' : ('-dfp', '--data_filepath'),
            'help': 'JSON data file path for context extraction',
            'type':str,
            'default': '../../Data/JSON_data/All_laws.json'
        },





}