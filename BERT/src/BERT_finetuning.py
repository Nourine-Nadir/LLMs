# Import packages
from transformers import BertTokenizerFast, BertForPreTraining, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Import modules
from parser import Parser
from args_config import PARSER_CONFIG
from utils import get_data, prepare_nsp_data, NSPDataset

try:
    parser = Parser(prog='BERT Fine Tuning',
                    description='BERT fine tuning with MLM and NSP')
    args = parser.get_args(
        PARSER_CONFIG
    )
except ValueError as e:
    args = None
    print(e)

# Get Data
json_path = args.data_filepath
texts, metadatas = get_data(json_path)

sentence_pairs = prepare_nsp_data(texts, load_data=True, save_data=False)
print(f'len of sents pairs : {len(sentence_pairs)}')
# Initialize model
train= True
if train:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForPreTraining.from_pretrained("bert-base-uncased")

    dataset = NSPDataset(sentence_pairs, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./bert_mlm_nsp",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=2000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none"  # disable WandB or others if not used
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    print('Start training...')
    trainer.train()

    model.save_pretrained("./bert_mlm_nsp")
    tokenizer.save_pretrained("./bert_mlm_nsp")
