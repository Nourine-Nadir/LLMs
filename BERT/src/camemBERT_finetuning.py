# Import packages
from transformers import  Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import CamembertTokenizer, CamembertForMaskedLM

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
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base", return_overflowing_tokens=False)
    model = CamembertForMaskedLM.from_pretrained("camembert-base")

    dataset = NSPDataset(sentence_pairs, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./camemBERT_mlm_nsp",
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

    model.save_pretrained("./camemBERT_mlm_nsp")
    tokenizer.save_pretrained("./camemBERT_mlm_nsp")
