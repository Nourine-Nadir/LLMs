from transformers import BertTokenizerFast, BertForPreTraining
import torch

tokenizer = BertTokenizerFast.from_pretrained("./bert_mlm_nsp")
model = BertForPreTraining.from_pretrained("./bert_mlm_nsp")
model.eval()

# Sentence with mask token
masked_sentence = "Le code des impôts indirects comporte l’ensemble des dispositions [MASK]."

def predict_mask(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.prediction_logits

    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    print(f"Input: {sentence}")
    print("Predictions:")
    for token_id in top_tokens:
        token = tokenizer.decode([token_id])
        print(f"  - {token}")

# Run test
predict_mask(masked_sentence)
