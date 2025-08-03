from transformers import CamembertTokenizer, CamembertForMaskedLM
import torch

# Load your fine-tuned model
tokenizer = CamembertTokenizer.from_pretrained("./bert_mlm_nsp")
model = CamembertForMaskedLM.from_pretrained("./bert_mlm_nsp")
model.eval()



def predict_mask(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    print("Tokenized input IDs:", inputs["input_ids"])
    print("Decoded tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    print("Mask token ID:", tokenizer.mask_token_id)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # CamemBERT uses `.logits` not `prediction_logits`

    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

    print(f"Input: {sentence}")
    print("Predictions:")
    for token_id in top_tokens:
        token = tokenizer.decode([token_id])
        print(f"  - {token}")

# Run test
# Sentence with [MASK]
masked_sentence = f"Les autorisations d‘achats en franchise de la {tokenizer.mask_token} sur la valeur ajoutée sont"
predict_mask(masked_sentence)


