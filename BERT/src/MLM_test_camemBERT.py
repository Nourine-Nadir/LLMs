from transformers import CamembertTokenizer, CamembertForMaskedLM
import torch

# Load your fine-tuned model
tokenizer = CamembertTokenizer.from_pretrained("./camemBERT_mlm_nsp")
model = CamembertForMaskedLM.from_pretrained("./camemBERT_mlm_nsp")
model.eval()

def predict_mask_all(sentence, top_k=5):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    mask_token_id = tokenizer.mask_token_id

    # Find all mask positions
    mask_token_indices = torch.where(input_ids == mask_token_id)[1]

    if mask_token_indices.nelement() == 0:
        print("No [MASK] token found in input.")
        return

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"Input: {sentence}")
    print("Predictions:")

    for i, mask_index in enumerate(mask_token_indices):
        mask_logits = logits[0, mask_index, :]
        top_token_ids = torch.topk(mask_logits, top_k, dim=-1).indices.tolist()
        decoded_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_token_ids]
        print(f"\nMask {i+1} at position {mask_index.item()}:")
        for token in decoded_tokens:
            print(f"  - {token}")

# Example input with multiple [MASK] tokens
masked_sentence = f"Si l'acquéreur n'a pas la qualité de commerçant, le nantissement est soumis auxdispositions des articles 151 à 159, 161 et 162 ci-dessus et celles du présent article. L'inscriptionprévue à l'article 153 du présent code est alors prise au greffe du tribunal dans le ressort duquelest domicilié {tokenizer.mask_token} du bien grevé.A défaut de payement à l'échéance, le créancier {tokenizer.mask_token} du privilège établi par le présentcode, peut faire {tokenizer.mask_token} à la vente publique sur la valeur ajoutée sont"
predict_mask_all(masked_sentence)
