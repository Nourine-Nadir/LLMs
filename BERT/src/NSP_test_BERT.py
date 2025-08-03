from transformers import BertTokenizerFast, BertForPreTraining
import torch

# Load fine-tuned model
tokenizer = BertTokenizerFast.from_pretrained("./bert_mlm_nsp")
model = BertForPreTraining.from_pretrained("./bert_mlm_nsp")
model.eval()

# Sample sentence pairs
sentence_a = "Les produits imposables réceptionnés par les entrepositaires doivent être immédiatement"
sentence_b_true = "pris encharge dans les comptes matières et l’acquit-à-caution ayant légitimé leur"
sentence_b_false = "Le chat est beau"

def predict_nsp(sent1, sent2):
    encoding = tokenizer(
        sent1,
        sent2,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.seq_relationship_logits
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs).item()

    print(f"Sentence A: {sent1}")
    print(f"Sentence B: {sent2}")
    print(f"Is Next Sentence: {'Yes' if pred_label == 1 else 'No'} (Confidence: {probs[0][pred_label]:.4f})")

# Run tests
predict_nsp(sentence_a, sentence_b_true)   # Expect Yes
predict_nsp(sentence_a, sentence_b_false)  # Expect No