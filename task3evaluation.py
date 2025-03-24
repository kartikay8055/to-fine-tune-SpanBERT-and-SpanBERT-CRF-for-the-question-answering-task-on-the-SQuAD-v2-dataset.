import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset
from torchcrf import CRF  # Ensure you have installed pytorch-crf (pip install pytorch-crf)

########################################
# Custom Model: SpanBERT-CRF Definition
########################################

class SpanBERT_CRF(torch.nn.Module):
    def __init__(self, base_model_path):
        super(SpanBERT_CRF, self).__init__()
        # Load the base model (this directory must contain config.json, etc.)
        self.base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_path)
        # Initialize the CRF layer (using 2 tags: 0 for non-answer, 1 for answer span)
        self.crf = CRF(num_tags=2, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Stack the start and end logits to shape (batch_size, seq_len, 2)
        logits = torch.stack([outputs.start_logits, outputs.end_logits], dim=2)
        
        # In inference, we decode the predicted label sequence using the CRF
        if start_positions is None and end_positions is None:
            decoded = self.crf.decode(logits, mask=attention_mask.bool())
            return decoded
        else:
            # For training (not used here), create labels and compute loss
            batch_size, seq_len = input_ids.shape
            labels = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
            for i in range(batch_size):
                start = start_positions[i].item() if isinstance(start_positions[i], torch.Tensor) else start_positions[i]
                end = end_positions[i].item() if isinstance(end_positions[i], torch.Tensor) else end_positions[i]
                if 0 <= start < seq_len and 0 <= end < seq_len and start <= end:
                    labels[i, start:end+1] = 1
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return {"loss": loss}

########################################
# Evaluation Helper Functions
########################################

def make_predictions_base(model, dataset, tokenizer):
    """Generates predictions using the base model."""
    predictions = []
    for example in dataset:
        inputs = tokenizer(
            example["question"],
            example["context"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=384,
            return_offsets_mapping=True
        )
        # Save offset mapping and remove it from inputs (model doesn't use it)
        offset_mapping = inputs.pop("offset_mapping")
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits

        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)

        # Use the offset mapping to extract the answer span from the original context.
        start_char = offset_mapping[0][start_pred.item()][0]
        end_char = offset_mapping[0][end_pred.item()][1]
        answer = example["context"][start_char:end_char]
        predictions.append(answer)
    return predictions

def make_predictions_crf(model, dataset, tokenizer):
    """Generates predictions using the CRF model."""
    predictions = []
    for example in dataset:
        inputs = tokenizer(
            example["question"],
            example["context"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=384,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")
        with torch.no_grad():
            # The CRF model returns a list of label sequences.
            label_seq = model(**inputs)
            labels = label_seq[0]  # since batch size is 1

        # Identify the contiguous block of tokens labeled as 1 (the predicted answer span)
        start_idx = None
        end_idx = None
        for idx, label in enumerate(labels):
            if label == 1 and start_idx is None:
                start_idx = idx
            elif label == 0 and start_idx is not None:
                end_idx = idx - 1
                break
        if start_idx is not None:
            if end_idx is None:
                end_idx = len(labels) - 1
            start_char = offset_mapping[0][start_idx][0]
            end_char = offset_mapping[0][end_idx][1]
            answer = example["context"][start_char:end_char]
        else:
            answer = ""
        predictions.append(answer)
    return predictions

def exact_match_score(predictions, references):
    """Calculates the Exact Match (EM) score (as a percentage)."""
    assert len(predictions) == len(references), "Prediction and reference lists must be the same length"
    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    return matches / len(references) * 100

# Main Evaluation Code

if __name__ == "__main__":
    # Load the validation dataset (SQuAD v2)
    print("Loading validation dataset...")
    val_dataset = load_dataset("squad_v2", split="validation")
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    
    # Prepare the reference answers.
    # For examples with no answer, assign an empty string.
    references = {example['id']: (example['answers']['text'][0] if len(example['answers']['text']) > 0 else "") 
                  for example in val_dataset}

    ########################################
    # Load and Evaluate Base SpanBERT Model
    ########################################

    print("Loading saved Base SpanBERT model...")
    # Update the path below to where your base model is saved (should include config.json)
    model_base = AutoModelForQuestionAnswering.from_pretrained("./qa_model_base")
    
    print("Making predictions with Base SpanBERT model...")
    predictions_base = make_predictions_base(model_base, val_dataset, tokenizer)
    em_base = exact_match_score(predictions_base, [references[example["id"]] for example in val_dataset])
    print(f"Exact Match Score for Base SpanBERT: {em_base:.2f}%")
    
    ########################################
    # Load and Evaluate SpanBERT-CRF Model
    ########################################

    print("Loading saved SpanBERT-CRF model...")
    # Initialize the CRF model with the base model path
    model_crf = SpanBERT_CRF("./qa_model_base")
    # Load the saved CRF weights (update the path if necessary)
    model_crf.load_state_dict(torch.load("./qa_model_crf/spanbert_crf.pt"))
    
    print("Making predictions with SpanBERT-CRF model...")
    predictions_crf = make_predictions_crf(model_crf, val_dataset, tokenizer)
    em_crf = exact_match_score(predictions_crf, [references[example["id"]] for example in val_dataset])
    print(f"Exact Match Score for SpanBERT-CRF: {em_crf:.2f}%")
