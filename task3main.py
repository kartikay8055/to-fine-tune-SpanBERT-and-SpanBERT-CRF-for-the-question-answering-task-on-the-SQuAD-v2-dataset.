import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
from torchcrf import CRF  # Ensure you install via: pip install pytorch-crf

########################################
# Preprocessing Functions
########################################

def preprocess_train(examples):
    outputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = outputs.pop("overflow_to_sample_mapping")
    offset_mapping = outputs.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            answer_start = answers["answer_start"][0]
            answer_text = answers["text"][0]
            
            token_start_index = 0
            while token_start_index < len(offsets) and offsets[token_start_index][0] == 0:
                token_start_index += 1
            token_end_index = len(offsets) - 1
            while token_end_index >= 0 and offsets[token_end_index][1] == 0:
                token_end_index -= 1
                
            if not (offsets[token_start_index][0] <= answer_start and 
                    offsets[token_end_index][1] >= answer_start + len(answer_text)):
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_index = None
                end_index = None
                for idx, (start, end) in enumerate(offsets):
                    if start_index is None and start <= answer_start < end:
                        start_index = idx
                    if start < answer_start + len(answer_text) <= end:
                        end_index = idx
                        break
                if start_index is None:
                    start_index = 0
                if end_index is None:
                    end_index = 0
                start_positions.append(start_index)
                end_positions.append(end_index)
    
    outputs["start_positions"] = start_positions
    outputs["end_positions"] = end_positions
    return outputs

def preprocess_validation(examples):
    outputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = outputs.pop("overflow_to_sample_mapping")
    new_offset_mapping = []
    orig_ids = []
    orig_contexts = []
    orig_answers = []
    for i, sample_idx in enumerate(sample_mapping):
        example_id = examples["id"][sample_idx] if "id" in examples else str(sample_idx)
        orig_ids.append(example_id)
        orig_contexts.append(examples["context"][sample_idx])
        orig_answers.append(examples["answers"][sample_idx])
        new_offset_mapping.append(outputs["offset_mapping"][i])
    outputs["orig_id"] = orig_ids
    outputs["orig_context"] = orig_contexts
    outputs["orig_answers"] = orig_answers
    outputs["offset_mapping"] = new_offset_mapping
    return outputs

########################################
# Postprocessing Functions
########################################

def postprocess_qa_predictions(features, raw_predictions, tokenizer, max_answer_length=30):
    start_logits, end_logits = raw_predictions
    predictions = {}
    for i, feature in enumerate(features):
        example_id = feature["orig_id"][0]
        context = feature["orig_context"][0]
        offsets = feature["offset_mapping"]
        pred_start = int(np.argmax(start_logits[i]))
        pred_end = int(np.argmax(end_logits[i]))
        if pred_end < pred_start or (pred_end - pred_start + 1) > max_answer_length:
            answer = ""
        else:
            start_char = offsets[pred_start][0]
            end_char = offsets[pred_end][1]
            answer = context[start_char:end_char]
        if example_id not in predictions:
            predictions[example_id] = answer
    return predictions

def postprocess_crf_predictions(features, crf_predictions, tokenizer, max_answer_length=30):
    predictions = {}
    for i, feature in enumerate(features):
        example_id = feature["orig_id"][0]
        context = feature["orig_context"][0]
        offsets = feature["offset_mapping"]
        labels = crf_predictions[i]  # Predicted label sequence per token
        answer = ""
        start_idx = None
        for j, label in enumerate(labels):
            if label == 1 and start_idx is None:
                start_idx = j
            elif label == 0 and start_idx is not None:
                end_idx = j - 1
                break
        if start_idx is not None:
            end_idx = j - 1 if j > start_idx else start_idx
            start_char = offsets[start_idx][0]
            end_char = offsets[end_idx][1]
            answer = context[start_char:end_char]
        predictions[example_id] = answer
    return predictions

def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(references) * 100

########################################
# Custom Model: SpanBERT-CRF
########################################

class SpanBERT_CRF(torch.nn.Module):
    def __init__(self, base_model_name):
        super(SpanBERT_CRF, self).__init__()
        self.base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
        self.crf = CRF(num_tags=2, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = torch.stack([outputs.start_logits, outputs.end_logits], dim=2)  # (batch_size, seq_len, 2)
        
        if start_positions is not None and end_positions is not None:
            batch_size, seq_len = input_ids.shape
            labels = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
            for i in range(batch_size):
                start = start_positions[i].item() if isinstance(start_positions[i], torch.Tensor) else start_positions[i]
                end = end_positions[i].item() if isinstance(end_positions[i], torch.Tensor) else end_positions[i]
                if 0 <= start < seq_len and 0 <= end < seq_len and start <= end:
                    labels[i, start:end+1] = 1
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return {"loss": loss}
        else:
            decoded = self.crf.decode(logits, mask=attention_mask.bool())
            return decoded

########################################
# Training Metric Functions
########################################

def compute_metrics_qa(eval_pred):
    return {}

def compute_metrics_crf(eval_pred):
    return {}

########################################
# Execution Block
########################################

if __name__ == "__main__":
    # 1. Data Loading and Preprocessing
    print("Loading datasets...")
    train_dataset = load_dataset("squad_v2", split="train[:15000]")
    val_dataset = load_dataset("squad_v2", split="validation")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    
    print("Preprocessing training data...")
    train_dataset_processed = train_dataset.map(preprocess_train, batched=True, remove_columns=train_dataset.column_names)
    
    print("Preprocessing validation data...")
    val_dataset_processed = val_dataset.map(
        preprocess_validation,
        batched=True,
        batch_size=1,
        remove_columns=val_dataset.column_names
    )
    
    ########################################
    # 2. Fine-tuning Base SpanBERT for QA
    ########################################
    print("Loading base SpanBERT model...")
    model_spanbert = AutoModelForQuestionAnswering.from_pretrained("SpanBERT/spanbert-base-cased")
    
    training_args_base = TrainingArguments(
        output_dir="./qa_model_base",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        save_strategy="epoch",
    )
    
    trainer_base = Trainer(
        model=model_spanbert,
        args=training_args_base,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_qa,
    )
    
    print("Training base SpanBERT model...")
    trainer_base.train()
    model_spanbert.save_pretrained("./qa_model_base")
    
    ########################################
    # 3. Evaluation for Base SpanBERT
    ########################################
    print("Evaluating base SpanBERT model...")
    raw_preds_base = trainer_base.predict(val_dataset_processed)
    predictions_base = postprocess_qa_predictions(val_dataset_processed, raw_preds_base.predictions, tokenizer)
    
    references_base = {}
    for example in val_dataset:
        example_id = example["id"] if "id" in example else str(example["context"])
        references_base[example_id] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
    
    common_ids = set(references_base.keys()).intersection(set(predictions_base.keys()))
    em_base = exact_match_score([predictions_base[id] for id in common_ids],
                                [references_base[id] for id in common_ids])
    print(f"Exact Match Score for Base SpanBERT: {em_base:.2f}%")
    
    # Plot training loss for base model.
    train_losses_base = [log["loss"] for log in trainer_base.state.log_history if "loss" in log]
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_base, label="SpanBERT Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss for Base SpanBERT")
    plt.legend()
    plt.savefig("training_loss_spanbert.png")
    plt.show()
    
    ########################################
    # 4. Fine-tuning SpanBERT-CRF
    ########################################
    print("Initializing SpanBERT-CRF model...")
    spanbert_crf_model = SpanBERT_CRF("./qa_model_base")
    
    training_args_crf = TrainingArguments(
        output_dir="./qa_model_crf",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer_crf = Trainer(
        model=spanbert_crf_model,
        args=training_args_crf,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_crf,
    )
    
    print("Training SpanBERT-CRF model...")
    trainer_crf.train()
    os.makedirs("./qa_model_crf", exist_ok=True)
    torch.save(spanbert_crf_model.state_dict(), "./qa_model_crf/spanbert_crf.pt")
    
    ########################################
    # 5. Evaluation for SpanBERT-CRF
    ########################################
    print("Evaluating SpanBERT-CRF model...")
    crf_raw_preds = trainer_crf.predict(val_dataset_processed)
    crf_predictions = crf_raw_preds.predictions  # Already decoded label sequences.
    predictions_crf = postprocess_crf_predictions(val_dataset_processed, crf_predictions, tokenizer)
    
    references_crf = {}
    for example in val_dataset:
        example_id = example["id"] if "id" in example else str(example["context"])
        references_crf[example_id] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
    
    common_ids_crf = set(references_crf.keys()).intersection(set(predictions_crf.keys()))
    em_crf = exact_match_score([predictions_crf[id] for id in common_ids_crf],
                               [references_crf[id] for id in common_ids_crf])
    print(f"Exact Match Score for SpanBERT-CRF: {em_crf:.2f}%")
    
    # Plot training loss for CRF model.
    train_losses_crf = [log["loss"] for log in trainer_crf.state.log_history if "loss" in log]
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_crf, label="SpanBERT-CRF Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss for SpanBERT-CRF")
    plt.legend()
    plt.savefig("training_loss_spanbert_crf.png")
    plt.show()
    
    ########################################
    # 6. Comparative Training Loss Plot
    ########################################
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_base, label="Base SpanBERT")
    plt.plot(train_losses_crf, label="SpanBERT-CRF")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig("training_loss_comparison.png")
    plt.show()
