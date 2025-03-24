# Fine-Tuning SpanBERT and SpanBERT-CRF for Question Answering

## Objective
This project fine-tunes **SpanBERT** and **SpanBERT-CRF** for the **question-answering task** on the **SQuAD v2** dataset. The objective is to extract the **answer span** from the given context as per the question asked.

## 1. Preprocessing

### Tokenization
- The **question** and **context** are tokenized together using the **SpanBERT tokenizer**, ensuring proper handling of the input format.

### Overflow Handling
- If a **context is too long**, it is split into multiple chunks, and each chunk is processed independently while maintaining **context offsets**.

### Answer Span Identification
- For each tokenized sample, the **start and end positions** of the answer span are determined using the **offsets** and the **original answer's character positions**.

### Handling No-Answer Cases
- If a question has **no answer**, the **start and end positions** are set to **(0, 0)** to indicate no span.

### Data Preparation
- The processed data is structured to include:
  - **Tokenized inputs**
  - **Attention masks**
  - **Answer span positions** (start, end)

### Batch Processing
- Preprocessing is done in **batches** to optimize **memory usage** and **processing speed**.

### Output
- The pre-processed data is **ready for model input** during training and evaluation.

## 2. Justification of Model Choices and Hyperparameters

| Parameter           | Value |
|--------------------|-------|
| Learning Rate      | 2e-5  |
| Batch Size        | 8     |
| Epochs           | 6     |
| Weight Decay      | 0.01  |
| Max Sequence Length | 384   |
| Logging Steps      | 50    |

- A **low learning rate** (2e-5) prevents overfitting while fine-tuning the pre-trained model.
- **Weight decay (0.01)** is applied as a **regularization technique**.
- The **maximum sequence length** is set to **384** (question + context).

## 3. Training and Validation Plots
(TODO: Add plots here)

## 4. Exact Match Scores
| Model          | Exact Match (EM) Score |
|---------------|----------------------|
| SpanBERT      | 47.87%               |
| SpanBERT-CRF  | 50.80%               |

## 5. Comparative Analysis
### SpanBERT vs SpanBERT-CRF
- The **CRF layer** in the **SpanBERT-CRF** model improves its ability to extract **more accurate answer spans**.
- SpanBERT-CRF achieves a **higher EM score** than SpanBERT, making it a **more robust model** for **question answering**.

## 6. Conclusion
Based on the **Exact Match (EM) scores**, the **SpanBERT-CRF model** provides a **more accurate** solution for question-answering on **SQuAD v2**.

### Future Work
- Further **hyperparameter tuning**.
- Exploring **more sophisticated models** to enhance performance.

## 7. How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SpanBERT_QA.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SpanBERT_QA
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python task3.py
   ```

## 8. Acknowledgments
- **SpanBERT** and **SpanBERT-CRF** implementations.
- **SQuAD v2 dataset** for QA tasks.
- **Hugging Face Transformers library** for model fine-tuning.
