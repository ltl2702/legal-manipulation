# Manipulation Detection in Legal Conversations

Khóa luận tốt nghiệp ngành Khoa học Máy tính — Trường Đại học Công Nghệ, ĐHQGHN

Nghiên cứu đề xuất một framework tự động phát hiện và phân tích hành vi thao túng trong hội thoại pháp lý, cụ thể là các phiên xét xử tại tòa án. Framework kết hợp kiến trúc đa tác tử (multi-agent) dựa trên mô hình ngôn ngữ lớn Mistral-7B được fine-tune với mô hình Legal-BERT chuyên biệt cho văn bản pháp lý, nhằm giải quyết đồng thời ba tác vụ: (1) phát hiện sự hiện diện của thao túng, (2) xác định chủ thể thao túng, và (3) phân loại các kỹ thuật thao túng được sử dụng.

## Project Structure
 
```
KLTN/
├── data/
│   ├── raw-data/                       # Dữ liệu gốc chưa xử lý
│   │   ├── train_split_raw.csv
│   │   ├── val_split_raw.csv
│   │   └── test_split_raw.csv
│   ├── data-clean/                     # Dữ liệu sau tiền xử lý
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   └── test_split.csv
│   ├── raw-intent/                     # Ý định trích xuất từ raw data
│   │   ├── train_intent_raw.csv
│   │   ├── val_intent_raw.csv
│   │   └── test_intent_raw.csv
│   ├── intent/                         # Ý định trích xuất từ cleaned data
│   │   ├── train_intent.csv
│   │   ├── val_intent.csv
│   │   └── test_intent.csv
│   └── LegalCon-Dataset.csv            # Dataset gốc
│
├── model/
│   ├── finetune-mistral.ipynb          # Fine-tune Mistral-7B với LoRA
│   └── train-legalbert.ipynb           # Huấn luyện Legal-BERT classifier
│
├── baseline_experiments/
│   ├── zero-shot.ipynb                 # Pretrained & Fine-tuned LLM Zero-shot
│   ├── few-shot.ipynb                  # Pretrained & Fine-tuned LLM Few-shot
│   └── claim.ipynb                     # CLAIM baseline
│
├── output/
│   ├── manipulation_results.csv            # Proposed framework — cleaned data
│   ├── manipulation_results_raw.csv        # Proposed framework — raw data
│   ├── ablation_llm_only_results.csv       # Ablation A — không có multi-agent
│   ├── ablation_no_intent_results.csv      # Ablation B — không có intent module
│   ├── ablation_no_bert_results.csv        # Ablation C — không có Legal-BERT
│   ├── results_zeroshot_pretrained.csv     # Baseline pretrained zero-shot
│   ├── results_zeroshot_finetuned.csv      # Baseline finetuned zero-shot
│   ├── results_fewshot_pretrained.csv      # Baseline pretrained few-shot
│   ├── results_fewshot_finetuned.csv       # Baseline finetuned few-shot
│   └── claim_results.csv                   # Baseline CLAIM
│
├── data_preprocessing.ipynb           # Bước 1 — Làm sạch và phân chia dữ liệu
├── stage1_intent.ipynb                # Bước 2 — Trích xuất ý định
├── proposed_framework.ipynb           # Bước 3 — Full proposed framework
├── rawdata_framework.ipynb            # Framework chạy trên raw data
├── ablation_study.ipynb               # Thí nghiệm cắt bỏ (Ablation A, B, C)
├── analysis.ipynb                     # Phân tích lỗi và trực quan hóa
├── .gitignore
└── README.md
```
 
## Running Order
 
| Bước | Notebook | Output |
|------|----------|--------|
| 1 | `data_preprocessing.ipynb` | `data/data-clean/` |
| 2 | `stage1_intent.ipynb` | `data/intent/`, `data/raw-intent/` |
| 3 | `model/finetune-mistral.ipynb` | Fine-tuned Mistral weights |
| 4 | `model/train-legalbert.ipynb` | Legal-BERT weights |
| 5 | `proposed_framework.ipynb` | `output/manipulation_results.csv` |
| 6 | `rawdata_framework.ipynb` | `output/manipulation_results_raw.csv` |
| 7 | `baseline_experiments/zero-shot.ipynb` | `output/results_zeroshot_*.csv` |
| 8 | `baseline_experiments/few-shot.ipynb` | `output/results_fewshot_*.csv` |
| 9 | `baseline_experiments/claim.ipynb` | `output/claim_results.csv` |
| 10 | `ablation_study.ipynb` | `output/ablation_*.csv` |
| 11 | `analysis.ipynb` | Figures và error analysis |
 
## Output Files
 
| File | Mô tả |
|------|--------|
| `manipulation_results.csv` | Framework đề xuất trên cleaned data |
| `manipulation_results_raw.csv` | Framework đề xuất trên raw data |
| `ablation_llm_only_results.csv` | Ablation A — no multi-agent |
| `ablation_no_intent_results.csv` | Ablation B — no intent module |
| `ablation_no_bert_results.csv` | Ablation C — no Legal-BERT |
| `results_zeroshot_pretrained.csv` | Pretrained LLM zero-shot |
| `results_zeroshot_finetuned.csv` | Fine-tuned LLM zero-shot |
| `results_fewshot_pretrained.csv` | Pretrained LLM few-shot |
| `results_fewshot_finetuned.csv` | Fine-tuned LLM few-shot |
| `claim_results.csv` | CLAIM baseline |
 

## Notes
- Tất cả thí nghiệm chạy trên Kaggle với GPU Tesla T4 (15.6 GB VRAM).
- Model weights không được lưu trong repo do giới hạn dung lượng. Mỗi notebook sử dụng bộ weights được fine-tune riêng trên dữ liệu tương ứng. Trước khi chạy, cần tải về từ Kaggle Dataset và đặt đúng đường dẫn trong từng notebook.

| Notebook | Mistral adapter | Legal-BERT |
|----------|----------------|------------|
| `proposed_framework.ipynb` | cleaned data + intent | cleaned data + intent |
| `rawdata_framework.ipynb` | raw data + intent | raw data + intent |
| `ablation_study.ipynb` — Ablation A | cleaned data + intent | cleaned data + intent |
| `ablation_study.ipynb` — Ablation B | cleaned data, no intent | cleaned data, no intent |
| `ablation_study.ipynb` — Ablation C | cleaned data + intent | — (không dùng) |

- Base model `mistralai/Mistral-7B-Instruct-v0.3` được tải tự động từ HuggingFace Hub, yêu cầu HuggingFace token (`HF_TOKEN`) được cấu hình trong Kaggle Secrets.