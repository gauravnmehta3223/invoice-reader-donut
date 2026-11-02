# Invoice Reader using Donut

This project fine-tunes [Donut](https://huggingface.co/naver-clova-ix/donut-base) (Document Understanding Transformer) to read invoice images and return their fields as JSON. Donut needs no separate OCR step: it reads the image and generates the structured output directly.

**Example output** for one invoice:
```json
{
  "SELLER_NAME": "Huff-Bryan",
  "DATE": "Date: 08-Oct-1993",
  "DUE_DATE": "Due Date : 28-Mar-1998",
  "SUB_TOTAL": "SUB_TOTAL : 638.17  USD",
  "DISCOUNT": "DISCOUNT(3.32%): (-)  21.19"
}
```

## Data

- **10,000 synthetic invoices:** 50 layout templates with 200 invoices each. The names, addresses and amounts are fake.
- **32 labeled field types**, for example `SELLER_NAME`, `DATE`, `TOTAL`, `TAX`, `BUYER` and `DUE_DATE`.
- The raw images and annotations are not in this repository, because they are too large. Place them in `data/raw/images/` and `data/raw/annotations/`.
- The processed labels are included:
  - `data/processed/` has one 70 / 15 / 15 train / validation / test split across all invoices.
  - `invoice_data/template_XX/` has an 80 / 20 train / test split for each template.

## Approach

1. Convert each annotation file to a `{"image", "ground_truth"}` record (`src/data_scripts/`).
2. Fine-tune `naver-clova-ix/donut-base` with PyTorch Lightning on a GPU, using 16-bit mixed precision and AdamW at a learning rate of 5e-5 (`src/models/`).
3. Evaluate on the test split:
   - exact match of the full JSON, plus BLEU and ROUGE scores (`evaluate_donut.py`)
   - precision, recall and F1 for each field (`evaluate_stuructured.py`)

## Run it

You need Python 3.9+ and an NVIDIA GPU with CUDA.

```bash
git clone https://github.com/Gmehta604/Invoice_Reader_Using_Donut.git
cd Invoice_Reader_Using_Donut
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root:

| Step | Command |
| --- | --- |
| Build the processed labels from `data/raw/annotations` | `python src/data_scripts/Data_preperation.py` |
| Split into train / validation / test | `python src/data_scripts/Train_val_test_split.py` |
| Train one model on all invoices | `PYTHONPATH=src/models python src/models/train_donut.py` |
| Train and evaluate a model for each template | `PYTHONPATH=src/models python src/models/Train_all_templates.py` |
| Write predictions for each template's test set | `PYTHONPATH=src/models python src/models/generate_predictions.py` |
| Score the predictions field by field | `python src/models/evaluate_stuructured.py` |

Checkpoints are saved to `checkpoints/`, which git ignores.

## Project structure

```
├── data/processed/        # labels for the 70/15/15 split across all invoices
├── invoice_data/          # 80/20 split for each of the 50 templates
├── src/
│   ├── data_scripts/      # annotation conversion and splitting
│   ├── models/            # dataset, training, evaluation, prediction, export
│   └── utils/             # label counts and a GPU check
└── requirements.txt
```
