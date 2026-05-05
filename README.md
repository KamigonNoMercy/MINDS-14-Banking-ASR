# 🎙️ MINDS-14 Banking ASR - wav2vec2 vs Whisper Fine-Tuning

This repository contains my **end-to-end ASR (Automatic Speech Recognition) project** comparing **wav2vec2-base-960h** and **Whisper-base** on the **MINDS-14 (en-US)** banking telephony dataset. Both models are evaluated as zero-shot baselines, then fine-tuned on the same data, and finally compared using **WER** (Word Error Rate) and **CER** (Character Error Rate).

👉 **Note:** The dataset belongs to its original authors (PolyAI / MINDS-14 team on Hugging Face).
👉 The **EDA, preprocessing pipeline, augmentation, fine-tuning, and evaluation** are my own work.

---

## ✨ Features

- End-to-end ASR pipeline: **EDA -> resampling -> augmentation -> stratified split -> two-processor preprocessing -> baseline -> fine-tuning -> final evaluation**
- **Two model architectures** compared head-to-head:
  - **wav2vec2-base-960h** (CTC head, frame-level decoding)
  - **whisper-base** (encoder-decoder seq2seq, autoregressive `.generate()`)
- **Audio preprocessing pipeline:**
  - Resampling 8 kHz -> 16 kHz with `torchaudio.transforms.Resample`
  - Manual audio augmentation (Gaussian noise + time-stretch + pitch-shift) - doubles train set from 394 -> 788
  - Stratified train/val/test split by `intent_class` (70 / 15 / 15)
- **Two separate processors handled correctly:**
  - wav2vec2: CTC tokenizer with **uppercase fix** (vocab is A-Z only) + dynamic padding
  - Whisper: log-mel 80×3000 fixed 30s window + BPE tokenizer (lowercase native)
- **Domain-aware fine-tuning:**
  - wav2vec2: 20 epochs, lr=1e-4, frozen feature encoder (~95.6% trainable)
  - Whisper: 10 epochs, lr=1e-5, full fine-tune with `Seq2SeqTrainer` and `predict_with_generate=True`
- **Evaluation with WER + CER** at every epoch, plus final test-set evaluation and a side-by-side `predictions_all.xlsx` for qualitative analysis

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Deep Learning:** `torch`, `torchaudio`
- **Audio I/O:** `soundfile`, `librosa`
- **HF Ecosystem:** `transformers` (Wav2Vec2/Whisper processors + models + Trainer + Seq2SeqTrainer), `datasets`, `accelerate`
- **Metrics:** `jiwer` (WER, CER)
- **Data Handling:** `pandas`, `numpy`, `pyarrow`
- **Modeling Utilities:** `scikit-learn` (stratified split)
- **Visualization:** `matplotlib`, `seaborn`
- **Hardware:** Trained on a single GPU (~8 GB VRAM, fp16 mixed precision)

---

## 📊 Dataset

- **Source:** [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) on Hugging Face (en-US split only) - see the [Download the Dataset](#-download-the-dataset) section below for setup steps
- **License:** CC BY 4.0
- **Size:** 563 audio clips, 14 banking intent classes
- **Sampling rate:** Native 8 kHz (telephone audio) -> resampled to 16 kHz to match both models' pre-training
- **Duration:** mean 8.58s, median 6.40s, range 1.71s to 58.45s (right-skewed)
- **Intent balance:** Min 34 (`address`/`abroad`/`latest_transactions`), max 48 (`cash_deposit`), max/min ratio 1.41 (well-balanced)
- **Total audio:** ~1.34 hours

The 14 intents cover real banking queries: `abroad`, `address`, `app_error`, `atm_limit`, `balance`, `business_loan`, `card_issues`, `cash_deposit`, `direct_debit`, `freeze`, `high_value_payment`, `joint_account`, `latest_transactions`, `pay_bill`.

---

## 🔍 Key Insights from EDA

1. **8 kHz telephony is the elephant in the room.** Both wav2vec2 and Whisper were pre-trained on **16 kHz** audio (LibriSpeech and 680k hours of mixed audio respectively), but MINDS-14 is **8 kHz telephone-quality**. Resampling 8 -> 16 kHz is mandatory but can't *create* high frequencies that were never recorded - this is the root cause of the gap between MINDS-14 WER and what these models achieve on cleaner benchmarks.

2. **Audio is technically clean, just acoustically narrow.** No clipping (max amplitude peaks around 0.75-0.90, not 1.0), no silent files, no garbage. The challenge is the **8 kHz bandwidth**, not the recording quality.

3. **Duration is heavily right-skewed.** Median 6.4s but max is 58.45s - a few users gave essay-length answers. This matters for Whisper because its log-mel input is **fixed at 30 seconds** - any clip longer than 30s gets truncated. Luckily 99%+ of MINDS-14 fits, so the impact is negligible.

4. **All 14 intents look similar in mel-space.** Per-intent mean mel-spectrograms have the same general shape (energy concentrated in low frequencies) - because they're all human speech in the same language. The differences are in *content*, not in *acoustic signature*. This is exactly why ASR models (which actually decode the words) work better here than purely acoustic intent classifiers.

5. **Stratifying on intent matters even for ASR.** Even though the task is speech-to-text (not classification), splitting stratified by `intent_class` ensures all 14 banking topics are proportionally represented in train/val/test - so the test WER reflects performance across the full domain, not just the most common intents.

---

## 🧪 Results

All numbers on the held-out **test set (85 samples)**. Lower is better.

### Baseline (zero-shot, no fine-tuning)

| Model | WER (%) | CER (%) | Notes |
|---|:---:|:---:|---|
| wav2vec2-base-960h | 64.60 | 46.42 | Strong domain mismatch (LibriSpeech studio -> bank telephony) |
| **whisper-base** | **52.85** | **41.16** | Better OOTB - Whisper saw way more diverse audio in pre-training |

👉 Whisper wins baseline by ~12 WER points - exactly what you'd expect given its 680k-hour multi-domain pre-training vs LibriSpeech's 960h of clean studio reads.

### After Fine-Tuning

| Model | Epochs | WER (%) | CER (%) | Δ WER vs baseline |
|---|:---:|:---:|:---:|:---:|
| wav2vec2-ft | 20 | 42.28 | 33.60 | **-22.32 pts** |
| **whisper-ft** | 10 | **33.89** | **30.08** | **-18.96 pts** |

👉 **Both models improved a lot.** wav2vec2 saw a *bigger absolute drop* (22.32 vs 18.96 WER points) because it started from a worse baseline - more room to improve. Whisper is still the winner overall by ~8 WER points.

### Training dynamics

- **wav2vec2** Validation WER: 45.69% (epoch 1) -> 25.33% (epoch 17-19). Trained ~62 minutes on GPU.
- **Whisper** Validation WER: 27.63% (epoch 1) -> 23.28% (epoch 8, best). Trained ~46 minutes on GPU.
- Whisper converges faster *and* lower, on fewer epochs - the multi-domain pre-training pays off again.

### Why ground-truth WER is still ~30% even after fine-tuning

A few characteristic error modes survive:
- **Acoustic confusions** at 8 kHz (fricatives `s/z/sh/f` blur together: `purchase` -> `purchaf`, `received` -> `was seezed`).
- **Function-word swaps** (`the` -> `that`, dropped articles). These cost a full word in WER but are barely audible.
- **Semantic-but-wrong substitutions** (`correctly` -> `properly`, `working` -> `walking`). Whisper's strong language model fills in plausible words even when audio is ambiguous.
- **Whisper baseline only:** auto-punctuation (`address.` vs reference `address`) - a *format* mismatch, not a *content* mismatch. Fine-tuning fixes this entirely.

---

## 🧠 Lessons Learned

- **Two text targets, two tokenizers, two collators.** wav2vec2's CTC vocab is **uppercase-only A-Z** (32 tokens). Forgetting `.upper()` before tokenizing turns every label into `<unk>` and training silently fails. Whisper's BPE tokenizer is fine with lowercase. The dataset has to be **rebuilt twice** - once per processor - this is one of the most error-prone steps in ASR fine-tuning and it's worth a verification cell that decodes a sample label and checks the `<unk>` ratio.
- **Whisper's encoder-decoder architecture is more forgiving on small datasets.** It needed only 10 epochs vs 20 for wav2vec2, with a much smaller learning rate (1e-5 vs 1e-4). Fewer steps, lower LR, lower final WER - that's a clear architectural win for low-resource fine-tuning.
- **Augmentation actually helps when the train set is tiny.** Doubling 394 -> 788 with noise + time-stretch + pitch-shift gave the model more variation to generalize from. With only 394 raw samples and frozen feature encoder, every bit of variation matters.
- **WER alone is misleading; pair it with CER.** The Whisper baseline's 52.85% WER looks bad until you see the predictions: most are correct except for added punctuation. CER (41.16%) tells the truer story. After fine-tuning, both metrics drop together - confirming the model fixed *content*, not just *format*.
- **Always verify before training.** I added a "pre-training verification" cell that decodes 3 sample labels, checks `<unk>` ratio, runs the data collator on a tiny batch, and prints free GPU memory. Catching a tokenizer mismatch in 2 seconds beats discovering it after a 1-hour training run.

---

## 🔮 Future Work

1. **Hyperparameter search** - grid / Optuna over learning rate, warmup steps, weight decay, and dropout. Currently hand-picked from HF tutorial defaults.
2. **Bigger Whisper variants** - `whisper-small` (244M) and `whisper-medium` (769M) should reach much lower WER if the GPU budget allows.
3. **More data** - augment MINDS-14 with **Common Voice** or **Switchboard** to broaden the telephony domain. 394 raw training samples is the real bottleneck here.
4. **Multilingual fine-tune** - MINDS-14 has de-DE, fr-FR, es-ES, and more. Whisper's multilingual pre-training is well-suited for this.
5. **Streaming / latency-aware inference** for actual call-center deployment. Right now the pipeline is offline batch-style.
6. **Error analysis dashboard** - cluster the ~30% errors by type (acoustic confusion vs function-word drop vs semantic substitution) to figure out where the next improvement should come from.

---

---

## Contributor (NLP Group C)
- Surya Dharma Putra
- Krisna Fery Rahmantya
- Agil Setiawan
- Khaerani Arista Dewi​
---

## 🗂️ Repository Structure

```
minds14-asr-wav2vec2-whisper/
├─ README.md
├─ Project3.ipynb              # Full end-to-end notebook (EDA -> preprocess -> baseline -> fine-tune -> compare)
├─ baseline_predictions.xlsx   # Test-set predictions: reference / wav2vec2-base / whisper-base
├─ predictions_all.xlsx        # All four conditions side-by-side: ref / w2v2-base / w2v2-ft / whisper-base / whisper-ft
└─ requirements.txt
```

The notebook also writes these directories during training (not committed to the repo - they're large):

```
./ds_wav2vec2_asr/             # Preprocessed HF DatasetDict for wav2vec2 (uppercase labels)
./ds_whisper_asr/              # Preprocessed HF DatasetDict for Whisper (lowercase BPE labels)
./wav2vec2-minds14-final/      # Fine-tuned wav2vec2 weights + processor
./whisper-minds14-final/       # Fine-tuned Whisper weights + processor
```

---

## 📒 Notebook Overview

| Section | What's inside |
|---|---|
| **EDA** | Load parquet, intent label mapping, audio stats (duration, sampling rate, RMS, max amplitude), waveform / mel-spectrogram / MFCC visualizations, per-intent mean mel-spectrograms |
| **Preprocessing** | Lowercase + strip transcription, decode bytes -> numpy, resample 8 -> 16 kHz with `torchaudio`, manual augmentation (noise + time-stretch + pitch-shift), stratified 70/15/15 split, two-processor preparation (wav2vec2 + Whisper), data collators, save to disk |
| **Baseline** | Zero-shot evaluation on test set with both pre-trained models, WER + CER, save `baseline_predictions.xlsx` |
| **Fine-tune wav2vec2** | Rebuild dataset with `.upper()` fix, freeze feature encoder, `TrainingArguments` (20 epochs, lr=1e-4), `Trainer.train()`, evaluate on test |
| **Fine-tune Whisper** | Set generation config (en + transcribe), `Seq2SeqTrainingArguments` (10 epochs, lr=1e-5, `predict_with_generate=True`), `Seq2SeqTrainer.train()`, evaluate on test |
| **Final comparison** | 4-row results table (baseline vs fine-tuned × 2 models), improvement table in WER/CER points, save `predictions_all.xlsx`, conclusions + future work |

---

## 📥 Download the Dataset

The notebook expects a single file in the working directory: **`train-00000-of-00001.parquet`** (the en-US split of MINDS-14, ~471 MB after audio decode, 563 rows).

Pick whichever option works for you:

**Option A - Direct download from Hugging Face (recommended).** The Parquet file is hosted on the auto-converted parquet branch:

```bash
huggingface-cli download PolyAI/minds14 \
    --repo-type dataset \
    --revision refs/convert/parquet \
    --include "en-US/*" \
    --local-dir ./minds14_en_us
# then move the parquet file to the working directory
mv ./minds14_en_us/en-US/train/0000.parquet ./train-00000-of-00001.parquet
```

**Option B - Use `datasets` library inside the notebook.** Replace the `pd.read_parquet(...)` cell with:

```python
from datasets import load_dataset
ds = load_dataset("PolyAI/minds14", "en-US", split="train")
df = ds.to_pandas()
```

**Option C - Manual.** Visit [huggingface.co/datasets/PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14), click on the **en-US** subset, and download the parquet file from the Files tab.

> ⚠️ The dataset card lists 14 language configurations (cs-CZ, de-DE, en-AU, en-GB, **en-US**, es-ES, fr-FR, it-IT, ko-KR, nl-NL, pl-PL, pt-PT, ru-RU, zh-CN). This project uses **en-US only**.

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/minds14-asr-wav2vec2-whisper.git
cd minds14-asr-wav2vec2-whisper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (see "Download the Dataset" section above)

# 4. Run the notebook end-to-end
jupyter notebook Project3.ipynb
```

⚠️ **Hardware note:** Fine-tuning both models takes ~1h 50min total on a single GPU with ~8 GB VRAM (fp16). On CPU this is not realistic - use Colab / Kaggle / a cloud GPU.

---

## License

- Code in this repository is released under the **MIT License**.
- The **MINDS-14 dataset** is provided by [PolyAI](https://huggingface.co/datasets/PolyAI/minds14) and is licensed under **CC BY 4.0**. This repository does **not include** the dataset audio files - download them from Hugging Face directly.
- Pre-trained model weights belong to their original authors:
  - `facebook/wav2vec2-base-960h` - Meta AI / Facebook AI Research, Apache 2.0 ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477))
  - `openai/whisper-base` - OpenAI, MIT License ([Radford et al., 2022](https://arxiv.org/abs/2212.04356))
- Fine-tuned model weights produced by this notebook are released under MIT, but inherit the licensing constraints of their respective base models when redistributed.
