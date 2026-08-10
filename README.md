# research-paper-lora

Hands-on exploratory project: LoRA fine-tuning a small open-source LLM (Qwen2.5-1.5B-Instruct)
on a custom instruction dataset built from my own published research spanning
precision agriculture, aquaculture, and remote sensing, run on a free-tier Colab T4 GPU.

**Status: exploratory / learning project, not production work.** This is my first hands-on
experience with LLM post-training (SFT + LoRA/PEFT). I built it specifically to move
that skill from "conceptual understanding" to "actually did it," and to document what
I learned honestly, including a result that didn't go the way I initially expected.

## Why this project

I have 12+ years of applied ML/CV background (YOLO detection, hyperspectral imaging,
sensor analytics — see my [portfolio](https://kothawadegs.github.io)) but no prior
hands-on experience with transformer post-training. Rather than just reading about
SFT/LoRA/RLHF, I picked the most practical technique (LoRA/QLoRA supervised
fine-tuning) and built something small but real: fine-tuning a small model to answer
technical questions about my own published papers, with a rigorous before/after eval.

## What's actually in this repo

- `data/qa_seed.jsonl` — 42 instruction/output pairs built from six of my published
  papers (see Data Sources below). Every fact was pulled directly from the paper text,
  not from memory or approximation.
- `train.py` — the full LoRA SFT pipeline: 4-bit quantized base model,
  chat-template formatting, LoRA config, training loop, and a before/after eval harness.
- `results/eval_results.md` — the three-stage before/after evaluation, including the
  full model outputs, not just a summary.

## Data sources (my own published, peer-reviewed work)

| Paper | My role |
|---|---|
| Kothawade & Khot et al., "Feasibility of Little Cherry/X-Disease Detection... Using FAIMS," *Sensors* 2025 | First author |
| Kothawade & Ranjan et al., "Snapshot hyperspectral imaging for production improvements in Atlantic salmon farming," *Smart Agricultural Technology* 2026 | First author |
| Ranjan, Kothawade et al., "Does YOLO26 Truly Offer Advantages Over Its Predecessors for Edge Deployment?" | Co-author |
| Mohite, Sawant, Kothawade, Pappula, "Ensemble Learning for Spatio-Temporal Rice Area Mapping Using Landsat 8" | Co-author |
| Tolomelli, Kothawade et al., "Aerial-RGB imagery based 3D canopy reconstruction... of grapevines" | Co-author |
| Dhakar, Amogi, Kothawade, Khot, "Simplified mechanistic model for estimating leaf wetness" | Co-author |

Every Q&A pair traces back to a specific fact in these papers' actual published text
(not the associated dissertation, which was deliberately excluded from this dataset).

## Method

- **Base model:** Qwen2.5-1.5B-Instruct, loaded in 4-bit (NF4, double quantization)
- **Fine-tuning technique:** LoRA (rank 16, alpha 32, targeting attention + MLP
  projection layers), via `peft` + `trl`'s `SFTTrainer`
- **Hardware:** free-tier Google Colab, single T4 GPU (15.6 GB VRAM)
- **Dataset size:** 42 instruction/output pairs (intentionally small — see Findings)
- **Training runs:** two — an initial 3-epoch run, and a follow-up 10-epoch run after
  the first run's results were diagnosed as underfit

## Findings — including what didn't work

I ran the same three held-out evaluation questions at three points: before any
training, after a 3-epoch fine-tune, and after a 10-epoch fine-tune. Full outputs are
in [`results/eval_results.md`](results/eval_results.md); summary below.

**Before fine-tuning:** the base model correctly declined to answer, stating it had no
access to the specific dataset or paper — an honest "I don't know," not a hallucination.

**After 3 epochs:** the model started producing fluent, confidently-formatted numeric
answers — and every single number was wrong (e.g., claimed 92% accuracy for a result
that was actually 81.8%). This was a valuable, if uncomfortable, finding: a small
amount of fine-tuning can make a model sound more authoritative without making it more
correct. Diagnostic tests confirmed this was underfitting — the model got even its own
*training* examples wrong.

**After 10 epochs:** the model reproduced training examples almost exactly when asked
with matching phrasing (e.g., "81.9%" vs. ground truth "81.8%" — likely decoding
noise). However, on the same three held-out questions with different phrasing, it
still confabulated — including one case where it invented an "Extra Trees (XT)"
model for the rice-mapping paper, a fact that only exists in a different paper in the
dataset. This is a well-documented failure mode for small-scale SFT: the model
memorizes surface patterns from a handful of examples before it has enough repetition
per fact to generalize under paraphrase.

**Takeaway:** 42 examples and 10 epochs is enough to prove the LoRA/SFT pipeline works
end-to-end and to observe underfitting-to-confabulation transition, but not enough to
produce a fine-tune that reliably generalizes. The standard fix — more examples per
fact, phrased multiple ways — is a natural next step I've deliberately left undone for
now, to keep this project honestly scoped to what I've actually built and verified.

## What I'd do next (not yet done)

- Expand the dataset so each fact appears with 2-3 paraphrased question variants
- Add a proper held-out train/eval split instead of ad hoc spot checks
- Try a DPO pass on top of SFT as a stretch goal

## Setup

```bash
pip install transformers peft trl bitsandbytes datasets accelerate
python train.py
```

Runs on a free Colab T4 GPU; no paid compute required.
