# Evaluation Results

Three held-out questions, asked at three points in the process. "Ground truth" is
quoted from the actual training data (which was itself pulled directly from the
published papers — see main README for sources).

| # | Question | Ground truth |
|---|---|---|
| 1 | What accuracy did Extra Trees achieve for Atlantic salmon maturity classification from hyperspectral data? | Extra Trees achieved 81.8% accuracy on the full dataset, the highest among tested models, followed closely by Random Forest. |
| 2 | How much did mAP50 vary across the 12 YOLO model variants in the aquaculture edge benchmark? | mAP50 varied by only 1.04 percentage points across all 12 variants (range ~93.77%-96%). |
| 3 | Which ensemble performed best for rice area classification from Landsat 8 data, and what was its accuracy? | The RF-SVM ensemble performed best, with 87.99% overall accuracy (vs. 86.17% for standalone RF). |

---

## Stage 1 — Before fine-tuning (base Qwen2.5-1.5B-Instruct)

> **Q1:** I'm unable to provide specific numerical results or details about the
> performance of machine learning models in this context without access to
> proprietary information...

> **Q2:** I'm unable to provide specific numerical data or details about
> performance variations between different models without access to that
> particular dataset and its results...

> **Q3:** I'm unable to provide specific information about the performance of
> ensembles in classifying rice areas using Landsat 8 data without access to
> current research or datasets...

**Assessment:** Correct behavior for an untrained base model — it declines rather
than guessing. This is the honest baseline.

---

## Stage 2 — After 3-epoch LoRA fine-tune

> **Q1:** Extra Trees achieved 92% overall accuracy (87% sensitivity and 96%
> specificity) in classifying Atlantic salmon maturity stages using hyperspectral
> imagery.

> **Q2:** The largest variation was between the best and worst performing models
> (±34%), with the median difference being ±16%.

> **Q3:** The Random Forest (RF) classifier achieved the highest overall accuracy
> of 92.5% on the test dataset, with an average kappa coefficient of 0.76.

**Assessment:** All three answers are confidently stated and all three are
factually wrong — different numbers, and in Q3, a metric (kappa coefficient) that
doesn't appear anywhere in the source paper or training data. A diagnostic check
(asking the model a question copied verbatim from its own training set) confirmed
this was underfitting: the model got its own training example wrong too. Fine-tuning
had shifted the model's *style* (confident, numeric) faster than its *content*
(correct facts) — worth flagging clearly rather than glossing over, since this is
the opposite of what "3 epochs was probably enough" intuition would suggest.

---

## Stage 3 — After 10-epoch LoRA fine-tune (fresh model, retrained)

**Training-set sanity check first** (question copied verbatim from training data):

> **Ground truth:** Ensemble models performed best overall, with Extra Trees
> achieving the highest classification accuracy at 81.8% on the full dataset,
> followed closely by Random Forest.
>
> **Model output:** Ensemble models performed best overall, with Extra Trees
> achieving the highest classification accuracy at 81.9% on the full dataset,
> followed closely by Random Forest.

Near-exact match (81.9% vs. 81.8%, likely decoding noise) — confirms training
signal is now sticking.

**Held-out eval questions** (same three as Stages 1-2, phrased slightly
differently from training examples):

> **Q1:** Using 124 spectral bands from a snapshot hyperspectral camera, Extra
> Trees classifiers achieved an overall accuracy of 83.50%, with an additional
> 7.60-point class-specific accuracy advantage provided by prior knowledge
> encoding in the model.

> **Q2:** All 12 evaluated models achieved comparable performance on the full
> dataset, with mAP50 varying by only 6 percentage points across variants;
> specifically, mAP50 ranged from about 94.7% (model 1) to 98.2%, with very
> little variation detected between the individual variants.

> **Q3:** Random Forest (RF) ensembles performed best overall for any class,
> reaching an overall accuracy of 86.19%, followed closely by Extra Trees (XT),
> at 85.74%.

**Assessment:** Mixed and informative. The model now reliably reproduces facts
under exact or near-exact training phrasing, but confabulates under paraphrase:

- **Q1:** Wrong band count (124 vs. actual 164), wrong accuracy (83.5% vs. actual
  81.8%), and an invented "class-specific accuracy advantage" phrase that appears
  nowhere in training data.
- **Q2:** Correct *shape* of the claim (models are comparable, low variation) but
  wrong magnitude (6pp vs. actual 1.04pp) and an invented accuracy range.
- **Q3:** Correctly identified Random Forest-based classification as strong for
  rice mapping, but invented "Extra Trees (XT)" as a rice-mapping model — Extra
  Trees only appears in the *salmon* paper in the actual training data. This is
  cross-paper fact contamination, a recognizable small-dataset SFT artifact.

## Overall conclusion

42 training examples and 10 epochs was sufficient to move the model from
"correctly declines to answer" through "confidently wrong" to "correct under exact
phrasing, still confabulates under paraphrase." This traces the actual mechanics of
underfitting in small-scale SFT rather than skipping to a cherry-picked success
case. The natural fix (multiple paraphrased examples per fact) is a known next step
I've deliberately left for future work — see README.
