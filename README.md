# indic-asr-eval-audit

[![CI](https://github.com/berserkersap/indic-asr-eval-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/berserkersap/indic-asr-eval-audit/actions/workflows/ci.yml)

Auditing how Indic ASR is evaluated: number-aware metrics and synthetic-vs-real test sets.

Hindi ASR is usually compared by word error rate (WER) after some text
normalisation. This repo asks two questions about that practice, on real
Hindi speech, with confidence intervals:

- **RQ1.** How much of a reported WER comes from the scoring convention rather
  than from recognition, and does the convention change which system looks
  best? In particular, does WER rank systems the same way as the error that
  matters for voice payments and forms: getting the numbers right?
- **RQ2.** Does a TTS-generated test set of the same sentences rank systems the
  way real speech does?

The work is updated as new experiments land; see the [research log](#research-log).

## Key takeaway

<!-- takeaway -->
On FLEURS Hindi, with five open ASR systems and 95% cluster-bootstrap intervals:

- **WER does not tell you which system gets the numbers right.** On held-out
  utterances with numbers, MMS-1B has the lowest WER of all five systems, yet
  misses about half of the numbers in the references, more than twice as many
  as Whisper large-v3. Both reversals against the Whisper large models are
  significant on both metrics. For a voice payment or form-filling product,
  the WER winner is the wrong choice.
- **The stock Whisper normaliser makes Hindi WER look 8 to 12 points lower**
  for every system, because it deletes vowel signs and so splits words. It
  shifts all systems alike, so it changes the numbers people report, not the
  ranking.
- **Spelling and number normalisation move WER by only 1 to 3 points.** WER is
  dominated by other words, so it is nearly blind to whether the numbers, the
  part users act on, came out right.

The tables below are generated from [`results/`](results/) by `asr-audit report`.
<!-- /takeaway -->

## Summary of results

### RQ1: scoring conventions (FLEURS Hindi, held-out utterances with numbers)

WER (%) at four normalisation levels and the numeric value error (%), with 95%
cluster-bootstrap intervals. Sorted by WER at the most lenient level.

<!-- rq1-wer -->
| system | WER whisper | WER basic | WER spelling | WER numbers | numeric value error |
|---|---|---|---|---|---|
| `mms-1b-all` | 16.2 [13.4, 20.1] | 25.1 [22.6, 27.9] | 23.7 [21.1, 26.6] | 23.1 [20.4, 25.9] | 49.6 [39.1, 60.1] |
| `whisper-large-v3` | 20.9 [17.7, 25.0] | 29.3 [26.8, 32.1] | 27.4 [24.3, 30.6] | 26.7 [23.5, 30.0] | 18.5 [7.8, 29.9] |
| `whisper-large-v3-turbo` | 24.1 [19.6, 29.7] | 33.6 [29.0, 40.6] | 31.7 [26.6, 39.0] | 31.3 [26.3, 38.6] | 23.7 [13.3, 35.2] |
| `whisper-medium` | 38.2 [30.7, 48.1] | 50.4 [41.3, 63.9] | 49.4 [39.9, 62.8] | 48.9 [39.5, 62.4] | 35.6 [23.9, 46.8] |
| `whisper-small` | 68.3 [55.9, 83.4] | 76.4 [66.9, 87.8] | 75.6 [66.0, 87.1] | 75.0 [65.5, 86.5] | 78.5 [69.2, 86.3] |
<!-- /rq1-wer -->

Does WER rank systems the way the numeric value error does? Pairs of systems
that the two metrics put in opposite order; a reversal is significant when the
paired intervals of both differences exclude zero:

<!-- rq1-disagreement -->
| subset | systems | Kendall tau | pairs reversed | significant reversals |
|---|---|---|---|---|
| held_out_numeric | 5 | 0.40 | 3 | whisper-large-v3/mms-1b-all, whisper-large-v3-turbo/mms-1b-all |
| numeric | 4 | 1.00 | 0 |  |
<!-- /rq1-disagreement -->

Paired change in WER (percentage points) from one normalisation level to the
next:

<!-- rq1-steps -->
| system | whisper->basic | basic->spelling | spelling->numbers |
|---|---|---|---|
| `mms-1b-all` | +8.8 [+5.5, +11.5] | -1.3 [-1.9, -0.8] | -0.7 [-1.1, -0.3] |
| `whisper-large-v3` | +8.4 [+4.7, +11.3] | -1.9 [-3.1, -0.8] | -0.7 [-1.2, -0.3] |
| `whisper-large-v3-turbo` | +9.5 [+5.5, +13.4] | -1.9 [-3.1, -0.9] | -0.4 [-0.7, -0.1] |
| `whisper-medium` | +12.3 [+7.3, +17.7] | -1.1 [-1.9, -0.4] | -0.4 [-0.8, -0.1] |
| `whisper-small` | +8.1 [+0.7, +14.5] | -0.8 [-1.2, -0.4] | -0.6 [-1.0, -0.3] |
<!-- /rq1-steps -->

All utterances with numbers (more data, but MMS is left out because it was
trained on FLEURS train):

<!-- rq1-wer-all-numeric -->
| system | WER whisper | WER basic | WER spelling | WER numbers | numeric value error |
|---|---|---|---|---|---|
| `whisper-large-v3` | 21.6 [20.3, 23.3] | 30.6 [29.4, 31.8] | 28.8 [27.7, 30.0] | 27.9 [26.8, 29.1] | 16.8 [13.6, 20.2] |
| `whisper-large-v3-turbo` | 24.9 [23.2, 26.8] | 35.2 [33.3, 37.6] | 33.6 [31.6, 36.0] | 32.7 [30.8, 35.1] | 27.0 [23.2, 31.2] |
| `whisper-medium` | 35.5 [33.2, 38.3] | 46.6 [44.3, 49.3] | 45.3 [43.0, 48.0] | 44.5 [42.2, 47.2] | 31.9 [27.6, 36.5] |
| `whisper-small` | 69.6 [64.4, 75.4] | 84.8 [78.2, 92.4] | 84.2 [77.6, 91.8] | 83.6 [76.9, 91.2] | 72.0 [67.7, 76.3] |
<!-- /rq1-wer-all-numeric -->

Does the ranking change with the level? Kendall tau against the stock Whisper
normaliser, and the number of system pairs whose order flips:

<!-- rq1-ranks -->
| subset | versus | n_systems | kendall_tau | pairs_flipped |
|---|---|---|---|---|
| held_out_numeric | whisper vs basic | 5 | 1.00 | 0 |
| held_out_numeric | whisper vs spelling | 5 | 1.00 | 0 |
| held_out_numeric | whisper vs numbers | 5 | 1.00 | 0 |
| numeric | whisper vs basic | 4 | 1.00 | 0 |
| numeric | whisper vs spelling | 4 | 1.00 | 0 |
| numeric | whisper vs numbers | 4 | 1.00 | 0 |
| control | whisper vs basic | 5 | 1.00 | 0 |
| control | whisper vs spelling | 5 | 1.00 | 0 |
| control | whisper vs numbers | 5 | 1.00 | 0 |
<!-- /rq1-ranks -->

![WER by normalisation level](results/rq1/fleurs_hi_in/wer_by_level.png)

### RQ2: synthetic versus real test sets

<!-- rq2 -->
In progress.
<!-- /rq2 -->

## Why this matters

Indian voice products turn speech into amounts, quantities, dates and names.
WER treats "53" and "तिरेपन" (fifty-three) as a substitution, treats a dropped
nukta as an error, and treats a wrong amount as one error out of many words.
Sarvam's own evaluation guide argues that WER penalises valid variation in
script, spelling and numerals and proposes LLM-judged alternatives
([Sarvam, 2026](https://www.sarvam.ai/blogs/evaluating-indian-language-asr)).
This repo takes the complementary, judge-free route: make each scoring
convention explicit, measure what it changes, and add a deterministic metric
for the values themselves.

## Experimental setup

**Data.** [FLEURS](https://huggingface.co/datasets/google/fleurs) Hindi
(CC BY 4.0), read Wikipedia sentences, pinned to revision `70bb2e8`. Numbers
are rare, so every utterance whose reference contains a digit is kept from all
splits (523 utterances), plus 200 seeded test-split controls without digits.
Recordings of 30 s or more are dropped (3 utterances, all with numbers), since
Whisper decodes 30 s windows and Sarvam's synchronous API rejects longer audio.
The exact set is fixed in [`manifests/fleurs_hi_in.jsonl`](manifests/fleurs_hi_in.jsonl).

**Subsets.** `held_out_numeric` (85 utterances from test and dev) is the
headline subset, because MMS was trained on FLEURS train
([Pratap et al., 2023](https://arxiv.org/abs/2305.13516), §5.4) and must not be
scored on it. `numeric` (523) gives tighter intervals for the other systems.
`control` (200) checks that effects are specific to numbers.

**Systems.** Whisper small, medium, large-v3 and large-v3-turbo, and MMS-1B-all
with the Hindi adapter, all pinned to a Hugging Face revision in
[`configs/systems.toml`](configs/systems.toml). Sarvam Saaras v4 in its
`transcribe` (numbers as digits) and `verbatim` (numbers as spoken) modes is
configured and will be added once it has run; it isolates the effect of an
output convention within one model.
Whisper is decoded greedily with a token limit of 25 tokens per second of
audio. The real FLEURS Hindi references need at most 19 tokens per second
(99th percentile 16.9) with Whisper's tokenizer, so the limit only stops
runaway repetition; outputs that reach it are counted in each run's
`.meta.json`.

**Normalisation levels** ([`normalize.py`](src/indic_asr_audit/normalize.py)):

| Level | What it does |
|---|---|
| `whisper` | Whisper's `BasicTextNormalizer`, the default in many open evaluations. It replaces every Unicode mark with a space, which deletes Hindi vowel signs and the virama ([Manohar et al., 2024](https://arxiv.org/abs/2409.02449)). |
| `basic` | Unicode NFC, lower case, punctuation removed, native digits mapped to ASCII, thousands separators removed. Vowel signs are kept. |
| `spelling` | `basic`, plus accepted spelling variants merged: nukta dropped, chandrabindu written as anusvara. |
| `numbers` | `spelling`, plus every number rewritten as digits: "उन्नीस सौ नब्बे" → 1990, "ढाई लाख" → 250000, "पंद्रहवीं" → "15 वीं". |

**Numeric value error.** For each number written as digits in the reference,
is its value present anywhere in the hypothesis, after both sides are parsed?
The parser ([`numbers.py`](src/indic_asr_audit/numbers.py)) handles the Indian
place-value system, fraction words (सवा, डेढ़, ढाई, साढ़े, पौने), decimals,
ordinals and digits mixed with words ("8 लाख"). Every number spelled by the
companion speller parses back to the same value (tested on 5,500 values).

**Statistics.** Corpus rates are sums of errors over sums of reference words.
Intervals are 95% percentile intervals from 2,000 bootstrap resamples of
sentences, not utterances, since FLEURS has up to two recordings per sentence.
Differences between levels or systems are paired: both sides use the same
resamples.

**Synthetic test sets (RQ2).** Each sentence is verbalised (digits, currency,
dates and times written as words), spoken by a TTS engine, checked and saved in
clean and telephony (300–3400 Hz, 8 kHz, G.711 μ-law) versions by a
[LangGraph](https://github.com/langchain-ai/langgraph) workflow
([`synth.py`](src/indic_asr_audit/synth.py)). The check deliberately uses no
ASR model, since filtering clips by one system's accuracy would favour that
system: a clip passes if its length is within 0.5–2× of the real recording and
it is neither silent nor clipped; otherwise the next voice is tried. Engines:
Meta MMS-TTS (open weights, CC BY-NC 4.0) and Sarvam Bulbul.

## Limitations

- FLEURS is read Wikipedia text, not conversational or transactional speech;
  its numbers are mostly years, counts and measurements, not prices.
- The held-out subset is small (85 utterances), so its intervals are wide.
- "Control" means no digit in the reference; a control can still contain a
  number written as a word.
- The numeric value error is strict about spelling. A number word with a
  phonetic misspelling ("पंद्रा" for पंद्रह, "सो" for सौ, "ग्यारवी" for
  ग्यारहवीं) counts as missing, although a reader would recover the value.
  Whisper models make this kind of miss; MMS misses are mostly garbled digits
  ("7त्तर" for सत्तर, "1093" for 1963). Measuring how often a lenient judge
  disagrees is the next experiment.
- The numeric value error matches values anywhere in the hypothesis. The
  article "एक" and the verb "दो" (give) parse as 1 and 2, which can hide a
  missed number; leading zeros are dropped.
- FLEURS has no speaker ids, so the bootstrap does not model speaker
  dependence and intervals may be slightly narrow.
- Whisper and Saaras may have seen FLEURS test or dev data; this cannot be
  ruled out, only flagged.
- Saaras is a hosted model that can change without notice; its cached
  transcripts in `results/transcripts/` are the reproducible artifact.

## Research log

| Date | Update |
|---|---|
| 2026-11-06 | Metrics, number parser and normaliser; FLEURS Hindi set; LangGraph synthesis pipeline and RQ2 analysis. RQ1 results for five open systems: the stock Whisper normaliser lowers Hindi WER by 8 to 12 points, and WER and the numeric value error rank MMS-1B against the Whisper large models in opposite orders. |

Next:

- RQ2 results with MMS-TTS and Bulbul, clean and telephony.
- Agreement between the numeric value error, an LLM judge in the style of
  Sarvam's `llm_wer`, and a small human-labelled set.
- RQ3: does LoRA fine-tuning on synthetic, number-dense speech reduce the
  numeric value error on real speech? (Fits an 8 GB GPU.)
- Conversational and code-mixed sets (Svarah, Lahaja, IndicVoices).

## Related work

- Manohar, Pillai and Sherly, *What is lost in Normalization? Exploring
  Pitfalls in Multilingual ASR Model Evaluations*, EMNLP 2024
  ([arXiv:2409.02449](https://arxiv.org/abs/2409.02449)): Whisper's
  normaliser strips Indic vowel signs and lowers WER artificially.
- Sarvam AI, *Evaluating Indian Language ASR*, 2026
  ([blog](https://www.sarvam.ai/blogs/evaluating-indian-language-asr);
  [llm_wer](https://github.com/sarvamai/llm_wer),
  [llm_intent_entity](https://github.com/sarvamai/llm_intent_entity)):
  LLM-WER, an intent score and an entity preservation score.
- Parulekar and Jyothi, *LASER: An LLM-based ASR Scoring and Evaluation
  Rubric*, EMNLP 2025 ([arXiv:2510.07437](https://arxiv.org/abs/2510.07437)).
- Kim et al., *Semantic Distance: A New Metric for ASR Performance Analysis
  Towards Spoken Language Understanding*
  ([arXiv:2104.02138](https://arxiv.org/abs/2104.02138)).
- Ugan et al., *PIER: A Novel Metric for Evaluating What Matters in
  Code-Switching*, ICASSP 2025 ([arXiv:2501.09512](https://arxiv.org/abs/2501.09512)).
- Peyser et al., *Improving Performance of End-to-End ASR on Numeric
  Sequences*, Interspeech 2019 ([arXiv:1907.01372](https://arxiv.org/abs/1907.01372)).
- Rossenbach et al., *Generating Synthetic Audio Data for Attention-Based
  Speech Recognition Systems*, ICASSP 2020
  ([arXiv:1912.09257](https://arxiv.org/abs/1912.09257)).
- Minixhofer, Klejch and Bell, *Beyond Oversmoothing: Evaluating DDPM and MSE
  for Scalable Speech Synthesis in ASR*
  ([arXiv:2410.12279](https://arxiv.org/abs/2410.12279)).

## Reproducibility

Developed on Windows 11, Python 3.14, an 8 GB RTX 5060 laptop GPU and 16 GB of
RAM. Run model jobs one at a time on a machine this size.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu128+ for RTX 50-series
pip install -e ".[asr,synth,dev]"

asr-audit prepare-fleurs                       # ~1.6 GB download, writes the manifest
asr-audit transcribe whisper-small whisper-medium whisper-large-v3 \
    whisper-large-v3-turbo mms-1b-all          # GPU, about 1.5 h in total
asr-audit rq1                                  # CPU, writes results/rq1/
asr-audit synthesize mms-tts                   # CPU, writes synthetic manifests
asr-audit transcribe whisper-small --dataset fleurs_hi_in-mms-tts-clean
asr-audit rq2 fleurs_hi_in-mms-tts-clean       # CPU, writes results/rq2/
asr-audit report                               # refreshes the tables in this README
pytest
```

Transcripts and result tables are committed, so `rq1`, `rq2` and `report` run
on a CPU-only machine without any model. Sarvam systems need `SARVAM_API_KEY`
(see [`.env.example`](.env.example)).

To add another language, see
[docs/extending-to-new-languages.md](docs/extending-to-new-languages.md).

## Repository layout

```
configs/      ASR systems and TTS engines, pinned to model revisions
docs/         how to add a language
manifests/    evaluation sets (real and synthetic), one utterance per line
results/      transcripts with run settings, result tables and figures
src/indic_asr_audit/
  languages/  language packs (Hindi)
  numbers.py  spoken-number parser and speller
  normalize.py  normalisation levels and TTS verbalisation
  metrics.py  word, character and numeric value errors
  stats.py    cluster bootstrap
  asr.py      ASR backends and the resumable runner
  synth.py    LangGraph synthesis workflow and TTS engines
  analysis.py, rq2.py  RQ1 and RQ2 tables
  report.py   fills the README tables from the result files
tests/        unit tests, including synthetic checks with known answers
```

## License

Code: MIT. FLEURS: CC BY 4.0. MMS-TTS weights: CC BY-NC 4.0, so synthetic
audio made with them is for research only.
