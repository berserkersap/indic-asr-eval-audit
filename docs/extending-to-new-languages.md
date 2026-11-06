# Extending to a new language

Everything language-specific lives in one `LanguagePack`
(`src/indic_asr_audit/languages/`). The normaliser, number parser, metrics and
analysis read only from the pack, so adding a language is mostly data entry.

## 1. Write the pack

Create `src/indic_asr_audit/languages/<code>.py`, using `hi.py` as the model.

| Field | What to put in it | Hindi example |
|---|---|---|
| `code`, `name` | ISO 639-1 code and English name | `"hi"`, `"Hindi"` |
| `native_digits` | The ten native digits, 0 to 9, or `""` | `"०१२३४५६७८९"` |
| `units` | A word for every number below 100, plus common spelling variants | `"तिरेपन": 53`, `"तिरपन": 53` |
| `scales` | Multiplier words | `"सौ": 100`, `"लाख": 100_000` |
| `fractions` | Words that are a value by themselves and can take a scale | `"ढाई": 2.5` (ढाई सौ = 250) |
| `offsets` | Prefixes that shift the following number | `"साढ़े": +0.5`, `"पौने": -0.25` |
| `decimal_points` | Words read for the decimal point | `"दशमलव"` |
| `ordinal_suffixes` | Suffixes that make an ordinal from a number word, most common first | `"वीं"` (पंद्रहवीं) |
| `symbols` | Symbols read as a word after the number | `"%": "प्रतिशत"` |
| `prefix_symbols` | Symbols written before a number but read after it | `"₹": "रुपये"` |
| `spelling_folds` | `(regex, replacement)` pairs that merge accepted spellings | drop the nukta |

Then register it in `languages/__init__.py`:

```python
LANGUAGES = {pack.code: pack for pack in (HINDI, MARATHI)}
```

The pack checks itself when it is built: every word must have one meaning, and
`native_digits` must have ten characters. `tests/test_languages.py` also
checks that every value below 100 has a word, for every registered pack.

## 2. Check the number parser fits the language

The parser handles languages that build numbers the way Hindi does: a distinct
word for each value below 100, then multipliers in the Indian system
(hundred, thousand, lakh, crore). Marathi, Gujarati, Punjabi, Bengali, Odia
and Nepali work this way, so they should need only data, but only Hindi has
been validated so far. Filling in 100 number words with their common spelling
variants is real work; check it with a native speaker.

Two cases need more than data:

- **Tens and units as separate words** (English "twenty five"). Already
  handled: a single digit may follow a multiple of ten.
- **Number words that fuse together or take case endings** (Tamil, Telugu,
  Kannada, Malayalam). For example, Telugu "నూట యాభై" (150) uses a bound
  form of "hundred". Either list the bound forms as extra scale entries, or
  pre-split them in a spelling fold. If neither works, the pack needs a
  custom splitting step before `find_numbers`; open an issue first.

Add parametrised cases to `tests/test_numbers.py` for years, Indian scales,
fractions and ordinals in the new language before trusting any results.

## 3. Get evaluation data

`asr-audit prepare-fleurs --lang <fleurs_config>` downloads any FLEURS
language, for example `mr_in`, `ta_in`, `te_in`, `bn_in` or `gu_in`. It keeps
every utterance whose reference contains a digit, plus a seeded sample of
test-split controls without numbers. Check the printed count of utterances
with numbers: some languages write few numbers as digits.

Two codes are in play: the FLEURS config (`mr_in`) names the dataset, and the
pack code (`mr`) names the language pack.

For another dataset, write a manifest: one JSON object per line with the
fields of `Utterance` in `src/indic_asr_audit/data.py`. `sentence_id` must be
shared by recordings of the same text, because the bootstrap resamples by it.

## 4. Configure the systems and run

Each system entry fixes one language, so make a systems file per language,
for example `configs/systems_mr.toml`, with the Whisper language code, the
MMS adapter code and the Sarvam `xx-IN` code. Check that each system supports
the language at all (Whisper, for example, has no Odia). Then:

```bash
asr-audit transcribe whisper-small mms-1b-all --dataset fleurs_mr_in --systems configs/systems_mr.toml
asr-audit rq1 --dataset fleurs_mr_in --lang mr --systems configs/systems_mr.toml
```

`rq1` reports and skips any system whose transcripts are incomplete.

## Checklist

- [ ] Every value 0 to 99 has a word, including the most common spelling variants ASR outputs
- [ ] Parser tests for years, scales, fractions, decimals and ordinals pass
- [ ] Spelling folds only merge spellings native readers accept as the same word
- [ ] Transcripts are complete for every system before running the analysis
