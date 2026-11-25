# MADRec

**MADRec: Reliable Multi-Agent Reasoning with Evidence-Grounded Debate for Recommendation**

MADRec reranks candidates from a trained sequential recommender. This implementation follows the candidate retrieval and ranking pipeline in Sections 3.2–3.5 of the manuscript. It accepts precomputed candidate pools from SASRec (default) or BERT4Rec using the same JSONL interface.

| Module | Implementation |
| --- | --- |
| **MAD** | The Interaction Agent observes previous debate rounds. The Semantic Agent also receives the current Interaction Agent output. Both score every eligible candidate in each round. |
| **EG-RAR** | The Semantic Agent returns a factual claim, a source quote, a preference score, and confidence for each candidate. The frozen Evidence Consistency Validator (ECV) supplies the probability of `Supported`. The semantic score is preference × confidence × support. |
| **JRC** | The Judge estimates one reliability weight per branch. Python combines the final-round scores for the union of both Top-K lists, including the score from the other branch for items selected by only one branch. |

The defaults are **30 retrieved candidates, 2 debate rounds, and Top-10 output**. Each nonempty pool uses five LLM calls at these defaults. Prompts use JSON, while Python validates scores and calculates rankings. Equal scores retain the original retrieval order.

## Quick start

Use Python 3.10 or later. The offline example uses fixed agent outputs and verification scores. It needs no API key or model weights.

```bash
python -m madrec demo --output demo.jsonl
```

The example shows a hit and a retrieval miss. It reports Recall@10 and NDCG@10 of 0.5 because both users are counted. The original script filename also works as a CLI entry point, for example `python chat_RecAct_horizontal_debate.py demo`.

## Run on exported candidates

```bash
python -m pip install -r requirements.txt
export MADREC_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export MADREC_MODEL="qwen-flash-2025-07-28"
# Set MADREC_API_KEY in your shell or secret manager.
python -m madrec recommend \
  --input data/users.jsonl \
  --evidence data/evidence.jsonl \
  --ecv-checkpoint checkpoints/ecv \
  --output results/run.jsonl
```

Use the base URL for your provider or deployment. MADRec appends `/chat/completions`. The result file contains round-by-round debate records, evidence checks, Judge weights, fused scores, and the final ranking for each user. Metrics are printed to stdout. Use `--candidate-count`, `--rounds`, and `--top-k` to override the defaults.

### User and candidate input

Export a row for every evaluation user, including users whose target was missed by the retriever. Each JSONL line is one object.

```json
{"user_id":"u1","history":[{"item_id":"17","title":"Night Patrol","rating":5,"timestamp":100}],"profile":"Enjoys investigative mysteries.","candidates":[{"item_id":"82","title":"The Archivist","retrieval_score":0.93},{"item_id":"42","title":"Ninth Avenue","retrieval_score":0.87}],"target_id":"82"}
```

- Supply the observed history from oldest to newest. Item IDs must use the same namespace in the user and evidence files. `profile` is optional and, if supplied, should use only observed data.
- Supply candidates in retrieval order or provide `retrieval_score` for all candidates. MADRec removes items in the observed history before taking the first 30 eligible candidates by default.
- `target_id` is optional and used only for evaluation. Each labeled user contributes to the denominator, and a retrieval miss contributes zero to Recall@K and NDCG@K.
- For the paper's evaluation protocol, train the retrieval model on the chronological prefix, use the second-most-recent interaction for validation, and reserve the latest for testing. Include the observed validation interaction in the test-time history when exporting. Use three independently sampled sets of 2,000 users per dataset, keep the users fixed across methods within each run, and summarize the per-run metrics with their mean and standard deviation.

### Evidence input

Each evidence row keeps original item text alongside its extracted semantic, functional, and contextual fields. Use item-side descriptions and metadata for the raw text.

```json
{"item_id":"82","raw_text":"An archivist investigates a decades-old disappearance.","semantic":"Investigative mystery","functional":"","contextual":""}
```

To extract the three structured fields, pass a JSONL file with `item_id` and `raw_text` to the offline builder. It uses the same LLM connection settings as the reranker.

```bash
python -m madrec build-evidence --input data/catalog.jsonl --output data/evidence.jsonl
```

A quote must be a substring of the source text for the same item to reach the verifier. The implementation assigns zero support when the source, claim, or quote is missing, or the quote cannot be found in the source. The interaction score remains available for fusion.

## Train the ECV

Use a pre-trained MADRec ECV checkpoint or fine-tune a new one from labeled pairs. Prepare separate JSONL files with item-disjoint train, validation, and test splits. Group items with duplicate source descriptions or passages before splitting, and pool the corresponding splits across datasets if using the paper's multi-domain setup.

```json
{"evidence":"An archivist investigates a decades-old disappearance.","claim":"The story follows an archivist's investigation.","label":"Supported"}
```

Use `Supported` when the passage supports every factual assertion and `NotSupported` otherwise.

```bash
python -m madrec train-ecv \
  --train data/ecv_train.jsonl \
  --validation data/ecv_valid.jsonl \
  --test data/ecv_test.jsonl \
  --output checkpoints/ecv
```

The defaults are DeBERTaV3-base, 10 epochs, batch size 16, AdamW with learning rate 2e-5 and weight decay 0.01, and seed 42. Both training and inference preserve the claim and truncate the evidence to fit the 512-token pair limit. The checkpoint with the best validation Macro-F1 is saved with its tokenizer and label mapping. Optional test metrics include Macro-F1, NotSupported precision and recall, and the Brier score.

Full experiments require your retrieval exports, annotated verification data or a trained ECV checkpoint, and LLM access. This package covers the ranking pipeline and ECV fine-tuning. Dataset preparation and retrieval-model training run separately.
