# SHL Grammar Scoring Engine (spoken audio -> grammar score 0-5)

Notebook: `shl_grammar_scoring.ipynb` (approach, pipeline, CV results, training RMSE, plots and limitations are inside).

## Approach
Audio -> Whisper transcript -> numeric fluency features + DeBERTa-v3 embedding (text model);
audio -> wav2vec2 / WavLM / HuBERT intermediate-layer embeddings, mean+std pooled (acoustic model).
Each part is a standardised Ridge regressor. Final score = 0.8 * mean(3 acoustic Ridge) + 0.2 * text Ridge, clipped to [0, 5].

## Run
* Kaggle notebook with the competition data attached. GPU is needed only if the cached files below are absent.
* Cached artefacts (optional, make the run take a few minutes on CPU): `transcripts.csv`, `emb_deberta-v3-base.npy`,
  `ac_wav2vec2-base-960h.npz`, `ac_wavlm-base-plus.npz`, `ac_hubert-base-ls960.npz`. Put them in `/kaggle/working` or attach as a Kaggle dataset.
* Output: `submission.csv` (columns `filename,label`). `submission_ac.csv` is the file that was submitted to Kaggle (public score 0.3746).
