# SHL Grammar Scoring Engine (spoken audio -> grammar score 0-5)

Everything (approach, pipeline, CV results, training RMSE, plots, limitations) is in `shl_grammar_scoring.ipynb`. The notebook is saved with its outputs, so it can be read without re-running.

## Approach
Audio -> Whisper transcript -> numeric fluency features + DeBERTa-v3 embedding (text model).
Audio -> wav2vec2 / WavLM / HuBERT intermediate-layer embeddings, mean+std pooled (acoustic model).
Each part is a standardised Ridge regressor. Final score = 0.8 * mean(3 acoustic Ridge) + 0.2 * text Ridge, clipped to [0, 5].

Results (5-fold CV on train): RMSE ~0.53, Pearson ~0.91. Training RMSE: 0.3495. Public leaderboard score: 0.3746.

## Files
* `shl_grammar_scoring.ipynb` - full pipeline with saved outputs
* `requirements.txt` - Python packages

The submission file was uploaded through the submission form. The audio data and the cached intermediate files (transcripts, embeddings) are not included in this public repository; they are available on request.

## How to run
1. Open the notebook on Kaggle with the competition data attached and a GPU enabled.
2. Run all cells. Without cached files the notebook transcribes the audio with Whisper and extracts the DeBERTa and speech-encoder embeddings itself, which takes a while on a single GPU.
3. If the cached files (`transcripts.csv`, `emb_deberta-v3-base.npy`, `ac_*.npz`) are placed in `/kaggle/working` or attached as a Kaggle dataset, the notebook finds them automatically and runs on CPU in a few minutes. It writes `submission.csv`.
