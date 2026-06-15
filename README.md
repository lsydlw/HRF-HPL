# HRF-HPL

GRU + hypergraph fusion model for next POI recommendation.

## Setup

```bash
pip install torch numpy pyyaml
```

Place datasets under `../ICASSP2024_ASTHL-main/datasets/{NYC|TKY|Gowalla}/`:

- `train_poi_zero.txt`
- `test_poi_zero.txt`
- `{Dataset}_pois_coos_poi_zero.pkl`

## Train

```bash
cd HRF-HPL
python train.py --dataset Gowalla
```

Common options: `--batch_size`, `--epochs`, `--device cuda`, `--save_dir logs`.

Training outputs (logs, args yaml, checkpoint) are saved under `--save_dir` (default: `logs/`).
