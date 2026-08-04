# Self-supervised training for high-resolution close-range multispectral remote sensing imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.11366-b31b1b.svg)](https://arxiv.org/abs/2607.11366)
[![Models](https://zenodo.org/badge/DOI/10.5281/zenodo.21532723.svg)](https://doi.org/10.5281/zenodo.21532723)

> **Official implementation** • Under Review, 2026 • [Paper](https://arxiv.org/abs/2607.11366)

Self-supervised learning (**MAE**, **MoCo v3**) on **MSUAV500K+N** for semantic segmentation on **WeedMap** dataset.

**Pre-trained Models:** [Zenodo](https://doi.org/10.5281/zenodo.21532723) • **Dataset:** [Finnish UAV Data](https://doi.org/10.5281/zenodo.18233335)

## Quick Start

```bash
# Install environment
conda env create -f environment.yml
conda activate ms-ssl-seg

# Configure paths
export DATA_ROOT=/path/to/data
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_v3_swin_s3.ckpt

# Run downstream training
cd downstream/specdeepmap
python train_loop.py --dataset sequoia
```

**See [docs/](docs/) for detailed instructions.**

## Documentation

- **[CHECKPOINTS.md](docs/CHECKPOINTS.md)** - Download and use pre-trained models
- **[DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md)** - Data organization and format
- **[ssl_pretraining.md](docs/ssl_pretraining.md)** - SSL training (MAE, MoCo v3)
- **[downstream_sequoia_rededge.md](docs/downstream_sequoia_rededge.md)** - Downstream segmentation

## Repository Structure

```
ssl_pretraining/     SSL training (MAE, MoCo v3)
downstream/          Semantic segmentation (SpecDeepMap)
preprocessing/       Data preparation scripts
configs/             Configuration examples
docs/                Detailed documentation
```

## Citation

```bibtex
@article{thomas2026selfsupervised,
  title={Self-supervised training for high-resolution close-range multispectral remote sensing imagery},
  author={Thomas, Leon-Friedrich and Änäkkälä, Mikko and Lajunen, Antti},
  journal={Under Review},
  year={2026},
  url={https://arxiv.org/abs/2607.11366}
}
```

## License

MIT License. SpecDeepMap components retain original license (see `downstream/specdeepmap/LICENSE_specdeepmap.md`).
