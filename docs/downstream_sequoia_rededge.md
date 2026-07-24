# Downstream Segmentation

Code: `downstream/specdeepmap/`

## Training

**Single run:**
```bash
cd downstream/specdeepmap
python train.py --input_folder Sequoia_train_loop/100p
```

**Experiment loops:**
```bash
# U-Net + Swin
python train_loop.py --dataset sequoia

# DPT + ViT
python train_loop_dpt_vit.py --dataset sequoia --vit_size small
```

**Switch dataset:**
```bash
--dataset rededge
```

## Evaluation

**All checkpoints:**
```bash
python testing_loop.py --preset swin --dataset sequoia
```

See [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for data organization.
