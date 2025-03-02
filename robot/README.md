# Dexterous Grasp Detection

This project is designed for dexterous grasp detection using robotic hands. The model takes multimodal inputs such as point clouds, grasp poses, and joint configurations and predicts the grasp success probability. It is built using **PyTorch Lightning** and **Hydra** frameworks for modularity and scalability.

---

## Setup Instructions

### Environment Requirements
- **Operating System**: Ubuntu 20.04 or higher (Windows users can use WSL2 with Ubuntu).
- **Python Version**: Python 3.9 or higher.
- **Dependencies**: Listed in `requirements.txt`.


## How to Run

### Training
To train the model, run:
```bash
python train.py
```

### Testing
Before testing, ensure you have a trained checkpoint saved in the `checkpoints` directory (e.g., `best.ckpt`). Then run:
```bash
python test.py
```
### Notes on Precomputing Contact Points
The first time you run the training or testing scripts, the dataset will automatically precompute and save the contact points and normals for each sample. **This process may take some time**, especially if your dataset is large. The computed contact points will be saved in files (`contact_points_train.npz` or `contact_points_test.npz`), which will be reused in subsequent runs to save time.

## Notes
- Ensure that `checkpoints/best.ckpt` exists before running `test.py`.
- Modify configurations in the `conf/` directory to fine-tune the model or experiment with different hyperparameters.

