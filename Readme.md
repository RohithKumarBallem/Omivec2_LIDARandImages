# OmniVec2 LiDAR + Images (nuScenes mini)

A minimal, Apple Silicon–friendly baseline that:
- Loads nuScenes v1.0-mini camera images and LiDAR point clouds. [web:109][web:105]
- Tokenizes each modality (image patches and LiDAR pillars) into sequences. [web:16]
- Fuses tokens via a tiny shared Transformer block and yields a fused embedding. [web:16]
- Includes a lightweight self-supervised training scaffold to validate gradients/end-to-end on MPS. [web:255]

## Environment

- Apple Silicon (M1–M4) with PyTorch MPS backend. [web:255][web:264]
- Python environment created earlier; packages installed via pip, including nuscenes-devkit. [web:432][web:422]

## Dataset

- Download nuScenes v1.0-mini and maps from the official portal, accept terms, and extract. [web:92][web:452]
- Expected structure at the configured dataroot:
  - samples/ (keyframe sensor data), sweeps/ (intermediate frames), maps/, v1.0-mini/ (JSON tables). [web:109][web:105]

Update config/dataset.yaml with the absolute dataroot path.

## Quickstart

- Verify device and basic ops:
  - PYTHONPATH=. python scripts/smoke_test.py [web:255]
- Verify dataset I/O:
  - PYTHONPATH=. python scripts/test_io.py [web:109]
- Run end-to-end tokenization and fusion:
  - PYTHONPATH=. python scripts/demo_tokens.py [web:16]
- Train a tiny contrastive alignment scaffold:
  - PYTHONPATH=. python scripts/train_contrastive.py [web:255]

If MPS memory errors occur, reduce image_size and increase LiDAR cell size; example settings are provided in config/dataset.yaml and scripts/demo_tokens.py. [web:255]

## Project Structure

- config/dataset.yaml
  - Holds nuScenes path, version, selected cameras, LiDAR sensor name, and image resolution for tokenization. [web:109][web:105]

- src/utils/device.py
  - Selects device prioritizing MPS on Apple Silicon with CPU fallback and optional MPS memory watermark relaxation. [web:255]

- src/data/nuscenes_loader.py
  - Minimal loader that instantiates a NuScenes devkit object, reads the first sample’s multi-view images, and loads the top LiDAR point cloud. [web:109]

- src/models/tokenizers.py
  - ImagePatchTokenizer: converts an image tensor to patch embeddings via Conv2d and LayerNorm. [web:16]
  - LidarPillarTokenizer: bins point cloud into a BEV grid (pillars), aggregates simple stats, and projects to token embeddings. [web:16]

- src/models/omnivec2_core.py
  - CrossModalBlock: two-way cross-attention between image and LiDAR tokens with small feed-forward layers. [web:16]
  - OmniVec2Tiny: stacks one or more blocks and returns a fused embedding by averaging pooled tokens and applying a head. [web:16]

- src/utils/training.py
  - ProjectionHead: small projection MLP for contrastive training. [web:16]
  - info_nce_loss: symmetric InfoNCE loss using cosine similarity of normalized projections. [web:16]

- scripts/smoke_test.py
  - Sanity check: prints device and runs a conv2d on selected accelerator to confirm MPS works. [web:255]

- scripts/test_io.py
  - Validates nuScenes I/O by printing the sample token, shapes of loaded images for configured cameras, and LiDAR point count. [web:109]

- scripts/demo_tokens.py
  - Full pipeline demo: loads one sample, tokenizes images and LiDAR with MPS-friendly settings, runs through OmniVec2Tiny, and prints the fused embedding shape. [web:16]

- scripts/train_contrastive.py
  - Minimal self-supervised loop over random samples; computes fused embeddings for image/LiDAR and applies a symmetric InfoNCE loss to verify backprop and checkpoint saving. [web:255]

## Notes and Limitations

- This is a didactic scaffold, not a full detection/segmentation model; tokenizers are simple and optimized for clarity and MPS, not accuracy or speed. [web:16]
- For stability and memory on MPS, image_size, patch size, LiDAR cell size, attention heads, and depth are kept small; adjust carefully if increasing. [web:255]
- The current contrastive setup projects the fused embedding and may yield near-zero loss; for a stronger signal, project pooled modality tokens pre-fusion and use those in the loss. [web:16]

## References

- Official nuScenes devkit tutorial and expected folder layout. [web:109][web:105]
- Apple Metal/MPS backend support for PyTorch. [web:255][web:264]
- nuScenes devkit installation and usage. [web:432]
