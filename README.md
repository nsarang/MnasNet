# MnasNet Tensorflow 2 Implementation
Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le.  **MnasNet: Platform-Aware Neural Architecture Search for Mobile**. CVPR 2019.
   Arxiv link: https://arxiv.org/abs/1807.11626

## Usage
Available implementations: **a1, b1, small, d1, d1_320**
```python
from MnasNet_models import Build_MnasNet

# Standard model
model = Build_MnasNet('a1')


# Change default parameters:
model = Build_MnasNet('a1', dict(input_shape=(128, 128, 3), dropout_rate=0.5))
```


## Pretrained models
| Model | Dataset | Input Size | Depth Multiplier | Top-1 Accuracy | Top-5 Accuracy | Pixel 1 latency (ms) | DownLoad Link |
| :---- | ------- | ---------- | ---------------- | -------------- | -------------- | -------------------- | ------------- |
| MnasNet-A1 | ImageNet | 224*224 | 1.0 | 75.2 | 95.2 | 78ms | [Google Drive](https://drive.google.com/file/d/1tGHQC8vwrCKsMTKVTJMK-7uElYgLeR20/view?usp=sharing)

## Reference
[MnasNet - Official implementation for Cloud TPU](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)
