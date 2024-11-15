# diffusers_tryon_from_scratch
diffusers版本实现的Virtual Try-on, 用于学习目的, 帮助更好的了解它的工作原理。

# 🏷️  Introduction

![framework](./assets/model_architecture.png)

## 🔧 Requirements

```bash
conda create -n tryon python=3.8.10
conda activate tryon
pip install -U pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

## 📝 Citation
```bibtex
@article{shen2024IMAGDressing-v1,
  title={IMAGDressing-v1: Customizable Virtual Dressing},
  author={Shen, Fei and Jiang, Xin and He, Xin and Ye, Hu and Wang, Cong, and Du, Xiaoyu, Li Zechao, and Tang, Jinhui},
  booktitle={arXiv preprint arXiv:2407.12705},
  year={2024}
}
```