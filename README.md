# Reference-Based 3D-Aware Image Editing with Triplanes

![Teaser](./assets/teaser.png)

[Bahri Batuhan Bilecen](https://three-bee.github.io), Yigit Yalin, [Ning Yu](https://ningyu1991.github.io/), and [Aysegul Dundar](http://www.cs.bilkent.edu.tr/~adundar/)

Generative Adversarial Networks (GANs) have emerged as powerful tools for high-quality image generation and real image editing by manipulating their latent spaces. Recent advancements in GANs include 3D-aware models such as EG3D, which feature efficient triplane-based architectures capable of reconstructing 3D geometry from single images. However, limited attention has been given to providing an integrated framework for 3D-aware, high-quality, reference-based image editing. This study addresses this gap by exploring and demonstrating the effectiveness of the triplane space for advanced reference-based edits. Our novel approach integrates encoding, automatic localization, spatial disentanglement of triplane features, and fusion learning to achieve the desired edits. Additionally, our framework demonstrates versatility and robustness across various domains, extending its effectiveness to animal face edits, partially stylized edits like cartoon faces, full-body clothing edits, and 360-degree head edits. Our method shows state-of-the-art performance over relevant latent direction, text, and image-guided 2D and 3D-aware diffusion and GAN methods, both qualitatively and quantitatively.

## 🛠️ Requirements and installation
* Make sure you have 64-bit Python 3.8, PyTorch 11.1 (or above), and CUDA 11.3 (or above).
* Preferably, create a new environment via [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html) and activate the environment.
* Install pip dependencies: ```pip install -r requirements.txt```

## :scissors: Dataset preparation
We follow [EG3D's dataset preparation](https://github.com/NVlabs/eg3d/?tab=readme-ov-file#preparing-datasets) for pose extraction and face alignment. Make sure that you **do not skip** the setup of [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21).
Then, run in-the-wild preprocessing code:
```
cd ./dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=YOUR_INPUT_IMAGE_FOLDER
```
This will generate aligned images and a ```dataset.json``` containing camera matrices in ```YOUR_INPUT_IMAGE_FOLDER/preprocessed/```.

**We have included example images and poses in ```./example/```**.

## :checkered_flag: Checkpoints
Put all downloaded files in ```./checkpoints/```.
|        **Network**        |         **Filename**        |
|:-------------------------:|:---------------------------:|
| [EG3D rebalanced generator](https://drive.google.com/drive/folders/12pTX5TKQcA8ElNW5jDkWURSPUyISggHs?usp=sharing) | ```ffhqrebalanced512-128.pkl```   |
|        [EG3D-GOAE encoders](https://drive.google.com/drive/folders/12pTX5TKQcA8ElNW5jDkWURSPUyISggHs?usp=sharing) | ```encoder_FFHQ.pt``` & ```afa_FFHQ.pt``` |
|  [Finetuned fusion encoder](https://drive.google.com/file/d/1cObOXsMVRd55KXyA17KNZxJf_WW9u65_/view?usp=sharing) | ```encoder_FFHQ_finetuned.pt```   |
|      [BiSeNet segmentation](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) | ```79999_iter.pth```              |
|       [IR-SE50 for ID loss](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) | ```model_ir_se50.pth```           |

## :rocket: Quickstart
Run ```demo.ipynb``` for various editing examples.

## :point_down: Citation
Our codebase utilizes the following great works: [EG3D](https://github.com/NVlabs/eg3d), [EG3D-GOAE](https://github.com/jiangyzy/GOAE), [TriPlaneNetv2](https://github.com/anantarb/triplanenet), [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch), and [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch). We thank the authors for providing them.
```
@misc{bilecen2024referencebased,
      title={Reference-Based 3D-Aware Image Editing with Triplanes}, 
      author={Bahri Batuhan Bilecen and Yigit Yalin and Ning Yu and Aysegul Dundar},
      year={2024},
      eprint={2404.03632},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Copyright 2024 [Bilkent DLR](https://dlr.bilkent.edu.tr/). Licensed under the Apache License, Version 2.0 (the "License").
