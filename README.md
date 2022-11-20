# StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment

Authors official PyTorch implementation of **[StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment](https://arxiv.org/abs/2209.13375)**. This paper has been accepted for publication at IEEE Conference on Automatic Face and Gesture Recognition, 2023.

<p align="center">
<img src="images/architecture.png" style="width: 750px"/>
</p>

>**StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment**<br>
> Stella Bounareli, Christos Tzelepis, Vasileios Argyriou, Ioannis Patras, Georgios Tzimiropoulos<br>
>
> **Abstract**: In this paper we address the problem of neural face reenactment, where, given a pair of a source and a target facial image, we need to transfer the target's pose (defined as the head pose and its facial expressions) to the source image, by preserving at the same time the source's identity characteristics (e.g., facial shape, hair style, etc), even in the challenging case where the source and the target faces belong to different identities. In doing so, we address some of the limitations of the state-of-the-art works, namely, a) that they depend on paired training data (i.e., source and target faces have the same identity), b) that they rely on labeled data during inference, and c) that they do not preserve identity in large head pose changes. More specifically, we propose a framework that, using unpaired randomly generated facial images, learns to disentangle the identity characteristics of the face from its pose by incorporating the recently introduced style space $\mathcal{S}$ of StyleGAN2, a latent representation space that exhibits remarkable disentanglement properties. By capitalizing on this, we learn to successfully mix a pair of source and target style codes using supervision from a 3D model. The resulting latent code, that is subsequently used for reenactment, consists of latent units corresponding to the facial pose of the target only and of units corresponding to the identity of the source only, leading to notable improvement in the reenactment performance compared to recent state-of-the-art methods. In comparison to state of the art, we quantitatively and qualitatively show that the proposed method produces higher quality results even on extreme pose variations. Finally, we report results on real images by first embedding them on the latent space of the pretrained generator. 

<a href="https://arxiv.org/abs/2209.13375"><img src="https://img.shields.io/badge/arXiv-2209.1337-b31b1b.svg" height=22.5></a>

## Face Reenactment Results
<br>


<p align="center">
<img src="images/source_target_gif.gif" style="width: 600px"/>
</p>

<p align="center">
<img src="images/source_target.png" style="width: 600px"/>
</p>

# Installation

* Python 3.5+ 
* Linux
* NVIDIA GPU + CUDA CuDNN
* Pytorch (>=1.5)
* [Pytorch3d](https://github.com/facebookresearch/pytorch3d)
* [DECA](https://github.com/YadiraF/DECA)

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).  

```
conda create -n python38 python=3.8
conda activate python38
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d 
pip install -r requirements.txt

```

# Pretrained Models

In order to use our method, make sure to download and save the required models under `./pretrained_models` path.

| Path | Description
| :--- | :----------
|[StyleGAN2-FFHQ-1024](https://drive.google.com/file/d/1I01HVu9UUyzAV7rNNnbzyFqPF0msebD7/view?usp=share_link)  | Official StyleGAN2 model trained on FFHQ 1024x1024 output resolution converted using [rosinality](FFHQ 1024x1024 output resolution).
|[e4e-FFHQ-1024](https://drive.google.com/file/d/1DexTMA3QMRNwQ3Xhdojki8UAYuY3g0uu/view?usp=share_link)  | Official e4e inversion model trained on FFHQ dataset taken from [e4e](https://github.com/omertov/encoder4editing). In case of using real images use this model to invert them into the latent space of StyleGAN2.
|[stylemask-model](https://drive.google.com/file/d/1_V_MnFB8rh5qrQ3zJk00fKXb81uHjIU8/view?usp=share_link)  | Our pretrained StyleMask model on FFQH 1024x1024 output resolution.

# Inference 
Given a pair of images or latent codes transfer the target facial pose into the source face.  
Source and target paths could be None (generate random latent codes pairs), latent code files, image files or directories with images or latent codes. In case of input paths are real images the script will use the e4e inversion model to get the inverted latent codes.
 
```
python run_inference.py --output_path ./results --save_grid
```

# Training 

We provide additional models needed during training.

| Path | Description
| :--- | :----------
|[IR-SE50 Model](https://drive.google.com/file/d/1s5pWag4AwqQyhue6HH-M_f2WDV4IVZEl/view?usp=sharing)  | Pretrained IR-SE50 model taken from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our identity loss.
|[DECA models](https://drive.google.com/file/d/1qZxbF13uuUpp-tCJ0kFIbSWH7h_Y-q2m/view?usp=sharing)  | Pretrained models taken from [DECA](https://github.com/YadiraF/DECA). Extract data.tar.gz under  `./libs/DECA/`.

By default, we assume that all pretrained models are downloaded and saved to the directory `./pretrained_models`. 

```
python run_trainer.py --experiment_path ./training_attempts/exp_v00 
``` 

## Citation

[1] Stella Bounareli, Christos Tzelepis, Argyriou Vasileios, Ioannis Patras, Georgios Tzimiropoulos. StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment. IEEE Conference on Automatic Face and Gesture Recognition (FG), 2023.

Bibtex entry:

```bibtex
@article{bounareli2022StyleMask,  
  author = {Bounareli, Stella and Tzelepis, Christos and Argyriou, Vasileios and Patras, Ioannis and Tzimiropoulos, Georgios},
  title = {StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment},
  journal = {IEEE Conference on Automatic Face and Gesture Recognition},
  year = {2023},
}
```



## Acknowledgment

This work was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.

