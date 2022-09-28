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

*Pretrained models and code are coming soon.*

## Face Reenactment Results
<br>


<p align="center">
<img src="images/source_target_gif.gif" style="width: 600px"/>
</p>

<p align="center">
<img src="images/source_target.png" style="width: 600px"/>
</p>

## Citation

[1] Stella Bounareli, Christos Tzelepis, Argyriou Vasileios, Ioannis Patras, Georgios Tzimiropoulos. StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment. IEEE Conference on Automatic Face and Gesture Recognition (FG), 2023.

Bibtex entry:

```bibtex
@article{bounareli2022StyleMask,  
  author = {Bounareli, Stella and Tzelepis, Christos and Argyriou, Vasileios and Patras, Ioannis and Tzimiropoulos, Georgios},
  title = {StyleMask: Disentangling the Style Space of StyleGAN2 for Neural Face Reenactment},
  journal = {arXiv preprint arXiv:2209.1337},
  year = {2022},
}
```

## Acknowledgment

This work was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.

