# AutoLens

Automated lens design from scratch using gradient backpropagation + deep learning. This project is built on the top of [DeepLens](https://github.com/singer-yang/DeepLens) framework.

We are planning to to build AutoLens as an open-source lens design software, imaging an open-source Zemax. Other algorithms (for example, end-to-end lens design and implicit representation) will be updated in DeepLens. Welcome to join us if you are interested in optical design! Contact Xinge Yang (xinge.yang@kaust.edu.sa)

## How to run

**Method 1**

1. git clone or download this repo
2. run ``python autolens.py``

**Method 2**

Run it in Google Colab  

<a target="_blank" href="https://colab.research.google.com/github/singer-yang/AutoLens/blob/main/autolens.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Method 3**

Use our packaged .exe file (upcoming)

## Lens Design Examples

1. Example automated lens design of (left) FoV 80deg, F/2.0, 4.55mm focal length, and (right) Full-frame, F/3.0, 50mm focal length.

<div style="text-align:center;">
    <img src="imgs/lens_design1.gif" alt="AutoLens" style="height:300px;"/>
    <img src="imgs/lens_design2.gif" alt="AutoLens" style="height:300px;"/>
</div>

1. 20 random automated lens design results for FoV 80deg, F/2.0, 4.55mm focal length.

<div style="text-align:center;">
    <img src="imgs/lens_design.png" alt="AutoLens"/>
</div>

3. An aspherical lens with outstanding optical performance.

<div style="text-align:center;">
    <img src="imgs/cellphone_example.png" alt="AutoLens"/>
</div>

## If you find this repo helpful, please cite our paper:

```
@article{yang2023curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={arXiv preprint arXiv:2302.01089},
  year={2023}
}
```
