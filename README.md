# AutoLens

Automated lens design from scratch.

#### How to run

1. clone or download this repo
2. run ```python hello_deeplens.py```
3. run ```python auto_lens_design.py```

or

[Click here](https://colab.research.google.com/github/singer-yang/AutoLens/blob/main/auto_lens_design.ipynb) to run it in Google Colab.

#### Lens Design Examples

1. A video of automated design of a FoV 80deg, F/2.0, 4.55mm focal length.

<div style="text-align:center;">
    <img src="imgs/lens_design.gif" alt="AutoLens" style="width:450px; height:400px;"/>
</div>

2. 20 random automated lens design results for FoV 80deg, F/2.0, 4.55mm focal length.

<div style="text-align:center;">
    <img src="imgs/lens_design.png" alt="AutoLens" style="width:800px; height:550px;"/>
</div>

3. An aspherical lens (optimizing for 50k iterations) with outstanding optical performance.

<div style="text-align:center;">
    <img src="imgs/cellphone.png" alt="AutoLens" style="width:500px"/>
</div>


#### News and updates

More code and demos will be updated at https://github.com/singer-yang/DeepLens, we are aiming for next-generation differentiable optical design!

#### If you find this repo helpful, please cite our paper:

```
@article{yang2023curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={arXiv preprint arXiv:2302.01089},
  year={2023}
}
```
