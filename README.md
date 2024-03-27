# AutoLens

Automated lens design from scratch using [DeepLens](https://github.com/singer-yang/DeepLens).

Why AutoLens performs better than Zemax/CodeV lens design: **gradient calculation + Adam optimizer = better optimization capability!**

#### How to run

1. clone or download this repo
2. run ``python auto_lens_design.py``

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
    <img src="imgs/cellphone_example.png" alt="AutoLens" style="width:800px"/>
</div>


#### If you find this repo helpful, please cite our paper:

```
@article{yang2023curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={arXiv preprint arXiv:2302.01089},
  year={2023}
}
```

#### License

`<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" />``</a><br />`This work is licensed under a `<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">`Creative Commons Attribution-NonCommercial 4.0 International License `</a>`.
