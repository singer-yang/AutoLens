""" 
"Hello. world!" for DeepLens. 

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.
"""
from deeplens import Lensgroup

def main():
    lens = Lensgroup(filename='./lens_zoo/cellphone.json')
    lens.analysis(render=True)

if __name__=='__main__':
    main()