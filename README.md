# PyTorch-Encoding-Layer

**Deep TEN: Texture Encoding Network** [[arXiv]](https://arxiv.org/pdf/1612.02844.pdf)  
  [Hang Zhang](http://hangzh.com/), [Jia Xue](http://jiaxueweb.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html)
```
@article{zhang2016deep,
  title={Deep TEN: Texture Encoding Network},
  author={Zhang, Hang and Xue, Jia and Dana, Kristin},
  journal={arXiv preprint arXiv:1612.02844},
  year={2016}
}
```
If you would like to reproduce the texture recognition benchmark in the paper, please visit our original [Torch implementation](https://github.com/zhanghang1989/Deep-Encoding).

## Installation
- :bangbang:Install PyTorch from source to the **`$HOME`** directory
	* Please follow the [PyTorch tutorial](https://github.com/pytorch/pytorch#install-pytorch). 
  * If you are not professional, please follow the instruction, otherwise see [this issue](https://github.com/zhanghang1989/PyTorch-Encoding/issues/6) to change the path manually. 

- Install this package
	* Clone the repo
	```bash
	git clone git@github.com:zhanghang1989/PyTorch-Encoding-Layer.git && cd PyTorch-Encoding-Layer
	```
	* On Linux
	```bash
	python setup.py install
	```
	* On OSX
	```bash
	MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
	```
