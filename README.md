# Defocus to focus: Photo-realistic bokeh rendering by fusing defocus and radiance priors
Xianrui Luo<sup>1</sup>, Juewen Peng<sup>1</sup>, Ke Xian<sup>1</sup>, Zijin Wu<sup>1</sup>, Zhiguo Cao<sup>1</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

![image](https://user-images.githubusercontent.com/44058627/224534575-da961bc9-3243-4d80-a89b-c76081f4ae8f.png)

This paper is an extension from conference paper on AIM 2020 Challenge on Rendering Realistic Bokeh ["Bokeh rendering from defocus estimation"](https://link.springer.com/chapter/10.1007/978-3-030-67070-2_15)

### [[Paper]](https://www.sciencedirect.com/science/article/pii/S1566253522001221)

## The link to the checkpoint trained on Ebb dataset
[Stage 1: Defocus Hallucination](https://1drv.ms/u/s!AiM1r33tcsmxpyUG7FiALBlPAVKK?e=cfppac) 

[Stage 2: Deep Poisson Fusion](https://1drv.ms/u/s!AiM1r33tcsmxpyZ3grOEo4Naupl3?e=fGAsFd)

This repository is the official PyTorch implementation of the Information Fusion paper "Defocus to focus: Photo-realistic bokeh rendering by fusing defocus and radiance priors". 

### Usage
`python test.py`

### Citation
If you find our work useful in your research, please cite our paper.
```
@article{luo2023defocus,
  title={Defocus to focus: Photo-realistic bokeh rendering by fusing defocus and radiance priors},
  author={Luo, Xianrui and Peng, Juewen and Xian, Ke and Wu, Zijin and Cao, Zhiguo},
  journal={Information Fusion},
  volume={89},
  pages={320--335},
  year={2023},
  publisher={Elsevier}
}
```
