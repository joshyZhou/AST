# Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration (CVPR 2024)

[Shihao Zhou](https://joshyzhou.github.io/), [Duosheng Chen](https://github.com/Calvin11311), [Jinshan Pan](https://jspan.github.io/), [Jinglei Shi](https://jingleishi.github.io/), and [Jufeng Yang](https://cv.nankai.edu.cn/)


#### News
- **Feb 27, 2024:** AST has been accepted to CVPR 2024 :tada: 

<hr />

<!-- > **Abstract:** *Transformer-based approaches have achieved promising performance in image restoration tasks, given their ability to model long-range dependencies, which is crucial for recovering clear images. Though diverse efficient attention mechanism designs have addressed the intensive computations associated with using transformers, they often involve redundant information and noisy interactions from irrelevant regions by considering all available tokens. In this work, we propose an <strong>A</strong>daptive <strong>S</strong>parse <strong>T</strong>ransformer (<strong>AST</strong>) to mitigate the noisy interactions of irrelevant areas and remove feature redundancy in both spatial and channel domains. AST comprises two core designs, i.e., an Adaptive Sparse Self-Attention (ASSA) block and a Feature Refinement Feed-forward Network (FRFN). Specifically, ASSA is adaptively computed using a two-branch paradigm, where the sparse branch is introduced to filter out the negative impacts of low query-key matching scores for aggregating features, while the dense one ensures sufficient information flow through the network for learning discriminative representations. Meanwhile, FRFN employs an enhance-and-ease scheme to eliminate feature redundancy in channels, enhancing the restoration of clear latent images. Experimental results on commonly used benchmarks have demonstrated the versatility and competitive performance of our method in <strong>6</strong> tasks, including deraining, dehazing, deraindrop, demoireing, desnowing and deshadowing.* 
<hr /> -->

## Package dependencies
The project is built with PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```
## Training
### Derain
To train AST on SPAD, you can run:
```sh
sh script/train_derain.sh
```
### Dehaze
To train AST on Densehaze, you can run:
```sh
sh script/train_dehaze.sh
```
### Raindrop
To train AST on AGAN, you can run:
```sh
sh script/train_raindrop.sh
```


## Evaluation
To evaluate AST, you can run:

```sh
sh script/test.sh
```
For evaluate on each dataset, you should uncomment corresponding line.


## Results
Experiments are performed for different image processing tasks including, rain streak removal, raindrop removal, and haze removal. 
Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Benchmark</th>
    <th align="center">Pretrained model</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">SPAD</td>
    <td align="center"><a href="https://pan.baidu.com/s/1uqsXeztA55ny8vnQHfxmlA?pwd=h68m">(code:h68m)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1KtAWESp3mzfrV4qy7BLaWg?pwd=wqdg">(code:wqdg)</a></td>
  </tr>
  <tr>
    <td align="left">AGAN</td>
    <td align="center"><a href="https://pan.baidu.com/s/1obhEpvWLV9FLF47FMcNX7g?pwd=astt">(code:astt)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1VewSQl6c7uWHpfCDVQeW4g?pwd=astt">(code:astt)</a></td>

  </tr>
  <tr>
    <td align="left">Dense-Haze</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PeSaf-ZRwnMBjlNHGEkvzg?pwd=astt">(code:astt)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/18JK8hQFLzSuiVZyianmJQw?pwd=astt">(code:astt)</a></td>
  </tr>

</table>

## Citation
If you find this project useful, please consider citing:

    @inproceedings{zhou2024AST,
      title={Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration},
      author={Zhou, Shihao and Chen, Duosheng and Pan, Jinshan and Shi, Jinglei and Yang, Jufeng},
      booktitle={CVPR},
      year={2024}
    }

## Acknowledgement

This code borrows heavily from [Uformer](https://github.com/ZhendongWang6/Uformer).
