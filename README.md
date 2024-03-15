# AST: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration (CVPR 2024)

[Shihao Zhou](https://joshyzhou.github.io/), [Duosheng Chen], [Jinshan Pan](https://jspan.github.io/), [Jinglei Shi](https://jingleishi.github.io/), and [Jufeng Yang](https://cv.nankai.edu.cn/)


#### News
- **March 15, 2024:** Paper accepted at CVPR 2024 :tada: 

<hr />

> **Abstract:** *  Transformer-based approaches have achieved promising performance in image restoration tasks, given their ability to model long-range dependencies, which is crucial for recovering clear images. Though diverse efficient attention mechanism designs have addressed the intensive computations associated with using transformers, they often involve redundant information and noisy interactions from irrelevant regions by considering all available tokens. In this work, we propose an <strong>A</strong>daptive <strong>S</strong>parse <strong>T</strong>ransformer (<strong>AST</strong>) to mitigate the noisy interactions of irrelevant areas and remove feature redundancy in both spatial and channel domains. AST comprises two core designs, i.e., an Adaptive Sparse Self-Attention (ASSA) block and a Feature Refinement Feed-forward Network (FRFN). Specifically, ASSA is adaptively computed using a two-branch paradigm, where the sparse branch is introduced to filter out the negative impacts of low query-key matching scores for aggregating features, while the dense one ensures sufficient information flow through the network for learning discriminative representations. Meanwhile, FRFN employs an enhance-and-ease scheme to eliminate feature redundancy in channels, enhancing the restoration of clear latent images. Experimental results on commonly used benchmarks have demonstrated the versatility and competitive performance of our method in \textbf{6} tasks, including deraining, dehazing, deraindrop, demoir{\`e}ing, desnowing and deshadowing.* 
<hr />


## Citation
If you find this project useful, please consider citing:

    @inproceedings{zhou2024AST,
      title={Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration},
      author={Zhou, Shihao and Chen, Duosheng and Pan, Jinshan and Shi, Jinglei and Yang, Jufeng},
      booktitle={CVPR},
      year={2024}
    }

