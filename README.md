# Spatial-Frequency Guided Pixel Transformer (SF-GPT) for NIR-to-RGB Translation

Official PyTorch implementation of the paper:

**Spatial-Frequency Guided Pixel Transformer for NIR-to-RGB Translation**  
*Infrared Physics & Technology*, 2025  
🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1350449525001847)  
🔗 [DOI: 10.1016/j.infrared.2025.105891](https://doi.org/10.1016/j.infrared.2025.105891)

## 📌 Introduction

Near-Infrared (NIR) imaging provides enhanced contrast and sensitivity but lacks the spatial and textural richness of RGB images. NIR-to-RGB translation is a challenging task due to:

- Spectral mapping ambiguity  
  <p align="center">
    <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/nir_dif_example.png" width="600"/>
  </p>

- Statistical weak correlation between NIR and RGB

  <p align="center">
    <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/Statistical_correlation.png" width="600"/>
  </p>


We propose **SF-GPT**, a novel deep learning architecture that leverages both spatial and frequency domains through transformer-based mechanisms.

## ✨ Key Contributions

<ol>
  <li><strong>SF-GPT:</strong> We propose a novel Spatial-Frequency Guided Pixel Transformer for NIR-to-RGB translation, combining spatial and frequency cues to capture both local textures and global context.</li>

  <li><strong>Dual-domain Feature Extraction:</strong> We incorporate DCT or DWT to extract low- and high-frequency features, while pixel-wise cues are obtained via PixelUnshuffle for fine-grained reconstruction.</li>

  <li><strong>SFG-MSA Module:</strong> We design a Spatial-Frequency Guided Multi-head Self-Attention mechanism that adaptively fuses pixel and frequency features, enhancing translation fidelity and feature discrimination.</li>

  <li><strong>State-of-the-art Performance:</strong> Extensive experiments validate the effectiveness of SF-GPT, outperforming existing methods in both visual quality and quantitative metrics.</li>
</ol>


## 🧠 Network Architecture

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/net.png" width="800"/>
</p>

<p align="center">
  <em>Figure: Overview of the proposed SF-GPT architecture, which integrates pixel and spatial-frequency domain features via the SFG-MSA module.</em>
</p>

