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

## ✨ Contributions

<ol>
  <li>We propose a Spatial-Frequency Guided Pixel Transformer (SF-GPT), a novel framework for NIR-to-RGB translation that effectively combines pixel and spatial-frequency features to capture both local textures and global context.</li>

  <li>Our method integrates Discrete Cosine Transform (DCT) or Discrete Wavelet Transform (DWT) decomposition to extract spatial-frequency domain features. These transformations decompose input images into distinct low-frequency and high-frequency components, while pixel features are extracted using the PixelUnshuffle operation.</li>

  <li>We introduce Spatial-Frequency Guided Multi-head Self-Attention (SFG-MSA), a novel self-attention mechanism that enhances the network's ability to extract critical features from both pixel and spatial-frequency representations, improving overall feature discrimination and translation accuracy.</li>

  <li>We conduct extensive experiments demonstrating the superior performance of our approach compared to state-of-the-art methods, showcasing the effectiveness of SF-GPT in accurately translating NIR images to RGB images.</li>
</ol>