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
  <li><strong>SF-GPT:</strong> We propose a novel <strong>Spatial-Frequency Guided Pixel Transformer</strong> for <strong>NIR-to-RGB translation</strong>, combining <strong>spatial</strong> and <strong>frequency-domain cues</strong> to capture both local textures and global context.</li>

  <li><strong>Dual-domain Feature Extraction:</strong> We incorporate <strong>DCT</strong> or <strong>DWT</strong> to extract <strong>low-frequency</strong> and <strong>high-frequency features</strong>, while <strong>pixel-level cues</strong> are obtained via <strong>PixelUnshuffle</strong> for fine-grained reconstruction.</li>

  <li><strong>SFG-MSA Module:</strong> We introduce a <strong>Spatial-Frequency Guided Multi-head Self-Attention</strong> mechanism that adaptively fuses pixel and frequency features, enhancing translation fidelity and feature discrimination.</li>

  <li><strong>State-of-the-art Performance:</strong> Extensive experiments confirm that <strong>SF-GPT surpasses existing techniques</strong> in both visual quality and quantitative metrics, achieving <strong>over +1.5 dB PSNR gain</strong> on benchmark datasets.</li>
</ol>



## 📈 PSNR Comparison on VCIP Test Dataset

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/psnr.png" width="800"/>
</p>
<p align="center">
</p>


## 🧠 Network Architecture

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/net.png" width="800"/>
</p>





## 🔍 Visualization of DCT and DWT Decomposition

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/nir_dwt_dct.jpg" width="800"/>
</p>


<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/DWT_DCT.png" width="800"/>
</p>

## 🖼️ Visual Comparison on VCIP Test Dataset

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/result_nir_vcip.png" width="800"/>
</p>

## 🖼️ Visual Comparison on SSMID Test Dataset

<p align="center">
  <img src="https://github.com/jianghongcheng/SF-GPT/blob/main/Figures/result_nir_ssmid.png" width="800"/>
</p>

## 📊 Quantitative Comparison on VCIP Test Dataset

<div align="center">

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>PSNR (↑)</th>
      <th>SSIM (↑)</th>
      <th>AE (↓)</th>
      <th>LPIPS (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>ATcycleGAN</td><td>19.59</td><td>0.59</td><td>4.33</td><td>0.295</td></tr>
    <tr><td>CoColor</td><td>23.54</td><td>0.69</td><td><u>2.68</u></td><td>0.233</td></tr>
    <tr><td>ColorMamba</td><td>24.56</td><td>0.71</td><td>2.81</td><td>0.212</td></tr>
    <tr><td>DCT-RCAN</td><td>22.15</td><td><u>0.77</u></td><td>3.40</td><td>0.214</td></tr>
    <tr><td>DRSformer</td><td>20.18</td><td>0.56</td><td>4.22</td><td>0.254</td></tr>
    <tr><td>HAT</td><td>19.42</td><td>0.69</td><td>3.98</td><td>0.298</td></tr>
    <tr><td>MCFNet</td><td>20.34</td><td>0.61</td><td>3.79</td><td>0.208</td></tr>
    <tr><td>MFF</td><td>17.39</td><td>0.61</td><td>4.69</td><td>0.318</td></tr>
    <tr><td>MPFNet</td><td>22.14</td><td>0.63</td><td>3.68</td><td>0.253</td></tr>
    <tr><td>NIR-GNN</td><td>17.50</td><td>0.60</td><td>5.22</td><td>0.384</td></tr>
    <tr><td>Restormer</td><td>19.43</td><td>0.54</td><td>4.41</td><td>0.267</td></tr>
    <tr><td>SPADE</td><td>19.24</td><td>0.59</td><td>4.59</td><td>0.283</td></tr>
    <tr><td>SST</td><td>14.26</td><td>0.57</td><td>5.61</td><td>0.361</td></tr>
    <tr><td>TTST</td><td>18.57</td><td>0.67</td><td>4.46</td><td>0.320</td></tr>
    <tr><td><b>SF-GPT (DCT)</b></td><td><b>26.09</b></td><td><u>0.77</u></td><td>2.72</td><td><u>0.132</u></td></tr>
    <tr><td><b>SF-GPT (DWT)</b></td><td><u>25.82</u></td><td><b>0.79</b></td><td><b>2.57</b></td><td><b>0.114</b></td></tr>
  </tbody>
</table>

<p><em>Table: Quantitative comparison with state-of-the-art methods on the VCIP test dataset. <b>Best</b> and <u>second-best</u> values are highlighted.</em></p>

</div>

---

## 📊 Quantitative Comparison on SSMID Test Dataset

<div align="center">

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>PSNR (↑)</th>
      <th>SSIM (↑)</th>
      <th>AE (↓)</th>
      <th>LPIPS (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>DGCAN</td><td>18.133</td><td>0.601</td><td>—</td><td>—</td></tr>
    <tr><td>Compressed DGCAN</td><td>16.973</td><td>0.565</td><td>—</td><td>—</td></tr>
    <tr><td>DRSformer</td><td>18.176</td><td>0.698</td><td>5.698</td><td>0.238</td></tr>
    <tr><td>Restormer</td><td>17.983</td><td>0.693</td><td>5.560</td><td>0.256</td></tr>
    <tr><td>HAT</td><td>17.677</td><td>0.692</td><td>5.803</td><td>0.249</td></tr>
    <tr><td>TTST</td><td>17.722</td><td>0.696</td><td>5.747</td><td>0.244</td></tr>
    <tr><td><b>SF-GPT (DCT)</b></td><td><u>19.011</u></td><td><u>0.699</u></td><td><u>5.541</u></td><td><u>0.185</u></td></tr>
    <tr><td><b>SF-GPT (DWT)</b></td><td><b>19.917</b></td><td><b>0.710</b></td><td><b>5.414</b></td><td><b>0.176</b></td></tr>
  </tbody>
</table>

<p><em>Table: Quantitative comparison with state-of-the-art methods on the SSMID test dataset. <b>Best</b> and <u>second-best</u> values are highlighted.</em></p>

</div>


## 📚 Citation

If you find this work helpful in your research, please cite:

```bibtex
@article{jiang2025spatial,
  title={Spatial-Frequency Guided Pixel Transformer for NIR-to-RGB Translation},
  author={Jiang, Hongcheng and Chen, ZhiQiang},
  journal={Infrared Physics \& Technology},
  year={2025},
  pages={105891},
  publisher={Elsevier},
  doi={10.1016/j.infrared.2025.105891},
  note={doi: \href{https://doi.org/10.1016/j.infrared.2025.105891}{10.1016/j.infrared.2025.105891}}
}

```


## 📬 Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out:

- 💻 Website: [jianghongcheng.github.io](https://jianghongcheng.github.io/)
- 📧 Email: [hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu)
- 🏫 Affiliation: University of Missouri–Kansas City (UMKC)


