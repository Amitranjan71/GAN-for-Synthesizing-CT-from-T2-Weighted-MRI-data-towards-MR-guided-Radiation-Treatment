# GAN for Synthesizing CT from T2-Weighted MRI data towards MR-guided Radiation Treatment
In medical domain, cross-modality image synthesis suffers from multiple issues , such as context-misalignment, image distortion, image blurriness, and loss of details. The fundamental objective behind this study is to address these issues in estimating synthetic Computed tomography (sCT) scans from T2-weighted Magnetic Resonance Imaging (MRI) scans to achieve MRI-guided Radiation Treatment (RT). 
# Materials and methods 
We proposed a conditional generative adversarial network (cGAN) with multiple residual blocks to estimate sCT from T2-weighted MRI scans using 367 paired brain MR-CT images dataset. Few state-of-the-art deep learning models were implemented to generate sCT including Pix2Pix model, U-Net model, autoencoder model and their results were compared, respectively. 
# Results 
Results with paired MR-CT image dataset demonstrate that the proposed model with nine residual blocks in generator architecture results in the smallest mean absolute error (MAE) value of 0.030±0.017, and mean squared error (MSE) value of 0.010±0.011, and produces the largest Pearson correlation coefficient (PCC) value of 0.954±0.041, SSIM value of 0.823±0.063 and peak signal-to-noise ratio (PSNR) value of 21.422±3.964, respectively. We qualitatively evaluated our result by visual comparisons of generated sCT to original CT of respective MRI input. 
# Discussion 
The quantitative and qualitative comparison of this work demonstrates that deep learning-based cGAN model can be used to estimate sCT scan from a reference T2 weighted MRI scan. The overall accuracy of our proposed model outperforms different state-of-the-art deep learning-based models.

# For more details go through the Research Article link given below:

<a href="https://link.springer.com/article/10.1007/s10334-021-00974-5"> GAN for synthesizing CT from T2-weighted MRI data towards MR-guided radiation treatment </a>
