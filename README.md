# CIRF: Importance of Related Features for Plausible Counterfactual Explanations

## Authors

* **Hee-Dong Kim**, **Yeong-Joon Ju**, **Jung-Ho Hong**, and **Seong-Whan Lee** 

## Abstract
Counterfactual explanation (CFE) has gained increasing interest in recent years because it provides actionable counterexamples and enhances the interpretability of the decision boundaries in deep neural networks. 
An ideal CFE should provide both plausible and practical examples that can alter the decision of a classifier as a plausible CFE grounded in the real-world.
We propose a framework i.e., CFE for identifying related features (CIRF) to improve the plausibility.
CIRF comprises the following two steps: i) searching for the direction vectors that contain class information; ii) investigating an optimal point using a projection-point, which determines the magnitude of manipulation along the direction. 
CIRF utilizes related features and the property of a latent space in a generative model, thereby highlighting the importance of related features. 
The versatility of CIRF is validated by performing experiments using various domains and datasets, and the two interchangeable steps.
In addition, our approach provides an advantage over the counterfactual and semi-factual explanation manners, thereby enhancing the human comprehension of underlying reasons of decisions.
CIRF exhibits remarkable performance in terms of plausibility across various domains, including tabular and image datasets.

One paragraph of project description goes here.
![Alt text](figure_2.png "Optional title")


## Setting
Our code was implemented in Ubuntu OS.


### Library
To install the necessary libraries, run the following commands:
```bash
torch==1.8.1
ctgan==0.6.0
```

## Quick start
See quick_start.ipynb

## Results
The modifications of related features when altering class to "Gray" hair.
![Alt text](figure_6.png "Optional title")

The modifications of relatively low related features when altering class to "Brown" and "Blond" hair.

![Alt text](figure_7.png "Optional title")
