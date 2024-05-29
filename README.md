# DNNcausal 
The R-package DNNcausal implements estimators of average causal effect and average causal effect on the treated,
combining AIPW with deep neural networks fits of nuisance functions.
A possible use of the package is to use one-dimensional convolutional neural networks for time-series data inputs
(first introduced in Ghasempour et al, 2023).
Tensorflow and Keras are used to implement the neural network models.

Examples of use can be found at [Causal_CNN](https://github.com/stat4reg/Causal_CNN)


# References


Ghasempour, M, Moosavi, N, de Luna, X. (2023). Convolutional neural networks for valid and efficient causal inference. Journal of Computational and Graphical Statistics. Open access: [DOI: 10.1080/10618600.2023.2257247](https://doi.org/10.1080/10618600.2023.2257247).
On arXiv: https://doi.org/10.48550/arXiv.2301.11732

# Installing

To install and load this package in R from GitHub, run the following commands:
```
install.packages("devtools")
library(devtools) 
install_github("stat4reg/DNNcausal")
library(DNNcausal)
keras::install_keras()
```

Mac users might have to run the following command in terminal:

```
$ xcode-select --install
```
