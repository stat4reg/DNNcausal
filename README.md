# DNNCausal 
The R-package DNNCausal performs estimations of average causal effect and average treatment causal on the treated,
with AIPW by fitting deep neural networks to estimate nuisance models.
A particular use of the package is to use one-dimensional convolutional neural networks for time-series data inputs
(first introduced in Convolutional neural networks for valid and efficient causal inference).
Tensorflow and Keras are used to implement the neural network models.

Examples of use can be found at [Causal_CNN](https://github.com/stat4reg/Causal_CNN)


# References

Ghasempour, M., Moosavi, N. and de Luna, X. (2022). Convolutional neural networks for valid and efficient causal inference. ArXiv version soon available

# Installing

To install and load this package in R from GitHub, run the following commands:
```
install.packages("devtools")
library(devtools) 
install_github("stat4reg/DNNCausal")
library(DNNCausal)
keras::install_keras()
```

Mac users might have to run the following command in terminal:

```
$ xcode-select --install
```
