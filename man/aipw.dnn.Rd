% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/aipw.dnn.R
\name{aipw.dnn}
\alias{aipw.dnn}
\title{Deep Neural Network Augmented Inverse Probability Weighting (AIPW) Estimator for Average treatment effect (ATE)}
\usage{
aipw.dnn(
  Y,
  T,
  X_t,
  X = NULL,
  rescale_outcome = TRUE,
  model = NULL,
  optimizer = NULL,
  loss = NULL,
  epochs = 256,
  batch_size = 200,
  compile_outcome_model = NULL,
  fit_outcome_model = NULL,
  compile_propensity_score = NULL,
  fit_propensity_score = NULL,
  verbose = TRUE,
  debugging = FALSE,
  propensity_score = NULL
)
}
\arguments{
\item{Y}{numerical vector of observed outcomes of length \code{n}.}

\item{T}{logical vector of treatment statuses of length \code{n}.}

\item{X_t}{set of covariates. If covariates are time series, it should be a list of \code{k} different \code{n * p} matrixes. Here \code{p} is the length of the time series, and \code{k} is the number of different covariates. If covariates are single valued, it should be a matrix of size \code{n * k}.}

\item{X}{single-valued covariates are possible to be included besides the time series. It is supposed to be a \code{k’ * n} dimensional matrix. The default value is \code{NULL}.}

\item{rescale_outcome}{logical; if \code{TRUE}, the outcome values are standardized to have zero mean and one standard deviation. The default is \code{TRUE}.}

\item{model}{the nuisance models. A vector of size 3 (for ATE) or 2 (for ATT) can be used to assign it, or a single model can be used. In the latter case, that one model will be used for all nuisance models. The default value for Model is \code{NULL}, and in this case, the functions will use a specific predefined network.  Each model should be defined using the package Keras.}

\item{optimizer}{the optimization algorithms that are used for fitting the nuisance models. It can be assigned by a vector containing two optimizers or can be just one optimizer. If there are two optimizers, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{NULL}, and in this case, adam optimizer is used.}

\item{loss}{the loss functions for defining the fitting problem of nuisance models. Acceptable options are a vector containing two loss functions or just a single loss function. In the presence of two functions, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{NULL}, and in this case, mean square error and cross-entropy error are used for the outcome models and propensity score, respectively.}

\item{epochs}{numeric vector of the number of epochs in fitting processes. It is also acceptable to use a single number for all problems. The default value is 256.}

\item{batch_size}{numeric vector of the batch sizes in fitting processes. A single value is also acceptable. The default value is 200.}

\item{compile_outcome_model}{list of parameters and their values can be used to add to the Keras compile function when it is used for the outcome models. The list can be used to overwrite existing parameters, so it would be a way to compile the Keras model in a completely arbitrary way. The default is \code{NULL}.}

\item{fit_outcome_model}{list of parameters and their values can be used to add to the Keras fit function when it is used for the outcome models. The list can be used to overwrite existing parameters. The default is \code{NULL}.}

\item{compile_propensity_score}{list of parameters and their values can be used to add to the Keras compile function when it is used for the propensity score. The list can be used to overwrite existing parameters, so it would be a way to compile the Keras model in a completely arbitrary way. The default is \code{NULL}.}

\item{fit_propensity_score}{list of parameters and their values can be used to add to the Keras fit function when it is used for the propensity score. The list can be used to overwrite existing parameters. The default is \code{NULL}.}

\item{verbose}{vector of logical variables or just one. It can be used to control the verbosity of the fitting process. The default is \code{FALSE}.}

\item{debugging}{logical; if \code{TRUE}, the function will return the vectors of estimations for the outcome models and the propensity score. The default is \code{FALSE}.}

\item{propensity_score}{numerical vector of treatment probability of length \code{n}. It is optional to use it to consider a pre-estimated propensity score. If it is assigned, this function will not make the neural network model for estimating the propensity score, and the predefined values will be used in the final estimation. The default value is \code{NULL}, and it means a model will be fitted for the propensity score.}
}
\value{
DNN-AIPW Estimate and Inference of ATE
}
\description{
Deep Neural Network Augmented Inverse Probability Weighting (AIPW) Estimator for Average treatment effect (ATE)
}
\examples{

library(DNNcausal)
# simulate covariates, treatment assignment mechanism, and potential outcomes.
x = matrix(rnorm(100 * 5), nrow = 100)
p = 1/ (1 + exp(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)))
T = rbinom(100,1,p)
m1 = 1 + tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
m0 = tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
# obtain the observed outcome
y = T*m1 + (1-T)*m0
# call the ATE estimator function
aipw.dnn(y,T,x)


}
