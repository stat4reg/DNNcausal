#' loss function that is used in outcome model estimation
#'
#' @param y_true two-column matrix of observed outcomes and exposure status
#' @param y_pred two-column matrix of predicted values for \eqn{\hat{\mu}_0} and \eqn{\hat{\tau}}
#' @return loss value for predicted \eqn{\hat{\mu}_0} and \eqn{\hat{\tau}}, considering the loss function introduced in the reference.
#'
#' @import keras
#'
#' @references Max H. Farrell and Tengyuan Liang and Sanjog Misra (2021) \emph{Deep Neural Networks for Estimation and Inference}. The Econometric Society.
#'
causal.loss <- function(y_true, y_pred){
   K        <- keras::backend()
  # calculate the metric
  loss <- K$sum(.5 * (K$pow(y_true[,1] - y_pred[,1] - (y_true[,2] * y_pred[,2]), 2)))
  #loss <- K$mean( (K$pow((y_true[,1] - y_pred[,1] -  y_pred[,2])*y_true[,2] , 2)) + (K$pow((y_true[,1] - y_pred[,1]) * (1 - y_true[,2]), 2)))
  return(loss)
}
attr(causal.loss, "py_function_name") <- "causal.loss"


#' loss function sum of squars of errors
#'
#' @param y_true true labels
#' @param y_pred predicted vector
#' @return sum of squars of errors.
#'
#' @import keras
#'
#'
loss_sum_square_error <- function(y_true, y_pred){
  K        <- keras::backend()
  # calculate the metric
  loss <- K$sum(.5 * (K$pow((y_true - y_pred), 2)))
  #loss <- K$mean( (K$pow((y_true[,1] - y_pred[,1] -  y_pred[,2])*y_true[,2] , 2)) + (K$pow((y_true[,1] - y_pred[,1]) * (1 - y_true[,2]), 2)))
  return(loss)
}
attr(loss_sum_square_error, "py_function_name") <- "loss_sum_square_error"

#' the metric of mean of absolute percentage error
#'
#' @param y_true two-column matrix of observed outcomes and exposure status
#' @param y_pred two-column matrix of predicted values for \eqn{\hat{\mu}_0} and \eqn{\hat{\tau}}
#' @return mean of absolute percentage error for \eqn{\hat{\tau}}.
#'
#' @import keras
#'
#'
metric_mean_absolute_percentage_error2 <- function(y_true, y_pred){
  K        <- keras::backend()
  # calculate the metric
  loss <- K$mean( K$abs((y_true[,1] - y_pred[,1] - (y_true[,2] * y_pred[,2]) )/ sqrt(y_true[,1]**2 + 1) ))
  return(loss)
}
attr(metric_mean_absolute_percentage_error2, "py_function_name") <- "mean_absolute_percentage_error2"
