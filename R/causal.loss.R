#' loss function that is used in outcome model estimations
#'
#' @param y_true real labels
#' @param y_pred predicted labels
#' @return loss function of \eqn{\hat{\mu}_0} and \eqn{\hat{\tau}}
#'
#' @import keras
#'
causal.loss <- function(y_true, y_pred){
   K        <- keras::backend()
  # calculate the metric
  loss <- K$sum(.5 * (K$pow(y_true[,1] - y_pred[,1] - (y_true[,2] * y_pred[,2]), 2)))
  #loss <- K$mean( (K$pow((y_true[,1] - y_pred[,1] -  y_pred[,2])*y_true[,2] , 2)) + (K$pow((y_true[,1] - y_pred[,1]) * (1 - y_true[,2]), 2)))
  return(loss)
}
attr(causal.loss, "py_function_name") <- "causal.loss"

loss_sum_square_error <- function(y_true, y_pred){
  K        <- keras::backend()
  # calculate the metric
  loss <- K$sum(.5 * (K$pow((y_true - y_pred), 2)))
  #loss <- K$mean( (K$pow((y_true[,1] - y_pred[,1] -  y_pred[,2])*y_true[,2] , 2)) + (K$pow((y_true[,1] - y_pred[,1]) * (1 - y_true[,2]), 2)))
  return(loss)
}
attr(causal.loss, "py_function_name") <- "loss_sum_square_error"

metric_mean_absolute_percentage_error2 <- function(y_true, y_pred){
  K        <- keras::backend()
  # calculate the metric
  loss <- K$mean( K$abs((y_true[,1] - y_pred[,1] - (y_true[,2] * y_pred[,2]) )/ sqrt(y_true[,1]**2 + 1) ))
  return(loss)
}
attr(metric_mean_absolute_percentage_error2, "py_function_name") <- "mean_absolute_percentage_error2"
