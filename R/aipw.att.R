#' Deep Neural Network Augmented Inverse Probability Weighting (AIPW) Estimator for Average treatment effects on the treated (ATT)
#'
#' @param	Y numerical vector of observed outcomes of length \code{n}.
#' @param T logical vector of treatment statuses of length \code{n}.
#' @param X_t set of covariates. If covariates are time series, it should be a list of \code{k} different \code{n * p} matrixes. Here \code{p} is the length of the time series, and \code{k} is the number of different covariates. If covariates are single valued, it should be a matrix of size \code{n * k}.
#' @param X single-valued covariates are possible to be included besides the time series. It is supposed to be a \code{k’ * n} dimensional matrix. The default value is \code{NULL}.
#' @param rescale_outcome logical; if \code{TRUE}, the outcome values are standardized to have zero mean and one standard deviation. The default is \code{TRUE}.
#' @param do_standardize a character string determining the orientation of standardizing the covariates. This must be \code{"row"} for row-wise or \code{"column"} for column-wise standardizing. The default \code{NULL} is equivalent to using the original covariates.
#' @param use_scalers logical; determines whether to use single valued covariates in an additional parallel neural network or not. The default is \code{TRUE}.
#' @param rescale_treated logical; in the presence of scaler covariates, to standardize them, it determines whether only treated ones are considered for the scaling or all. The default is \code{TRUE}.
#' @param model the nuisance models. A vector of size 2 can be used to assign it, or a single model can be used. In the latter case, that one model will be used for all nuisance models. The default value is \code{NULL}, and in this case, the functions will use a specific predefined network.  Each model should be defined using the package Keras.
#' @param optimizer the optimization algorithms that are used for fitting the nuisance models. It can be assigned by a vector containing two optimizers or can be just one optimizer. If there are two optimizers, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{NULL}, and in this case, adam optimizer is used.
#' @param loss the loss functions for defining the fitting problem of nuisance models. Acceptable options are a vector containing two loss functions or just a single loss function. In the presence of two functions, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{NULL}, and in this case, mean square error and cross-entropy error are used for the outcome models and propensity score, respectively.
#' @param epochs numeric vector of the number of epochs in fitting processes. It is also acceptable to use a single number for all problems. The default value is 256.
#' @param batch_size numeric vector of the batch sizes in fitting processes. A single value is also acceptable. The default value is 200.
#' @param propensity_score numerical vector of treatment probability of length \code{n}. It is optional to use it to consider a pre-estimated propensity score. If it is assigned, this function will not make the neural network model for estimating the propensity score, and the predefined values will be used in the final estimation. The default value is \code{NULL}, and it means a model will be fitted for the propensity score.
#' @param debugging logical; if \code{TRUE}, the function will return the vectors of estimations for the outcome models and the propensity score. The default is \code{FALSE}.
#' @param verbose vector of logical variables or just one. It can be used to control the verbosity of the fitting process. The default is \code{FALSE}.
#' @param compile_outcome_model list of parameters and their values can be used to add to the Keras compile function when it is used for the outcome models. The list can be used to overwrite existing parameters, so it would be a way to compile the Keras model in a completely arbitrary way. The default is \code{NULL}.
#' @param compile_propensity_score list of parameters and their values can be used to add to the Keras compile function when it is used for the propensity score. The list can be used to overwrite existing parameters, so it would be a way to compile the Keras model in a completely arbitrary way. The default is \code{NULL}.
#' @param fit_outcome_model list of parameters and their values can be used to add to the Keras fit function when it is used for the outcome models. The list can be used to overwrite existing parameters. The default is \code{NULL}.
#' @param fit_propensity_score list of parameters and their values can be used to add to the Keras fit function when it is used for the propensity score. The list can be used to overwrite existing parameters. The default is \code{NULL}.
#' @param truncate_ps logical; it determines whether a truncation over the estimation of propensity scores should be considered or not. The default is \code{TRUE}.
#'
#' @return DNN-AIPW Estimate and Inference of ATT
#' @export
#'
#'
#' @import keras abind
#' @importFrom stats predict
#'
#' @examples
#'
#' library(DNNcausal)
#' # simulate covariates, treatment assignment mechanism, and potential outcomes.
#' x = matrix(rnorm(100 * 5), nrow = 100)
#' p = 1/ (1 + exp(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)))
#' T = rbinom(100,1,p)
#' m1 = 1 + tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
#' m0 = tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
#' # obtain the observed outcome
#' y = T*m1 + (1-T)*m0
#' # call the ATT estimator function
#' aipw.att(y,T,x)
#'
#'
#'
#'
aipw.att = function(Y,T,X_t,X = NULL,rescale_treated=TRUE,rescale_outcome=TRUE,model=NULL,optimizer=NULL,loss= NULL,epochs=256,batch_size=200, compile_outcome_model=NULL,fit_outcome_model=NULL,compile_propensity_score=NULL,fit_propensity_score=NULL,verbose =TRUE, debugging= FALSE, propensity_score= NULL, use_scalers = TRUE, do_standardize = NULL,truncate_ps = TRUE){



  state = data.frame()
  if (is.list(X_t)){
    if(is.vector(X_t)){
      if (length(unique(lapply(X_t,function(x){dim(x)})))==1){
        k = length(X_t)
        n = dim(X_t[[1]])[1]
        p = dim(X_t[[1]])[2]
        Xt_t = list()
        if(!is.null(do_standardize) && do_standardize == "Row"){
        for (i in 1:k) {
          x_t = data.matrix(X_t[[i]])
          mean = rowMeans(x_t,na.rm = TRUE)
          sd = sqrt(rowMeans((x_t-mean)^2,na.rm = TRUE))
          sd[sd==0] = 1
          X_t[[i]]= (x_t - mean)/sd
          Xt_t[[i]] = X_t[[i]][T==0,]

          if (dim(state)[1]==0){
            state = data.frame(mean)
            state = cbind(state,sd)
          }else{
            state = cbind(state,mean,sd)
          }}
        }else if(!is.null(do_standardize) && do_standardize == "Column") {
          for (i in 1:k) {
            x_t = data.matrix(X_t[[i]])
            mean = colMeans(x_t,na.rm = TRUE)
            sd = sqrt(colMeans(sweep(x_t,2,mean)^2,na.rm = TRUE))
            sd[sd==0] = 1
            X_t[[i]]= sweep(sweep(x_t ,2, mean),2,sd,"/")
            Xt_t[[i]] = X_t[[i]][T==0,]

            }
        }else{
          for (i in 1:k) {
            Xt_t[[i]] = X_t[[i]][T==0,]
            }
        }
        X_t = abind::abind(X_t,along = 3)
        Xt_t = abind::abind(Xt_t,along = 3)
      }else if (length(unique(lapply(X_t,function(x){dim(x)})))==2){
        stop("Under construction...")
        k = length(X_t)
        n = dim(X_t[1])[1]
        p = dim(X_t[1])[2]
      }else {
        stop("The data list contains datasets with too many different dimensions.")
      }

    }else{
      n = dim(X_t)[1]
      k = dim(X_t)[2]
      p = 1
      X_t = scale(X_t)
      X_t[is.na(X_t)]=0
    }
  }#todo add normalizing for the arrays
  else if(is.array(X_t)|is.matrix(X_t)){
    if (length(dim(X_t))==1){
      n = dim(X_t)
      k = 1
      p = 1
      if(!is.null(do_standardize) && (do_standardize == "Column" || do_standardize == TRUE)){
        mean = mean(X_t,na.rm = TRUE)
        sd = sd(X_t,na.rm = TRUE)
        sd[sd==0] = 1
        X_t= (X_t - mean)/sd
        Xt_t = X_t[T==0,]
      }else{
      Xt_t = X_t[T==0]}
    }else if (length(dim(X_t))==2){
      n = dim(X_t)[1]
      k = dim(X_t)[2]
      p = 1
      if(!is.null(do_standardize) && do_standardize == "Row"){
        mean = rowMeans(X_t,na.rm = TRUE)
        sd = sqrt(rowMeans((X_t-mean)^2,na.rm = TRUE))
        sd[sd==0] = 1
        X_t= (X_t - mean)/sd
        Xt_t = X_t[T==0,]
        if (dim(state)[1]==0){
          state = data.frame(mean)
          state = cbind(state,sd)
        }else{
          state = cbind(state,mean,sd)
        }
      }else if(!is.null(do_standardize) && do_standardize == "Column"){
          mean = colMeans(X_t,na.rm = TRUE)
          sd = sqrt(colMeans(sweep(X_t,2,mean)^2,na.rm = TRUE))
          sd[sd==0] = 1
          X_t= sweep(sweep(X_t ,2, mean),2,sd,"/")
          Xt_t = X_t[T==0,]
      }else{
      Xt_t = X_t[T==0,]}
    }else if (length(dim(X_t))==3){
      n = dim(X_t)[1]
      p = dim(X_t)[2]
      k = dim(X_t)[3]
      Xt_t = X_t[T==0,]
    }else if (length(dim(X_t))==4){
      n = dim(X_t)[1]
      p1 = dim(X_t)[2]
      p2 = dim(X_t)[3]
      k = dim(X_t)[4]
      p = c(p1,p2)
    }
  }else if(is.vector(X_t)){
    n = length(X_t)
    k = 1
    p = 1
  }


  if (is.null(X)){
    if(ncol(state)!=0) X = as.matrix(state)
  } else if(is.data.frame(X)){
    if(n == nrow(state)) X =as.matrix(cbind(X,state))
  }

cat('inputs are defined \n')

  #
  # if (is.list(X)){
  #   if(is.vector(X)){
  #     if (length(unique(lapply(X,function(x){dim(x)})))==1){
  #       k = length(X)
  #       n = dim(X[[1]])[1]
  #       p = dim(X[[1]])[2]
  #
  #       if(rescale_treated){
  #         X = abind::abind(lapply(X,function(x){
  #           x = data.matrix(x)
  #           xt = x[T==0,]
  #           xt = scale(xt)
  #           xtscale = attr(xt,'scaled:scale')
  #           xtcenter = attr(xt,'scaled:center')
  #           xt[is.na(xt)]=0
  #
  #           x=sweep(sweep(x,2,xtcenter),2,xtscale,'/')
  #           x[is.na(x)]=0
  #           x[is.infinite(x)]=0
  #           return(x)
  #         }),along = 3)
  #         Xt = X[T==0,]
  #         }else{
  #         X = abind::abind(lapply(X,function(x){
  #           x = data.matrix(x)
  #           x = scale(x)
  #           x[is.na(x)]=0
  #           return(x)
  #         }),along = 3)
  #         Xt = X[T==0,]
  #       }
  #
  #       X = abind::abind(lapply(X,function(x){data.matrix(x)}),along = 3)
  #     }else if (length(unique(lapply(X,function(x){dim(x)})))==2){
  #       stop("Under construction...")
  #       k = length(X)
  #       n = dim(X[1])[1]
  #       p = dim(X[1])[2]
  #     }else {
  #       stop("The data list contains datasets with too many different dimensions.")
  #     }
  #
  #   }else{
  #     n = dim(X)[1]
  #     k = dim(X)[2]
  #     p = 1
  #   }
  # }else if(is.array(X)|is.matrix(X)){
  #   if (length(dim(X))==1){
  #     n = dim(X)
  #     k = 1
  #     p = 1
  #   }else if (length(dim(X))==2){
  #     n = dim(X)[1]
  #     k = dim(X)[2]
  #     p = 1
  #   }else if (length(dim(X))==3){
  #     n = dim(X)[1]
  #     p = dim(X)[2]
  #     k = dim(X)[3]
  #   }else if (length(dim(X))==4){
  #     n = dim(X)[1]
  #     p1 = dim(X)[2]
  #     p2 = dim(X)[3]
  #     k = dim(X)[4]
  #     p = c(p1,p2)
  #   }
  # }else if(is.vector(X)){
  #   n = length(X)
  #   k = 1
  #   p = 1
  # }


  if (is.null(model)) {

    model_m = keras::keras_model_sequential()
    model_p = keras::keras_model_sequential()
    if(length(p)==2){
      model_m =  keras::layer_conv_2d(model_m,30,c(p1-2,p2-2),padding = 'same',activation = "relu",input_shape = c(p1, p2 ))
      model_m =  keras::layer_conv_2d(model_m,20,c(p1-2,p2-2),padding = 'same',activation = "relu")
      model_m =  keras::layer_flatten(model_m)

      model_p =  keras::layer_conv_2d(model_p,15,c(p1-2,p2-2),padding = 'same',activation = "relu",input_shape = c(p1, p2 ))
      model_p =  keras::layer_conv_2d(model_p,10,c(p1-2,p2-2),padding = 'same',activation = "relu")
      model_p =  keras::layer_flatten(model_p)
    }else if(length(p)==1 & p!=1){
      model_m =  keras::layer_conv_1d(model_m,30,p-2,padding = 'same',activation = "relu",input_shape =c(p,k))
      model_m =  keras::layer_conv_1d(model_m,20,p-2,padding = 'same',activation = "relu")
      model_m =  keras::layer_flatten(model_m)

      model_p =  keras::layer_conv_1d(model_p,15,p-2,padding = 'same',activation = "relu",input_shape =c(p,k))
      model_p =  keras::layer_conv_1d(model_p,10,p-2,padding = 'same',activation = "relu")
      model_p =  keras::layer_flatten(model_p)
    }else if(length(p)==1 & p==1){
      model_m =  keras::layer_dense(model_m,30,activation = "relu",input_shape = k)
      model_m =  keras::layer_dense(model_m,20,activation = "relu")

      model_p =  keras::layer_dense(model_p,15,activation = "relu",input_shape = k)
      model_p =  keras::layer_dense(model_p,10,activation = "relu")
    }
  } else if(is.vector(model)){
    if (length(model)==1) {
      model_m = model[1]
      model_p = model[1]
    }
    else if (length(model)==2) {
      model_m = model[[1]]
      model_p = model[[2]]
    }
    else if (length(model)==3) {
      stop("Using three models is not supported yet.")
      model_m0 = model[[1]]
      model_m1 = model[[2]]
      model_p = model[[3]]
    }
  } else {
    model_p = model
    model_m = model
  }
  if(!is.null(X) & (ncol(X)!=0 && use_scalers)){
    px = dim(X)[2]
    model_mx =  keras::keras_model_sequential()
    model_mx =  keras::layer_dense(model_mx,30,activation = "relu",input_shape = px)
    model_mx =  keras::layer_dense(model_mx,20,activation = "relu")

    model_px =  keras::keras_model_sequential()
    model_px =  keras::layer_dense(model_px,30,activation = "relu",input_shape = px)
    model_px =  keras::layer_dense(model_px,20,activation = "relu")


    m_m =  keras::layer_dense(keras::layer_concatenate(list(model_m$output,model_mx$output)),1)
    m_p =  keras::layer_dense(keras::layer_concatenate(list(model_p$output,model_px$output)),2,activation = keras::activation_softmax)
    if(truncate_ps){
      m_p =  keras::layer_lambda(m_p, f = function(x){x*log(99)})
      m_p =  keras::layer_activation(m_p, activation = keras::activation_softmax)
    }



    model_m = keras::keras_model(list(model_m$input,model_mx$input),m_m)
    model_p = keras::keras_model(list(model_p$input,model_px$input),m_p)
  }
cat('model is defined \n')
  if (is.null(optimizer)) {
    optimizer_m = keras::optimizer_adam(lr = 0.003)
    optimizer_p = keras::optimizer_adam(lr = 0.003)
  } else if(is.vector(optimizer)){
    if (length(optimizer)==1) {
      optimizer_m = optimizer[[1]]
      optimizer_p = optimizer[[1]]
    }
    else if (length(optimizer)==2) {
      optimizer_m = optimizer[[1]]
      optimizer_p = optimizer[[2]]
    }
    else if (length(optimizer)==3) {
      stop("Using three models is not supported yet.")
      optimizer_m0 = optimizer[[1]]
      optimizer_m1 = optimizer[[2]]
      optimizer_p = optimizer[[3]]
    }
  } else {
    optimizer_p = optimizer
    optimizer_m = optimizer
  }
  if (is.null(loss)) {
    loss_m = keras::loss_mean_squared_error
    loss_p = keras::loss_categorical_crossentropy
  } else if(is.vector(loss)){
    if (length(loss)==1) {
      loss_m = keras::loss_mean_squared_error
      loss_p = loss[[1]]
    }
    else if (length(loss)==2) {
      loss_m = loss[[1]]
      loss_p = loss[[2]]
    }
    else if (length(loss)==3) {
      stop("Using three models is not supported yet.")
      loss_m0 = loss[[1]]
      loss_m1 = loss[[2]]
      loss_p = loss[[3]]
    }
  } else {
    loss_m = keras::loss_mean_squared_error
    loss_p = loss
  }

  if(is.vector(epochs)){
    if (length(epochs)==1) {
      epochs_m = epochs[1]
      epochs_p = epochs[1]
    }
    else if (length(epochs)==2) {
      epochs_m = epochs[1]
      epochs_p = epochs[2]
    }
    else if (length(epochs)==3) {
      stop("Using three models is not supported yet.")
    }
  } else {
    epochs_p = epochs
    epochs_m = epochs
  }


  if(is.vector(batch_size)){
    if (length(batch_size)==1) {
      batch_size_m = batch_size[1]
      batch_size_p = batch_size[1]
    }
    else if (length(batch_size)==2) {
      batch_size_m = batch_size[1]
      batch_size_p = batch_size[2]
    }
    else if (length(batch_size)==3) {
      stop("Using three models is not supported yet.")
    }
  } else {
    batch_size_p = batch_size
    batch_size_m = batch_size
  }

  if(is.vector(verbose)){
    if (length(verbose)==1) {
      verbose_m = verbose[1]
      verbose_p = verbose[1]
    }
    else if (length(verbose)==2) {
      verbose_m = verbose[1]
      verbose_p = verbose[2]
    }
    else if (length(verbose)==3) {
      stop("Using three models is not supported yet.")
    }
  } else {
    verbose_p = verbose
    verbose_m = verbose
  }
  if(is.logical(verbose_p)){
    verbose_p = as.numeric(verbose_p)
  }
  if(is.logical(verbose_m)){
    verbose_m = as.numeric(verbose_m)
  }
  #### outcome models
cat('training starts \n')
  if(!is.null(X) &(ncol(X)!=0 && use_scalers)){outcome_model = model_m}else{
    outcome_model = keras::layer_dense(object = model_m,units = 1)}

  if (is.null(compile_outcome_model)) {
    keras::compile(outcome_model,
                   loss = loss_m,
                   optimizer = optimizer_m,
                   metrics = keras::metric_mean_absolute_percentage_error)
} else{
      if(!'object'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('object'=outcome_model))
      } else{
        outcome_model = compile_outcome_model$object
      }
      if(!'loss'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('loss'=loss_m))
      }
      if(!'metrics'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('metrics'=keras::metric_mean_absolute_percentage_error))
      }
      if(!'optimizer'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('optimizer'=optimizer_m))
      }
      do.call(keras::compile,compile_outcome_model)
    }






  if (!is.null(X) &(ncol(X)!=0 && use_scalers)) {
  if(rescale_treated){
    Xt = X[T==0,]
    Xt = scale(Xt)
    xtscale = attr(Xt,'scaled:scale')
    xtcenter = attr(Xt,'scaled:center')
    Xt[is.na(Xt)]=0

    X=sweep(sweep(X,2,xtcenter),2,xtscale,'/')
    X[is.na(X)]=0
    X[is.infinite(X)]=0
  }else{
    X = scale(X)
    X[is.na(X)]=0
    Xt = X[T==0,]
  }
    x_train =list( X_t, X)
    xt_train <- list(Xt_t, Xt)
  }else{
    x_train <- X_t
    xt_train <- Xt_t
    }
  #y_train <- keras::array_reshape(cbind(Y,T), c(length(Y),2))

  Yt = Y[T==0]

  if(rescale_outcome){
    Yt = scale(Yt)
    ytscale = attr(Yt,'scaled:scale')
    ytcenter = attr(Yt,'scaled:center')
  }else{
    ytscale = 1
    ytcenter = 0
  }



  #x_train <- keras::array_reshape(X, c(nrow(X), ncol(X)))
  #xt_train <- keras::array_reshape(Xt, c(nrow(Xt), ncol(Xt)))
  #x_train <- X
  #xt_train <- Xt
  yt_train <- keras::array_reshape(Yt, c(length(Yt)))


  if (is.null(fit_outcome_model)) {
    keras::fit(outcome_model,
               xt_train, yt_train,
               batch_size = batch_size_m,
               epochs = epochs_m,verbose=verbose_m)
  } else{
    if(!'object'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('object'=outcome_model))
    } else{
      outcome_model = fit_outcome_model$object
    }
    if(!'x'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('x'=xt_train))
    }
    if(!'y'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('y'=yt_train))
    }
    if(!'batch_size'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('batch_size'=batch_size_m))
    }
    if(!'epochs'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('epochs'=epochs_m))
    }
    if(!'verbose'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('verbose'=verbose_m))
    }
    do.call(keras::fit,fit_outcome_model)
  }

  predictions <- keras::predict_on_batch(outcome_model,x_train)

  #### propensity score
  if( is.null(propensity_score)){
  if(!is.null(X) &(ncol(X)!=0  && use_scalers)){ps_model = model_p}else{
    ps_model = keras::layer_dense(model_p,units = 2,activation = keras::activation_softmax)#,activation = keras::activation_sigmoid
  if(truncate_ps){
    ps_model = keras::layer_lambda(ps_model, f = function(x){x*log(99)},)
    ps_model = keras::layer_activation(ps_model, activation = keras::activation_softmax)
  }
    }
  if (is.null(compile_propensity_score)) {
    keras::compile(ps_model,
                   loss = loss_p,
                   optimizer = optimizer_p,
                   metrics = c(keras::metric_binary_accuracy, keras::metric_binary_crossentropy)
    )
    } else{
      if(!'object'%in%names(compile_propensity_score)){
        compile_propensity_score = append(compile_propensity_score, list('object'=ps_model))
      } else{
        ps_model = compile_propensity_score$object
      }
      if(!'loss'%in%names(compile_propensity_score)){
        compile_propensity_score = append(compile_propensity_score, list('loss'=loss_p))
      }
      if(!'metrics'%in%names(compile_propensity_score)){
        compile_propensity_score = append(compile_propensity_score, list('metrics'=c(keras::metric_binary_accuracy, keras::metric_binary_crossentropy)))
      }
      if(!'optimizer'%in%names(compile_propensity_score)){
        compile_propensity_score = append(compile_propensity_score, list('optimizer'=optimizer_p))
      }
      do.call(keras::compile,compile_propensity_score)
    }

  t_train <- keras::to_categorical(T,2)



  if (is.null(fit_propensity_score)) {
    keras::fit(ps_model,
               x_train, t_train ,
               batch_size = batch_size_p,
               epochs = epochs_p, verbose=verbose_p
    )
  } else{
    if(!'object'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('object'=ps_model))
    } else{
      ps_model = fit_propensity_score$object
    }
    if(!'x'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('x'=x_train))
    }
    if(!'y'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('y'=t_train))
    }
    if(!'batch_size'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('batch_size'=batch_size_p))
    }
    if(!'epochs'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('epochs'=epochs_p))
    }
    if(!'verbose'%in%names(fit_propensity_score)){
      fit_propensity_score = append(fit_propensity_score, list('verbose'=verbose_p))
    }
    do.call(keras::fit,fit_propensity_score)
  }

  if(!is.null(X) &(ncol(X)!=0  && use_scalers)){predictions_ps <- keras::predict_on_batch(ps_model,x_train)}else{
    predictions_ps <- tryCatch({ predict(ps_model,x_train)}, error = function(cond){
      return(keras::predict_proba(ps_model,x_train))})
  }
  p_one = predictions_ps[,2]
  }else{
    p_one = propensity_score
}

  miu_hat_zero = (predictions*ytscale)+ytcenter

  p_one[p_one==0] = min(p_one[p_one!=0])
  p_one[p_one==1] = max(p_one[p_one!=1])

  p = sum(T)/length(T)
  psi_11 = Y * T/p
  psi_01 = p_one*(1 - T)*(Y - miu_hat_zero)/((1 - p_one)* p) + T * (miu_hat_zero - mean(miu_hat_zero))/p  #
  att = mean(psi_11 - psi_01)
  sigma_att_hat = mean((psi_11 - psi_01)**2)- att**2
  att = att - mean(miu_hat_zero)
  lower_95_bound_att = att - stats::qnorm(.975) * sqrt(sigma_att_hat/ length(Y))
  upper_95_bound_att = att + stats::qnorm(.975) * sqrt(sigma_att_hat/ length(Y))

  res = list( 'EIF_ATT'=att, 'lower_bound_att'=lower_95_bound_att, 'upper_bound_att'=upper_95_bound_att, 'Standard_error'= sqrt(sigma_att_hat/ length(Y)) )

  if(debugging){
    res=c(res, 'p_hat'=as.data.frame(p_one))
    res=c(res, 'm0_hat'=as.data.frame(miu_hat_zero))
  }
  return(res)
}
