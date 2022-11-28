#' Deep Neural Network AIPW Estimator
#'
#'
#' @param	Y is a numerical vector of observed outcomes of length \code{n}.
#' @param T is a logical vector of treatment statuses of length \code{n}.
#' @param X_t is the set of covariates. If covariates are time series, it should be a list of \code{k} different \code{n * p} matrixes. Here \code{p} is the length of the time series, and \code{k} is the number of different covariates. If covariates are single valued, it should be a matrix of size \code{n * k}.
#' @param X if there are some single value covariates besides the time series, it can be considered here. It has to be \code{k’ * n} matrix. The default value is \code{Null}.
#' @param rescale_outcome determine if scaling the outcome values to have zero mean and one standard deviation. Default is \code{True}.
#' @param model It can be defined by a vector of size 3 (for ate function) or size 2 (for att function), or it can be just one model. In the latter case, that one model will be used for all nuisance models. The default value for Model is \code{Null}, and in this case, the functions will use a specific predefined network.  Each model should be defined using the package Keras.
#' @param optimizer can be a vector contain two optimizers or can be just one. In the presence of two optimizers, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{Null}, and in this case, adam optimizer is used.
#' @param loss can be a vector contain two loss functions or can be just one. In the presence of two functions, the first one will be used for the outcome models, and the second one will be used for the propensity score estimation. The default is \code{Null}, and in this case, mean square error and cross-entropy error are used for the outcome models and propensity score, respectively.
#' @param epochs default value is 256. A vector is also acceptable.
#' @param batch_size default value is 200. A vector is also acceptable.
#' @param propensity_score can be used to determine a pre-estimated propensity score. By using it, the functions will not make the neural network for estimating the propensity score, and the predefined values will be used in the final estimation.
#' @param debugging is a logical variable, and if it is \code{True}, the function will return the estimated vectors for the outcome models and the propensity score.
#' @param verbose can be a vector of logical variables or just one. That will control the verbosity of the fitting process.
#' @param compile_outcome_model a list of parameters and their values can be used to add to Keras compile function when it is used for the outcome models. the list can be used to overwrite existing parameters, so it would be considered as a way to run Keras in a completely arbitrary way.
#' @param compile_propensity_score a list of parameters and their values can be used to add to Keras compile function when it is used for the propensity score. the list can be used to overwrite existing parameters, so it would be considered as a way to run Keras in a completely arbitrary way.
#' @param fit_outcome_model a list of parameters and their values can be used to add to Keras fit function when it is used for the outcome models. the list can be used to overwrite existing parameters.
#' @param fit_propensity_score a list of parameters and their values can be used to add to Keras fit function when it is used for the propensity score. the list can be used to overwrite existing parameters.
#'
#'
#'
#' @return DNN AIPW Estimate and Inference of ATE
#' @export
#'
#' @import keras abind
#'
#' @examples
#'
#' library(DNNCausal)
#' # simulate covariates, treatment assignment mechanism, and potential outcomes.
#' x = matrix(rnorm(100 * 5), nrow = 100)
#' p = 1/ (1 + exp(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)))
#' T = rbinom(100,1,p)
#' m1 = 1 + tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
#' m0 = tan(0.1*( (x[,5] - x[,1])^2 + (x[,4] - x[,2])^2 - x[,3]^2)) + rnorm(100,sd = 0.1)
#' # obtain the observed outcome
#' y = T*m1 + (1-T)*m0
#' # call the ATE estimator function
#' aipw.dnn(y,T,x)
#'
#'
aipw.dnn = function(Y,T,X_t,X = NULL,rescale_outcome=TRUE,model=NULL,optimizer=NULL,loss= NULL,epochs=256,batch_size=200, compile_outcome_model=NULL,fit_outcome_model=NULL,compile_propensity_score=NULL,fit_propensity_score=NULL, verbose=TRUE, debugging= FALSE, propensity_score= NULL){

  state = data.frame()
  if (is.list(X_t)){
    if(is.vector(X_t)){
      if (length(unique(lapply(X_t,function(x){dim(x)})))==1){
        k = length(X_t)
        n = dim(X_t[[1]])[1]
        p = dim(X_t[[1]])[2]
        for (i in 1:k) {
          x_t = data.matrix(X_t[[i]])
          mean = rowMeans(x_t,na.rm = TRUE)
          sd = sqrt(rowMeans((x_t-mean)^2,na.rm = TRUE))
          sd[sd==0] = 1
          X_t[[i]]= (x_t - mean)/sd
          if (dim(state)[1]==0){
            state = data.frame(mean)
            state = cbind(state,sd)
          }else{
          state = cbind(state,mean,sd)
        }}
        X_t = abind::abind(X_t,along = 3)
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
    }else if (length(dim(X_t))==2){
      n = dim(X_t)[1]
      k = dim(X_t)[2]
      p = 1
    }else if (length(dim(X_t))==3){
      n = dim(X_t)[1]
      p = dim(X_t)[2]
      k = dim(X_t)[3]
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
      model_m = model[[1]]
      model_p = model[[1]]
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
  if(!is.null(X)) {
    if (ncol(X)!=0){
    px = dim(X)[2]
    model_mx =  keras::keras_model_sequential()
    model_mx =  keras::layer_dense(model_mx,30,activation = "relu",input_shape = px)
    model_mx =  keras::layer_dense(model_mx,20,activation = "relu")

    model_px =  keras::keras_model_sequential()
    model_px =  keras::layer_dense(model_px,30,activation = "relu",input_shape = px)
    model_px =  keras::layer_dense(model_px,20,activation = "relu")

    m_m =  keras::layer_dense(keras::layer_concatenate(list(model_m$output,model_mx$output)),2)
    m_p =  keras::layer_dense(keras::layer_concatenate(list(model_p$output,model_px$output)),2,activation = keras::activation_softmax)

    model_m = keras::keras_model(list(model_m$input,model_mx$input),m_m)
    model_p = keras::keras_model(list(model_p$input,model_px$input),m_p)

    }
}

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
    loss_m = causal.loss
    loss_p = keras::loss_categorical_crossentropy
  } else if(is.vector(loss)){
    if (length(loss)==1) {
      loss_m = causal.loss
      loss_p = loss[1]
    }
    else if (length(loss)==2) {
      loss_m = loss[1]
      loss_p = loss[2]
    }
    else if (length(loss)==3) {
      stop("Using three models is not supported yet.")
      loss_m0 = loss[1]
      loss_m1 = loss[2]
      loss_p = loss[3]
    }
  } else {
    loss_m = causal.loss
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
  if(!is.null(X)){
    if(ncol(X)!=0){outcome_model = model_m}else{
    outcome_model = keras::layer_dense(model_m,2)}
  }else{outcome_model = keras::layer_dense(model_m,2)}
  if (is.null(compile_outcome_model)) {
    keras::compile(outcome_model,
    loss = loss_m,
    optimizer = optimizer_m,
    metrics = c(metric_mean_absolute_percentage_error2)
    )} else{
      if(!'object'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('object'=outcome_model))
      } else{
        outcome_model = compile_outcome_model$object
      }
      if(!'loss'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('loss'=loss_m))
      }
      if(!'metrics'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('metrics'=metric_mean_absolute_percentage_error2))
      }
      if(!'optimizer'%in%names(compile_outcome_model)){
        compile_outcome_model = append(compile_outcome_model, list('optimizer'=optimizer_m))
      }
      do.call(keras::compile,compile_outcome_model)
    }

  if(length(p)==2){


    }

  #x_train <- keras::array_reshape(X, c(nrow(X), ncol(X)))
  x_train = X_t
  if (!is.null(X)){
   if(ncol(X)!=0) {
    X = scale(X)
    X[is.na(X)] = 0
    x_train =list( X_t, X)}
  }
  Yn=Y
  if(rescale_outcome){
  Yt = Y[T==1]
  Yu = Y[T==0]
  Yt = scale(Yt)
  Yu = scale(Yu)
  ytscale = attr(Yt,'scaled:scale')
  ytcenter = attr(Yt,'scaled:center')
  yuscale = attr(Yu,'scaled:scale')
  yucenter = attr(Yu,'scaled:center')
  Yn[T==1] = Yt
  Yn[T==0] = Yu
  }else{
    yuscale = 1
    ytscale = 1
    yucenter = 0
    ytcenter = 0
  }
  y_train <- keras::array_reshape(cbind(Yn,T), c(length(Yn),2))

  if (is.null(fit_outcome_model)) {
    keras::fit(outcome_model,
               x_train, y_train,
               batch_size = batch_size_m,
               epochs = epochs_m,verbose=verbose_m)
  } else{
    if(!'object'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('object'=outcome_model))
    } else{
      outcome_model = fit_outcome_model$object
    }
    if(!'x'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('x'=x_train))
    }
    if(!'y'%in%names(fit_outcome_model)){
      fit_outcome_model = append(fit_outcome_model, list('y'=y_train))
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
  if(!is.null(X)){
   if(ncol(X)!=0){ps_model = model_p}else{
    ps_model = keras::layer_dense(model_p,2,activation = keras::activation_softmax)
   }}
    else{
      ps_model = keras::layer_dense(model_p,2,activation = keras::activation_softmax)
  }
  if (is.null(compile_propensity_score)) {
    keras::compile(ps_model,
                   loss = loss_p,
                   optimizer = optimizer_p,
                   metrics = c(keras::metric_binary_accuracy, keras::metric_binary_crossentropy)
    )} else{
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
               epochs = epochs_p,verbose=verbose_p
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

  if(!is.null(X)){
    if(ncol(X)!=0){
      predictions_ps <- keras::predict_on_batch(ps_model,x_train)
      }else{

    predictions_ps <- tryCatch({ keras::predict(ps_model,x_train)}, error = function(cond){return(keras::predict_proba(ps_model,x_train))})
      }
  }else{
    predictions_ps <- tryCatch({ keras::predict(ps_model,x_train)}, error = function(cond){return(keras::predict_proba(ps_model,x_train))})
  }


  p_one = predictions_ps[,2]
  }else{
    p_one = propensity_score
}



  #tau_hat = predictions[,2]
  miu_hat_zero = (predictions[,1]*yuscale)+yucenter
  miu_hat_one = ((predictions[,1] + predictions[,2])*ytscale)+ytcenter
  tau_hat =  miu_hat_one - miu_hat_zero

  p_one[p_one==0] = min(p_one[p_one!=0])
  p_one[p_one==1] = max(p_one[p_one!=1])
  phi_hat_one = (T* (Y - miu_hat_one) / p_one )+ miu_hat_one
  phi_hat_zero = ((1 - T)*(Y - miu_hat_zero)/ (1 - p_one) )+ miu_hat_zero
  tau_hat_sec = mean(phi_hat_one - phi_hat_zero)

  sigma_hat = mean((phi_hat_one - phi_hat_zero)**2)- tau_hat_sec**2
  lower_95_bound = tau_hat_sec - stats::qnorm(.975) * sqrt(sigma_hat/ length(Y))
  higher_95_bound = tau_hat_sec + stats::qnorm(.975) * sqrt(sigma_hat/ length(Y))

  p = sum(T)/length(T)
  psi_11 = Y * T/p
  psi_01 = p_one*(1 - T)*(Y - miu_hat_zero)/((1 - p_one)* p) + T * miu_hat_zero/p
  att = mean(psi_11 - psi_01)
  sigma_att_hat = mean((psi_11 - psi_01)**2)- att**2
  lower_95_bound_att = att - stats::qnorm(.975) * sqrt(sigma_att_hat/ length(Y))
  higher_95_bound_att = att + stats::qnorm(.975) * sqrt(sigma_att_hat/ length(Y))

  res = list('tau_hat'=tau_hat_sec,'lower_bound'=lower_95_bound,'higher_bound'=higher_95_bound, 'primitive_tau_hat'=mean(tau_hat), 'ATT'=att, 'lower_bound_att'=lower_95_bound_att, 'higher_bound_att'=higher_95_bound_att)
  if(debugging){
    res=c(res, 'p_hat'=as.data.frame(p_one))
    res=c(res, 'm0_hat'=as.data.frame(miu_hat_zero))
    res=c(res, 'm1_hat'=as.data.frame(miu_hat_one))
  }

  return(res)
}
