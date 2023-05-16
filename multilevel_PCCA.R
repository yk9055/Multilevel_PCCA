multilevel_pCCA <- function(data, # list of data sets
                            idVar, 
                            kb = 1, kw = 1, 
                            G=2,
                            lambda_b, lambda_w, # lambdas = matrix of size G * kb , G*kw respectively
                            weights_b, weights_w,
                            method = 0,
                            coord = 1, 
                            init = "svd",
                            gamma = 1, # for EBIC
                            epsilon = 1e-3,
                            maxit = 1500,
                            control = list(minimize = TRUE, 
                                           tol = 1e-7,
                                           intermed = TRUE,
                                           trace = FALSE)){
  
  n_features <- sapply(data, ncol) 
  X <- do.call(cbind, data) 
  uid <- unique(idVar)
  N <- length(uid) # The number of subjects (clusters)
  n <- nrow(X) # The number of observations
  p <- ncol(X)
  
  # -- Initialize
  # Random
  if(init == "random"){
    Wb <- matrix(1, nrow = p, ncol = kb)
    Ww <- matrix(1, nrow = p, ncol = kw)
    PSI_b <- diag(1, p)
    PSI_w <- diag(1, p)
  }else{
    svd.X <- svd(X, nu=0)
    Wb <- svd.X$v[,1:kb]
    Ww <- svd.X$v[,1:kw]
    d <- svd.X$d^2/(n-1)
    
    if(p < n){
      PSI_b <- diag( sum(d[(kb+1):p])/(p-kb), p)
      PSI_w <- diag( sum(d[(kw+1):p])/(p-kw), p)
    }else{
      PSI_b <- diag( sum(d[(kb+1):n])/(n-kb), p)
      PSI_w <- diag( sum(d[(kw+1):n])/(n-kw), p)
    }
  }
  
  sigma_w <- Ww%*%t(Ww) + PSI_w
  inv_sigma_w <- solve(sigma_w)
  
  sigma_b <- Wb%*%t(Wb) + PSI_b
  
  
  params <- c(c(Wb), c(Ww), diag(PSI_b), diag(PSI_w))
  computation.time <- Sys.time()
  
  if(method != 1){
    
    control.default <- list(K=1, square=TRUE, minimize=TRUE, method=3, step.min0=1, step.max0=1, mstep=4, kr=1, objfn.inc=1,
                            tol=1.e-07, maxiter=1500, trace=FALSE, intermed=FALSE)
    
    namc <- names(control)
    if (!all(namc %in% names(control.default)))
      stop("unknown names in control: ", namc[!(namc %in% names(control.default))])
    ctrl <- modifyList(control.default, control)
    
    sq_EM <- squarem(par = params,
                     fixptfn = mpCCA_EM_cpp,
                     objfn = mpCCA_ll_sqEM_cpp,
                     X = X, 
                     idVar = idVar, 
                     uid = uid, 
                     N = N, 
                     n = n,
                     kb = kb, kw = kw,
                     G = G, 
                     n_features = n_features,
                     lambda_b = matrix(0, nrow = G, ncol = kb), 
                     lambda_w = matrix(0, nrow = G, ncol = kw),
                     weights_b = matrix(1, nrow = p, ncol = kb), 
                     weights_w = matrix(1, nrow = p, ncol = kw),
                     method = 0, coord = 0,
                     control = ctrl)
    
    sq_EM$computation.time <- Sys.time() - computation.time
    
    sq_EM$Wb <- matrix(sq_EM$par[1:(p*kb)], nrow = p)
    sq_EM$Ww <- matrix(sq_EM$par[(p*kb+1):(p*(kb+kw))], nrow = p)
    sq_EM$PSI.b <- diag(sq_EM$par[(p*(kb+kw)+1):((p*(kb+kw+1)))])
    sq_EM$PSI.w <- diag(sq_EM$par[((p*(kb+kw+1)+1)):length(sq_EM$par)])
    
    #sq_EM$lambda_b <- lambda_b
    #sq_EM$lambda_w <- lambda_w
    
    n_nonzero <- sum(abs(sq_EM$Wb) >= 1e-5) + sum(abs(sq_EM$Ww) >= 1e-5) 
    sq_EM$EBIC <- -2 * -sq_EM$value.objfn + (n_nonzero + 2 * p) * log(N) + 2 * gamma * log( choose(p * (kb + kw), n_nonzero) )
    
    return(sq_EM)
    
  }else{
    
    out <- sparse_mpCCA_EM_cpp(params = params, 
                               X = X,
                               idVar = idVar,
                               uid = uid,
                               N = N,
                               n = n,
                               kb = kb,
                               kw = kw,
                               epsilon = epsilon,
                               G = G,
                               n_features = n_features,
                               lambda_b = lambda_b,
                               lambda_w = lambda_w,
                               weights_b = weights_b,
                               weights_w = weights_w,
                               maxit = maxit,
                               method = 1,
                               coord = coord)
    
    
    computation.time <- Sys.time() - computation.time
    
    out$conv <- ifelse(out$it < maxit, 0, 1)
    n_nonzero <- sum(out$Wb != 0) + sum(out$Ww != 0) 
    out$EBIC <- -2 * out$pll + (n_nonzero + 2 * p) * log(N) + 2 * gamma * log( choose(p * (kb + kw), n_nonzero) )
    
    out$computation.time <- computation.time
    return(out)
  }
}
