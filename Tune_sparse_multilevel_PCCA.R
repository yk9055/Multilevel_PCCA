tune_sparse_multilevelPCCA <- function(data,
                                       idVar,
                                       G = 2, # the number of data sets
                                       kb = 1,
                                       kw = 1,
                                       maxit = 1500,
                                       epsilon = 1e-3,
                                       init = "svd",
                                       coord = 1,
                                       weighted = TRUE,
                                       phi = 1, # power of adaptive weights
                                       lambda_range = NULL, # a list of length G; each element is a 2*(kb + kw) matrix where first row is min and second row is max 
                                       tune_length = 100,
                                       p_frac = 1, 
                                       l_min_r = 0.01,
                                       gamma = 1,
                                       mc.silent = TRUE, 
                                       mc.cores = 2, 
                                       mc.preschedule = FALSE,
                                       ...){
  
  if(!("parallel" %in% loadedNamespaces())){ library(parallel) }
  
  uid <- unique(idVar) 
  n_features <- sapply(data, ncol) 
  p <- sum(n_features) 
  X <- do.call(cbind, data) 
  
  computation.time <- Sys.time() 
  if(is.null(lambda_range)){
    # Fit a model to full data with all lambdas = 0 (i.e., MLE)
    fit.full.0 <- multilevel_pCCA(data,
                                  idVar = idVar,
                                  kb = kb,
                                  kw = kw,
                                  G = G,
                                  init = init,
                                  lambda_b = matrix(0, nrow = G, ncol = kb),
                                  lambda_w = matrix(0, nrow = G, ncol = kw),
                                  weights_b = matrix(1, nrow = p, ncol = kb),
                                  weights_w = matrix(1, nrow = p, ncol = kw),
                                  method = 0,
                                  coord = 0,
                                  gamma = gamma)
  }
  
  if(weighted){
    weights_b <- abs(1 / fit.full.0$Wb)^phi
    weights_w <- abs(1 / fit.full.0$Ww)^phi
  }
  
  # For each lambda, find max of relevant weights
  # And Set ranges
  lambda_range <- lapply(1:G, 
                         function(g){
                           weights <- abs(cbind(fit.full.0$Wb, fit.full.0$Ww)) # p * (kb+kw) where p = p1 + p2
                           if(g == 1){
                             l.max <- apply(weights[1:n_features[g], ], 2, function(x){quantile(x, p_frac)})
                           }else{
                             l.max <- apply(weights[(sum(n_features[1:(g-1)])+1):sum(n_features[1:g]), ], 2, function(x){quantile(x, p_frac)})}
                           
                           rbind(l.max * l_min_r, l.max)
                         })
  
  #print(lambda_range)
  
  ## -- random search of lambdas: each candidate is a matrix of size G*(kb+kw)
  # Draw each lambda from uniform distribution with the corresponding range
  # lambdas is a list of length = tune_length and each element is a matrix of size G*(kb+kw)
  #set.seed(1)
  lambdas <- lapply(1:tune_length, function(x){
    t(sapply(lambda_range, function(bounds){
      apply(bounds, 2, function(x){exp(sample(runif(10, min = log(x[1]), max = log(x[2])), 1))})
    }) )
  })
  
  
  out <- mclapply(lambdas, 
                  function(lambdas.i){
                    fit.i <- try(multilevel_pCCA(data, 
                                                 G = G,
                                                 idVar = idVar,
                                                 kb = kb,
                                                 kw = kw,
                                                 maxit = maxit,
                                                 epsilon = epsilon,
                                                 init = init,
                                                 lambda_b = lambdas.i[, 1:kb, drop = FALSE],
                                                 lambda_w = lambdas.i[,-c(1:kb), drop = FALSE],
                                                 coord=coord,
                                                 method = 1,
                                                 weights_b = weights_b,
                                                 weights_w = weights_w,
                                                 gamma = gamma), 
                                 TRUE)
                    return(list(EBIC = fit.i$EBIC,
                                it.each.tune = fit.i$it, 
                                computation.time.each.tune = as.numeric(fit.i$computation.time, units = "mins"),
                                conv.each.tune = fit.i$conv))
                  },
                  mc.silent = mc.silent, mc.cores = mc.cores, mc.preschedule = mc.preschedule)
  
  out1 <- unlist(lapply(out, function(x){return(x$EBIC)})) # a vector of EBIC; length = tune_length
  print(out1)
  best.lambdas <- lambdas[[which.min(out1)]] 
  
  
  
  best.fit <- multilevel_pCCA(data, 
                              G = G, 
                              idVar = idVar,
                              kb = kb, kw = kw,
                              maxit = maxit, epsilon = epsilon,
                              init = init,
                              lambda_b = best.lambdas[,1:kb, drop = FALSE],
                              lambda_w = best.lambdas[,-c(1:kb), drop = FALSE],
                              coord = coord,
                              method = 1,
                              weights_b = weights_b,
                              weights_w = weights_w,
                              gamma = gamma) 
  
  
  best.fit$which.non.zero <- list( Wb = which( best.fit$Wb != 0 ),
                                   Ww = which( best.fit$Ww != 0 ) )
  best.fit$non.zero <- sapply(best.fit$which.non.zero, length)
  
  computation.time <- Sys.time() - computation.time
  
  return(list(EBIC = out1,
              computation.time.each.tune = unlist(lapply(out, function(x){return(x$computation.time.each.tune)})),
              it.each.tune = unlist(lapply(out, function(x){return(x$it.each.tune)})),
              conv.each.tune = unlist(lapply(out, function(x){return(x$conv.each.tune)})),
              lambdas = lambdas,
              best.lambdas = best.lambdas,
              best.fit = best.fit,
              computation.time = computation.time,
              fit.mle = fit.full.0)
  )
}

