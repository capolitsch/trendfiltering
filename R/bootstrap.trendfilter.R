#' Bootstrap the optimized trend filtering estimator to obtain variability bands
#'
#' @description \loadmathjax \code{bootstrap.trendfilter} implements a
#' parametric bootstrap algorithm to obtain one or both of the following: 
#' \enumerate{
#' \item{Standard errors of the optimized trend filtering point estimator}
#' \item{Percentile-based `1-alpha` variability bands of the optimized trend
#' filtering point estimator.}}
#' @param obj An object of class '\link{SURE.trendfilter}'.
#' @param alpha Specifies the width of the `1-alpha` pointwise variability 
#' bands. Defaults to `alpha = 0.05`.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100`. Increase this for more precise
#' bands (e.g. for the final analysis you intend to publish).
#' @param return.full.ensemble Logical. If `TRUE`, the full trend filtering 
#' bootstrap ensemble is returned as an \mjeqn{n \times B}{ascii} matrix, less 
#' any columns potentially pruned post-hoc (see `prune` below). Defaults to 
#' `return.full.ensemble = FALSE`.
#' @param prune Logical. If `TRUE`, then the trend filtering bootstrap 
#' ensemble is examined for rare instances in which the optimization has 
#' stopped at zero knots (most likely erroneously), and removes them from the 
#' ensemble. Defaults to `TRUE`. Do not change this unless you know what you are
#' doing!
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected.
#' @return An object of class 'bootstrap.trendfilter'. This is a comprehensive
#' list containing all of the analysis' important information, data, and
#' results:
#' \item{x.eval}{(Inherited from `obj`) The grid of inputs the trend 
#' filtering estimate and variability bands were evaluated on.}
#' \item{tf.estimate}{The trend filtering estimate of the signal, evaluated on 
#' \code{x.eval}.}
#' \item{tf.standard.errors}{The standard errors of the optimized trend 
#' filtering point estimator.}
#' \item{bootstrap.lower.band}{Vector of lower bounds for the 
#' \code{1-alpha} pointwise variability band, evaluated on `x.eval`.}
#' \item{bootstrap.upper.band}{Vector of upper bounds for the 
#' \code{1-alpha} pointwise variability band, evaluated on `x.eval`.}
#' \item{bootstrap.algorithm}{The string specifying the bootstrap algorithms 
#' that was used. Here, always "parametric".}
#' \item{alpha}{The 'level' of the variability bands, i.e. `alpha`
#' produces a `100*(1-alpha)`\% pointwise variability band.}
#' \item{B}{The number of bootstrap samples used to estimate the pointwise
#' variability bands.}
#' \item{tf.bootstrap.ensemble}{(Optional) If `return.full.ensemble = TRUE`, the 
#' full trend filtering bootstrap ensemble as an \mjeqn{n \times B}{ascii} 
#' matrix, less any columns potentially pruned post-hoc (if `prune = TRUE`). 
#' If `return.full.ensemble = FALSE`, then this will return `NULL`.}
#' \item{edf.boots}{An integer vector of the estimated number of effective 
#' degrees of freedom of each trend filtering bootstrap estimate. These should
#' all be relatively close to `edf.min` (below).}
#' \item{prune}{Logical. If `TRUE`, then the trend filtering bootstrap 
#' ensemble is examined for rare instances in which the optimization has 
#' stopped at zero knots (most likely erroneously), and removes them from the 
#' ensemble.}
#' \item{n.pruned}{The number of badly-converged bootstrap trend filtering 
#' estimates pruned from the ensemble.}
#' \item{x}{(Inherited from `obj`) The vector of the observed inputs.}
#' \item{y}{(Inherited from `obj`) The vector of the observed outputs.}
#' \item{weights}{(Inherited from `obj`) A vector of weights for the observed
#' outputs. These are defined as `weights = 1 / sigma^2`, where `sigma` is a
#' vector of standard errors of the uncertainty in the output measurements.}
#' \item{residuals}{(Inherited from `obj`) `residuals = y - fitted.values`}
#' \item{k}{(Inherited from `obj`) The degree of the trend filtering estimator.}
#' \item{gammas}{(Inherited from `obj`) Vector of hyperparameter values tested
#' during validation.}
#' \item{gammas.min}{(Inherited from `obj`) Hyperparameter value that minimizes
#' the validation error curve.}
#' \item{edf}{(Inherited from `obj`}) Integer vector of effective degrees of
#' freedom for trend filtering estimators fit during validation.}
#' \item{edf.min}{(Inherited from `obj`) The effective degrees of freedom of the
#' optimally-tuned trend filtering estimator.}
#' \item{i.min}{(Inherited from `obj`) The index of `gammas` that minimizes the
#' validation error.}
#' \item{validation.method}{"SURE"}
#' \item{error}{(Inherited from `obj`) Vector of hyperparameter validation
#' errors, inherited from `obj` (an object of class 'SURE.trendfilter').}
#' \item{optimization.params}{(Inherited from `obj`) a list of parameters that
#' control the trend filtering convex optimization.}
#' \item{n.iter}{(Inherited from `obj`) Vector of the number of iterations 
#' needed for the ADMM algorithm to converge within the given tolerance, for 
#' each hyperparameter value. If many of these are exactly equal to `max_iter`,
#' then their solutions have not converged with the tolerance specified by
#' `obj_tol`. In which case, it is often prudent to increase `max_iter`.}
#' \item{n.iter.boots}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each bootstrap trend
#' filtering estimate.}
#' \item{x.scale, y.scale, data.scaled}{For internal use.}
#' @details See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Politsch et al. (2020a)} for the full parametric bootstrap algorithm. 
#' 
#' @export bootstrap.trendfilter
#' @author Collin A. Politsch, Ph.D., \email{collinpolitsch@@gmail.com}
#' @seealso \code{\link{SURE.trendfilter}}
#' 
#' @references 
#' \enumerate{
#' \item{Politsch et al. (2020a). Trend filtering – I. A modern 
#' statistical tool for time-domain astronomy and astronomical spectroscopy. 
#' \emph{Monthly Notices of the Royal Astronomical Society}, 492(3), 
#' p. 4005-4018.
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{[Link]}} \cr
#' \item{Politsch et al. (2020b). Trend Filtering – II. Denoising 
#' astronomical signals with varying degrees of smoothness. \emph{Monthly 
#' Notices of the Royal Astronomical Society}, 492(3), p. 4019-4032.
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{[Link]}}}

#' @importFrom glmgen trendfilter
#' @importFrom dplyr %>% mutate case_when n
#' @importFrom tidyr tibble
#' @importFrom parallel mclapply detectCores
#' @importFrom stats quantile rnorm
bootstrap.trendfilter <- function(obj,
                                  alpha = 0.05, 
                                  B = 100L, 
                                  bootstrap.algorithm = c("nonparametric","parametric","wild"),
                                  return.full.ensemble = FALSE,
                                  prune = TRUE,
                                  mc.cores = detectCores()){
  
  stopifnot( class(obj) %in% c("SURE.trendfilter","cv.trendfilter") )
  bootstrap.algorithm <- match.arg(bootstrap.algorithm)
  stopifnot( is.numeric(alpha) & alpha > 0 & alpha < 1 )
  stopifnot( B >= 10 )
  
  if ( !prune ) warning("I hope you know what you are doing!")
  if ( mc.cores < detectCores() ){
    warning(paste0("Your machine only has ", detectCores(), " cores. Consider increasing `mc.cores` for computational speedups."))
  }
  if ( mc.cores > detectCores() ){
    warning(paste0("Your machine only has ", detectCores(), " cores. Adjusting mc.cores accordingly."))
    mc.cores <- detectCores()
  }
  
  sampler <- case_when(
    bootstrap.algorithm == "nonparametric" ~ list(nonparametric.resampler),
    bootstrap.algorithm == "parametric" ~ list(parametric.sampler),
    bootstrap.algorithm == "wild" ~ list(wild.sampler)
  )[[1]]
  
  obj$prune <- prune
  data.scaled <- obj$data.scaled
  
  par.out <- mclapply(1:B, bootstrap.estimator, mc.cores = mc.cores)
  tf.boot.ensemble <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["tf.estimate"]]) %>%
    unlist %>% 
    matrix(nrow = length(obj$x.eval)) 
  
  obj$edf.boots <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["edf"]]) %>%
    unlist %>%
    as.integer
  obj$n.iter.boots <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["n.iter"]]) %>%
    unlist %>%
    as.integer
  
  obj$n.pruned <- (B - ncol(tf.boot.ensemble)) %>% as.integer
  obj$tf.standard.errors <- apply(tf.boot.ensemble, 1, sd) 
  obj$bootstrap.lower.band <- apply(tf.boot.ensemble, 1, quantile, probs = alpha / 2) 
  obj$bootstrap.upper.band <- apply(tf.boot.ensemble, 1, quantile, probs = 1 - alpha / 2)
  
  obj <- c(obj, list(bootstrap.algorithm = bootstrap.algorithm, alpha = alpha, B = B))
  
  if ( return.full.ensemble ){
    obj$tf.bootstrap.ensemble <- tf.boot.ensemble
  }else{
    obj <- c(obj, list(tf.bootstrap.ensemble = NULL))
  }
  
  obj <- obj[c("x.eval","tf.estimate","tf.standard.errors","bootstrap.lower.band",
               "bootstrap.upper.band","bootstrap.algorithm","alpha","B",
               "edf.boots","tf.bootstrap.ensemble","prune","n.pruned","x","y",
               "weights","fitted.values","residuals","k","gammas","gamma.min",
               "edfs","edf.min","i.min","validation.method","errors",
               "optimization.params","n.iter","n.iter.boots","x.scale","y.scale",
               "data.scaled")]
  class(obj) <- "bootstrap.trendfilter"
  return(obj)
}


bootstrap.estimator <- function(b){
  tf.estimator(data = sampler(data.scaled), obj = obj, mode = "edf")
}


tf.estimator <- function(data, obj, mode = "gamma"){
  
  if ( mode == "edf" ){
    tf.fit <- trendfilter(x = data$x,
                          y = data$y,
                          weights = data$weights,
                          k = obj$k,
                          lambda = obj$gammas,
                          thinning = obj$thinning,
                          control = obj$optimization.params)
    
    i.min <- which.min( abs(tf.fit$df - obj$edf.min) )
    gamma.min <- obj$gammas[i.min]
    edf.min <- tf.fit$df[i.min]
    n.iter <- tf.fit$iter[i.min]
    
    if ( obj$prune && edf.min <= 2 ){
      return(list(tf.estimate = integer(0), df = NA, n.iter = NA))
    }
  }
  
  if ( mode == "gamma" ){
    tf.fit <- trendfilter(x = data$x,
                          y = data$y,
                          weights = data$weights,
                          k = obj$k,
                          lambda = obj$gamma.min,
                          thinning = obj$thinning,
                          control = obj$optimization.params)
    
    gamma.min <- obj$gamma.min
    edf.min <- tf.fit$df
    n.iter <- as.integer(tf.fit$iter)
  }
  
  tf.estimate <- glmgen:::predict.trendfilter(object = tf.fit, 
                                              x.new = obj$x.eval / obj$x.scale, 
                                              lambda = gamma.min) %>% 
    as.numeric
  
  return(list(tf.estimate = tf.estimate * obj$y.scale, edf = edf.min, n.iter = n.iter))
}


########################################################


#' Resampling functions for various bootstrap algorithms
#' 
#' @param data Data frame / tibble with minimal column set: `x`, `y`, `weights`
#' (for `parametric.sampler`), `fitted.values` (for `parametric.sampler`), and
#' `residuals` (for `wild.sampler`)
#' 
#' @return Bootstrap sample returned in the same format as the input data frame
#' / tibble.


#' @importFrom dplyr %>% mutate n
#' @rdname samplers
#' @export
parametric.sampler <- function(data){
  data %>% mutate(y = fitted.values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @importFrom dplyr %>% slice_sample n
#' @rdname samplers
#' @export
nonparametric.resampler <- function(data){
  data %>% slice_sample(n = n(), replace = TRUE)
}


#' @importFrom dplyr %>% mutate n
#' @rdname samplers
#' @export
wild.sampler <- function(data){
  data %>% mutate(y = fitted.values + residuals *
                    sample(x = c(
                      (1 + sqrt(5)) / 2, 
                      (1 - sqrt(5)) / 2
                    ), 
                    size = n(), replace = TRUE,
                    prob = c(
                      (1 + sqrt(5)) / (2 * sqrt(5)),
                      (sqrt(5) - 1) / (2 * sqrt(5))
                    )
                    )
  )
}
