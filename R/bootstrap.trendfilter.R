#' Obtain pointwise uncertainty bands by bootstrapping the optimized trend
#' filtering estimator.
#'
#' \loadmathjax `bootstrap.trendfilter` implements
#' 
#' @param obj An object of class '\link{SURE.trendfilter}' or
#' '\link{cv.trendfilter}'.
#' @param level The level of the pointwise variability bands. Defaults to
#' `level = 0.95`.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100`.
#' @param bootstrap.algorithm A string specifying which variation of the 
#' bootstrap to use. One of `c("nonparametric","parametric","wild")`. See
#' details below for recommendations on when each option is appropriate.
#' @param return.full.ensemble Logical. If `TRUE`, the full trend filtering 
#' bootstrap ensemble is returned as an \mjseqn{n \times B} matrix, less 
#' any columns from post-hoc pruning (see `prune` below). Defaults to 
#' `return.full.ensemble = FALSE`.
#' @param prune Logical. If `TRUE`, then the trend filtering bootstrap 
#' ensemble is examined for rare instances in which the optimization has 
#' stopped at zero knots (likely erroneously), and removes them from the 
#' ensemble that is used to compute the variability bands. Defaults to `TRUE`.
#' Do not change this unless you know what you are doing!
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected.
#' 
#' @return An object of class 'bootstrap.trendfilter'. This is a comprehensive
#' list containing all of the analysis important information, data, and
#' results:
#' \item{x.eval}{(Inherited from `obj`) The grid of inputs the optimized trend
#' filtering estimate was evaluated on.}
#' \item{tf.estimate}{The trend filtering estimate of the signal, evaluated on 
#' `x.eval`.}
#' \item{tf.standard.errors}{The standard errors of the optimized trend 
#' filtering point estimator.}
#' \item{bootstrap.lower.band}{Vector of lower bounds for the 
#' pointwise variability bands, evaluated on `x.eval`.}
#' \item{bootstrap.upper.band}{Vector of upper bounds for the 
#' pointwise variability bands, evaluated on `x.eval`.}
#' \item{bootstrap.algorithm}{A string specifying which variation of the 
#' bootstrap was used to obtain the variability bands.}
#' \item{level}{The level of the pointwise variability bands.}
#' \item{B}{The number of bootstrap samples used to estimate the pointwise
#' variability bands.}
#' \item{tf.bootstrap.ensemble}{If `return.full.ensemble = TRUE`, the 
#' full trend filtering bootstrap ensemble as an \mjseqn{n \times B}
#' matrix, less any columns from post-hoc pruning (if `prune = TRUE`). 
#' If `return.full.ensemble = FALSE`, then this will return `NULL`.}
#' \item{edf.boots}{An integer vector of the estimated number of effective 
#' degrees of freedom of each trend filtering bootstrap estimate. These should
#' all be relatively close to `edf.min` (below).}
#' \item{prune}{Logical. If `TRUE`, then the trend filtering bootstrap 
#' ensemble is examined for rare instances in which the optimization has 
#' stopped at zero knots (likely erroneously), and removes them from the 
#' ensemble.}
#' \item{n.pruned}{The number of poorly-converged bootstrap trend filtering 
#' estimates pruned from the ensemble.}
#' \item{x}{(Inherited from `obj`) Vector of observed inputs.}
#' \item{y}{(Inherited from `obj`) Vector of observed outputs.}
#' \item{weights}{(Inherited from `obj`) A vector of weights for the observed
#' outputs. These are defined as `weights = 1 / sigma^2`, where `sigma` is a
#' vector of standard errors of the uncertainty in the output measurements.}
#' \item{residuals}{(Inherited from `obj`) `residuals = y - fitted.values`}
#' \item{k}{(Inherited from `obj`) The degree of the trend filtering estimator.}
#' \item{gammas}{(Inherited from `obj`) Vector of hyperparameter values tested
#' during validation (always returned in descending order).}
#' \item{gamma.min}{(Inherited from `obj`) Hyperparameter value that minimizes
#' the validation error curve.}
#' \item{edf}{(Inherited from `obj`) Integer vector of effective degrees of
#' freedom for trend filtering estimators fit during validation.}
#' \item{edf.min}{(Inherited from `obj`) The effective degrees of freedom of the
#' optimally-tuned trend filtering estimator.}
#' \item{i.min}{(Inherited from `obj`) The index of `gammas` that minimizes the
#' validation error.}
#' \item{validation.method}{Either `"SURE"` or `paste0(V,"-fold CV")`.}
#' \item{errors}{(Inherited from `obj`) Vector of hyperparameter validation
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
#' 
#' @details \loadmathjax See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Politsch et al. (2020a)} for more details.
#' 
#' \itemize{
#' \item{The inputs are irregularly sampled}
#' \item{The inputs are regularly sampled and the noise distribution is known}
#' \item{The inputs are regularly sampled and the noise distribution is unknown}
#' }
#' 
#' Parametric bootstrap for fixed-input uncertainty quantification (when noise
#' distribution \mjseqn{\epsilon_i\sim Q_i} is known a priori)
#' Require: Training Data \mjseqn{(x_1, y_1),\dots,(x_n, y_n)}, hyperparameters
#' \mjseqn{\gamma} and \mjseqn{k}, assumed noise distribution
#' \mjseqn{\epsilon_i \sim Q_i}, prediction input grid \mjseqn{x_1',\dots,x_m'}
#' Compute the trend filtering point estimate at the observed inputs:
#' \mjseqn{(x_1, y_1),\dots, (x_n, y_n)}
#' For \mjseqn{b in 1:B}
#' Define a bootstrap sample by sampling from the assumed noise distribution:
#' \mjsdeqn{y_i = \widehat{f}(x_i) + \epsilon_i \quad \quad \text{where }\epsilon_i \sim Q_i, \quad i=1,\dots,n}
#' Let \mjseqn{\widehat{f}(x_1'), \dots, \widehat{f}(x_m')} denote the trend
#' filtering estimate fit on the bootstrap sample and evaluated on the
#' prediction grid \mjseqn{x_1',\dots,x_m'}
#' 
#' Given the full trend filtering bootstrap ensemble provided by the relevant
#' bootstrap algorithm, for any \mjseqn{\alpha\in(0,1)}, a
#' \mjseqn{(1-\alpha)\cdot100}% pointwise variability band is
#' given by 
#' \mjsdeqn{B_{1-\alpha}(x_i') = \left(\widehat{f}_{\alpha/2}(x_i'),\;\widehat{f}_{1-\alpha/2}(x_i')\right), \quad\quad i = 1,\dots,m}
#' where
#' \mjsdeqn{\widehat{f}_{\beta}(x_i') = \inf_{g}\left\{g : \frac{1}{B} \sum_{b=1}^{B} \mathbbm{1}\big\{\widehat{f}_{b}(x_i') \leq g\big\} \geq \beta \right\}, \quad\quad \beta\in(0,1).}
#' 
#' @export bootstrap.trendfilter
#' 
#' @author
#' \bold{Collin A. Politsch, Ph.D.}
#' ---
#' Email: collinpolitsch@@gmail.com \cr
#' Website: [collinpolitsch.com](https://collinpolitsch.com/) \cr
#' GitHub: [github.com/capolitsch](https://github.com/capolitsch/) \cr \cr
#' 
#' @references
#' \bold{Companion references} 
#' \enumerate{
#' \item{Politsch et al. (2020a). 
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Trend filtering – I. A modern statistical tool for time-domain astronomy and
#' astronomical spectroscopy}. \emph{MNRAS}, 492(3), p. 4005-4018.} \cr
#' \item{Politsch et al. (2020b). 
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' Trend Filtering – II. Denoising astronomical signals with varying degrees of
#' smoothness}. \emph{MNRAS}, 492(3), p. 4019-4032.}}
#' 
#' \bold{The Bootstrap and variations}
#' \enumerate{
#' \item{Efron and Tibshirani (1986). 
#' \href{https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full}{
#' Bootstrap Methods for Standard Errors, Confidence Intervals, and Other
#' Measures of Statistical Accuracy}.
#' \emph{Statistical Science}, 1(1), p. 54-75.} \cr
#' \item{Mammen (1993). 
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-21/issue-1/Bootstrap-and-Wild-Bootstrap-for-High-Dimensional-Linear-Models/10.1214/aos/1176349025.full}{
#' Bootstrap and Wild Bootstrap for High Dimensional Linear Models}. \emph{The
#' Annals of Statistics}, 21(1), p. 255-285.} \cr
#' \item{Efron (1979). 
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full}{
#' Bootstrap Methods: Another Look at the Jackknife}.
#' \emph{The Annals of Statistics}, 7(1), p. 1-26.}}
#' 
#' @seealso {\link{SURE.trendfilter}}, \code{\link{cv.trendfilter}}
#' 
#' @examples 
#' # A quasar is an extremely luminous galaxy with an active central black 
#' # hole. Absorptions in the spectra of quasars at vast 
#' # cosmological distances from our galaxy reveal the presence of a gaseous 
#' # medium permeating the entirety of intergalactic space -- appropriately 
#' # named the 'intergalactic medium'. These absorptions allow astronomers to 
#' # study the structure of the Universe using the distribution of these 
#' # absorptions in quasar spectra. Particularly important is the 'forest' of 
#' # absorptions that arise from the Lyman-alpha spectral line, which traces 
#' # the presence of neutral hydrogen gas in intergalactic space.
#' #
#' # Here, we are interested in denoising the Lyman-alpha forest of a quasar 
#' # spectroscopically measured by the Sloan Digital Sky Survey. SDSS spectra 
#' # are equally spaced in log10 wavelength space, aside from some instances of 
#' # masked pixels.
#' 
#' data(quasar_spec)
#' 
#' # Run a parametric bootstrap on the optimized trend filtering estimator to 
#' # obtain uncertainty bands
#' 
#' opt <- SURE.trendfilter(spec$log10.wavelength, spec$flux, spec$weights)
#' boot.out <- bootstrap.trendfilter(SURE.out, bootstrap.algorithm = "parametric")


#' @importFrom glmgen trendfilter
#' @importFrom dplyr %>% mutate case_when select n
#' @importFrom tidyr tibble
#' @importFrom parallel mclapply detectCores
#' @importFrom stats quantile rnorm
bootstrap.trendfilter <- function(obj,
                                  level = 0.95, 
                                  B = 100L, 
                                  bootstrap.algorithm = c("nonparametric","parametric","wild"),
                                  return.full.ensemble = FALSE,
                                  prune = TRUE,
                                  mc.cores = detectCores()){
  
  stopifnot( class(obj) %in% c("SURE.trendfilter","cv.trendfilter") )
  bootstrap.algorithm <- match.arg(bootstrap.algorithm)
  stopifnot( is.numeric(level) & level > 0 & level < 1 )
  stopifnot( B >= 10 )
  
  if ( !prune ) warning("I hope you know what you are doing!")
  
  if ( mc.cores < detectCores() ){
    warning(paste0("Your machine has ", detectCores(), " cores. Consider increasing mc.cores to speed up computation."))
  }
  
  if ( mc.cores > detectCores() ) mc.cores <- detectCores()
  
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
  obj$bootstrap.lower.band <- apply(tf.boot.ensemble, 1, quantile, probs = (1 - level) / 2) 
  obj$bootstrap.upper.band <- apply(tf.boot.ensemble, 1, quantile, probs = 1 - (1 - level) / 2)
  
  obj <- c(obj, list(bootstrap.algorithm = bootstrap.algorithm, level = level, B = B))
  
  if ( return.full.ensemble ){
    obj$tf.bootstrap.ensemble <- tf.boot.ensemble
  }else{
    obj <- c(obj, list(tf.bootstrap.ensemble = NULL))
  }
  
  obj <- obj[c("x.eval","tf.estimate","tf.standard.errors","bootstrap.lower.band",
               "bootstrap.upper.band","bootstrap.algorithm","level","B",
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
