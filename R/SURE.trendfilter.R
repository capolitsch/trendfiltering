#' Optimize the trend filtering hyperparameter (with respect to Stein's 
#' unbiased risk estimate)
#'
#' @description \loadmathjax{} \code{SURE.trendfilter} estimates the fixed-input 
#' mean-squared error of a trend filtering estimator (via Stein's unbiased risk 
#' estimate) on a grid of values for the hyperparameter \code{gamma}, and 
#' returns the full error curve and the optimized trend filtering estimate 
#' within a larger list with useful ancillary information.
#' @param x The vector of observed values of the input variable (a.k.a. the 
#' predictor, covariate, explanatory variable, regressor, independent variable, 
#' control variable, etc.)
#' @param y The vector of observed values of the output variable (a.k.a. the
#' response, target, outcome, regressand, dependent variable, etc.).
#' @param weights \emph{\strong{Must be passed.}} A vector of weights for the 
#' observed outputs. These are defined as \code{weights = 1 / sigma^2}, where 
#' \code{sigma} is a vector of standard errors of the uncertainty in the 
#' observed outputs. \code{weights} should either have length equal to 1 
#' (corresponding to observations with a constant (scalar) variance of 
#' \code{sigma = 1/sqrt(weights)}) or length equal to 
#' \code{length(y)} (i.e. general heteroskedastic noise). 
#' @param k The degree of the trend filtering estimator. Defaults to 
#' \code{k = 2} (quadratic trend filtering). Must be one of \code{k = 0,1,2,3},
#' although \code{k = 3} is discouraged due to algorithmic instability (and is
#' visually indistinguishable from \code{k = 2} anyway).
#' @param ngammas Integer. The number of trend filtering hyperparameter values 
#' to run the grid search over.
#' @param gammas Overrides \code{ngammas} if passed. A user-supplied vector of 
#' trend filtering hyperparameter values to run the grid search over. It is
#' advisable to let the vector be equally-spaced in log-space and provided in 
#' descending order. The function output will contain the sorted hyperparameter
#' vector regardless of the input ordering, and all related output objects 
#' (e.g. the \code{errors} vector) will correspond to the sorted ordering. 
#' Unless you know what you are doing, it is best to leave this alone.
#' @param x.eval A user-supplied grid of inputs to evaluate the optimized trend 
#' filtering estimate on. Defaults to the observed inputs.
#' @param nx.eval Overrides \code{x.eval} if passed. A length for an
#' equally-spaced \code{x} grid between the minimum and maximum observed inputs
#' to evaluate the optimized trend filtering estimate on.
#' @param optimization.params a named list of parameters that
#' contains all parameter choices (user-supplied or defaults) to be passed to 
#' the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the documentation for the 
#' \pkg{glmgen} function \code{\link[glmgen]{trendfilter.control.list}} for 
#' full details. 
#' No technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{ 
#' \item{\code{max_iter}}: Maximum iterations allowed for the trend filtering 
#' convex optimization. Defaults to \code{max_iter = 600L}. Increase this if 
#' the trend filtering estimate does not appear to have fully converged to a 
#' reasonable estimate of the signal.
#' \item{\code{obj_tol}}: The tolerance used in the convex optimization stopping 
#' criterion; when the relative change in the objective function is less than 
#' this value, the algorithm terminates. Decrease this if the trend filtering 
#' estimate does not appear to have fully converged to a reasonable estimate of 
#' the signal.
#' \item{\code{thinning}}: logical. If \code{TRUE}, then the data are 
#' preprocessed so that a smaller, better conditioned data set is used for 
#' fitting. When left \code{NULL}, the default, the optimization will 
#' automatically detect whether thinning should be applied (i.e., cases in 
#' which the numerical fitting algorithm will struggle to converge). This 
#' preprocessing procedure is controlled by the \code{x_tol} argument of 
#' \code{\link[glmgen]{trendfilter.control.list}}.
#' \item{x_tol}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins 
#' of size \code{x_tol} and find at least two \code{x}'s which fall into the 
#' same bin, then we thin the data.
#' }
#' @return An object of class 'SURE.trendfilter'. This is a list with the 
#' following elements:
#' \item{x.eval}{The grid of inputs the optimized trend filtering estimate was 
#' evaluated on.}
#' \item{tf.estimate}{The optimized trend filtering estimate of the signal, 
#' evaluated on \code{x.eval}.}
#' \item{validation.method}{"SURE"}
#' \item{gammas}{Vector of hyperparameter values tested during validation
#' (always returned in descending order).}
#' \item{errors}{Vector of SURE error estimates corresponding to the 
#' *descending* set of gamma values tested during validation.}
#' \item{gamma.min}{Hyperparameter value that minimizes the SURE error curve.}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{edf.min}{The effective degrees of freedom of the optimally-tuned trend 
#' filtering estimator.}
#' \item{i.min}{The index of \code{gammas} (descending order) that minimizes 
#' the SURE error curve.}
#' \item{x}{The vector of the observed inputs.}
#' \item{y}{The vector of the observed outputs.}
#' \item{weights}{A vector of weights for the observed outputs. These are
#' defined as \code{weights = 1 / sigma^2}, where \code{sigma} is a vector of 
#' standard errors of the uncertainty in the observed outputs.}
#' \item{fitted.values}{The optimized trend filtering estimate of the signal, 
#' evaluated at the observed inputs \code{x}.}
#' \item{residuals}{\code{residuals = y - fitted.values}.}
#' \item{k}{The degree of the trend filtering estimator.}
#' \item{optimization.params}{a list of parameters that control the trend
#' filtering convex optimization.}
#' \item{n.iter}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each hyperparameter
#' value. If many of these are exactly equal to \code{max_iter}, then their
#' solutions have not converged with the tolerance specified by \code{obj_tol}.
#' In which case, it is often prudent to increase \code{max_iter}.}
#' \item{thinning}{logical. If \code{TRUE}, then the data are preprocessed so 
#' that a smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{for internal use.}
#' @details Further details...
#' @export SURE.trendfilter
#' @author Collin A. Politsch, \email{collinpolitsch@@gmail.com}
#' @seealso \code{\link{bootstrap.trendfilter}}
#' @references 
#' \enumerate{
#' \item{Politsch et al. (2020a). Trend filtering – I. A modern 
#' statistical tool for time-domain astronomy and astronomical spectroscopy. 
#' \emph{Monthly Notices of the Royal Astronomical Society}, 492(3), 
#' p. 4005-4018. 
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{\strong{[Link]}}}
#' \item{Politsch et al. (2020b). Trend Filtering – II. Denoising 
#' astronomical signals with varying degrees of smoothness. \emph{Monthly 
#' Notices of the Royal Astronomical Society}, 492(3), p. 4019-4032.
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' \strong{[Link]}}}
#' }

#' @importFrom tidyr drop_na tibble
#' @importFrom magrittr %$%
#' @importFrom dplyr %>% arrange filter
#' @importFrom glmgen trendfilter.control.list
SURE.trendfilter <- function(x,
                             y,
                             weights,
                             x.eval = x,
                             ngammas = 250L,
                             nx.eval,
                             gammas,
                             k = 2L,
                             thinning = NULL,
                             optimization.params = list(max_iter = 600L, obj_tol = 1e-10)
){
  
  if ( missing(x) || is.null(x) ) stop("x must be passed.")
  if ( missing(y) || is.null(y) ) stop("y must be passed.")
  if ( length(x) != length(y) ) stop("x and y must have the same length.")
  if ( missing(weights) || !is.numeric(weights) ){
    stop(paste0("Currently, the user must pass weights to compute SURE."))
  }
  if ( !(length(weights) %in% c(1,length(y))) ){
    stop("weights must either be have length 1 or length(y).")
  }
  if ( length(y) < k + 2 ){
    stop("y must have length >= k+2 for kth order trend filtering.")
  }
  if ( k < 0 || k != round(k) ){
    stop("k must be a nonnegative integer. k=2 recommended")
  }
  if ( k > 3 ){
    stop(paste0("Large k leads to generally worse conditioning.\n", 
                "k = 0,1,2 are the most stable choices."))
  }
  if ( k == 3 ){
    warning(paste0("k = 3 can have poor conditioning...\n", 
                   "k = 2 is more stable and visually indistinguishable."))
  }
  if ( !missing(gammas) ){
    if ( min(gammas) < 0L ){
      stop("All specified gamma values must be positive.")
    }
    if ( length(gammas) < 25L ) stop("gammas must have length >= 25.")
  }
  if ( !missing(nx.eval) ){
    if ( nx.eval != round(nx.eval) ) stop("nx.eval must be a positive integer.")
  }else{
    if ( any(x.eval < min(x) || x.eval > max(x)) ){
      stop("x.eval should all be in range(x).")
    }
  }
  if ( length(weights) == 1 ){
    weights <- rep(weights, length(x))
  }

  data <- tibble(x = x, y = y, weights = weights) %>% 
    arrange(x) %>% 
    filter( weights != 0 ) %>%
    drop_na
  
  rm(x,y,weights)

  optimization.params <- trendfilter.control.list(max_iter = optimization.params$max_iter,
                                                  obj_tol = optimization.params$obj_tol)
  
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  optimization.params$x_tol <- optimization.params$x_tol / x.scale
  
  data.scaled <- data %>%
    mutate(x = x / x.scale,
           y = y / y.scale,
           weights = weights * y.scale ^ 2) %>%
    select(x,y,weights)
  
  if ( missing(gammas) ){
    gammas <- seq(16, -10, length = ngammas) %>% exp 
  }else{
    gammas <- sort(gammas, decreasing = TRUE)
  }
  
  out <- glmgen::trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gammas,
                             k = k,
                             thinning = thinning,
                             control = optimization.params
  )
  
  training.error <- colMeans( (out$beta - data.scaled$y) ^ 2 ) 
  optimism <- 2 * out$df / nrow(data) * mean(1 / data.scaled$weights)
  errors <- as.numeric(training.error + optimism)
  edfs <- out$df
  n.iter <- out$iter
  
  i.min <- as.integer(which.min(errors))
  gamma.min <- gammas[i.min]
  
  if ( !missing(x.eval) ){
    x.eval <- seq(min(data$x), max(data$x), length = nx.eval)
  }else{
    x.eval <- sort(x.eval)
  }
  
  optimization.params$obj_tol <- optimization.params$obj_tol * 1e-2
  
  out <- glmgen::trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gamma.min,
                             k = k,
                             thinning = thinning,
                             control = optimization.params
  )
  
  optimization.params$obj_tol <- optimization.params$obj_tol * 1e2
  
  tf.estimate <- glmgen:::predict.trendfilter(out, 
                                              lambda = gamma.min, 
                                              x.new = x.eval / x.scale) %>%
    as.numeric
  
  data.scaled$fitted.values <- glmgen:::predict.trendfilter(out, 
                                                            lambda = gamma.min, 
                                                            x.new = data.scaled$x) %>% 
    as.numeric
  
  data.scaled <- data.scaled %>% mutate(residuals = y - fitted.values)
  
  obj <- structure(list(x.eval = x.eval,
                        tf.estimate = tf.estimate * y.scale,
                        validation.method = "SURE",
                        gammas = gammas, 
                        gamma.min = gamma.min,
                        edfs = edfs,
                        edf.min = out$df,
                        i.min = i.min,
                        errors = errors * y.scale ^ 2,
                        x = data$x,
                        y = data$y,
                        weights = data$weights,
                        fitted.values = data.scaled$fitted.values * y.scale,
                        residuals = data.scaled$residuals * y.scale,
                        k = as.integer(k),
                        optimization.params = optimization.params,
                        n.iter = n.iter,
                        x.scale = x.scale, 
                        y.scale = y.scale,
                        data.scaled = data.scaled
  ),
  class = "SURE.trendfilter"
  )
  
  return(obj)
}
