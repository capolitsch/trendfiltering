#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased 
#' risk estimate
#'
#' @description \loadmathjax{} \code{SURE.trendfilter} optimizes the trend
#' filtering hyperparameter by running a grid search over the vector, `gammas`,
#' of candidate hyperparameter values, and then selects the value that minimizes
#' an unbiased estimate of the model's generalization error. The full 
#' generalization error curve and the optimized trend filtering estimate of the
#' signal are then returned within a list that also includes useful ancillary
#' information.
#' 
#' @param x The vector of observed values of the input variable (a.k.a. the 
#' predictor, covariate, explanatory variable, regressor, independent variable, 
#' control variable, etc.)
#' @param y The vector of observed values of the output variable (a.k.a. the
#' response, target, outcome, regressand, dependent variable, etc.)
#' @param weights \strong{Must be passed.} A vector of weights for the 
#' observed outputs, defined as the reciprocal of the variance of the error
#' distribution. That is, `weights = 1 / sigmas^2`, where `sigmas` is a vector
#' of standard errors of the uncertainty in the observed outputs. `weights`
#' should either have length equal to 1 (corresponding to an error distribution 
#' with a constant variance) or length equal to `length(y)`
#' (i.e. heteroskedastic errors). 
#' @param k The degree of the trend filtering estimator. More precisely, with
#' the trend filtering estimator defined as a piecewise function of polynomials
#' smoothly connected at a set of "knots", `k` controls the degree of the
#' polynomials that build up the trend filtering estimator.
#' Defaults to `k = 2` (i.e. a piecewise quadratic estimate). Must be one of
#' `k = 0,1,2,3`. However, `k = 3` is discouraged due to algorithmic
#' instability, and `k = 2` typically gives a visually indistinguishable
#' estimate anyway.
#' @param ngammas Integer. The number of trend filtering hyperparameter values 
#' to run the grid search over. In this default case, the hyperparameter values
#' are automatically chosen by `SURE.trendfilter` and `ngammas` simply controls
#' the granularity of the grid.
#' @param gammas Overrides `ngammas` if passed. A vector of trend filtering
#' hyperparameter values to run the grid search over. It is advisable to let
#' the vector be equally-spaced in log-space and passed to `SURE.trendfilter`
#' in descending order. The function output will contain the sorted
#' hyperparameter vector regardless of the user-supplied ordering, and all
#' related output objects (e.g. the `errors` vector) will correspond to this
#' descending ordering. It's best to leave this
#' argument alone unless you know what you are doing.
#' @param x.eval A grid of inputs to evaluate the optimized trend filtering 
#' estimate on. Defaults to the observed inputs, `x`.
#' @param nx.eval Integer. If passed, then `x.eval` is overridden with \cr
#' `x.eval = seq(min(x), max(x), length = nx.eval)`
#' @param optimization.params A named list of parameters that contains all
#' parameter choices to be passed to the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the documentation for the \pkg{glmgen} function 
#' \code{\link[glmgen]{trendfilter.control.list}} for full details. 
#' No technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{ 
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering 
#' convex optimization. Defaults to `max_iter = 600L`. Increase this if 
#' the trend filtering estimate does not appear to have fully converged to a 
#' reasonable estimate of the signal.
#' \item{`obj_tol`}: The tolerance used in the convex optimization stopping 
#' criterion; when the relative change in the objective function is less than 
#' this value, the algorithm terminates. Decrease this if the trend filtering 
#' estimate does not appear to have fully converged to a reasonable estimate of 
#' the signal.
#' \item{`thinning`}: Logical. If `TRUE`, then the data are 
#' preprocessed so that a smaller, better conditioned data set is used for 
#' fitting. When left `NULL` (the default choice), the optimization will 
#' automatically detect whether thinning should be applied (i.e. cases in 
#' which the numerical fitting algorithm will struggle to converge). This 
#' preprocessing procedure is controlled by the `x_tol` argument below.
#' \item{`x_tol`}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins 
#' of size `x_tol` and find at least two elements of `x` that fall into the 
#' same bin, then we thin the data.}
#' 
#' @return An object of class 'SURE.trendfilter'. This is a list with the 
#' following elements:
#' \item{x.eval}{The grid of inputs the optimized trend filtering estimate was 
#' evaluated on.}
#' \item{tf.estimate}{The optimized trend filtering estimate of the signal, 
#' evaluated on `x.eval`.}
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
#' defined as `weights = 1 / sigmas^2`, where `sigmas` is a vector of 
#' standard errors of the uncertainty in the observed outputs.}
#' \item{fitted.values}{The optimized trend filtering estimate of the signal, 
#' evaluated at the observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted.values`}
#' \item{k}{The degree of the trend filtering estimator.}
#' \item{optimization.params}{A list of parameters that control the trend
#' filtering convex optimization.}
#' \item{n.iter}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each hyperparameter
#' value. If many of these are exactly equal to `max_iter`, then their
#' solutions have not converged with the tolerance specified by `obj_tol`.
#' In which case, it is often prudent to increase `max_iter`.}
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so 
#' that a smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{For internal use}
#' 
#' @details \code{SURE.trendfilter} estimates the fixed-input
#' mean-squared error of a trend filtering estimator by computing Stein's
#' unbiased risk estimate (a.k.a. SURE) over a grid of hyperparameter
#' values, which should typically be equally-spaced in log-space. The full error
#' curve and the optimized trend filtering estimate are returned within a
#' list that also includes useful ancillary information.
#' 
#' Given the choice of $k$, the hyperparameter $\gamma>0$ is used to tune the 
#' complexity (i.e. the wiggliness) of the trend filtering estimate by 
#' weighting the tradeoff between the complexity of the estimate and the size 
#' of the squared residuals. Obtaining an accurate estimate is therefore 
#' intrinsically tied to finding an optimal choice of $\gamma$. The selection 
#' of $\gamma$ is typically done by minimizing an estimate of the mean-squared 
#' prediction error (MSPE) of the trend filtering estimator. Here, there are 
#' two different notions of error to consider, namely, \emph{fixed-input} error 
#' and \emph{random-input} error. As the names suggest, the distinction between 
#' which type of error to consider is made based on how the inputs are sampled. 
#' As a general rule-of-thumb, we recommend optimizing with respect to 
#' fixed-input error when the inputs are regularly-sampled and optimizing with 
#' respect to random-input error on irregularly-sampled data.
#'
#' Recall the DGP stated in (link) and let it be denoted by 
#' $Q$ so that $\mathbb{E}_Q[\cdot]$ is the mathematical expectation with respect 
#' to the randomness of the DGP. Further, let 
#' $\sigma_i^2 = \text{Var}(\epsilon_i)$. The fixed-input MSPE is given by
#' \begin{align}
#' R(\gamma) &= \frac{1}{n}\sum_{i=1}^{n}\mathbb{E}_{Q}\Big[\big(f(t_i) - \widehat{f}_0(t_i;\gamma)\big)^2\;\Big|\;t_1,\dots,t_n\Big] \\
#' &= \frac{1}{n}\sum_{i=1}^{n}\Big(\mathbb{E}_{Q}\Big[\big(f_0(t_i) - \widehat{f}_0(t_i;\gamma)\big)^2\;\Big|\;t_1,\dots,t_n\Big] + \sigma_i^2\Big)
#' \end{align}
#' and the random-input MSPE is given by
#' \begin{equation}
#' \widetilde{R}(\gamma) = \mathbb{E}_{Q}\Big[\big(f(t) - \widehat{f}_0(t;\gamma)\big)^2\Big],
#' \end{equation}
#' where, in the latter, $t$ is considered to be a random component of the DGP 
#' with a marginal probability density $p_t(t)$ supported on the observed input 
#' interval. In each case, the theoretically optimal choice of $\gamma$ is 
#' defined as the minimizer of the respective choice of error. Empirically, we 
#' estimate the theoretically optimal choice of $\gamma$ by minimizing an 
#' estimate of (link) or (link). For fixed-input 
#' error we recommend Stein's unbiased risk estimate 
#' (SURE; (link)) and for random-input error we recommend 
#' $K$-fold cross validation with $K=10$. We elaborate on SURE here and refer 
#' the reader to (link) for $K$-fold cross validation. 
#' 
#' The SURE formula provides an unbiased estimate of the fixed-input MSPE of a 
#' statistical estimator:
#' \begin{align}
#' \widehat{R}_0(\gamma) &= \frac{1}{n}\sum_{i=1}^{n}\big(f(t_i) - \widehat{f}_0(t_i; \gamma)\big)^2 + \frac{2\overline{\sigma}^{2}\text{df}(\widehat{f}_0)}{n},
#' \end{align}
#' where $\overline{\sigma}^{2} = n^{-1}\sumin \sigma_i^2$ and
#' $\text{df}(\widehat{f}_0)$ is defined above. A formula for the
#' effective degrees of freedom of the trend filtering estimator is available
#' via the generalized lasso results of (link); namely,
#' \begin{align}
#' \text{df}(\widehat{f}_0) &= \mathbb{E}[\text{number of knots in $\widehat{f}_0$}] + k + 1.
#' \end{align}
#' We then obtain our hyperparameter estimate $\widehat{\gamma}$ by minimizing the 
#' following plug-in estimate for (link):
#' \begin{equation}
#' \widehat{R}(\gamma) = \frac{1}{n}\sum_{i=1}^{n}\big(f(t_i) - \widehat{f}_0(t_i; \gamma)\big)^2 + \frac{2\widehat{\overline{\sigma}}^{2}\widehat{\text{df}}(\widehat{f}_0)}{n},
#' \end{equation}
#' where $\widehat{\text{df}}$ is the estimate for the effective degrees of 
#' freedom that is obtained by replacing the expectation in (link) with 
#' the observed number of knots, and $\widehat{\overline{\sigma}}^2$ is an 
#' estimate of $\overline{\sigma}^2$. If a reliable estimate of 
#' $\overline{\sigma}^2$ is not available \emph{a priori}, a data-driven 
#' estimate can be constructed (see, e.g., (link)).
#' 
#' @export SURE.trendfilter
#' @author Collin A. Politsch, Ph.D., \email{collinpolitsch@@gmail.com}
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
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{\strong{[Link]}}}}

#' @importFrom glmgen trendfilter.control.list
#' @importFrom tidyr drop_na tibble
#' @importFrom dplyr %>% arrange filter
#' @importFrom magrittr %$%
SURE.trendfilter <- function(x,
                             y,
                             weights,
                             ngammas = 250L,
                             gammas,
                             x.eval = x,
                             nx.eval,
                             k = 2L,
                             optimization.params = list(max_iter = 600L,
                                                        obj_tol = 1e-10,
                                                        thinning = NULL)
                             ){
  
  if ( missing(x) || is.null(x) ) stop("x must be passed.")
  if ( missing(y) || is.null(y) ) stop("y must be passed.")
  if ( length(x) != length(y) ) stop("x and y must have the same length.")
  
  if ( missing(weights) || !(class(weights) %in% c("numeric","integer")) ){
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

  data <- tibble(x, y, weights) %>% 
    arrange(x) %>% 
    filter( weights != 0 ) %>%
    drop_na
  
  rm(x,y,weights)
  
  thinning <- optimization.params$thinning
  optimization.params <- trendfilter.control.list(max_iter = optimization.params$max_iter,
                                                  obj_tol = optimization.params$obj_tol)
  
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  optimization.params$x_tol <- optimization.params$x_tol / x.scale
  
  data.scaled <- data %>%
    mutate(x = x / x.scale,
           y = y / y.scale,
           weights = weights * y.scale ^ 2) %>%
    select(x, y, weights)
  
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
  
  tf.estimate <- glmgen:::predict.trendfilter(out, lambda = gamma.min, 
                                              x.new = x.eval / x.scale) %>%
    as.numeric
  
  data.scaled$fitted.values <- glmgen:::predict.trendfilter(out, lambda = gamma.min, 
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
                        thinning = thinning,
                        n.iter = n.iter,
                        x.scale = x.scale, 
                        y.scale = y.scale,
                        data.scaled = data.scaled),
                   class = "SURE.trendfilter"
                   )
  return(obj)
}
