#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased 
#' risk estimate
#'
#' `SURE.trendfilter` optimizes the trend filtering hyperparameter by running a
#' grid search over a vector, `gammas`, of candidate hyperparameter values, and
#' selecting the value that minimizes an unbiased estimate of the model's
#' generalization error. The full generalization error curve and the optimized
#' trend filtering estimate are then returned as elements of a list object that
#' comprehensively summarizes the analysis.
#' 
#' @param x The vector of observed values of the input variable (a.k.a. the 
#' predictor, covariate, explanatory variable, regressor, independent variable, 
#' control variable, etc.)
#' @param y The vector of observed values of the output variable (a.k.a. the
#' response, target, outcome, regressand, dependent variable, etc.).
#' @param weights **Must be passed.** A vector of weights for the observed
#' outputs, defined as the reciprocal of the variance of the error distribution.
#' That is, `weights = 1 / sigmas^2`, where `sigmas` is a vector of standard
#' errors of the uncertainty in the observed outputs. `weights` should either
#' have length equal to 1 (corresponding to an error distribution with a
#' constant variance) or length equal to `length(y)` (i.e. heteroskedastic
#' errors). 
#' @param k The degree of the trend filtering estimator. More precisely, with
#' the trend filtering estimator defined as a piecewise function of polynomials
#' smoothly connected at a set of "knots", `k` controls the degree of the
#' polynomials that build up the trend filtering estimator. Defaults to `k = 2`
#' (i.e. a piecewise quadratic estimate). Must be one of `k = 0,1,2,3`. However,
#' `k = 3` is discouraged due to algorithmic instability, and `k = 2` typically
#' gives a visually indistinguishable estimate anyway.
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
#' descending ordering. It's best to leave this argument alone unless you know
#' what you are doing.
#' @param x.eval A grid of inputs to evaluate the optimized trend filtering 
#' estimate on. Defaults to the observed inputs, `x`.
#' @param nx.eval Integer. If passed, overrides `x.eval` with
#' `seq(min(x), max(x), length = nx.eval)`.
#' @param optimization.params A named list of parameters that contains all
#' parameter choices to be passed to the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the 
#' [glmgen::trendfilter.control.list()]
#' documentation for full details. No technical understanding of the ADMM
#' algorithm is needed and the default parameter choices will almost always
#' suffice. However, the following parameters may require some adjustments to
#' ensure that your trend filtering estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering convex
#' optimization. Defaults to `max_iter = 600L`. Increase this if the trend
#' filtering estimate does not appear to have fully converged to a reasonable
#' estimate of the signal.
#' \item{`obj_tol`}: The tolerance used in the convex optimization stopping 
#' criterion; when the relative change in the objective function is less than 
#' this value, the algorithm terminates. Decrease this if the trend filtering 
#' estimate does not appear to have fully converged to a reasonable estimate of 
#' the signal.
#' \item{`thinning`}: Logical. If `TRUE`, then the data are preprocessed so that 
#' a smaller, better conditioned data set is used for fitting. When left `NULL`
#' (the default setting), the optimization will automatically detect whether
#' thinning should be applied (i.e. cases in which the numerical fitting
#' algorithm will struggle to converge). This preprocessing procedure is
#' controlled by the `x_tol` argument below.
#' \item{`x_tol`}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then we thin the data.}
#' @param ... Additional named arguments to be passed to 
#' [glmgen::trendfilter.control.list()].
#' 
#' @details \loadmathjax As a general rule-of-thumb, we recommend optimizing the
#' trend filtering hyperparameter by minimizing Stein's unbiased risk estimate
#' (SURE) when the inputs are regularly-sampled (either on the raw `x` scale or
#' some transformation of it) and optimizing the hyperparameter by
#' \mjseqn{V}-fold cross validation (using \code{\link{cv.trendfilter}}) when
#' the inputs are irregularly-sampled. A regularly-sampled data set with some
#' discarded pixels (either sporadically or in large consecutive chunks) is
#' still considered regularly sampled. When the inputs are regularly
#' sampled on a transformed scale, we recommend transforming to that
#' scale and carrying out the full trend filtering analysis (using SURE) on that
#' scale. See the example below for a case when the inputs are evenly sample on
#' the `log10(x)` scale.
#' 
#' Below we describe the general motivation for trend filtering with SURE. See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{Politsch et al. (2020a)}
#' for more details. \cr \cr
#' 
#' Suppose we observe noisy measurements of a response variable of interest
#' (e.g., flux, magnitude, photon counts) according to the data generating
#' process (DGP)
#' \mjsdeqn{y_i = f(x_i) + \epsilon_i, \quad\quad x_1,\dots,x_n\in(a,b),}
#' where \mjseqn{y_i} is a noisy observation of a signal \mjseqn{f(x_i)} and the
#' \mjseqn{\epsilon_i} have mean zero with variance
#' \mjseqn{\sigma_{i}^{2} = \text{Var}(\epsilon_{i})}. Let
#' \mjseqn{\widehat{f}(\cdot\;; \gamma)} denote the trend filtering estimator of
#' order \mjseqn{k} with tunable hyperparameter \mjseqn{\gamma}. The fixed-input
#' mean-squared prediction error (MSPE) of the estimator \mjseqn{\widehat{f}}
#' is defined as
#' \mjsdeqn{R(\gamma) = \frac{1}{n}\sum_{i=1}^{n}\;\mathbb{E}\left\[\left(y_i - \widehat{f}(x_{i};\gamma)\right)^2\;|\;x_{1},\dots,x_{n}\right\]}
#' \mjsdeqn{= \frac{1}{n}\sum_{i=1}^{n}\left(\mathbb{E}\left\[\left(f(x_i) - \widehat{f}(x_i;\gamma)\right)^2\;|\;x_1,\dots,x_n\right\] + \sigma_i^2\right).}
#' Stein's unbiased risk estimate (SURE) provides an unbiased estimate of the
#' fixed-input MSPE via the following formula:
#' \mjsdeqn{\widehat{R}(\gamma) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \widehat{f}(x_i; \gamma)\big)^2 + \frac{2\overline{\sigma}^{2}\text{df}(\widehat{f})}{n},}
#' where \mjseqn{\overline{\sigma}^{2} = n^{-1}\sum_{i=1}^{n} \sigma_i^2}
#' and \mjseqn{\text{df}(\widehat{f})} is the effective degrees of
#' freedom of the trend filtering estimator (with a fixed choice of
#' hyperparameter). The generalized lasso results of 
#' [Tibshirani and Taylor (2012)](https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full)
#' provide the following formula for the effective degrees of freedom of a trend
#' filtering estimator (with a fixed hyperparameter choice):
#' \mjsdeqn{\text{df}(\widehat{f}) = \mathbb{E}\left\[\text{number of knots in}\;\widehat{f}\right\] + k + 1.}
#' The optimal hyperparameter value is then defined as
#' \mjsdeqn{\widehat{\gamma} = \arg\min_{\gamma} \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \widehat{f}(x_i; \gamma)\big)^2 + \frac{2\widehat{\overline{\sigma}}^{2}\widehat{\text{df}}(\widehat{f})}{n},}
#' where \mjseqn{\widehat{\text{df}}} is the estimate for the effective 
#' degrees of freedom that is obtained by replacing the expectation with the
#' observed number of knots, and \mjseqn{\widehat{\overline{\sigma}}^2}
#' is an estimate of \mjseqn{\overline{\sigma}^2}. We define 
#' \mjseqn{\overline{\sigma}^2} as `mean(1 / weights)`, so `weights` must be
#' passed in order to use `SURE.trendfilter`. If a reliable estimate of
#' \mjseqn{\overline{\sigma}^2} is not available a priori, a data-driven
#' estimate can be constructed, e.g. see 
#' [Wasserman (2004)](https://link.springer.com/book/10.1007/978-0-387-21736-9)
#' or
#' [Hastie, Tibshirani, and Friedman (2009)](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf).
#' 
#' @return An object of class 'SURE.trendfilter'. This is a list with the 
#' following elements:
#' \item{x.eval}{The grid of inputs the optimized trend filtering estimate was 
#' evaluated on.}
#' \item{tf.estimate}{The optimized trend filtering estimate of the signal, 
#' evaluated on `x.eval`.}
#' \item{validation.method}{"SURE"}
#' \item{gammas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{errors}{Vector of SURE error estimates corresponding to the 
#' **descending** set of gamma values tested during validation.}
#' \item{gamma.min}{Hyperparameter value that minimizes the SURE error curve.}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{edf.min}{The effective degrees of freedom of the optimally-tuned trend 
#' filtering estimator.}
#' \item{i.min}{The index of `gammas` (descending order) that minimizes the
#' SURE error curve.}
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
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so that a
#' smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{For internal use.}
#' 
#' @export SURE.trendfilter
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
#' \bold{Trend filtering theory}
#' \enumerate{
#' \item{Tibshirani (2014).
#' \href{https://projecteuclid.org/euclid.aos/1395234979}{Adaptive piecewise
#' polynomial estimation via trend filtering}. \emph{The Annals of Statistics}.
#' 42(1), p. 285-323.} \cr
#' \item{Tibshirani (2020). \href{https://arxiv.org/abs/2003.03886}{Divided
#' Differences, Falling Factorials, and Discrete Splines: Another Look at Trend
#' Filtering and Related Problems}. arXiv preprint.}}
#' 
#' \bold{Stein's unbiased risk estimate}
#' \enumerate{
#' \item{Tibshirani and Wasserman (2015). 
#' \href{http://www.stat.cmu.edu/~larry/=sml/stein.pdf}{Stein’s Unbiased Risk 
#' Estimate}. \emph{36-702: Statistical Machine Learning course notes}
#' (Carnegie Mellon University).} \cr
#' \item{Efron (2014). 
#' \href{https://www.tandfonline.com/doi/abs/10.1198/016214504000000692}{
#' The Estimation of Prediction Error: Covariance Penalties 
#' and Cross-Validation}. \emph{Journal of the American Statistical
#' Association}. 99(467), p. 619-632.} \cr
#' \item{Stein (1981).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-6/Estimation-of-the-Mean-of-a-Multivariate-Normal-Distribution/10.1214/aos/1176345632.full}{
#' Estimation of the Mean of a Multivariate Normal Distribution}.
#' \emph{The Annals of Statistics}. 9(6), p. 1135-1151.}}
#' 
#' \bold{Effective degrees of freedom for trend filtering}
#' \enumerate{
#' \item{Tibshirani and Taylor (2012)}.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full}{
#' Degrees of freedom in lasso problems}. \emph{The Annals of Statistics},
#' 40(2), p. 1198-1232.}
#' 
#' @seealso \code{\link{cv.trendfilter}}, \code{\link{bootstrap.trendfilter}}
#' 
#' @examples 
#' #############################################################################
#' ##                    Quasar Lyman-alpha forest example                    ##
#' #############################################################################
#' # A quasar is an extremely luminous galaxy with an active supermassive black 
#' # hole at its center. Absorptions in the spectra of quasars at vast 
#' # cosmological distances from our galaxy reveal the presence of a gaseous 
#' # medium permeating the entirety of intergalactic space -- appropriately 
#' # named the 'intergalactic medium'. These absorptions allow astronomers to 
#' # study the structure of the Universe using the distribution of these 
#' # absorptions in quasar spectra. Particularly important is the 'forest' of 
#' # absorptions that arise from the Lyman-alpha spectral line, which traces 
#' # the presence of electrically neutral hydrogen in the intergalactic medium.
#' #
#' # Here, we are interested in denoising the Lyman-alpha forest of a quasar 
#' # spectroscopically measured by the Sloan Digital Sky Survey. SDSS spectra 
#' # are equally spaced in log10 wavelength space, aside from some instances of 
#' # masked pixels.
#' 
#' data(quasar_spec)
#'
#' # head(data)
#' #
#' # | log10.wavelength|       flux|   weights|
#' # |----------------:|----------:|---------:|
#' # |           3.5529|  0.4235348| 0.0417015|
#' # |           3.5530| -2.1143005| 0.1247811|
#' # |           3.5531| -3.7832341| 0.1284383|
#' 
#' SURE.out <- SURE.trendfilter(x = data$log10.wavelength, 
#'                              y = data$flux, 
#'                              weights = data$weights)
#' 
#' 
#' # Extract the estimated hyperparameter error curve and optimized trend 
#' # filtering estimate from the `SURE.trendfilter` output, and transform the 
#' # input grid to wavelength space (in Angstroms).
#' 
#' log.gammas <- log(SURE.out$gammas)
#' errors <- SURE.out$errors
#' log.gamma.min <- log(SURE.out$gamma.min)
#' 
#' wavelength <- 10 ^ (SURE.out$x)
#' wavelength.eval <- 10 ^ (SURE.out$x.eval)
#' tf.estimate <- SURE.out$tf.estimate
#' 
#' 
#' # Plot the results
#'
#' par(mfrow = c(2,1), mar = c(5,4,2.5,1) + 0.1)
#' plot(x = log.gammas, y = errors, main = "SURE error curve", 
#'      xlab = "log(gamma)", ylab = "SURE error")
#' abline(v = log.gamma.min, lty = 2, col = "blue3")
#' text(x = log.gamma.min, y = par("usr")[4], 
#'      labels = "optimal gamma", pos = 1, col = "blue3")
#' 
#' plot(x = wavelength, y = SURE.out$y, type = "l", 
#'      main = "Quasar Lyman-alpha forest", 
#'      xlab = "Observed wavelength (Angstroms)", ylab = "Flux")
#' lines(wavelength.eval, tf.estimate, col = "orange", lwd = 2.5)
#' legend(x = "topleft", lwd = c(1,2), lty = 1, col = c("black","orange"), 
#'        legend = c("Noisy quasar Lyman-alpha forest", 
#'                   "Trend filtering estimate"))


#' @importFrom glmgen trendfilter trendfilter.control.list
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
                                                        thinning = NULL),
                             ...){
  
  if ( missing(x) || is.null(x) ) stop("x must be passed.")
  if ( missing(y) || is.null(y) ) stop("y must be passed.")
  if ( length(x) != length(y) ) stop("x and y must have the same length.")
  
  if ( missing(weights) || !(class(weights) %in% c("numeric","integer")) ){
    stop(paste0("weights must be passed to compute SURE."))
  }
  
  if ( !(length(weights) %in% c(1,length(y))) ){
    stop("weights must either be have length 1 or length(y)")
  }
  
  if ( length(y) < k + 2 ){
    stop("length(y) must be >= k + 2")
  }
  
  if ( k < 0 || k != round(k) ){
    stop("k must be a nonnegative integer.")
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
                                                  obj_tol = optimization.params$obj_tol,
                                                  ...)
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  optimization.params$x_tol <- optimization.params$x_tol / x.scale
  
  data.scaled <- data %>%
    mutate(x = x / x.scale,
           y = y / y.scale,
           weights = weights * y.scale ^ 2) %>%
    select(x, y, weights)
  
  if ( missing(gammas) ){
    gammas <- exp(seq(16, -10, length = ngammas))
  }else{
    gammas <- sort(gammas, decreasing = T)
  }
  
  out <- trendfilter(x = data.scaled$x,
                     y = data.scaled$y,
                     weights = data.scaled$weights,
                     lambda = gammas,
                     k = k,
                     thinning = thinning,
                     control = optimization.params)
  
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
  
  # Increase the algorithmic precision for the optimized TF estimate
  optimization.params$obj_tol <- optimization.params$obj_tol * 1e-2
  
  out <- trendfilter(x = data.scaled$x,
                     y = data.scaled$y,
                     weights = data.scaled$weights,
                     lambda = gamma.min,
                     k = k,
                     thinning = thinning,
                     control = optimization.params)
  
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
                        thinning = thinning,
                        n.iter = n.iter,
                        x.scale = x.scale, 
                        y.scale = y.scale,
                        data.scaled = data.scaled),
                   class = "SURE.trendfilter"
                   )
  return(obj)
}
