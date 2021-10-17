#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased
#' risk estimate
#'
#' [`sure_trendfilter()`] optimizes the trend filtering hyperparameter via a grid
#' search over a vector of candidate hyperparameter settings and selects the
#' value that minimizes an unbiased estimate of the model's generalization
#' error. See details for when to use [`sure_trendfilter()`] vs.
#' [cv_trendfilter()].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights Weights for the observed outputs, defined as the reciprocal
#' variance of the additive noise that contaminates the signal. `weights` can be
#' passed as a scalar when the noise is expected to have equal variance for all
#' observations. Otherwise, `weights` must have the same length as `x` and `y`.
#' @param k Degree of the piecewise polynomials that make up the trend
#' filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas The number of hyperparameter settings to test during
#' validation. When nothing is passed to `lambdas` (highly recommended for
#' general use), the grid is automatically constructed by
#' [`sure_trendfilter()`], with `nlambdas` controlling the granularity of the
#' grid.
#' @param lambdas (Optional) Overrides `nlambdas` if passed. The vector of trend
#' filtering hyperparameter values for the grid search. Use of this argument is
#' discouraged unless you know what you are doing.
#' @param nx_eval Integer. If nothing is passed to `x_eval`, then it is defined
#' as `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @param x_eval (Optional) A grid of inputs to evaluate the optimized trend
#' filtering estimate on. May be ignored, in which case the grid is determined
#' by `nx_eval`.
#' @param optimization_params (Optional) A named list of parameter choices to be
#' passed to the trend filtering ADMM algorithm ([Ramdas and Tibshirani 2016](
#' http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf)). See the
#' [glmgen::trendfilter.control.list()] documentation for full details. No
#' technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering convex
#' optimization. Defaults to `max_iter = 600L`. See the `n_iter` element of the
#' function output for the actual number of iterations taken for every
#' hyperparameter choice in `lambdas`. If any of the elements of `n_iter` are
#' equal to `max_iter`, the objective function's tolerance has not been
#' achieved and `max_iter` may need to be increased.
#' \item{`obj_tol`}: The tolerance used in the convex optimization stopping
#' criterion; when the relative change in the objective function is less than
#' this value, the algorithm terminates. Thus, decreasing this setting will
#' increase the precision of the solution returned by the optimization. Defaults
#' to `obj_tol = 1e-10`. If the returned trend filtering estimate does not
#' appear to have fully converged to a reasonable estimate of the signal, this
#' issue can be resolve by some combination of decreasing `obj_tol` and
#' increasing `max_iter`.
#' \item{`thinning`}: Logical. If `TRUE`, then the data are preprocessed so that
#' a smaller, better conditioned data set is used for fitting. When left `NULL`
#' (the default setting), the optimization will automatically detect whether
#' thinning should be applied (i.e. cases in which the numerical fitting
#' algorithm will struggle to converge). This preprocessing procedure is
#' controlled by the `x_tol` argument below.
#' \item{`x_tol`}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then we thin the data.}
#'
#' @details \loadmathjax Our recommendations for when to use [cv_trendfilter()]
#' vs. [sure_trendfilter()] are shown in the table below.
#'
#' A regularly-sampled data set with some discarded pixels (either sporadically
#' or in large consecutive chunks) is still considered regularly sampled. When
#' the inputs are regularly sampled on a transformed scale, we recommend
#' transforming to that scale and carrying out the full trend filtering analysis
#' on that scale. See the example below for a case when the inputs are evenly
#' sampled on the `log10(x)` scale.
#'
#' | Scenario                                                            |  Hyperparameter optimization  |
#' | :---                                                                |                         :---: |
#' | `x` is irregularly sampled                                          |      [`cv_trendfilter()`]     |
#' | `x` is regularly sampled and reciprocal variances are not available |      [`cv_trendfilter()`]     |
#' | `x` is regularly sampled and reciprocal variances are available     |      [`sure_trendfilter()`]   |
#'
#' # Trend filtering with Stein's unbiased risk estimate
#' Here we describe the general motivation for optimizing a trend filtering
#' estimator with respect to Stein's unbiased risk estimate. See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Politsch et al. (2020a)} for more details. \cr
#'
#' Suppose we observe noisy measurements of an output variable of interest
#' (e.g., flux, magnitude, photon counts) according to the data generating
#' process
#' \mjsdeqn{y_i = f(x_i) + \epsilon_i, \quad\quad x_1,\dots,x_n\in(a,b),}
#' where \mjseqn{y_i} is a noisy observation of a signal \mjseqn{f(x_i)} and the
#' \mjseqn{\epsilon_i} have mean zero with variance
#' \mjseqn{\sigma_{i}^{2} = \text{Var}(\epsilon_{i})}. Let
#' \mjseqn{\hat{f}(\cdot\;; \lambda)} denote the trend filtering estimator of
#' order \mjseqn{k} with tuneable hyperparameter \mjseqn{\lambda}. The
#' fixed-input mean-squared prediction error (MSPE) of the estimator
#' \mjseqn{\hat{f}} is defined as
#' \mjsdeqn{R(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\;\mathbb{E}\left\[\left(y_i -
#' \hat{f}(x_{i};\lambda)\right)^2\;|\;x_{1},\dots,x_{n}\right\]}
#' \mjsdeqn{= \frac{1}{n}\sum_{i=1}^{n}\left(\mathbb{E}\left\[\left(f(x_i) -
#' \hat{f}(x_i;\lambda)\right)^2\;|\;x_1,\dots,x_n\right\] + \sigma_i^2\right).}
#' Stein's unbiased risk estimate (SURE) provides an unbiased estimate of the
#' fixed-input MSPE via the following formula:
#' \mjsdeqn{\hat{R}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{f}(x_i; \lambda)\big)^2 +
#' \frac{2\overline{\sigma}^{2}\text{df}(\hat{f})}{n},}
#' where \mjseqn{\overline{\sigma}^{2} = n^{-1}\sum_{i=1}^{n} \sigma_i^2}
#' and \mjseqn{\text{df}(\hat{f})} is the effective degrees of
#' freedom of the trend filtering estimator (with a fixed choice of
#' hyperparameter). The generalized lasso results of
#' [Tibshirani and Taylor (2012)](
#' https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full)
#' provide the following formula for the effective degrees of freedom of a trend
#' filtering estimator (with a fixed hyperparameter choice):
#' \mjsdeqn{\text{df}(\hat{f}) =
#' \mathbb{E}\left\[\text{number of knots in}\;\hat{f}\right\] + k + 1.}
#' The optimal hyperparameter value is then defined as
#' \mjsdeqn{\hat{\lambda} = \arg\min_{\lambda}
#' \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{f}(x_i; \lambda)\big)^2 +
#' \frac{2\hat{\overline{\sigma}}^{2}\hat{\text{df}}(\hat{f})}{n},}
#' where \mjseqn{\hat{\text{df}}} is the estimate for the effective
#' degrees of freedom that is obtained by replacing the expectation with the
#' observed number of knots, and \mjseqn{\hat{\overline{\sigma}}^2}
#' is an estimate of \mjseqn{\overline{\sigma}^2}. We define
#' \mjseqn{\overline{\sigma}^2} as `mean(1 / weights)`, so `weights` must be
#' passed in order to use `sure_trendfilter`. If a reliable estimate of
#' \mjseqn{\overline{\sigma}^2} is not available a priori, a data-driven
#' estimate can be constructed, e.g. see
#' [Wasserman (2004)](https://link.springer.com/book/10.1007/978-0-387-21736-9)
#' or
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf).
#' \cr \cr
#'
#' @return An object of class [`sure_tf`][sure_trendfilter()]. This is a list
#' with the following elements:
#' \item{x_eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf_estimate}{Optimized trend filtering estimate, evaluated at
#' `x_eval`.}
#' \item{validation_method}{"SURE"}
#' \item{lambdas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{generalization_errors}{Vector of SURE generalization error estimates,
#' corresponding to the descending-ordered `lambdas` vector.}
#' \item{lambda_min}{Hyperparameter value that minimizes the SURE generalization
#' error curve.}
#' \item{i_min}{Index of `lambdas` that minimizes the SURE error curve.}
#' \item{edf_min}{Effective degrees of freedom of the optimized trend
#' filtering estimator.}
#' \item{n_iter}{The number of iterations needed for the ADMM algorithm to
#' converge within the given tolerance, for each hyperparameter value. If many
#' of these are exactly equal to `max_iter`, then their solutions have not
#' converged with the tolerance specified by `obj_tol`. In which case, it is
#' often prudent to increase `max_iter`.}
#' \item{training_errors}{Mean-squared error between the observed outputs `y`
#' and the trend filtering estimate, for every hyperparameter choice.}
#' \item{optimisms}{SURE-estimated optimisms, i.e.
#' `optimisms = generalization_errors - training_errors`.}
#' \item{x}{Vector of observed inputs.}
#' \item{y}{Vector of observed outputs.}
#' \item{weights}{Weights for the observed outputs, defined as the reciprocal
#' variance of the additive noise that contaminates the signal.}
#' \item{fitted_values}{Optimized trend filtering estimate, evaluated at the
#' observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted_values`}
#' \item{k}{Degree of the trend filtering estimator.}
#' \item{admm_params}{List of parameter settings for the trend filtering ADMM
#' algorithm, constructed by passing the `optimization_params` list to
#' [glmgen::trendfilter.control.list()].}
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so that a
#' smaller, better conditioned data set is used for fitting.}
#' \item{x_scale, y_scale, data_scaled}{For internal use.}
#'
#' @export sure_trendfilter
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
#' @seealso [cv_trendfilter()], [bootstrap_trendfilter()]
#'
#' @examples
#' data(quasar_spectrum)
#' head(spec)
#'
#' \dontrun{
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' }
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom tidyr drop_na tibble
#' @importFrom dplyr %>% arrange filter select
#' @importFrom magrittr %$%
sure_trendfilter <- function(x, y, weights,
                             k = 2L, nlambdas = 250L, lambdas,
                             nx_eval = 1500L, x_eval,
                             optimization_params) {
  if (missing(x) || is.null(x)) stop("x must be passed.")
  if (missing(y) || is.null(y)) stop("y must be passed.")
  if (length(x) != length(y)) stop("x and y must have equal length.")
  if (length(y) < k + 2) stop("Must have >= k + 2 observations.")
  if (k < 0 || k != round(k)) stop("k must be a nonnegative integer.")
  if (k > 2) {
    stop("k > 2 are algorithmically unstable and do not improve upon k = 2.")
  }

  if (missing(lambdas)) {
    if (nlambdas < 0 || nlambdas != round(nlambdas)) {
      stop("nlambdas must be a positive integer")
      nlambdas <- nlambdas %>% as.integer()
    }
  } else {
    if (min(lambdas) <= 0L) {
      stop("All specified lambda values must be positive.")
    }
    if (length(lambdas) < 25L) {
      warning("Recommended to provide more candidate hyperparameter values.")
    }
    if (!all(lambdas == sort(lambdas, decreasing = T))) {
      warning("Sorting lambdas to descending order.")
    }
  }

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
  } else {
    if (any(x_eval < min(x) || x_eval > max(x))) {
      stop("x_eval should all be in range(x).")
    }
  }

  if (missing(weights) || !(class(weights) %in% c("numeric", "integer"))) {
    stop("weights must be passed to use sure_trendfilter.")
  }

  if (!(length(weights) %in% c(1, length(y)))) {
    stop(
      "weights must either have length 1 or length(y) to use sure_trendfilter."
    )
  }

  if (length(weights) == 1) weights <- rep(weights, length(x))

  x <- x %>% as.double()
  y <- y %>% as.double()
  weights <- weights %>% as.double()
  k <- k %>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()
  rm(x, y, weights)

  if (missing(lambdas)) {
    lambdas <- exp(seq(16, -10, length = nlambdas))
  } else {
    lambdas <- lambdas %>%
      as.double() %>%
      sort(decreasing = T)
  }

  if (missing(optimization_params)) {
    optimization_params <- list(max_iter = 600L, obj_tol = 1e-10)
  }
  thinning <- optimization_params$thinning
  optimization_params$thinning <- NULL
  admm_params <- do.call(trendfilter.control.list, optimization_params)
  x_scale <- median(diff(data$x))
  y_scale <- median(abs(data$y)) / 10
  admm_params$x_tol <- admm_params$x_tol / x_scale

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    ) %>%
    select(x, y, weights)

  out <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  training_errors <- (out$beta - data_scaled$y)^2 %>%
    colMeans() %>%
    as.double()
  optimisms <- 2 * out$df / nrow(data) * mean(1 / data_scaled$weights) %>%
    as.double()
  generalization_errors <- training_errors + optimisms
  edfs <- out$df %>% as.integer()
  n_iter <- out$iter %>% as.integer()
  i_min <- min(which.min(generalization_errors)) %>% as.integer()
  lambda_min <- lambdas[i_min]

  if (missing(x_eval)) {
    x_eval <- seq(min(data$x), max(data$x), length = nx_eval)
  } else {
    x_eval <- x_eval %>%
      as.double() %>%
      sort()
  }

  # Increase the TF solution's algorithmic precision for the optimized estimate
  admm_params$obj_tol <- admm_params$obj_tol * 1e-2

  out <- trendfilter(data_scaled$x, data_scaled$y, data_scaled$weights,
    lambda = lambda_min, k = k,
    thinning = thinning, control = admm_params
  )

  # Return the objective tolerance to its previous setting
  admm_params$obj_tol <- admm_params$obj_tol * 1e2

  tf_estimate <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_min,
    x.new = x_eval / x_scale
  ) %>%
    as.double()

  data_scaled$fitted_values <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_min,
    x.new = data_scaled$x
  ) %>%
    as.double()

  data_scaled <- data_scaled %>% mutate(residuals = y - fitted_values)

  structure(list(
    x_eval = x_eval,
    tf_estimate = tf_estimate * y_scale,
    validation_method = "SURE",
    lambdas = lambdas,
    generalization_errors = generalization_errors * y_scale^2,
    lambda_min = lambda_min,
    edfs = edfs,
    edf_min = out$df,
    i_min = i_min,
    n_iter = n_iter,
    training_errors = training_errors * y_scale^2,
    optimisms = optimisms * y_scale^2,
    x = data$x,
    y = data$y,
    weights = data$weights,
    fitted_values = data_scaled$fitted_values * y_scale,
    residuals = data_scaled$residuals * y_scale,
    k = k,
    admm_params = admm_params,
    thinning = thinning,
    x_scale = x_scale,
    y_scale = y_scale,
    data_scaled = data_scaled
  ),
  class = c("sure_tf", "list")
  )
}
