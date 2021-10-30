#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased
#' risk estimate
#'
#' [`sure_trendfilter()`] optimizes the trend filtering hyperparameter via a
#' grid search over a vector of candidate hyperparameter values and selects
#' the value that minimizes an unbiased estimate of the model's generalization
#' mean-squared error (at the observed inputs). See details below for when you
#' should use [`sure_trendfilter()`] vs. [cv_trendfilter()].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights Weights for the observed outputs, defined as the reciprocal
#' variance of the additive noise that contaminates the signal in `y`.
#' When the noise is expected to have equal variance for all observations,
#' `weights` can be passed as a scalar. Otherwise, `weights` must have the same
#' length as `x` and `y`.
#' @param k Degree of the piecewise polynomials that make up the trend
#' filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas The number of hyperparameter settings to test during
#' validation. The hyperparameter grid is dynamically constructed to be
#' representative of the full model space between a single polynomial solution
#' and an interpolating solution, with `nlambdas` controlling the granularity
#' of the grid.
#' @param nx_eval Integer. The length of the input grid that the optimized
#' trend filtering estimate is evaluated on; i.e. if nothing is passed to
#' `x_eval`, then it is defined as
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @param x_eval (Optional) Overrides `nx_eval` if passed. A grid of inputs to
#' evaluate the optimized trend filtering estimate on.
#' @param optimization_params (Optional) A named list of optimization parameter
#' values to be passed to the trend filtering ADMM algorithm of
#' [Ramdas and Tibshirani 2016](
#' http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf) (implemented in the
#' `glmgen` package). See the [glmgen::trendfilter.control.list()] documentation
#' for full details. The default parameter choices will almost always suffice,
#' but when adjustments are necessary, no technical understanding of the ADMM
#' algorithm is needed in order to do so. The
#' following parameters may require some adjustments to ensure that your trend
#' filtering estimate has sufficiently converged:
#' \describe{
#' \item{`obj_tol`}{The objective tolerance that, together with `max_iter`,
#' determines the ADMM algorithm's stopping criterion. The algorithm will stop
#' either (1) when the relative change in the objective function is less than
#' `obj_tol`; or (2) when the number of iterations has reached `max_iter`.
#' This argument defaults to `obj_tol = 1e-10`. Therefore, when necessary, the
#' precision of the approximate solution given by the ADMM algorithm can be
#' increased by decreasing `obj_tol` and/or increasing `max_iter`.}
#' \item{`max_iter`}{Maximum iterations allowed for the trend filtering
#' optimization. Defaults to `max_iter = length(y)`. See the
#' `n_iter` element of the `sure_trendfilter()` output for the actual number of
#' iterations the ADMM algorithm took, for every candidate hyperparameter value
#' in `lambdas`. If any of the elements of `n_iter` are equal to `max_iter`,
#' the objective function's tolerance has not been reached and `max_iter` may
#' need to be increased.}
#' \item{`thinning`}{Logical. If `TRUE`, then the data are preprocessed so that
#' a smaller, better conditioned data set is used for fitting. When left `NULL`
#' (the default setting), the optimization will automatically detect whether
#' thinning should be applied (i.e. cases in which the numerical fitting
#' algorithm will struggle to converge). This preprocessing procedure is
#' controlled by the `x_tol` argument below.}
#' \item{`x_tol`}{Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then the data is thinned.
#' }}
#'
#' @details Our recommendations for when to use [cv_trendfilter()]
#' vs. [sure_trendfilter()] are shown in the table below.
#'
#' | Scenario                                                         |  Hyperparameter optimization  |
#' | :---                                                             |                         :---: |
#' | `x` is unevenly sampled                                          |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and reciprocal variances are not available |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and reciprocal variances are available     |      [`sure_trendfilter()`]   |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in large consecutive chunks) is still considered to
#' be evenly sampled. When the inputs are evenly sampled on a transformed scale,
#' we recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See the example below for a case when the
#' inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class [`sure_tf`][sure_trendfilter()]. This is a list
#' with the following elements:
#' \describe{
#' \item{x_eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf_estimate}{Optimized trend filtering estimate, evaluated at `x_eval`.
#' }
#' \item{validation_method}{"SURE"}
#' \item{lambdas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{generalization_errors}{Vector of SURE generalization error estimates,
#' corresponding to the descending-ordered `lambdas` vector.}
#' \item{lambda_min}{Hyperparameter value that minimizes the SURE generalization
#' error curve.}
#' \item{edf_min}{Effective degrees of freedom of the optimized trend
#' filtering estimator.}
#' \item{i_min}{Index of `lambdas` that minimizes the SURE error curve.}
#' \item{cost_change}{The relative change in the cost functional values
#' between the ADMM algorithm's penultimate and final iterations, for
#' every hyperparameter choice.}
#' \item{n_iter}{The number of iterations taken by the ADMM algorithm along its
#' approximate solution path.
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
#' \item{thinning}{Logical. If `TRUE`, then the data were preprocessed such
#' that a reduced subset was passed to the trend filtering ADMM algorithm in
#' order to make for a more tractable/stable problem and solution.}
#' \item{x_scale, y_scale, data_scaled}{For internal use.}
#' }
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
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom tidyr drop_na tibble
#' @importFrom dplyr arrange filter select first last
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom stats median approx
sure_trendfilter <- function(x,
                             y,
                             weights,
                             k = 2L,
                             nlambdas = 250L,
                             nx_eval = 1500L,
                             x_eval,
                             optimization_params) {
  if (missing(x) || is.null(x)) stop("x must be passed.")
  if (missing(y) || is.null(y)) stop("y must be passed.")
  if (length(x) != length(y)) stop("x and y must have equal length.")
  if (length(y) < k + 2) stop("Must have >= k + 2 observations.")
  if (k < 0 || k != round(k)) stop("k must be a nonnegative integer.")
  if (k > 2) {
    stop("k > 2 are algorithmically unstable and do not improve upon k = 2.")
  }

  if (nlambdas < 100 || nlambdas != round(nlambdas)) {
    stop("nlambdas must be an integer >=100")
  } else {
    nlambdas %<>% as.integer()
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

  x %<>% as.double()
  y %<>% as.double()
  weights %<>% as.double()
  k %<>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()
  rm(x, y, weights)

  if (missing(optimization_params)) {
    optimization_params <- list(
      max_iter = nrow(data),
      obj_tol = 1e-10
    )
    thinning <- NULL
  } else {
    if (!("max_iter" %in% names(optimization_params))) {
      optimization_params$max_iter <- nrow(data)
    }
    if (!("obj_tol" %in% names(optimization_params))) {
      optimization_params$obj_tol <- 1e-10
    }
    if (!("thinning" %in% names(optimization_params))) {
      thinning <- NULL
    } else {
      thinning <- optimization_params$thinning
      optimization_params$thinning <- NULL
    }
  }

  admm_params <- do.call(trendfilter.control.list, optimization_params)
  x_scale <- diff(data$x) %>% median()
  y_scale <- median(abs(data$y)) / 10
  admm_params$x_tol <- admm_params$x_tol / x_scale

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    ) %>%
    select(x, y, weights)

  if (nlambdas >= 150) {
    nlambdas_start <- 100
  } else {
    nlambdas_start <- 50
  }

  out <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda.min.ratio = 1e-16,
    nlambda = nlambdas_start,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  lambdas <- c(
    approx(
      x = out$df,
      y = log(out$lambda),
      xout = seq(
        min(out$df),
        max(out$df),
        length = nlambdas - nlambdas_start - 2
      )[
        -c(1, nlambdas - nlambdas_start - 2)
      ]
    )[["y"]] %>%
      suppressWarnings() %>%
      exp(),
    out$lambda
  ) %>%
    unique() %>%
    sort(decreasing = TRUE)

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
  cost_change <- out$obj[,length(lambdas)]

  out <- trendfilter(
    data_scaled$x,
    data_scaled$y,
    data_scaled$weights,
    lambda = lambda_min,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  if (missing(x_eval)) {
    x_eval <- seq(min(data$x), max(data$x), length = nx_eval)
  } else {
    x_eval %<>%
      as.double() %>%
      sort()
  }

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

  data_scaled %<>% mutate(residuals = y - fitted_values)

  structure(
    list(
      x_eval = x_eval,
      tf_estimate = tf_estimate * y_scale,
      validation_method = "SURE",
      lambdas = lambdas,
      generalization_errors = generalization_errors * y_scale^2,
      lambda_min = lambda_min,
      edfs = edfs,
      edf_min = out$df,
      i_min = i_min,
      cost_change = cost_change,
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
