#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased
#' risk estimate
#'
#' For every candidate hyperparameter value, compute an unbiased estimate of the
#' trend filtering model's predictive mean-squared error. See the details
#' section for guidelines on when [`sure_trendfilter()`] should be used versus
#' [cv_trendfilter()].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights (Optional) Weights for the observed outputs, defined as the
#' reciprocal variance of the additive noise that contaminates the output
#' signal. When the noise is expected to have an equal variance,
#' \mjseqn{\sigma^2}, for all observations, a scalar can be passed to `weights`,
#' namely `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must be a vector
#' with the same length as `x` and `y`.
#' @param k Degree of the polynomials that make up the piecewise-polynomial
#' trend filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas Number of hyperparameter values to test during validation.
#' Defaults to `nlambdas = 250`. The hyperparameter grid is internally
#' constructed to span the full trend filtering model space lying by a global
#' polynomial solution (i.e. a power law) and an interpolating solution, with
#' `nlambdas` controlling the granularity of the model space that is
#' @param optimization_params (Optional) A named list of parameter values to be
#' passed to the trend filtering ADMM algorithm of
#' [Ramdas and Tibshirani (2016)](
#' http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf), which is implemented in
#' the `glmgen` R package. See the [glmgen::trendfilter.control.list()]
#' documentation for a full list of the algorithm's parameters. The default
#' parameter choices will almost always suffice, but when adjustments do need to
#' be made, one can do so without any technical understanding of the ADMM
#' algorithm. In particular, the four parameter descriptions below should serve
#' as sufficient working knowledge.
#' \describe{
#' \item{obj_tol}{A stopping threshold for the ADMM algorithm. If the relative
#' change in the algorithm's cost functional between two consecutive steps is
#' less than `obj_tol`, the algorithm terminates. The algorithm's termination
#' can also result from it reaching the maximum tolerable iterations set
#' by the `max_iter` parameter (see below). The `obj_tol` parameter defaults to
#' `obj_tol = 1e-10`. The `cost_functional` vector, returned within the
#' `sure_trendfilter()` output, gives the relative change in the trend filtering
#' cost functional over the algorithm's final iteration, for every candidate
#' hyperparameter value.}
#' \item{max_iter}{Maximum number of ADMM iterations that we will tolerate.
#' Defaults to `max_iter = length(y)`. The actual number of iterations performed
#' by the algorithm, for every candidate hyperparameter value, is returned in
#' the `n_iter` vector, within the `sure_trendfilter()` output. If any of the
#' elements of `n_iter` are equal to `max_iter`, the tolerance defined by
#' `obj_tol` has not been attained and `max_iter` may need to be increased.}
#' \item{thinning}{Logical. If `thinning = TRUE`, then the data are preprocessed
#' so that a smaller data set is used to fit the trend filtering estimate, which
#' will ease the ADMM algorithm's convergence. This can be
#' very useful when a signal is so well-sampled that very little additional
#' information / predictive accuracy is gained by fitting the trend filtering
#' estimate on the full data set, compared to some subset of it. See the
#' [`cv_trendfilter()`] examples for a case study of this nature. When nothing
#' is passed to `thinning`, the algorithm will automatically detect whether
#' thinning should be applied. This preprocessing procedure is controlled by the
#' `x_tol` parameter below.}
#' \item{x_tol}{Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then the data is thinned.
#' }}
#'
#' @details Our recommendations for when to use [cv_trendfilter()] versus
#' [sure_trendfilter()] are shown in the table below.
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
#' @return An object of class `'sure_tf'`. This is a list with the following
#' elements:
#' \describe{
#' \item{lambdas}{Vector of candidate hyperparameter values (always returned in
#' descending order).}
#' \item{edfs}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambdas`.}
#' \item{sure_errors}{Vector of mean-squared prediction errors estimated
#' by SURE, for every hyperparameter value in `lambdas`.}
#' \item{se_sure_errors}{Vector of estimated standard errors for the
#' `sure_errors`.}
#' \item{lambda_min}{Hyperparameter value that minimizes the SURE validation
#' error curve.}
#' \item{lambda_1se}{The largest hyperparameter value (corresponding to the
#' smoothest trend filtering estimate) that yields a SURE error within
#' one standard error of `min(sure_errors)`. We call this the
#' "1-standard-error rule" hyperparameter, and it serves as an Occam's
#' razor-esque heuristic. That is, given two models with approximately equal
#' performance (here, in terms of MSE), it may be wise to opt for the simpler
#' model, i.e. the model with fewer effective degrees of freedom.}
#' \item{edf_min}{Number of effective degrees of freedom in the minimum-SURE
#' trend filtering estimator.}
#' \item{edf_1se}{Number of effective degrees of freedom in the 1-stand-error
#' rule trend filtering estimator.}
#' \item{i_min}{Index of `lambdas` that minimizes the SURE error curve.}
#' \item{i_1se}{Index of `lambdas` that gives the 1-standard-error rule
#' hyperparameter.}
#' \item{cost_functional}{The relative change in the cost functional over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambdas`.}
#' \item{n_iter}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambdas`. If an element of `n_iter`
#' is exactly equal to `max_iter`, then the ADMM algorithm stopped before
#' reaching the tolerance set by `obj_tol`. In these cases, you may need
#' to increase `max_iter` to ensure the trend filtering solution has
#' converged to satisfactory precision.}
#' \item{training_errors}{In-sample mean-squared error between the observed
#' outputs `y` and the trend filtering estimate, for every hyperparameter value
#' in `lambdas`.}
#' \item{optimisms}{SURE-estimated optimisms, i.e.
#' `optimisms = sure_errors - training_errors`.}
#' \item{tf_model}{A list of objects that is used internally by other
#' functions that operate on the `'sure_tf'` object.}
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
#' @importFrom dplyr arrange filter select
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom matrixStats rowSds
#' @importFrom stats median
sure_trendfilter <- function(x,
                             y,
                             weights,
                             k = 2L,
                             nlambdas = 250L,
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

  if (missing(weights) || !(class(weights) %in% c("numeric", "integer"))) {
    stop("weights must be passed to use sure_trendfilter.")
  }

  if (!(length(weights) %in% c(1, length(y)))) {
    stop(
      "weights must either have length 1 or length(y) to use sure_trendfilter()."
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

  if (missing(optimization_params)) {
    optimization_params <- NULL
  }

  opt_params <- get_optimization_params(optimization_params, n = length(x))
  optimization_params <- opt_params$optimization_params
  thinning <- opt_params$thinning
  admm_params <- do.call(trendfilter.control.list, optimization_params)
  x_scale <- data$x %>%
    diff() %>%
    median()
  y_scale <- median(abs(data$y)) / 10

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    ) %>%
    select(x, y, weights)

  admm_params$x_tol <- admm_params$x_tol / x_scale

  lambdas <- get_lambdas(nlambdas, data_scaled, k, thinning, admm_params)

  out <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  squared_residuals_mat <- (out$beta - data_scaled$y)^2
  optimisms_mat <- 2 / (data_scaled$weights * nrow(data_scaled)) * matrix(
    rep(out$df, each = nrow(data_scaled)),
    nrow = nrow(data_scaled)
  )

  sure_errors_mat <- (squared_residuals_mat + optimisms_mat) * y_scale^2
  sure_errors <- sure_errors_mat %>% colMeans()
  i_min <- min(which.min(sure_errors)) %>% as.integer()

  se_sure_errors <- replicate(
    5000,
    sure_errors_mat[sample(1:nrow(data_scaled), replace = TRUE), ] %>%
      colMeans()
  ) %>%
    rowSds()

  i_1se <- which(
    sure_errors <= sure_errors[i_min] + se_sure_errors[i_min]
  ) %>% min()

  tf_model <- structure(
    list(
      model_fit = out,
      x = x,
      y = y,
      weights = weights,
      x_scale = x_scale,
      y_scale = y_scale,
      data_scaled = data_scaled,
      k = k,
      admm_params = admm_params,
      thinning = thinning
    ),
    class = c("tf_model", "list")
  )

  structure(
    list(
      lambdas = lambdas,
      edfs = out$df %>% as.integer(),
      sure_errors = sure_errors,
      se_sure_errors = se_sure_errors,
      lambda_min = lambdas[i_min],
      lambda_1se = lambdas[i_1se],
      edf_min = out$df[i_min] %>% as.integer(),
      edf_1se = out$df[i_1se] %>% as.integer(),
      i_min = i_min,
      i_1se = i_1se,
      cost_functional = out$obj[nrow(out$obj), ],
      n_iter = out$iter %>% as.integer(),
      tf_model = tf_model
    ),
    class = c("sure_tf", "list")
  )
}
