#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased
#' risk estimate
#'
#' `SURE.trendfilter` optimizes the trend filtering hyperparameter via a grid
#' search over a vector of candidate hyperparameter settings and selects the
#' value that minimizes an unbiased estimate of the model's generalization
#' error. See details for when to use `SURE.trendfilter` vs.
#' \code{\link{cv.trendfilter}}.
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
#' general use), the grid is automatically constructed by `SURE.trendfilter`,
#' with `nlambdas` controlling the granularity of the grid.
#' @param lambdas (Optional) Overrides `nlambdas` if passed. The vector of trend
#' filtering hyperparameter values for the grid search. Use of this argument is
#' discouraged unless you know what you are doing.
#' @param x.eval (Optional) A grid of inputs to evaluate the optimized trend
#' filtering estimate on. May be ignored, in which case the grid is determined
#' by `nx.eval`.
#' @param nx.eval Integer. If nothing is passed to `x.eval`, then it is defined
#' as `x.eval = seq(min(x), max(x), length = nx.eval)`.
#' @param optimization.params A named list of parameter choices to be passed to
#' the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the [glmgen::trendfilter.control.list()]
#' documentation for full details. No technical understanding of the ADMM
#' algorithm is needed and the default parameter choices will almost always
#' suffice. However, the following parameters may require some adjustments to
#' ensure that your trend filtering estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering convex
#' optimization. Defaults to `max_iter = 600L`. See the `n.iter` element
#' of the function output for the actual number of iterations taken for every
#' hyperparameter choice in `lambdas`. If any of the elements of `n.iter` are
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
#' @param seed Random number seed (for reproducible results).
#'
#' @details \loadmathjax Our recommendations for when to use
#' \code{\link{cv.trendfilter}} vs. `SURE.trendfilter`, as well as each of the
#' available settings for `bootstrap.algorithm` are shown in the table below.
#'
#' A regularly-sampled data set with some discarded pixels (either sporadically
#' or in large consecutive chunks) is still considered regularly sampled. When
#' the inputs are regularly sampled on a transformed scale, we recommend
#' transforming to that scale and carrying out the full trend filtering analysis
#' on that scale. See the example below for a case when the inputs are evenly
#' sampled on the `log10(x)` scale.
#'
#' | Scenario                                                          | Hyperparameter optimization | `bootstrap.algorithm` |
#' | :------------                                                     |     ------------:           |         ------------: |
#' | `x` is irregularly sampled                                        | `cv.trendfilter`            | "nonparametric"       |
#' | `x` is regularly sampled & reciprocal variances are not available | `cv.trendfilter`            | "wild"                |
#' | `x` is regularly sampled & reciprocal variances are available     | `SURE.trendfilter`          | "parametric"          |
#'
#'
#' # Trend filtering with Stein's unbiased risk estimate
#' Here we describe the general motivation for optimizing a trend filtering
#' estimator with respect to Stein's unbiased risk estimate. See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{Politsch et al. (2020a)}
#' for more details. \cr
#'
#' Suppose we observe noisy measurements of an output variable of interest
#' (e.g., flux, magnitude, photon counts) according to the data generating
#' process
#' \mjsdeqn{y_i = f(x_i) + \epsilon_i, \quad\quad x_1,\dots,x_n\in(a,b),}
#' where \mjseqn{y_i} is a noisy observation of a signal \mjseqn{f(x_i)} and the
#' \mjseqn{\epsilon_i} have mean zero with variance
#' \mjseqn{\sigma_{i}^{2} = \text{Var}(\epsilon_{i})}. Let
#' \mjseqn{\widehat{f}(\cdot\;; \lambda)} denote the trend filtering estimator of
#' order \mjseqn{k} with tunable hyperparameter \mjseqn{\lambda}. The fixed-input
#' mean-squared prediction error (MSPE) of the estimator \mjseqn{\widehat{f}}
#' is defined as
#' \mjsdeqn{R(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\;\mathbb{E}\left\[\left(y_i - \widehat{f}(x_{i};\lambda)\right)^2\;|\;x_{1},\dots,x_{n}\right\]}
#' \mjsdeqn{= \frac{1}{n}\sum_{i=1}^{n}\left(\mathbb{E}\left\[\left(f(x_i) - \widehat{f}(x_i;\lambda)\right)^2\;|\;x_1,\dots,x_n\right\] + \sigma_i^2\right).}
#' Stein's unbiased risk estimate (SURE) provides an unbiased estimate of the
#' fixed-input MSPE via the following formula:
#' \mjsdeqn{\widehat{R}(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \widehat{f}(x_i; \lambda)\big)^2 + \frac{2\overline{\sigma}^{2}\text{df}(\widehat{f})}{n},}
#' where \mjseqn{\overline{\sigma}^{2} = n^{-1}\sum_{i=1}^{n} \sigma_i^2}
#' and \mjseqn{\text{df}(\widehat{f})} is the effective degrees of
#' freedom of the trend filtering estimator (with a fixed choice of
#' hyperparameter). The generalized lasso results of
#' [Tibshirani and Taylor (2012)](https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full)
#' provide the following formula for the effective degrees of freedom of a trend
#' filtering estimator (with a fixed hyperparameter choice):
#' \mjsdeqn{\text{df}(\widehat{f}) = \mathbb{E}\left\[\text{number of knots in}\;\widehat{f}\right\] + k + 1.}
#' The optimal hyperparameter value is then defined as
#' \mjsdeqn{\widehat{\lambda} = \arg\min_{\lambda} \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \widehat{f}(x_i; \lambda)\big)^2 + \frac{2\widehat{\overline{\sigma}}^{2}\widehat{\text{df}}(\widehat{f})}{n},}
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
#' \cr \cr
#'
#' @return An object of class 'SURE.trendfilter'. This is a list with the
#' following elements:
#' \item{x.eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf.estimate}{Optimized trend filtering estimate, evaluated at `x.eval`.}
#' \item{validation.method}{"SURE"}
#' \item{lambdas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{generalization.errors}{Vector of SURE generalization error estimates,
#' corresponding to the descending-ordered `lambdas` vector.}
#' \item{lambda.min}{Hyperparameter value that minimizes the SURE generalization
#' error curve.}
#' \item{i.min}{Index of `lambdas` that minimizes the SURE error curve.}
#' \item{edf.min}{Effective degrees of freedom of the optimized trend
#' filtering estimator.}
#' \item{n.iter}{The number of iterations needed for the ADMM algorithm to
#' converge within the given tolerance, for each hyperparameter value. If many
#' of these are exactly equal to `max_iter`, then their solutions have not
#' converged with the tolerance specified by `obj_tol`. In which case, it is
#' often prudent to increase `max_iter`.}
#' \item{training.errors}{Mean-squared error between the observed outputs `y`
#' and the trend filtering estimate, for every hyperparameter choice.}
#' \item{optimisms}{SURE-estimated optimisms, i.e.
#' `optimisms = generalization.errors - training.errors`.}
#' \item{x}{Vector of observed inputs.}
#' \item{y}{Vector of observed outputs.}
#' \item{weights}{Weights for the observed outputs, defined as the reciprocal
#' variance of the additive noise that contaminates the signal.}
#' \item{fitted.values}{Optimized trend filtering estimate, evaluated at the
#' observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted.values`}
#' \item{k}{Degree of the trend filtering estimator.}
#' \item{ADMM.params}{List of parameter settings for the trend filtering ADMM
#' algorithm, constructed by passing the `optimization.params` list to
#' [glmgen::trendfilter.control.list()].}
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so that a
#' smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{For internal use.}
#'
#' @export SURE.trendfilter
#'
#' @author
#' \emph{\bold{Collin A. Politsch, Ph.D.}} \cr
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
<<<<<<< HEAD
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
#' Filtering and Related Problems}. arXiv: 2003.03886.}}
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
#' data(quasar_spectrum)
#' head(spec)
#'
#' # | log10.wavelength|       flux|   weights|
#' # |----------------:|----------:|---------:|
#' # |           3.5529|  0.4235348| 0.0417015|
#' # |           3.5530| -2.1143005| 0.1247811|
#' # |           3.5531| -3.7832341| 0.1284383|
#'
#' opt <- SURE.trendfilter(spec$log10.wavelength, spec$flux, spec$weights)
#'
#' plot(log(opt$lambdas), opt$generalization.errors, type = "l", lwd = 1.5)
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom tidyr drop_na tibble
#' @importFrom dplyr %>% arrange filter select
#' @importFrom magrittr %$%
SURE.trendfilter <- function(x, y, weights,
                             k = 2L, nlambdas = 250L, lambdas,
                             x.eval, nx.eval = 1500L,
                             optimization.params = list(max_iter = 600L, obj_tol = 1e-10),
                             seed = 1) {
  if (missing(x) || is.null(x)) stop("x must be passed.")
  if (missing(y) || is.null(y)) stop("y must be passed.")
  if (length(x) != length(y)) stop("x and y must have equal length.")
  if (length(y) < k + 2) stop("Must have >= k + 2 observations.")
  if (k < 0 || k != round(k)) stop("k must be a nonnegative integer.")
  if (k > 2) stop("k > 2 are algorithmically unstable and do not improve upon k = 2.")

  if (missing(lambdas)) {
    if (nlambdas < 0 || nlambdas != round(nlambdas)) {
      stop("nlambdas must be a positive integer")
      nlambdas <- nlambdas %>% as.integer()
    }
  } else {
    if (min(lambdas) <= 0L) stop("All specified lambda values must be positive.")
    if (length(lambdas) < 25L) warning("Recommended to provide more candidate hyperparameter values.")
    if (!all(lambdas == sort(lambdas, decreasing = T))) warning("Sorting lambdas to descending order.")
  }

  if (missing(x.eval)) {
    if (!(class(nx.eval) %in% c("numeric", "integer")) || nx.eval < 1 || nx.eval != round(nx.eval)) stop("nx.eval must be a positive integer.")
  } else {
    if (any(x.eval < min(x) || x.eval > max(x))) stop("x.eval should all be in range(x).")
  }

  if (missing(weights) || !(class(weights) %in% c("numeric", "integer"))) {
    stop("weights must be passed to use SURE.trendfilter.")
  }

  if (!(length(weights) %in% c(1, length(y)))) {
    stop("weights must either have length 1 or length(y) to use SURE.trendfilter.")
  }

  if (length(weights) == 1) weights <- rep(weights, length(x))

  x <- x %>% as.double()
  y <- y %>% as.double()
  weights <- weights %>% as.double()
  k <- k %>% as.integer()

  if (!missing(seed)) set.seed(seed)

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

  thinning <- optimization.params$thinning
  optimization.params$thinning <- NULL
  ADMM.params <- do.call(trendfilter.control.list, optimization.params)
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  ADMM.params$x_tol <- ADMM.params$x_tol / x.scale

  data.scaled <- data %>%
    mutate(
      x = x / x.scale,
      y = y / y.scale,
      weights = weights * y.scale^2
    ) %>%
    select(x, y, weights)

  out <- trendfilter(
    x = data.scaled$x,
    y = data.scaled$y,
    weights = data.scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = ADMM.params
  )

  training.errors <- (out$beta - data.scaled$y)^2 %>%
    colMeans() %>%
    as.double()
  optimisms <- 2 * out$df / nrow(data) * mean(1 / data.scaled$weights) %>% as.double()
  generalization.errors <- training.errors + optimisms
  edfs <- out$df %>% as.integer()
  n.iter <- out$iter %>% as.integer()
  i.min <- min(which.min(generalization.errors)) %>% as.integer()
  lambda.min <- lambdas[i.min]

  if (missing(x.eval)) {
    x.eval <- seq(min(data$x), max(data$x), length = nx.eval)
  } else {
    x.eval <- x.eval %>%
      as.double() %>%
      sort()
  }

  # Increase the TF solution's algorithmic precision for the optimized estimate
  ADMM.params$obj_tol <- ADMM.params$obj_tol * 1e-2

  out <- trendfilter(data.scaled$x, data.scaled$y, data.scaled$weights,
    lambda = lambda.min, k = k,
    thinning = thinning, control = ADMM.params
  )

  # Return the objective tolerance to its previous setting
  ADMM.params$obj_tol <- ADMM.params$obj_tol * 1e2

  tf.estimate <- glmgen:::predict.trendfilter(out,
    lambda = lambda.min,
    x.new = x.eval / x.scale
  ) %>%
    as.double()

  data.scaled$fitted.values <- glmgen:::predict.trendfilter(out,
    lambda = lambda.min,
    x.new = data.scaled$x
  ) %>%
    as.double()

  data.scaled <- data.scaled %>% mutate(residuals = y - fitted.values)

  structure(list(
    x.eval = x.eval,
    tf.estimate = tf.estimate * y.scale,
    validation.method = "SURE",
    lambdas = lambdas,
    generalization.errors = generalization.errors * y.scale^2,
    lambda.min = lambda.min,
    edfs = edfs,
    edf.min = out$df,
    i.min = i.min,
    n.iter = n.iter,
    training.errors = training.errors * y.scale^2,
    optimisms = optimisms * y.scale^2,
    x = data$x,
    y = data$y,
    weights = data$weights,
    fitted.values = data.scaled$fitted.values * y.scale,
    residuals = data.scaled$residuals * y.scale,
    k = k,
    ADMM.params = ADMM.params,
    thinning = thinning,
    x.scale = x.scale,
    y.scale = y.scale,
    data.scaled = data.scaled
  ),
  class = "SURE.trendfilter"
  )
}
