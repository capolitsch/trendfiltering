#' Fit a trend filtering model (front-end function focused on ease of use)
#'
#' Fit a trend filtering model. Generic functions such as [`predict()`],
#' [`fitted.values()`], and [`residuals()`] may be called on the
#' [`.trendfilter()`] output.
#'
#' @param x
#'   Vector of observed values for the input variable.
#' @param y
#'   Vector of observed values for the output variable.
#' @param weights
#'   Weights for the output measurements. Output weights are defined as the
#'   inverse variance of the additive noise that contaminates the output signal.
#'   When the noise is expected to have a constant variance \mjseqn{\sigma^2}
#'   over all outputs, a scalar may be passed to `weights`, i.e.
#'   `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must be a vector
#'   with the same length as `x` and `y`.
#' @param lambda
#'   One or more hyperparameter values to fit a trend filtering estimate for.
#' @param edf
#'   (Not yet available) Alternative hyperparametrization for the trend
#'   filtering model(s). Vector of the desired number of effective degrees of
#'   freedom in each model.
#' @param k
#'   Degree of the polynomials that make up the piecewise-polynomial trend
#'   filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#'   estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#'   disallowed since they yield no statistical benefit over `k = 2` and their
#'   use can lead to instability in the convex optimization.
#' @param obj_tol
#'   Stopping criterion for the trend filtering convex optimization. If the
#'   relative change in the trend filtering objective function between two
#'   successive iterations is less than `obj_tol`, the algorithm terminates.
#'   Defaults to `obj_tol = 1e-10`.
#' @param max_iter
#'   Maximum number of iterations that we will tolerate for the trend filtering
#'   convex optimization algorithm. Defaults to `max_iter = length(y)`.
#' @param ...
#'   Additional named arguments. Currently unused.
#'
#' @return An object of class `'trendfilter'`. This is a list with the
#' following elements:
#' \describe{
#' \item{`x`}{Vector of observed values for the input variable.}
#' \item{`y`}{Vector of observed values for the output variable (if originally
#' present, observations with `is.na(y)` or `weights == 0` are dropped).}
#' \item{`weights`}{Vector of weights for the observed outputs.}
#' \item{`k`}{Degree of the trend filtering estimate.}
#' \item{`lambda`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edf`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambda`.}
#' \item{`fitted_values`}{The fitted values of the trend filtering estimate(s).
#' If `length(lambda) == 1`, fitted values for the single fit are returned as a
#' numeric vector. Otherwise, fitted values are returned in a matrix with
#' `length(lambda)` columns, with `fitted_values[,i]` corresponding to the trend
#' filtering estimate with hyperparameter `lambda[i]`.}
#' \item{`admm_params`}{A list of the parameter values used by the ADMM
#' algorithm used to solve the trend filtering convex optimization.}
#' \item{`obj_func`}{The relative change in the objective function over the
#' ADMM algorithm's final iteration, for every hyperparameter value in
#' `lambda`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every hyperparameter value in `lambda`. If an element of `n_iter` is exactly
#' equal to `admm_params$max_iter`, then the ADMM algorithm stopped before
#' reaching the objective tolerance `admm_params$obj_tol`. In these situations,
#' you may need to increase the maximum number of tolerable iterations by
#' passing a `max_iter` argument to `cv_trendfilter()` in order to ensure that
#' the ADMM solution has converged to satisfactory precision.}
#' \item{`status`}{For internal use. Output from the C solver.}
#' \item{`call`}{The function call.}
#' \item{`scale`}{For internal use.}
#' }
#'
#' @references
#' 1. Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#'    for time-domain astronomy and astronomical spectroscopy. *MNRAS*, 492(3),
#'    p. 4005-4018.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#'    [[arXiv](https://arxiv.org/abs/1908.07151)].
#' 2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#'    signals with varying degrees of smoothness. *MNRAS*, 492(3), p. 4019-4032.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)]
#'    [[arXiv](https://arxiv.org/abs/2001.03552)].
#'
#' @examples
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' fit <- .trendfilter(
#'   x,
#'   y,
#'   weights,
#'   lambda = exp(10),
#'   obj_tol = 1e-6,
#'   max_iter = 1e4
#' )

#' @importFrom glmgen .tf_thin .tf_fit .tf_predict
#' @importFrom dplyr tibble filter mutate select arrange case_when group_split
#' @importFrom dplyr bind_rows
#' @importFrom tidyr drop_na
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom matrixStats rowSds
#' @importFrom stats median
#' @aliases dot-trendfilter
#' @export
.trendfilter <- function(x,
                         y,
                         weights = NULL,
                         lambda,
                         edf = NULL,
                         k = 2L,
                         obj_tol = 1e-10,
                         max_iter = length(y),
                         ...) {
  tf_call <- match.call()
  extra_args <- list(...)

  if (!is.null(edf)) {
    stop(
      "Functionality for specifying trend filtering models via `edf` is ",
      "not yet available. \nPlease use `lambda` instead."
    )
  }

  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  stopifnot(is.numeric(k) && length(k) == 1 && k == round(k))
  k %<>% as.integer()
  if (!any(k == 0:2)) stop("`k` must be equal to 0, 1, or 2.")

  n <- length(y)
  weights <- weights %||% rep_len(1, n)
  stopifnot(is.numeric(weights))
  stopifnot(length(weights) %in% c(1L, n))
  stopifnot(all(weights >= 0L))
  if (length(weights) == 1) weights <- rep_len(weights, n)

  if (missing(lambda)) {
    stop("Must pass at least one hyperparameter value to `lambda`.")
  } else {
    stopifnot(is.numeric(lambda))
    stopifnot(all(lambda >= 0L))

    if (any(duplicated.default(lambda))) {
      warning(
        "Duplicated values passed to `lambda`. Retaining only unique values.",
        call. = FALSE
      )
      lambda %<>%
        unique.default() %>%
        sort.default(decreasing = TRUE)
    } else {
      lambda %<>%
        sort.default(decreasing = TRUE)
    }
  }

  stopifnot(is.numeric(obj_tol) & obj_tol > 0L & length(obj_tol) == 1L)
  stopifnot(is.numeric(max_iter) & max_iter == round(max_iter))
  stopifnot(length(max_iter) == 1L)

  dat <- tibble(
    x = as.double(x),
    y = as.double(y),
    weights = as.double(weights)
  ) %>%
    drop_na() %>%
    arrange(x) %>%
    filter(weights > 0)

  rm(x, y, weights)
  n <- nrow(dat)

  if ("scaling" %in% names(extra_args)) {
    scaling <- extra_args$scaling
    extra_args$scaling <- NULL
  } else {
    scaling <- TRUE
  }

  if (!scaling) {
    x_scale <- 1
    y_scale <- 1
  } else {
    x_scale <- median(diff(dat$x))
    y_scale <- median(abs(dat$y)) / 10
  }

  dat_scaled <- dat %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    )

  admm_params <- get_admm_params(obj_tol, max(max_iter, n, 200L))
  admm_params$x_tol <- admm_params$x_tol / x_scale

  if (min(diff(dat_scaled$x)) <= admm_params$x_tol) {
    thin_out <- .tf_thin(
      x = dat_scaled$x,
      y = dat_scaled$y,
      weights = dat_scaled$weights,
      k = k,
      admm_params = admm_params
    )

    dat_scaled <- tibble(x = thin_out$x, y = thin_out$y, weights = thin_out$w)
  }

  fit <- .tf_fit(
    x = dat_scaled$x,
    y = dat_scaled$y,
    weights = dat_scaled$weights,
    k = k,
    admm_params = admm_params,
    lambda = lambda
  )

  scale <- c(x_scale, y_scale)
  names(scale) <- c("x","y")

  invisible(
    structure(
      list(
        x = dat_scaled$x * x_scale,
        y = dat_scaled$y * y_scale,
        weights = dat_scaled$weights / y_scale^2,
        k = as.integer(k),
        lambda = lambda,
        edf = as.integer(fit$df),
        fitted_values = drop(fit$beta) * y_scale,
        admm_params = admm_params,
        obj_func = drop(fit$obj),
        n_iter = as.integer(fit$iter),
        status = fit$status,
        call = tf_call,
        scale = scale
      ),
      class = c("trendfilter", "trendfiltering")
    )
  )
}


#' Fit a trend filtering model (back-end function with more options for expert
#' users)
#'
#' Fit a trend filtering model. Generic functions such as [`predict()`],
#' [`fitted.values()`], and [`residuals()`] may be called on the
#' [`trendfilter()`] output.
#'
#' @param x
#'   Vector of observed values for the input variable.
#' @param y
#'   Vector of observed values for the output variable.
#' @param weights
#'   Weights for the output measurements. Output weights are defined as the
#'   inverse variance of the additive noise that contaminates the output signal.
#'   When the noise is expected to have a constant variance \mjseqn{\sigma^2}
#'   over all outputs, a scalar may be passed to `weights`, i.e.
#'   `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must be a vector with
#'   the same length as `x` and `y`.
#' @param lambda
#'   One or more hyperparameter values to fit a trend filtering estimate for.
#' @param edf
#'   (Not yet available) Alternative hyperparametrization for the trend
#'   filtering model(s). Vector of the desired number of effective degrees of
#'   freedom in each model.
#' @param ...
#'   Additional named arguments to pass to the internal/expert function
#'   [`.trendfilter()`].
#'
#' @return An object of class `'trendfilter'`.
#'
#' @references
#' 1. Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#'    for time-domain astronomy and astronomical spectroscopy. *MNRAS*, 492(3),
#'    p. 4005-4018.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#'    [[arXiv](https://arxiv.org/abs/1908.07151)].
#' 2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#'    signals with varying degrees of smoothness. *MNRAS*, 492(3), p. 4019-4032.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)]
#'    [[arXiv](https://arxiv.org/abs/2001.03552)].
#'
#' @examples
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' fit <- trendfilter(x, y, weights, lambda = exp(10))

#' @export
trendfilter <- function(x,
                        y,
                        weights = NULL,
                        lambda,
                        edf = NULL,
                        ...) {
  extra_args <- list(...)
  do.call(
    .trendfilter,
    c(
      list(x = x, y = y, weights = weights, lambda = lambda, edf = edf),
      extra_args
    )
  )
}
