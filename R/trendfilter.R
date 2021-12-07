#' Fit a trend filtering model
#'
#' Fit a trend filtering model.
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
#' fit <- .trendfilter(x,
#'                     y,
#'                     weights,
#'                     lambda = exp(2),
#'                     obj_tol = 1e-6,
#'                     max_iter = 1e4
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
        "Duplicated values passed to `lambda`. ",
        "Retaining only unique values."
      )
      lambda %<>% unique.default() %>% sort.default(decreasing = TRUE)
    } else {
      lambda %<>% sort.default(decreasing = TRUE)
    }
  }

  stopifnot(is.numeric(obj_tol) & obj_tol > 0L & length(obj_tol) == 1L)
  stopifnot(is.numeric(max_iter) & max_iter == round(max_iter))
  stopifnot(length(max_iter) == 1L)

  df <- tibble(x = as.double(x),
               y = as.double(y),
               weights = as.double(weights)) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()

  x_scale <- median(diff(df$x))
  y_scale <- median(abs(df$y)) / 10

  df_scaled <- df %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    )

  max_iter <- max(max_iter, nrow(df), 200L)
  admm_params <- get_admm_params(obj_tol, max_iter)
  admm_params$x_tol <- admm_params$x_tol / x_scale

  if (min(diff(df_scaled$x)) <= admm_params$x_tol) {
    thin_out <- .Call("thin_R",
      sX = as.double(df_scaled$x),
      sY = as.double(df_scaled$y),
      sW = as.double(df_scaled$weights),
      sN = nrow(df_scaled),
      sK = as.integer(k),
      sControl = admm_params,
      PACKAGE = "glmgen"
    )

    df_scaled <- tibble(x = thin_out$x, y = thin_out$y, weights = thin_out$w)
  }

  fit <- df_scaled %$% .tf_fit(x, y, weights, k, lambda, admm_params)

  admm_params$x_tol <- admm_params$x_tol * x_scale

  invisible(
    structure(
      list(
        x = df$x,
        y = df$y,
        weights = df$weights,
        k = k,
        lambda = lambda,
        edf = fit$df,
        beta = fit$beta,
        obj_func = fit$obj,
        status = fit$status,
        n_iter = fit$iter,
        admm_params = admm_params,
        call = tf_call
      ),
      class = c("trendfilter", "trendfiltering")
    )
  )
}


#' Fit a trend filtering model
#'
#' Fit a trend filtering model.
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
#' fit <- trendfilter(x, y, weights, lambda = exp(2))

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