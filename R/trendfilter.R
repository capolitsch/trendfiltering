#' Fit a trend filtering model
#'
#' @param x
#'   Vector of observed values for the input variable.
#' @param y
#'   Vector of observed values for the output variable.
#' @param weights
#'   (Optional) Weights for the observed output measurements. The weights are
#'   defined as the inverse variance of the additive noise that contaminates the
#'   observed output signal. When the noise is expected to have an equal
#'   variance \mjseqn{\sigma^2} for all observations, a scalar may be passed to
#'   `weights`, i.e. `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must
#'   be a vector with the same length as `x` and `y`.
#' @param lambdas
#'   Vector of one or more hyperparameter values to fit a model for.
#' @param k
#'   Degree of the polynomials that make up the piecewise-polynomial trend
#'   filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#'   estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#'   disallowed since they yield no predictive benefits over `k = 2` and their
#'   use can lead to instability in the convex optimization.
#' @param obj_tol
#'   (Optional) Stopping criterion for the ADMM algorithm. If the relative
#'   change in the trend filtering objective function between two successive
#'   iterations is less than `obj_tol`, the algorithm terminates. The
#'   algorithm's termination can also result from it reaching the maximum
#'   tolerable iterations set by the `max_iter` parameter below. The `obj_tol`
#'   parameter defaults to `obj_tol = 1e-10`. The `obj_func` vector returned
#'   within the `trendfilter()` output gives the relative change in the trend
#'   filtering objective function over the algorithm's final iteration, for
#'   every candidate hyperparameter value.
#' @param max_iter
#'   (Optional) Maximum number of ADMM iterations that we will tolerate.
#'   Defaults to `max_iter = length(y)`. The actual number of iterations
#'   performed by the algorithm, for every candidate hyperparameter value, is
#'   returned in the `n_iter` vector within the `trendfilter()` output. If any
#'   of the elements of `n_iter` are equal to `max_iter`, the tolerance
#'   defined by `obj_tol` has not been attained and `max_iter` may need to be
#'   increased.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to the number
#'   of cores detected, minus 4.
#' @param ...
#'   Additional named arguments. Currently unused.
#'
#' @return An object of class `'trendfilter'`. This is a list with the following
#' elements:
#'
#' 1. `x`
#' 2. `y`
#'
#' @references
#' 1. Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#'    for time-domain astronomy and astronomical spectroscopy. *MNRAS*, 492(3),
#'    p. 4005-4018.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#'    [[arXiv](https://arxiv.org/abs/1908.07151)]
#'    [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].
#' 2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#'    signals with varying degrees of smoothness. *MNRAS*, 492(3), p. 4019-4032.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)]
#'    [[arXiv](https://arxiv.org/abs/2001.03552)]
#'    [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].

#' @importFrom dplyr filter mutate select arrange case_when group_split
#' @importFrom dplyr bind_rows
#' @importFrom tidyr tibble drop_na
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom stringi stri_wrap
#' @importFrom matrixStats rowSds
#' @importFrom stats median
#' @export
trendfilter <- function(x,
                        y,
                        weights = NULL,
                        lambdas,
                        k = 2L,
                        obj_tol = 1e-10,
                        max_iter = length(y),
                        mc_cores = parallel::detectCores() - 4,
                        ...) {
  tf_call <- match.call()

  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  stopifnot(is.numeric(k) && length(k) == 1 && k == round(k))
  k %<>% as.integer()
  if (!k %in% 0:2) stop("`k` must be equal to 0, 1, or 2.")

  n <- length(y)
  weights <- weights %||% rep_len(1, n)
  stopifnot(is.numeric(weights))
  stopifnot(!length(weights) %in% c(1L, n))
  stopifnot(all(weights >= 0L))
  if (length(weights) == 1) weights <- rep_len(weights, n)

  if (missing(lambdas)) {
    stop("Must provide at least one hyperparameter value to `lambdas`.")
  } else {
    stopifnot(is.numeric(lambdas))
    stopifnot(all(lambdas >= 0L))
  }

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()

  n <- nrow(data)
  x_scale <- median(diff(data$x))
  y_scale <- median(abs(data$y)) / 10

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    ) %>%
    select(x, y, weights)

  max_iter <- max(n, max_iter %||% 200L)
  obj_tol <- obj_tol %||% 1e-10
  admm_params <- get_admm_params(n, obj_tol, max_iter)
  admm_params$x_tol <- admm_params$x_tol / x_scale

  data_scaled <- expert_thin(
    data_scaled$x,
    data_scaled$y,
    data_scaled$weights,
    admm_params
  )

  tf_out <- data_scaled %$% expert_tf(x, y, weights, lambdas, admm_params)

  invisible(
    structure(
      list(
        x = x,
        y = y,
        weights = weights,
        k = k,
        lambdas = tf_out$lambda,
        edfs = tf_out$df,
        beta = tf_out$beta,
        obj = tf_out$obj,
        status = tf_out$status,
        iter = tf_out$iter,
        call = tf_call
      ),
      class = c("trendfilter", "trendfiltering")
    )
  )
}


#' @noRd
get_admm_params <- function(n = NULL, obj_tol = NULL, max_iter = NULL) {
  list(
    obj_tol = obj_tol,
    max_iter = max_iter,
    x_tol = 1e-6,
    rho = 1,
    obj_tol_newton = 1e-5,
    max_iter_newton = 50L,
    alpha_ls = 0.5,
    gamma_ls = 0.8,
    max_iter_ls = 30L,
    tridiag = 0
  )
}


tfMultiply <- function(x, y, k = 2L) {
  z <- .Call("matMultiply_R",
             x = as.numeric(x),
             sB = as.numeric(y),
             sK = as.integer(k),
             sMatrixCode = 0L,
             PACKAGE = "glmgen"
  )

  z[1:(length(z) - k)]
}


#' @useDynLib glmgen thin_R
#' @importFrom tidyr tibble
#' @noRd
expert_thin <- function(x,
                        y,
                        weights,
                        admm_params,
                        k = 2L) {
  mindx <- min(diff(x))

  if (mindx <= admm_params$x_tol) {
    c_thin <- .Call("thin_R",
                    sX = x,
                    sY = y,
                    sW = weights,
                    sN = length(y),
                    sK = k,
                    sControl = admm_params,
                    PACKAGE = "glmgen"
    )

    tibble(x = c_thin$x, y = c_thin$y, weights = c_thin$w)
  } else {
    tibble(x = x, y = y, weights = weights)
  }
}


#' @useDynLib glmgen tf_R
#' @noRd
expert_tf <- function(x,
                      y,
                      weights,
                      lambdas,
                      admm_params,
                      k = 2L) {
  lambda_min_ratio <- 0.5 * max(lambdas) / min(lambdas)

  .Call("tf_R",
        sX = x,
        sY = y,
        sW = weights,
        sN = length(y),
        sK = k,
        sFamily = 0L,
        sMethod = 0L,
        sBeta0 = NULL,
        sLamFlag = 1L,
        sLambda = lambdas,
        sNlambda = length(lambdas),
        sLambdaMinRatio = lambda_min_ratio,
        sVerbose = 0L,
        sControl = admm_params,
        PACKAGE = "glmgen"
  )
}
