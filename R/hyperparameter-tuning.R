#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' \loadmathjax For every candidate hyperparameter value, estimate the trend
#' filtering model's out-of-sample error by \mjseqn{V}-fold cross validation.
#' Many common regression loss functions are defined internally, and each
#' returns its own cross validation curve. Custom loss functions may also be
#' passed via the `user_loss_funcs` argument. See the *Loss functions* section
#' below for definitions of the internal loss functions, and for guidelines on
#' when [`cv_trendfilter()`] should be used versus [`sure_trendfilter()`]. The
#' [`cv_trendfilter()`] output object is compatible with generic
#' [`stats`][stats::stats-package] functions such as [`predict()`],
#' [`fitted.values()`], [`residuals()`], etc.
#'
#' @param x
#'   Vector of observed values for the input variable.
#' @param y
#'   Vector of observed values for the output variable.
#' @param weights
#'   (Optional) Weights for the observed outputs, defined as the reciprocal
#'   variance of the additive noise that contaminates the output signal. When
#'   the noise is expected to have an equal variance \mjseqn{\sigma^2} for all
#'   observations, a scalar may be passed to `weights`, i.e.
#'   `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must be a vector with
#'   the same length as `x` and `y`.
#' @param nlambdas
#'   Number of hyperparameter values to test during cross validation. Defaults
#'   to `nlambdas = 250`. The hyperparameter grid is internally constructed to
#'   span the full trend filtering model space (which is bookended by a global
#'   polynomial solution and an interpolating solution), with `nlambdas`
#'   controlling the granularity of the hyperparameter grid.
#' @param V
#'   Number of folds that the data are partitioned into for V-fold cross
#'   validation. Must be at least 2 and no greater than 10. Defaults to
#'   `V = 10`.
#' @param user_loss_funcs
#'   (Optional) A named list of one or more functions, with each defining a loss
#'   function to be evaluated during cross validation. See the
#'   ''Loss functions'' section below for an example.
#' @param fold_ids
#'   (Optional) An integer vector defining a custom partition of the data for
#'   cross validation. `fold_ids` must have the same length as `x` and `y`, and
#'   only contain integer values `1`, ..., `V` designating the fold assignments.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to
#'   `mc_cores = V`.
#' @param ... Additional named arguments to pass to [`trendfilter()`].
#'
#' @details Our recommendations for when to use [`cv_trendfilter()`] versus
#' [`sure_trendfilter()`] are summarized in the table below. See Section 3.5 of
#' [Politsch et al. (2020a)](https://arxiv.org/abs/1908.07151) for more details.
#'
#' | Scenario                                                                  |  Hyperparameter optimization  |
#' | :---                                                                      |                         :---: |
#' | `x` is unevenly sampled                                                   |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and measurement variances for `y` are not available |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and measurement variances for `y` are available     |      [`sure_trendfilter()`]   |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in wide consecutive chunks) is still considered to
#' be evenly sampled. When `x` is evenly sampled on a transformed scale, we
#' recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See the [`sure_trendfilter()`] examples for
#' a case when the inputs are evenly sampled on the `log10(x)` scale.
#'
#' @section Loss functions:
#'
#' The following loss functions are automatically computed during cross
#' validation and their CV error curves are returned in the `errors` list
#' within the [`cv_trendfilter()`] output.
#'
#' 1. Mean absolute deviations error: \mjsdeqn{\text{MAE}(\lambda) =
#' \frac{1}{n} \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|}
#'
#' 2. Weighted mean absolute deviations error:
#' \mjsdeqn{\text{WMAE}(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#'
#' 3. Mean-squared error: \mjsdeqn{\text{MSE}(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|^2}
#'
#' 4. Weighted mean-squared error: \mjsdeqn{\text{WMSE}(\lambda)
#' = \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#'
#' 5. log-cosh error: \mjsdeqn{\text{logcosh}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}
#' \log\left(\cosh\left(Y_i - \hat{f}(x_i; \lambda)\right)\right)}
#'
#' 6. Weighted log-cosh error: \mjsdeqn{\text{wlogcosh}(\lambda) =
#' \sum_{i=1}^{n}
#' \log\left(\cosh\left((Y_i - \hat{f}(x_i; \lambda))\sqrt{w_i}\right)\right)}
#'
#' 7. Huber loss: \mjsdeqn{\text{Huber}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}L_{\lambda}(Y_i; \delta)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\lambda}(Y_i; \delta) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)| - \delta^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| > \delta$}}
#'
#' 8. Weighted Huber loss: \mjsdeqn{\text{wHuber}(\lambda) =
#' \sum_{i=1}^{n}L_{\lambda}(Y_i; \delta)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\lambda}(Y_i; \delta) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2w_i, &
#' $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} -
#' \delta^2, & $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} > \delta$}}
#'
#' 9. Mean-squared logarithmic error: \mjsdeqn{\text{MSLE}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}
#' \left|\log(Y_i + 1) - \log(\hat{f}(x_i; \lambda) + 1)\right|}
#'
#' where \mjseqn{w_i:=} `weights[i]`.
#'
#' When defining custom loss functions, each function within the named list
#' passed to `user_loss_funcs` should take three vector arguments --- `y`,
#' `tf_estimate`, and `weights` --- and return a single scalar value for the
#' validation loss. For example, if I wanted to optimize the hyperparameter by
#' minimizing an uncertainty-weighted median of the model's MAD errors, I would
#' pass the list below to `user_loss_funcs`:
#' ```{r, eval = FALSE}
#' MedAE <- function(tf_estimate, y, weights) {
#'   matrixStats::weightedMedian(abs(tf_estimate - y), sqrt(weights))
#' }
#'
#' my_loss_funcs <- list(MedAE = MedAE)
#' ```
#'
#' @return An object of class `'cv_trendfilter'` and subclass `'trendfilter'`.
#' This is a list with the following elements:
#' \describe{
#' \item{`lambdas`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edfs`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every candidate hyperparameter value in `lambdas`.}
#' \item{`errors`}{A named list of vectors, with each representing the
#' CV error curve for every loss function in `loss_funcs` (see below).}
#' \item{`se_errors`}{Standard errors for the `errors`, within a named list of
#' the same structure.}
#' \item{`lambda_min`}{A named vector with length equal to `length(errors)`,
#' containing the hyperparameter value that minimizes the CV error curve, for
#' every loss function in `loss_funcs`.}
#' \item{`lambda_1se`}{A named vector with length equal to `length(errors)`,
#' containing the "1-standard-error rule" hyperparameter, for every loss
#' function in `loss_funcs`. The "1-standard-error rule" hyperparameter is the
#' largest hyperparameter value in `lambdas` that has a CV error within one
#' standard error of `min(errors)`. It serves as an Occam's razor-like
#' heuristic. More precisely, given two models with approximately equal
#' performance (in terms of some loss function), it may be wise to opt for the
#' simpler model, i.e. the model with the larger hyperparameter value / fewer
#' effective degrees of freedom.}
#' \item{`edf_min`}{A named vector with length equal to `length(errors)`,
#' containing the number of effective degrees of freedom in the trend filtering
#' estimator that minimizes the CV error curve, for every loss function in
#' `loss_funcs`.}
#' \item{`edf_1se`}{A named vector with length equal to `length(errors)`,
#' containing the number of effective degrees of freedom in the
#' "1-standard-error rule" trend filtering estimator, for every loss function in
#' `loss_funcs`.}
#' \item{`i_min`}{A named vector with length equal to `length(errors)`,
#' containing the index of `lambdas` that minimizes the CV error curve, for
#' every loss function in `loss_funcs`.}
#' \item{`i_1se`}{A named vector with length equal to `length(errors)`,
#' containing the index of `lambdas` that gives the "1-standard-error rule"
#' hyperparameter value, for every loss function in `loss_funcs`.}
#' \item{`obj_func`}{The relative change in the objective function over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambdas`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambdas`. If an element of `n_iter`
#' is exactly equal to `model$admm_params$max_iter` (see below), then the ADMM
#' algorithm stopped before reaching the objective tolerance
#' `model$admm_params$obj_tol`. In these situations, you may need to increase
#' the maximum number of tolerable iterations via the
#' `optimization_params$max_iter` argument of `cv_trendfilter()` in order to
#' ensure that the ADMM solution has converged to satisfactory precision.}
#' \item{`loss_funcs`}{A named list of functions that defines all loss functions
#' evaluated during cross validation.}
#' \item{`V`}{The number of folds the data were split into for cross
#' validation.}
#' \item{`x`}{Vector of observed values for the input variable.}
#' \item{`y`}{Vector of observed values for the output variable (if originally
#' present, observations with `is.na(y)` or `weights == 0` are dropped).}
#' \item{`weights`}{Vector of weights for the observed outputs.}
#' \item{`model`}{A list containing the trend filtering model fit object, the
#' ADMM parameter settings, and other modeling objects that are useful to pass
#' along to functions that operate on the `cv_trendfilter()` output.}
#' }
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
#'
#' @examples
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' cv_tf <- cv_trendfilter(x, y, weights,
#'   optimization_params = list(
#'     max_iter = 1e4,
#'     obj_tol = 1e-6
#'   )
#' )
#' @useDynLib glmgen thin_R tf_R
#' @importFrom dplyr filter mutate select arrange case_when group_split bind_rows
#' @importFrom tidyr tibble drop_na
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom stringi stri_wrap
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom stats median sd
#' @export
cv_trendfilter <- function(x,
                           y,
                           weights = NULL,
                           nlambdas = 250L,
                           V = 10L,
                           mc_cores = V,
                           user_loss_funcs = NULL,
                           fold_ids = NULL,
                           ...) {
  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  extra_args <- list(...)

  if (any(names(extra_args) == "k")) {
    stopifnot(is.numeric(k) && k == round(k))
    stopifnot(length(k) == 1)
    if (!k %in% 0:2) stop("`k` must be equal to 0, 1, or 2.")
  } else {
    k <- 2L
  }

  V != round(V)
  if (V < 2L || V > 10L) stop("Must have `V >= 2` and `V <= 10`.")

  if (missing(weights)) weights <- rep_len(1, length(y))
  if (length(weights) == 1) weights <- rep_len(weights, length(y))

  if (!(class(weights) %in% c("numeric", "integer"))) {
    stop("`weights` must be a numeric vector with the same length as `y`.")
  }

  # All weights must be strictly greater than zero.

  if (nlambdas < 100 || nlambdas != round(nlambdas)) {
    stop("Must have `nlambdas >= 100`.")
  } else {
    nlambdas %<>% as.integer()
  }

  mc_cores <- max(c(1, floor(mc_cores)))
  mc_cores <- min(c(detectCores(), V, mc_cores))

  if (mc_cores < V & mc_cores < detectCores() / 2) {
    warning(
      stri_wrap(
        paste(
          "Your machine has", detectCores(), "cores, but you've only configured
          `cv_trendfilter()` to use", mc_cores, "of them. Letting `mc_cores = V`
          can significantly speed up the cross validation."
        ),
        prefix = " "
      ),
      call. = FALSE, immediate. = TRUE
    )
  }

  if (missing(loss_funcs)) {
    loss_funcs <- list(
      MAE = MAE,
      WMAE = WMAE,
      MSE = MSE,
      WMSE = WMSE,
      logcosh = logcosh,
      wlogcosh = wlogcosh,
      Huber = Huber,
      wHuber = wHuber,
      MSLE = MSLE
    )
  } else {
    if (class(loss_funcs) == "function") {
      stop("Custom loss function(s) must be passed within a named list.")
    }

    if (class(loss_funcs) != "list") {
      stop("Custom loss function(s) must be passed within a named list.")
    }

    if (class(loss_funcs) == "list") {
      if (is.null(names(loss_funcs)) ||
        any(names(loss_funcs) == "")) {
        stop(
          "Please name each of the functions in your `loss_funcs` list."
        )
      }

      for (X in 1:length(loss_funcs)) {
        if (!all(c("y", "tf_estimate", "weights") %in%
          names(formals(loss_funcs[[X]])))) {
          stop(
            stri_wrap(
              paste0(
                "Incorrect input argument structure for the function
                `loss_funcs[[", X, "]]`. Each custom loss function should have
                vector input arguments `y`, `tf_estimate`, `weights`, and then
                compute and return a scalar value for the validation error."
              ),
              prefix = " "
            )
          )
        }
      }
      loss_funcs <- c(
        list(
          MAE = MAE,
          WMAE = WMAE,
          MSE = MSE,
          WMSE = WMSE,
          logcosh = logcosh,
          wlogcosh = wlogcosh,
          Huber = Huber,
          wHuber = wHuber,
          MSLE = MSLE
        ),
        loss_funcs
      )
    }
  }

  if (missing(optimization_params)) {
    optimization_params <- NULL
  }

  opt_params <- get_optimization_params(optimization_params, n = length(x))
  optimization_params <- opt_params$optimization_params
  thinning <- opt_params$thinning
  admm_params <- do.call(trendfilter.control.list, optimization_params)

  x %<>% as.double()
  y %<>% as.double()
  weights %<>% as.double()
  k %<>% as.integer()
  V %<>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()

  x_scale <- data$x %>%
    diff() %>%
    median()
  y_scale <- median(abs(data$y)) / 10
  admm_params$x_tol <- admm_params$x_tol / x_scale

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    ) %>%
    select(x, y, weights)

  if (missing(fold_ids)) {
    fold_ids <- sample(rep_len(1:V, nrow(data_scaled)))
  } else {
    fold_ids %<>% as.integer() %>%
      unique.default() %>%
      sort.int()
    if (!all.equal(, 1:V)) {
      stop(
        stri_wrap(
          "`fold_ids` should only contain integer values 1:V, with no empty
          folds.",
          prefix = " "
        )
      )
    }

    if (length(fold_ids) == length(x)) {
      stop("")
    }
    stopifnot()

    counts <- table(fold_ids) %>%
      as.double() %>%
      sort()
    equal_counts <- table(rep_len(1:V, nrow(data_scaled))) %>%
      as.double() %>%
      sort()

    if (!all.equal(counts, equal_counts) & nrow(data_scaled) %% V == 0) {
      warning("Your cross validation folds are imbalanced.")
    }

    if (!all.equal(counts, equal_counts) & nrow(data_scaled) %% V != 0) {
      warning(
        stri_wrap(
          "Your cross validation folds are imbalanced, beyond what is simply due
          to `length(x)` not being divisible by `V`.",
          prefix = " "
        )
      )
    }
  }

  data_folded <- data_scaled %>%
    mutate(ids = fold_ids) %>%
    group_split(ids, .keep = FALSE)

  lambdas <- get_lambdas(nlambdas, data_scaled, k, thinning, admm_params)

  cv_out <- mclapply(
    1:V,
    FUN = validate_fold,
    data_folded = data_folded,
    lambdas = lambdas,
    k = k,
    thinning = thinning,
    admm_params = admm_params,
    loss_funcs = loss_funcs,
    y_scale = y_scale,
    mc.cores = mc_cores
  )

  cv_loss_mats <- lapply(
    X = 1:length(loss_funcs),
    FUN = function(X) {
      lapply(
        1:length(cv_out),
        FUN = function(itr) cv_out[[itr]][[X]]
      ) %>%
        unlist() %>%
        matrix(ncol = V)
    }
  )

  errors <- lapply(
    1:length(cv_loss_mats),
    FUN = function(X) {
      cv_loss_mats[[X]] %>%
        rowMeans() %>%
        as.double()
    }
  )

  i_min <- lapply(
    1:length(errors),
    FUN = function(X) {
      errors[[X]] %>%
        which.min() %>%
        min()
    }
  ) %>% unlist()

  se_errors <- lapply(
    1:length(errors),
    FUN = function(X) {
      rowSds(cv_loss_mats[[X]]) / sqrt(V) %>%
        as.double()
    }
  )

  i_1se <- lapply(
    X = 1:length(errors),
    FUN = function(X) {
      which(errors[[X]] <= errors[[X]][i_min[X]] + se_errors[[X]][i_min[X]]) %>%
        min()
    }
  ) %>% unlist()

  lambda_min <- lambdas[i_min]
  lambda_1se <- lambdas[i_1se]

  fit <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  edf_min <- fit$df[i_min] %>% as.integer()
  edf_1se <- fit$df[i_1se] %>% as.integer()

  names(lambda_min) <- names(loss_funcs)
  names(lambda_1se) <- names(loss_funcs)
  names(edf_min) <- names(loss_funcs)
  names(edf_1se) <- names(loss_funcs)
  names(errors) <- names(loss_funcs)
  names(i_min) <- names(loss_funcs)
  names(se_errors) <- names(loss_funcs)
  names(i_1se) <- names(loss_funcs)

  model <- structure(
    list(
      fit = fit,
      k = k,
      admm_params = admm_params,
      thinning = thinning,
      data_scaled = data_scaled,
      x_scale = x_scale,
      y_scale = y_scale
    ),
    class = c("tf_model", "list")
  )

  structure(
    list(
      lambdas = lambdas,
      edfs = fit$df %>% as.integer(),
      errors = errors,
      se_errors = se_errors,
      lambda_min = lambda_min,
      lambda_1se = lambda_1se,
      edf_min = edf_min,
      edf_1se = edf_1se,
      i_min = i_min,
      i_1se = i_1se,
      obj_func = fit$obj[nrow(fit$obj), ],
      n_iter = fit$iter %>% as.integer(),
      loss_funcs = loss_funcs,
      V = V,
      x = data$x,
      y = data$y,
      weights = data$weights,
      model = model
    ),
    class = c("cv_trendfilter", "trendfilter", "trendfiltering", "list")
  )
}


#' @importFrom dplyr bind_rows
#' @importFrom purrr map
validate_fold <- function(fold_id,
                          data_folded,
                          lambdas,
                          loss_funcs,
                          k,
                          admm_params,
                          y_scale) {

  data_train <- data_folded[-fold_id] %>% bind_rows()
  data_validate <- data_folded[[fold_id]]

  out <- trendfilter(
    x = data_train$x,
    y = data_train$y,
    weights = data_train$weights,
    k = k,
    lambda = lambdas,
    thinning = thinning,
    control = admm_params
  )

  tf_validate_preds <- glmgen:::predict.trendfilter(
    out,
    lambda = lambdas,
    x.new = data_validate$x
  ) %>%
    suppressWarnings()

  lapply(
    X = 1:length(loss_funcs),
    FUN = function(X) {
      apply(
        tf_validate_preds * y_scale,
        2,
        loss_funcs[[X]],
        y = data_validate$y * y_scale,
        weights = data_validate$weights / y_scale^2
      ) %>%
        as.double()
    }
  )
}


get_internal_loss_funcs <- function() {

  MAE <- function(tf_estimate, y, weights) {
    mean(abs(tf_estimate - y))
  }

  WMAE <- function(tf_estimate, y, weights) {
    sum(abs(tf_estimate - y) * sqrt(weights) / sum(sqrt(weights)))
  }

  MSE <- function(tf_estimate, y, weights) {
    mean((tf_estimate - y)^2)
  }

  WMSE <- function(tf_estimate, y, weights) {
    sum((tf_estimate - y)^2 * weights / sum(weights))
  }

  logcosh <- function(tf_estimate, y, weights) {
    mean(log(cosh(y - tf_estimate)))
  }

  wlogcosh <- function(tf_estimate, y, weights) {
    std_residuals <- (y - tf_estimate) * sqrt(weights)
    sum(log(cosh(std_residuals)))
  }

  #' @importFrom stats sd
  Huber <- function(tf_estimate, y, weights, n_stderr = 3) {
    stderr <- sd(y - tf_estimate)
    delta <- n_stderr * stderr
    sum(
      sapply(X = 1:length(y), FUN = function(X) {
        ifelse(
          abs(y[X] - tf_estimate[X]) <= delta,
          (y[X] - tf_estimate[X])^2,
          2 * delta * mean(abs(y[X] - tf_estimate[X])) - delta^2
        )
      })
    )
  }

  wHuber <- function(tf_estimate, y, weights, delta = 3) {
    sum(
      sapply(X = 1:length(y), FUN = function(X) {
        ifelse(
          abs(y[X] - tf_estimate[X]) * sqrt(weights[X]) <= delta,
          (y[X] - tf_estimate[X])^2 * weights[X],
          2 * delta * abs(y[X] - tf_estimate[X]) * sqrt(weights[X]) - delta^2
        )
      })
    )
  }

  MSLE <- function(tf_estimate, y, weights) {
    offset <- min(c(y, 0))
    mean((log(tf_estimate - offset + 1) - log(y - offset + 1))^2)
  }

  list(
    MAE = MAE,
    WMAE = WMAE,
    MSE = MSE,
    WMSE = WMSE,
    logcosh = logcosh,
    wlogcosh = wlogcosh,
    Huber = Huber,
    wHuber = wHuber,
    MSLE = MSLE
  )
}


#' Optimize the trend filtering hyperparameter by minimizing Stein's unbiased
#' risk estimate
#'
#' For every candidate hyperparameter value, compute an unbiased estimate of the
#' trend filtering model's predictive mean-squared error. See the details
#' section for guidelines on when [`sure_trendfilter()`] should be used versus
#' [`cv_trendfilter()`]. Generic functions like `predict()` and `residuals()`
#' work on `sure_trendfilter` output.
#'
#' @inheritParams cv_trendfilter
#'
#' @details Our recommendations for when to use [`sure_trendfilter()`] versus
#' [`cv_trendfilter()`] are summarized in the table below. See Section 3.5 of
#' [Politsch et al. (2020a)](https://arxiv.org/abs/1908.07151) for more details.
#'
#' | Scenario                                                                  |  Hyperparameter optimization  |
#' | :---                                                                      |                         :---: |
#' | `x` is unevenly sampled                                                   |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and measurement variances for `y` are not available |      [`cv_trendfilter()`]     |
#' | `x` is evenly sampled and measurement variances for `y` are available     |      [`sure_trendfilter()`]   |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in wide consecutive chunks) is still considered to
#' be evenly sampled. When `x` is evenly sampled on a transformed scale, we
#' recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See the [`sure_trendfilter()`] examples for
#' a case when the inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class `'sure_trendfilter'` and subclass `'trendfilter'`.
#' This is a list with the following elements:
#' \describe{
#' \item{`lambdas`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edfs`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambdas`.}
#' \item{`errors`}{Vector of mean-squared prediction errors estimated by SURE,
#' for every hyperparameter value in `lambdas`.}
#' \item{`se_errors`}{Vector of estimated standard errors for the `errors`.}
#' \item{`lambda_min`}{Hyperparameter value in `lambdas` that minimizes the SURE
#' validation error curve.}
#' \item{`lambda_1se`}{The largest hyperparameter value in `lambdas` that has a
#' SURE error within one standard error of `min(errors)`. We call this the
#' "1-standard-error rule" hyperparameter, and it serves as an Occam's
#' razor-esque heuristic. More precisely, given two models with approximately
#' equal performance (here, in terms of predictive MSE), it may be wise to opt
#' for the simpler model, i.e. the model with the larger hyperparameter value /
#' fewer effective degrees of freedom.}
#' \item{`edf_min`}{Number of effective degrees of freedom in the trend
#' filtering estimator with hyperparameter `lambda_min`.}
#' \item{`edf_1se`}{Number of effective degrees of freedom in the  trend
#' filtering estimator with hyperparameter `lambda_1se`.}
#' \item{`i_min`}{Index of `lambdas` that gives `lambda_min`.}
#' \item{`i_1se`}{Index of `lambdas` that gives `lambda_1se`.}
#' \item{`obj_func`}{The relative change in the objective function over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambdas`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambdas`. If an element of `n_iter`
#' is exactly equal to `model$admm_params$max_iter` (see below), then the ADMM
#' algorithm stopped before reaching the objective tolerance
#' `model$admm_params$obj_tol`. In these situations, you may need to increase
#' the maximum number of tolerable iterations via the
#' `optimization_params$max_iter` argument of `sure_trendfilter()` in order to
#' ensure that the ADMM solution has converged to satisfactory precision.}
#' \item{`training_errors`}{The "in-sample" MSE between the observed outputs `y`
#' and the trend filtering estimate, for every hyperparameter value in
#' `lambdas`.}
#' \item{`optimisms`}{SURE-estimated optimisms, i.e.
#' `optimisms = errors - training_errors`.}
#' \item{`x`}{Vector of observed values for the input variable.}
#' \item{`y`}{Vector of observed values for the output variable (if originally
#' present, observations with `is.na(y)` or `weights == 0` are dropped).}
#' \item{`weights`}{Vector of weights for the observed outputs.}
#' \item{`model`}{A list containing the trend filtering model fit object, the
#' ADMM parameter settings, and other modeling objects that are useful to pass
#' along to functions that operate on the `sure_trendfilter()` output.}
#' }
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
#'
#' @examples
#' data(quasar_spectrum)
#' head(spec)
#'
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' @importFrom dplyr filter mutate select arrange case_when group_split bind_rows
#' @importFrom tidyr tibble drop_na
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom stringi stri_wrap
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom stats median sd
#' @export
sure_trendfilter <- function(x,
                             y,
                             weights,
                             nlambdas = 250L,
                             ...) {
  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  extra_args <- list(...)

  if (any(names(extra_args) == "k")) {
    stopifnot(is.numeric(k) && k == round(k))
    stopifnot(length(k) == 1)
    if (!k %in% 0:2) stop("`k` must be equal to 0, 1, or 2.")
  } else {
    k <- 2L
  }

  if (nlambdas < 100 || nlambdas != round(nlambdas)) {
    stop("`nlambdas` must be an integer >=100.")
  } else {
    nlambdas %<>% as.integer()
  }

  if (missing(weights) || !(class(weights) %in% c("numeric", "integer"))) {
    stop("`weights` must be passed for `sure_trendfilter()`.")
  }

  if (length(weights) == 1) weights <- rep_len(weights, length(y))

  if (!(class(weights) == "numeric") | !(length(weights) %in% c(1, length(y)))) {
    stop("`weights` must be a numeric vector with length 1 or `length(y)`.")
  }

  x %<>% as.double()
  y %<>% as.double()
  weights %<>% as.double()
  k %<>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()

  if (missing(optimization_params)) optimization_params <- NULL
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

  fit <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  squared_residuals_mat <- (fit$beta - data_scaled$y)^2

  optimisms_mat <- 2 / (data_scaled$weights * nrow(data)) *
    matrix(rep(fit$df, each = nrow(data)), nrow = nrow(data))

  errors_mat <- (squared_residuals_mat + optimisms_mat) * y_scale^2
  errors <- errors_mat %>% colMeans()
  i_min <- min(which.min(errors)) %>% as.integer()

  se_errors <- replicate(
    5000,
    errors_mat[sample(1:nrow(data_scaled), replace = TRUE), ] %>%
      colMeans()
  ) %>%
    rowSds()

  i_1se <- which(
    errors <= errors[i_min] + se_errors[i_min]
  ) %>% min()

  model <- structure(
    list(
      fit = fit,
      k = k,
      admm_params = admm_params,
      thinning = thinning,
      data_scaled = data_scaled,
      x_scale = x_scale,
      y_scale = y_scale
    ),
    class = c("tf_model", "list")
  )

  invisible(
    structure(
      list(
        lambdas = lambdas,
        edfs = fit$df %>% as.integer(),
        errors = errors,
        se_errors = se_errors,
        lambda_min = lambdas[i_min],
        lambda_1se = lambdas[i_1se],
        edf_min = fit$df[i_min] %>% as.integer(),
        edf_1se = fit$df[i_1se] %>% as.integer(),
        i_min = i_min,
        i_1se = i_1se,
        obj_func = fit$obj[nrow(fit$obj), ],
        n_iter = fit$iter %>% as.integer(),
        x = data$x,
        y = data$y,
        weights = data$weights,
        model = model
      ),
      class = c("sure_trendfilter", "trendfilter", "trendfiltering", "list")
    )
  )
}
