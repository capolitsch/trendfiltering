#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' \loadmathjax For every candidate hyperparameter value, estimate the trend
#' filtering model's out-of-sample error by \mjseqn{V}-fold cross validation.
#' Many common regression loss functions are defined internally, and a cross
#' validation curve is returned for each. Custom loss functions may also be
#' passed via the `loss_funcs` argument. See the **Loss functions** section
#' below for definitions of the internal loss functions, and for guidelines on
#' when [`cv_trendfilter()`] should be used versus [`sure_trendfilter()`].
#' Generic functions such as [`predict()`], [`fitted.values()`],
#' and [`residuals()`] may be called on the [`cv_trendfilter()`] output.
#'
#' @param x
#'   Vector of observed values for the input variable.
#' @param y
#'   Vector of observed values for the output variable.
#' @param weights
#'   Weights for the observed outputs, defined as the reciprocal variance of the
#'   additive noise that contaminates the output signal. When the noise is
#'   expected to have an equal variance \mjseqn{\sigma^2} for all observations,
#'   a scalar may be passed to `weights`, i.e. `weights = `\mjseqn{1/\sigma^2}.
#'   Otherwise, `weights` must be a vector with the same length as `x` and `y`.
#' @param nlambda
#'   Number of hyperparameter values to test during cross validation. Defaults
#'   to `nlambda = 250`. The hyperparameter grid is internally constructed to
#'   span the full trend filtering model space (which is bookended by a global
#'   polynomial solution and an interpolating solution), with `nlambda`
#'   controlling the granularity of the hyperparameter grid.
#' @param V
#'   Number of folds that the data are partitioned into for V-fold cross
#'   validation. Must be at least 2 and no more than 10. Defaults to `V = 10`.
#' @param loss_funcs
#'   A named list of one or more functions, with each defining a loss function
#'   to be evaluated during cross validation. See the ''Loss functions'' section
#'   below for an example.
#' @param fold_ids
#'   An integer vector defining a custom partition of the data for cross
#'   validation. `fold_ids` must have the same length as `x` and `y`, and
#'   only contain integer values `1`, ..., `V` designating the fold assignments.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to
#'   `mc_cores = V`.
#' @param ... Additional named arguments to pass to [`.trendfilter()`].
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
#' validation and their CV error curves are returned in the `error` list
#' within the [`cv_trendfilter()`] output.
#'
#' 1. Mean absolute deviations error: \mjsdeqn{MAE(\lambda) =
#' \frac{1}{n} \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|}
#'
#' 2. Weighted mean absolute deviations error:
#' \mjsdeqn{WMAE(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#'
#' 3. Mean-squared error: \mjsdeqn{MSE(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|^2}
#'
#' 4. Weighted mean-squared error: \mjsdeqn{WMSE(\lambda)
#' = \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#'
#' 5. log-cosh error: \mjsdeqn{logcosh(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}
#' \log\left(\cosh\left(Y_i - \hat{f}(x_i; \lambda)\right)\right)}
#'
#' 6. Weighted log-cosh error: \mjsdeqn{wlogcosh(\lambda) =
#' \sum_{i=1}^{n}
#' \log\left(\cosh\left((Y_i - \hat{f}(x_i; \lambda))\sqrt{w_i}\right)\right)}
#'
#' 7. Huber loss: \mjsdeqn{Huber(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}L_{\lambda}(Y_i; \delta)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\lambda}(Y_i; \delta) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)| - \delta^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| > \delta$}}
#'
#' 8. Weighted Huber loss: \mjsdeqn{wHuber(\lambda) =
#' \sum_{i=1}^{n}L_{\lambda}(Y_i; \delta)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\lambda}(Y_i; \delta) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2w_i, &
#' $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} -
#' \delta^2, & $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} > \delta$}}
#'
#' 9. Mean-squared logarithmic error: \mjsdeqn{MSLE(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}
#' \left|\log(Y_i + 1) - \log(\hat{f}(x_i; \lambda) + 1)\right|}
#'
#' where \mjseqn{w_i:=} `weights[i]`.
#'
#' When defining custom loss functions, each function within the named list
#' passed to `loss_funcs` should take three vector arguments --- `y`,
#' `tf_estimate`, and `weights` --- and return a single scalar value for the
#' validation loss. For example, if we wanted to optimize the hyperparameter by
#' minimizing an uncertainty-weighted median of the model's MAD errors, we would
#' pass the list below to `loss_funcs`:
#' ```{r, eval = FALSE}
#' MedAE <- function(tf_estimate, y, weights) {
#'   matrixStats::weightedMedian(abs(tf_estimate - y), sqrt(weights))
#' }
#'
#' my_loss_funcs <- list(MedAE = MedAE)
#' ```
#'
#' @return An object of class `'cv_trendfilter'` and subclass `'trendfilter'`.
#' This is a list with the elements below,
#' as well as all elements from the '[`trendfilter`][`trendfilter()`]' call.
#' \describe{
#' \item{`lambda`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edf`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every candidate hyperparameter value in `lambda`.}
#' \item{`error`}{A named list of vectors, with each representing the
#' CV error curve for every loss function in `loss_funcs` (see below).}
#' \item{`se_error`}{Standard errors for the `error`, within a named list of
#' the same structure.}
#' \item{`lambda_min`}{A named vector with length equal to `length(lambda)`,
#' containing the hyperparameter value that minimizes the CV error curve, for
#' every loss function in `loss_funcs`.}
#' \item{`lambda_1se`}{A named vector with length equal to `length(lambda)`,
#' containing the "1-standard-error rule" hyperparameter, for every loss
#' function in `loss_funcs`. The "1-standard-error rule" hyperparameter is the
#' largest hyperparameter value in `lambda` that has a CV error within one
#' standard error of `min(error)`. It serves as an Occam's razor-like
#' heuristic. More precisely, given two models with approximately equal
#' performance (in terms of some loss function), it may be wise to opt for the
#' simpler model, i.e. the model with the larger hyperparameter value / fewer
#' effective degrees of freedom.}
#' \item{`edf_min`}{A named vector with length equal to `length(lambda)`,
#' containing the number of effective degrees of freedom in the trend filtering
#' estimator that minimizes the CV error curve, for every loss function in
#' `loss_funcs`.}
#' \item{`edf_1se`}{A named vector with length equal to `length(lambda)`,
#' containing the number of effective degrees of freedom in the
#' "1-standard-error rule" trend filtering estimator, for every loss function in
#' `loss_funcs`.}
#' \item{`i_min`}{A named vector with length equal to `length(lambda)`,
#' containing the index of `lambda` that minimizes the CV error curve, for
#' every loss function in `loss_funcs`.}
#' \item{`i_1se`}{A named vector with length equal to `length(lambda)`,
#' containing the index of `lambda` that gives the "1-standard-error rule"
#' hyperparameter value, for every loss function in `loss_funcs`.}
#' \item{`obj_func`}{The relative change in the objective function over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambda`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambda`. If an element of `n_iter`
#' is exactly equal to `admm_params$max_iter` (see below), then the
#' ADMM algorithm stopped before reaching the objective tolerance
#' `admm_params$obj_tol`. In these situations, you may need to
#' increase the maximum number of tolerable iterations by passing a
#' `max_iter` argument to `cv_trendfilter()` in order to ensure that the ADMM
#' solution has converged to satisfactory precision.}
#' \item{`loss_funcs`}{A named list of functions that defines all loss functions
#' --- both internal and user-passed --- evaluated during cross validation.}
#' \item{`V`}{The number of folds the data were split into for cross
#' validation.}
#' \item{`x`}{Vector of observed values for the input variable.}
#' \item{`y`}{Vector of observed values for the output variable (if originally
#' present, observations with `is.na(y)` or `weights == 0` are dropped).}
#' \item{`weights`}{Vector of weights for the observed outputs.}
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
#' cv_tf <- cv_trendfilter(x, y, weights, max_iter = 1e4, obj_tol = 1e-6)
#' @importFrom dplyr tibble filter mutate select arrange case_when group_split
#' @importFrom dplyr bind_rows
#' @importFrom tidyr drop_na expand_grid
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom stats median sd
#' @export
cv_trendfilter <- function(x,
                           y,
                           weights = NULL,
                           nlambda = 250L,
                           V = 10L,
                           mc_cores = V,
                           loss_funcs = NULL,
                           fold_ids = NULL,
                           ...) {
  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  cv_call <- match.call()
  extra_args <- list(...)

  if (any(names(extra_args) == "k")) {
    stopifnot(is.numeric(k) && k == round(k))
    stopifnot(length(k) == 1)
    if (!k %in% 0:2) stop("`k` must be equal to 0, 1, or 2.", call. = FALSE)
  } else {
    k <- 2L
  }

  if (any(names(extra_args) == "obj_tol")) {
    obj_tol <- extra_args$obj_tol
    extra_args$obj_tol <- NULL
    stopifnot(is.numeric(obj_tol) && obj_tol > 0L && length(obj_tol) == 1L)
  } else{
    obj_tol <- NULL
  }

  if (any(names(extra_args) == "max_iter")) {
    max_iter <- extra_args$max_iter
    extra_args$max_iter <- NULL
    stopifnot(is.numeric(max_iter) && max_iter == round(max_iter))
    stopifnot(length(max_iter) == 1L)
    max_iter %<>% as.integer()
  } else{
    max_iter <- 0
  }

  stopifnot(is.numeric(V) && V == round(V))
  if (V < 2L || V > 10L) {
    stop("Must have `V >= 2` and `V <= 10`.", call. = FALSE)
  }

  n <- length(y)
  weights <- weights %||% rep_len(1, n)
  stopifnot(is.numeric(weights))
  stopifnot(length(weights) %in% c(1L, n))
  stopifnot(all(weights >= 0L))
  if (length(weights) == 1) weights <- rep_len(weights, n)

  if (nlambda < 100 || nlambda != round(nlambda)) {
    stop("`nlambda` must be an integer >= 100`.", call. = FALSE)
  } else {
    nlambda %<>% as.integer()
  }

  mc_cores <- min(c(detectCores(), V, max(c(1, floor(mc_cores)))))

  if (mc_cores < V & mc_cores < detectCores() / 2) {
    warning(
      paste(
        "Your machine has", detectCores(), "cores, but you've only",
        "configured `cv_trendfilter()` to use", mc_cores, "of them. Letting",
        "`mc_cores = V` can significantly speed up cross validation."
      ),
      call. = FALSE, immediate. = TRUE
    )
  }

  internal_loss_funcs <- get_internal_loss_funcs()

  if (is.null(loss_funcs)) {
    loss_funcs <- internal_loss_funcs
  } else {
    if (class(loss_funcs) != "list") {
      stop(
        paste(
          "Loss function(s) must be passed within a named list to the",
          "`loss_funcs` argument."
        ),
        call. = FALSE
      )
    } else {
      classes <- lapply(X = 1:length(loss_funcs), class(loss_funcs[[X]]))
      if (!all(classes == "function")) {
        inds <- which(classes != "function")
        stop(
          paste(
            "Element(s)", paste(inds, collapse = ", "),
            " of `loss_funcs` are not of class 'function'."
          ),
          call. = FALSE
        )
      }

      if (is.null(names(loss_funcs)) || any(names(loss_funcs) == "")) {
        stop(
          "`loss_funcs` must be a named list of functions.",
          call. = FALSE
        )
      }

      for (X in 1:length(loss_funcs)) {
        if (!all(c("y", "tf_estimate", "weights") %in%
          names(formals(loss_funcs[[X]])))) {
          stop(
            paste0(
              "Incorrect input argument structure for the function ",
              "'", names(loss_funcs)[X], "'. Each loss function should have ",
              "vector input arguments `y`, `tf_estimate`, `weights`, and ",
              "return a single numeric value for the validation error."
            ),
            call. = FALSE
          )
        }
      }
      loss_funcs <- c(internal_loss_funcs, loss_funcs)
    }
  }

  k %<>% as.integer()
  V %<>% as.integer()

  if (is.null(fold_ids)) {
    fold_ids <- sample(rep_len(1:V, n))
  } else {
    unique_fold_ids <- fold_ids %>%
      as.integer() %>%
      unique.default() %>%
      sort.int()
    if (!all.equal(unique_fold_ids, 1:V)) {
      stop(
        "`fold_ids` must contain only integer values 1:V, with no empty folds.",
        call. = FALSE
      )
    }

    counts <- table(fold_ids) %>%
      as.integer() %>%
      sort.int()
    equal_counts <- table(rep_len(1:V, n)) %>%
      as.integer() %>%
      sort.int()

    if (!all.equal(counts, equal_counts)) {
      if (n %% V == 0L) {
        warning("Your cross validation folds are imbalanced.", call. = FALSE)
      } else {
        warning(
          paste(
            "Your cross validation folds are imbalanced, beyond what is simply",
            "due to `length(x)` not being divisible by `V`."
          ),
          call. = FALSE
        )
      }
    }
  }

  data <- tibble(x = as.double(x),
                 y = as.double(y),
                 weights = as.double(weights),
                 fold_id = as.integer(fold_ids)) %>%
    drop_na() %>%
    arrange(x) %>%
    filter(weights > 0)

  rm(x, y, weights)
  n <- nrow(data)

  x_scale <- median(diff(data$x))
  y_scale <- median(abs(data$y)) / 10

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    )

  admm_params <- get_admm_params(obj_tol, max(max_iter, n, 200L))
  admm_params$x_tol <- admm_params$x_tol / x_scale

  if (min(diff(data_scaled$x)) <= admm_params$x_tol) {
    thin_out <- .tf_thin(
      data_scaled$x,
      data_scaled$y,
      data_scaled$weights,
      k,
      admm_params
    )

    inds <- match(thin_out$x, data_scaled$x)
    data_scaled <- tibble(x = thin_out$x,
                          y = thin_out$y,
                          weights = thin_out$w,
                          fold_id = data_scaled$fold_id[inds])
  }

  data_folded <- data_scaled %>%
    mutate(ids = fold_id) %>%
    group_split(ids, .keep = FALSE)

  lambda <- get_lambda_grid_edf_spacing(data = data_scaled,
                                        admm_params = admm_params,
                                        nlambda = nlambda,
                                        k = k)

  cv_out <- mclapply(
    1:V,
    FUN = validate_fold,
    data_folded = data_folded,
    lambda = lambda,
    k = k,
    admm_params = admm_params,
    loss_funcs = loss_funcs,
    y_scale = y_scale,
    extra_args = extra_args,
    mc.cores = mc_cores
  )

  get_cv_mat <- function(loss_func, V) {
    lapply(
      1:V,
      function(X) cv_out[[X]][loss_func]
    ) %>%
      unlist() %>%
      matrix(ncol = V)
  }

  cv_loss_mats <- lapply(
    X = seq_along(loss_funcs),
    FUN = function(X) get_cv_mat(X, V)
  )

  error <- lapply(
    seq_along(cv_loss_mats),
    FUN = function(X) {
      cv_loss_mats[[X]] %>%
        rowMeans() %>%
        as.double()
    }
  )

  i_min <- lapply(
    seq_along(error),
    FUN = function(X) {
      error[[X]] %>%
        which.min() %>%
        min()
    }
  ) %>%
    unlist()

  se_error <- lapply(
    seq_along(error),
    FUN = function(X) {
      rowSds(cv_loss_mats[[X]]) / sqrt(V) %>%
        as.double()
    }
  )

  i_1se <- lapply(
    X = seq_along(error),
    FUN = function(X) {
      which(error[[X]] <= error[[X]][i_min[X]] + se_error[[X]][i_min[X]]) %>%
        min()
    }
  ) %>%
    unlist()

  lambda_min <- lambda[i_min]
  lambda_1se <- lambda[i_1se]

  args <- c(
    list(
      x = data_scaled$x,
      y = data_scaled$y,
      weights = data_scaled$weights,
      lambda = lambda,
      k = k,
      obj_tol = admm_params$obj_tol,
      max_iter = admm_params$max_iter
    ),
    extra_args
  )

  duplicated_args <- duplicated(names(args))
  if (any(duplicated_args)) args <- args[-duplicated_args]

  fit <- do.call(.trendfilter, args)

  edf_min <- as.integer(fit$edf[i_min])
  edf_1se <- as.integer(fit$edf[i_1se])

  names(lambda_min) <- names(loss_funcs)
  names(lambda_1se) <- names(loss_funcs)
  names(edf_min) <- names(loss_funcs)
  names(edf_1se) <- names(loss_funcs)
  names(error) <- names(loss_funcs)
  names(i_min) <- names(loss_funcs)
  names(se_error) <- names(loss_funcs)
  names(i_1se) <- names(loss_funcs)

  invisible(
    structure(
      list(
        lambda = fit$lambda,
        edf = fit$edf,
        error = error,
        se_error = se_error,
        lambda_min = lambda[i_min],
        lambda_1se = lambda[i_1se],
        edf_min = fit$edf[i_min],
        edf_1se = fit$edf[i_1se],
        i_min = i_min,
        i_1se = i_1se,
        obj_func = fit$obj_fun,
        n_iter = fit$n_iter,
        status = fit$status,
        loss_funcs = loss_funcs,
        V = V,
        x = data$x,
        y = data$y,
        weights = data$weights,
        k = k,
        fitted_values = fit$fitted_values * y_scale,
        admm_params = admm_params,
        call = cv_call,
        x_scale = x_scale,
        y_scale = y_scale
      ),
      class = c("cv_trendfilter", "trendfilter", "trendfiltering")
    )
  )
}


#' @importFrom dplyr bind_rows
#' @importFrom magrittr %>%
validate_fold <- function(fold_id,
                          data_folded,
                          lambda,
                          k,
                          admm_params,
                          loss_funcs,
                          y_scale,
                          extra_args) {
  data_train <- data_folded[-fold_id] %>% bind_rows()
  data_validate <- data_folded[[fold_id]]

  args <- c(
    list(
      x = data_train$x,
      y = data_train$y,
      weights = data_train$weights,
      k = k,
      lambda = lambda,
      obj_tol = admm_params$obj_tol,
      max_iter = admm_params$max_iter
    ),
    extra_args
  )

  duplicated_args <- duplicated(names(args))
  if (any(duplicated_args)) args <- args[-duplicated_args]

  fit <- do.call(.trendfilter, args)

  tf_validate_preds <- suppressWarnings(
    predict(fit, lambda = lambda, x_eval = data_validate$x)
  )

  out <- lapply(
    seq_along(loss_funcs),
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

  names(out) <- names(loss_funcs)
  out
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

  Huber <- function(tf_estimate, y, weights, n_stderr = 3) {
    stderr <- stats::sd(y - tf_estimate)
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
#' trend filtering model's predictive mean-squared error. See the **Details**
#' section for guidelines on when [`sure_trendfilter()`] should be used versus
#' [`cv_trendfilter()`]. Generic functions such as [`predict()`],
#' [`fitted.values()`], and [`residuals()`] may be called on the
#' [`sure_trendfilter()`] output.
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
#' @return An object of class '`sure_trendfilter`' and subclass
#' '[`trendfilter`][`trendfilter()`]'. This is a list with the elements below,
#' as well as all elements from the '[`trendfilter`][`trendfilter()`]' call.
#' \describe{
#' \item{`lambda`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edf`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambda`.}
#' \item{`error`}{Vector of mean-squared prediction errors estimated by SURE,
#' for every hyperparameter value in `lambda`.}
#' \item{`se_error`}{Vector of estimated standard errors for the `error`.}
#' \item{`lambda_min`}{Hyperparameter value in `lambda` that minimizes the SURE
#' validation error curve.}
#' \item{`lambda_1se`}{The largest hyperparameter value in `lambda` that has a
#' SURE error within one standard error of `min(error)`. We call this the
#' "1-standard-error rule" hyperparameter, and it serves as an Occam's
#' razor-esque heuristic. More precisely, given two models with approximately
#' equal performance (here, in terms of predictive MSE), it may be wise to opt
#' for the simpler model, i.e. the model with the larger hyperparameter value /
#' fewer effective degrees of freedom.}
#' \item{`edf_min`}{Number of effective degrees of freedom in the trend
#' filtering estimator with hyperparameter `lambda_min`.}
#' \item{`edf_1se`}{Number of effective degrees of freedom in the  trend
#' filtering estimator with hyperparameter `lambda_1se`.}
#' \item{`i_min`}{Index of `lambda` that gives `lambda_min`.}
#' \item{`i_1se`}{Index of `lambda` that gives `lambda_1se`.}
#' \item{`obj_func`}{The relative change in the objective function over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambda`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambda`. If an element of `n_iter` is
#' exactly equal to `admm_params$max_iter` (see below), then the ADMM algorithm
#' stopped before reaching the objective tolerance `admm_params$obj_tol`. In
#' these situations, you may need to increase the maximum number of tolerable
#' iterations by passing a `max_iter` argument to `cv_trendfilter()` in order to
#' ensure that the ADMM solution has converged to satisfactory precision.}
#' \item{`training_error`}{The "in-sample" MSE between the observed outputs `y`
#' and the trend filtering estimate, for every hyperparameter value in
#' `lambda`.}
#' \item{`optimism`}{SURE-estimated optimisms, i.e.
#' `optimism = error - training_error`.}
#' \item{`call`}{The function call.}
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
#' data("quasar_spectrum")
#' head(quasar_spectrum)
#'
#' x <- quasar_spectrum$log10_wavelength
#' y <- quasar_spectrum$flux
#' weights <- quasar_spectrum$weights
#'
#' sure_tf <- sure_trendfilter(x, y, weights)
#' @importFrom glmgen .tf_thin .tf_fit .tf_predict
#' @importFrom dplyr tibble filter mutate select arrange case_when group_split
#' @importFrom dplyr bind_rows
#' @importFrom tidyr drop_na
#' @importFrom purrr map
#' @importFrom magrittr %>% %$% %<>%
#' @importFrom rlang %||%
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom stats median sd
#' @export
sure_trendfilter <- function(x,
                             y,
                             weights,
                             nlambda = 250L,
                             ...) {
  if (missing(x)) stop("`x` must be passed.")
  if (missing(y)) stop("`y` must be passed.")
  stopifnot(is.numeric(x))
  stopifnot(is.numeric(y))
  stopifnot(length(x) == length(y))

  sure_call <- match.call()
  extra_args <- list(...)

  if (any(names(extra_args) == "k")) {
    stopifnot(is.numeric(k) && k == round(k))
    stopifnot(length(k) == 1)
    if (!k %in% 0:2) stop("`k` must be equal to 0, 1, or 2.", call. = FALSE)
  } else {
    k <- 2L
  }

  if (any(names(extra_args) == "obj_tol")) {
    obj_tol <- extra_args$max_iter
    stopifnot(is.numeric(obj_tol) && obj_tol > 0L && length(obj_tol) == 1L)
  } else{
    obj_tol <- NULL
  }

  if (any(names(extra_args) == "max_iter")) {
    max_iter <- extra_args$max_iter
    stopifnot(is.numeric(max_iter) && max_iter == round(max_iter))
    stopifnot(length(max_iter) == 1L)
    max_iter %<>% as.integer()
  } else{
    max_iter <- 0
  }

  if (missing(weights)) {
    stop("`weights` must be passed for `sure_trendfilter()`.")
  }
  n <- length(y)
  stopifnot(is.numeric(weights))
  stopifnot(length(weights) %in% c(1L, n))
  stopifnot(all(weights >= 0L))
  if (length(weights) == 1) weights <- rep_len(weights, n)

  if (nlambda < 100 || nlambda != round(nlambda)) {
    stop("`nlambda` must be an integer >= 100`.", call. = FALSE)
  } else {
    nlambda %<>% as.integer()
  }

  k %<>% as.integer()

  data <- tibble(x = as.double(x),
                 y = as.double(y),
                 weights = as.double(weights)) %>%
    drop_na() %>%
    arrange(x) %>%
    filter(weights > 0)

  rm(x, y, weights)
  n <- nrow(data)

  x_scale <- median(diff(data$x))
  y_scale <- median(abs(data$y)) / 10

  data_scaled <- data %>%
    mutate(
      x = x / x_scale,
      y = y / y_scale,
      weights = weights * y_scale^2
    )

  admm_params <- get_admm_params(obj_tol, max(max_iter, n, 200L))
  admm_params$x_tol <- admm_params$x_tol / x_scale

  if (min(diff(data_scaled$x)) <= admm_params$x_tol) {
    thin_out <- .tf_thin(
      data_scaled$x,
      data_scaled$y,
      data_scaled$weights,
      k,
      admm_params
    )

    data_scaled <- tibble(x = thin_out$x, y = thin_out$y, weights = thin_out$w)
  }

  lambda <- get_lambda_grid_edf_spacing(data = data_scaled,
                                        admm_params = admm_params,
                                        nlambda = nlambda,
                                        k = k)

  args <- c(
    list(
      x = data_scaled$x,
      y = data_scaled$y,
      weights = data_scaled$weights,
      lambda = lambda,
      k = k,
      obj_tol = admm_params$obj_tol,
      max_iter = admm_params$max_iter
    ),
    extra_args
  )

  duplicated_args <- duplicated(names(args))
  if (any(duplicated_args)) args <- args[-duplicated_args]

  fit <- do.call(.trendfilter, args)

  squared_residuals_mat <- (fit$fitted_values - data_scaled$y)^2
  optimism_mat <- 2 / (data_scaled$weights * n) *
    matrix(rep(fit$edf, each = n), nrow = n)

  error_mat <- (squared_residuals_mat + optimism_mat) * y_scale^2
  error <- colMeans(error_mat)
  i_min <- min(which.min(error)) %>% as.integer()

  se_error <- replicate(
    5000,
    error_mat[sample.int(n, replace = TRUE), ] %>%
      colMeans()
  ) %>%
    rowSds()

  i_1se <- which(
    error <= error[i_min] + se_error[i_min]
  ) %>%
    min()

  invisible(
    structure(
      list(
        lambda = fit$lambda,
        edf = fit$edf,
        error = error,
        se_error = se_error,
        lambda_min = lambda[i_min],
        lambda_1se = lambda[i_1se],
        edf_min = fit$edf[i_min],
        edf_1se = fit$edf[i_1se],
        i_min = i_min,
        i_1se = i_1se,
        obj_func = fit$obj_fun,
        status = fit$status,
        n_iter = fit$n_iter,
        x = data$x,
        y = data$y,
        weights = data$weights,
        k = k,
        fitted_values = fit$fitted_values,
        admm_params = admm_params,
        call = sure_call,
        x_scale = x_scale,
        y_scale = y_scale
      ),
      class = c("sure_trendfilter", "trendfilter", "trendfiltering")
    )
  )
}
