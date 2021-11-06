#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' For every candidate hyperparameter value, estimate the corresponding trend
#' filtering model's out-of-sample error by V-fold cross validation. Four
#' commonly-used types of error --- MSE, WMSE, MAE, WMAE --- are all evaluated
#' and returned, and the user has the option to pass additional error
#' functionals to be evaluated. See the details section for when you should use
#' [cv_trendfilter()] versus [`sure_trendfilter()`].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights Weights for the observed outputs, defined as the reciprocal
#' variance of the additive noise that contaminates the signal in `y`.
#' When the noise is expected to have equal variance for all observations,
#' `weights` can be passed as a scalar. Otherwise, `weights` must be a vector
#' with the same length as `x` and `y`.
#' @param k Degree of the polynomials that make up the piecewise-polynomial
#' trend filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas Number of hyperparameter values to test during validation.
#' Defaults to `nlambdas = 250`. The hyperparameter grid is dynamically
#' constructed to span the full model space lying between a single polynomial
#' solution (i.e. a power law) and an interpolating solution, with `nlambdas`
#' controlling the granularity of the hyperparameter grid.
#' @param V Number of folds that the data are partitioned into for the V-fold
#' cross validation. Defaults to `V = 10`.
#' @param custom_error_funcs (Optional) A named list of one or more functions,
#' with each defining an error functional to evaluate on held-out folds during
#' cross validation. Mean-squared error (MSE) and mean absolute error (MAE) are
#' both evaluated automatically, as well as weighted versions of each (with
#' reciprocal variances as weights) --- WMSE and WMAE. Therefore, the user does
#' not need to pass anything to `custom_error_funcs` unless they want to define
#' a validation error metric other than MSE, MAE, WMSE, and WMAE.
#'
#' In such a case, each function in the named list passed to
#' `custom_error_funcs` should take three vector arguments --- `y`,
#' `tf_estimate`, and `weights` --- and return a single scalar value for the
#' validation error. For example, if I also wanted to define a validation
#' error curve from the weighted median of the absolute errors, I could pass the
#' following list to `custom_error_funcs`:
#' ```{r, eval = FALSE}
#' list(MedAE = function(tf_estimate, y, weights) {
#'                matrixStats::weightedMedian(abs(tf_estimate - y), weights)
#'              }
#'     )
#' ```
#' @param mc_cores Multi-core computing using the
#' [`parallel`][`parallel::parallel-package`] R package: The number of cores to
#' utilize. Defaults to the number of cores detected, minus four.
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
#' filtering analysis on that scale. See the [`sure_trendfilter()`] examples for
#' a case when the inputs are evenly sampled on the `log10(x)` scale.
#'
#' The validation error functionals that we automatically compute CV curves for
#' are formally defined below.
#'
#' \loadmathjax
#' \mjsdeqn{MSE(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|^2}
#' \mjsdeqn{MAE(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|}
#' \mjsdeqn{WMSE(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#' \mjsdeqn{WMAE(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' where \mjseqn{w_i:=} `weights[i]`.
#'
#' If constant weights are passed, or if nothing is passed, then the weighted
#' and unweighted counterparts are equivalent.
#'
#' Briefly stated, weighting the validation error metric (with reciprocal
#' variances) helps prevent the error from being dominated by the model's
#' (in)accuracy on a (potentially very small) subset of the data that have
#' high variance; and absolute error is less sensitive to outliers than squared
#' error.
#'
#' @return An object of class `'cv_tf'`. This is a list with the following
#' elements:
#' \describe{
#' \item{lambdas}{Vector of candidate hyperparameter values (always returned in
#' descending order).}
#' \item{edfs}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambdas`.}
#' \item{validation_errors}{A named list of vectors, with each representing the
#' cross validation error curve for some definition of error. The first 4
#' vectors of the list correspond to WMAE, WMSE, MAE, MSE. If any custom
#' error functionals were passed to `custom_error_funcs`, their cross validation
#' curves will follow the first 4.}
#' \item{se_validation_errors}{Named list of estimated standard errors for each
#' of the cross validation error curves in `validation_errors`.}
#' \item{lambda_min}{A named vector with length equal to
#' `length(validation_errors)`, containing the hyperparameter value that
#' minimizes the cross validation error curve, for every type validation error.}
#' \item{lambda_1se}{A named vector with length equal to
#' `length(validation_errors)`, containing the "1-standard-error rule"
#' hyperparameter, for every type validation error. The "1-standard-error rule"
#' hyparameter is the largest hyperparameter value (corresponding to the
#' smoothest trend filtering estimate) that has a cross validation error
#' within one standard error of the minimum cross validation error. It serves as
#' an Occam's razor-esque heuristic. That is, given two models with
#' approximately equal performance, it may be wise to opt for the simpler model,
#' i.e. the model with fewer effective degrees of freedom.}
#' \item{edf_min}{A named vector with length equal to
#' `length(validation_errors)`, containing the number of effective degrees of
#' freedom in the trend filtering estimator that minimizes the cross validation
#' error curve, for every type of validation error.}
#' \item{edf_1se}{A named vector with length equal to
#' `length(validation_errors)`, containing the number of effective degrees of
#' freedom in the "1-standard-error rule" trend filtering estimator, for every
#' type of validation error.}
#' \item{i_min}{A named vector with length equal to
#' `length(validation_errors)`, containing the index of `lambdas` that minimizes
#' the CV error curve, for every type of validation error.}
#' \item{i_1se}{A named vector with length equal to
#' `length(validation_errors)`, containing the index of `lambdas` that
#' gives the "1-standard-error rule" hyperparameter value, for every
#' type of validation error.}
#' \item{validation_error_funcs}{A named list of functions that define the
#' types of error that were evaluated during cross validation.}
#' \item{cost_functional}{The relative change in the cost functional over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambdas`.}
#' \item{n_iter}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambdas`. If an element of `n_iter`
#' is exactly equal to `max_iter`, then the ADMM algorithm stopped before
#' reaching the tolerance set by `obj_tol`. In these cases, you may need
#' to increase `max_iter` to ensure the trend filtering solution has
#' converged to satisfactory precision.}
#' \item{V}{The number of folds the data were split into for cross validation.}
#' \item{tf_model}{A list of objects that is used internally by other
#' functions that operate on the `cv_trendfilter()` output.}
#' }
#'
#' @export cv_trendfilter
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
#' \bold{Cross validation}
#' \enumerate{
#' \item{Hastie, Tibshirani, and Friedman (2009).
#' \href{https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf}{
#' The Elements of Statistical Learning: Data Mining, Inference, and
#' Prediction}. 2nd edition. Springer Series in Statistics. (See Sections 7.10
#' and 7.12)}}
#'
#' @seealso [sure_trendfilter()], [bootstrap_trendfilter()]
#'
#' @examples
#' data(eclipsing_binary)
#' head(EB)
#'
#' cv_tf <- cv_trendfilter(
#'   x = EB$phase,
#'   y = EB$flux,
#'   weights = 1 / EB$std_err^2,
#'   optimization_params = list(
#'     max_iter = 1e4,
#'     obj_tol = 1e-6,
#'     thinning = TRUE
#'   )
#' )
#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom magrittr %$% %>% %<>%
#' @importFrom tidyr tibble drop_na
#' @importFrom stats median sd
cv_trendfilter <- function(x,
                           y,
                           weights,
                           k = 2L,
                           nlambdas = 250L,
                           V = 10L,
                           mc_cores = parallel::detectCores() - 4,
                           custom_error_funcs,
                           optimization_params) {
  stopifnot(mc_cores >= 1)
  if (missing(x) || is.null(x)) stop("x must be passed.")
  if (missing(y) || is.null(y)) stop("y must be passed.")
  if (length(x) != length(y)) stop("x and y must have equal length.")
  if (length(y) < k + 2) stop("Must have >= k + 2 observations.")
  if (k < 0 || k != round(k)) stop("k must be a nonnegative integer.")
  if (k > 2) {
    stop("k > 2 are algorithmically unstable and do not improve upon k = 2.")
  }
  if (V < 2 || V != round(V)) {
    stop("V must be an integer between 2 and length(x).")
  }

  if (!missing(weights)) {
    if (!(length(weights) %in% c(1, length(y)))) {
      stop(
        "If passed, weights must be numerically-valued with length(weights) = 1
        or the same length as x and y."
      )
    }
    if (!(class(weights) %in% c("numeric", "integer"))) {
      stop(
        "If passed, weights must be numerically-valued with length(weights) = 1
        or the same length as x and y."
      )
    }
  }

  if (nlambdas < 0 || nlambdas != round(nlambdas)) {
    stop("nlambdas must be a positive integer")
  } else {
    nlambdas <- nlambdas %>% as.integer()
  }

  if (mc_cores < V & mc_cores < detectCores() / 2) {
    warning(
      paste0(
        "Your machine has ", detectCores(), " cores available. ",
        "Setting `mc_cores = V` can speed up computation."
      )
    )
  }

  if (mc_cores > detectCores()) mc_cores <- detectCores()
  mc_cores <- min(floor(mc_cores), V)
  if (missing(weights)) weights <- rep(1, length(y))
  if (length(weights) == 1) weights <- rep(weights, length(y))

  x %<>% as.double()
  y %<>% as.double()
  weights %<>% as.double()
  k %<>% as.integer()
  V %<>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights > 0) %>%
    drop_na()

  if (missing(custom_error_funcs)) {
    validation_error_funcs <- list(
      WMAE = WMAE, WMSE = WMSE,
      MAE = MAE, MSE = MSE
    )
  } else {
    if (class(custom_error_funcs) == "function") {
      stop("Please pass your validation error function within a named list.")
    }

    if (class(custom_error_funcs) != "list") {
      stop("Please pass your validation error function(s) within a named list.")
    }

    if (class(custom_error_funcs) == "list") {
      if (is.null(names(custom_error_funcs)) |
        any(names(custom_error_funcs) == "")) {
        stop(paste0(
          "Please name each of the functions in your custom_error_funcs ",
          "list."
        ))
      }

      for (X in 1:length(list)) {
        if (!all(c("y", "tf_estimate", "weights") %in%
          names(formals(validation_error_funcs[[X]])))) {
          stop(paste0(
            "Incorrect input argument structure for function ",
            "validation_error_funcs[[", X, "]]."
          ))
        }
      }
      validation_error_funcs <- c(
        list(WMAE = WMAE, WMSE = WMSE, MAE = MAE, MSE = MSE),
        custom_error_funcs
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

  data_folded <- data_scaled %>%
    group_split(sample(rep_len(1:V, nrow(data_scaled))), .keep = FALSE)

  lambdas <- get_lambdas(nlambdas, data_scaled, k, thinning, admm_params)

  cv_errors <- mclapply(
    1:V,
    FUN = validate_trendfilter,
    data_folded = data_folded,
    lambdas = lambdas,
    k = k,
    thinning = thinning,
    admm_params = admm_params,
    validation_error_funcs = validation_error_funcs,
    y_scale = y_scale,
    mc.cores = mc_cores
  )

  cv_error_mats <- lapply(
    X = 1:length(validation_error_funcs),
    FUN = function(X) {
      lapply(
        1:length(cv_errors),
        FUN = function(itr) cv_errors[[itr]][[X]]
      ) %>%
        unlist() %>%
        matrix(ncol = V)
    }
  )

  validation_errors <- lapply(
    1:length(cv_error_mats),
    FUN = function(X) {
      cv_error_mats[[X]] %>%
        rowMeans() %>%
        as.double()
    }
  )

  i_min <- lapply(
    1:length(validation_errors),
    FUN = function(X) {
      validation_errors[[X]] %>%
        which.min() %>%
        min()
    }
  ) %>% unlist()

  se_validation_errors <- lapply(
    1:length(validation_errors),
    FUN = function(X) {
      rowSds(cv_error_mats[[X]]) / sqrt(V) %>% as.double()
    }
  )

  i_1se <- lapply(
    X = 1:length(validation_errors),
    FUN = function(X) {
      which(
        validation_errors[[X]] <= validation_errors[[X]][i_min[X]] +
          se_validation_errors[[X]][i_min[X]]
      ) %>% min()
    }
  ) %>% unlist()

  lambda_min <- lambdas[i_min]
  lambda_1se <- lambdas[i_1se]

  out <- trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  edf_min <- out$df[i_min] %>% as.integer()
  edf_1se <- out$df[i_1se] %>% as.integer()

  names(lambda_min) <- names(validation_error_funcs)
  names(lambda_1se) <- names(validation_error_funcs)
  names(edf_min) <- names(validation_error_funcs)
  names(edf_1se) <- names(validation_error_funcs)
  names(validation_errors) <- names(validation_error_funcs)
  names(i_min) <- names(validation_error_funcs)
  names(se_validation_errors) <- names(validation_error_funcs)
  names(i_1se) <- names(validation_error_funcs)

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
      validation_errors = validation_errors,
      se_validation_errors = se_validation_errors,
      lambda_min = lambda_min,
      lambda_1se = lambda_1se,
      edf_min = edf_min,
      edf_1se = edf_1se,
      i_min = i_min,
      i_1se = i_1se,
      validation_error_funcs = validation_error_funcs,
      cost_functional = out$obj[nrow(out$obj), ],
      n_iter = out$iter %>% as.integer(),
      V = V,
      tf_model = tf_model
    ),
    class = c("cv_tf", "list")
  )
}


#' @importFrom glmgen trendfilter
#' @importFrom dplyr bind_rows
validate_trendfilter <- function(validation_index,
                                 data_folded,
                                 lambdas,
                                 k,
                                 thinning,
                                 admm_params,
                                 validation_error_funcs,
                                 y_scale) {
  data_train <- data_folded[-validation_index] %>% bind_rows()
  data_validate <- data_folded[[validation_index]]

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
    X = 1:length(validation_error_funcs),
    FUN = function(X) {
      apply(
        tf_validate_preds * y_scale,
        2,
        validation_error_funcs[[X]],
        y = data_validate$y * y_scale,
        weights = data_validate$weights / y_scale^2
      ) %>%
        as.double()
    }
  )
}

####

# Functions for common validation error functionals

MSE <- function(tf_estimate, y, weights) {
  mean((tf_estimate - y)^2)
}

MAE <- function(tf_estimate, y, weights) {
  mean(abs(tf_estimate - y))
}

WMSE <- function(tf_estimate, y, weights) {
  sum((tf_estimate - y)^2 * weights / sum(weights))
}

WMAE <- function(tf_estimate, y, weights) {
  sum(abs(tf_estimate - y) * sqrt(weights) / sum(sqrt(weights)))
}
