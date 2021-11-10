#' Optimize the trend filtering hyperparameter by \emph{V}-fold cross validation
#'
#' \loadmathjax For every candidate hyperparameter value, estimate the trend
#' filtering model's out-of-sample error by \emph{V}-fold cross validation. CV
#' error curves are returned for four of the most common regression loss
#' metrics, as well as observation-weighted versions. Custom loss functions may
#' also be passed to the `loss_funcs` argument. See the details section for
#' definitions of the internal loss functions, and for guidelines on when
#' [cv_trendfilter()] should be used versus [`sure_trendfilter()`].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights (Optional) Weights for the observed outputs, defined as the
#' reciprocal variance of the additive noise that contaminates the output
#' signal. When the noise is expected to have an equal variance,
#' \mjseqn{\sigma^2}, for all observations, a scalar may be passed to `weights`,
#' namely `weights = `\mjseqn{1/\sigma^2}. Otherwise, `weights` must be a vector
#' with the same length as `x` and `y`.
#' @param k Degree of the polynomials that make up the piecewise-polynomial
#' trend filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas Number of hyperparameter values to test during validation.
#' Defaults to `nlambdas = 250`. The hyperparameter grid is internally
#' constructed to span the full trend filtering model space (which is bookended
#' by a global polynomial solution and an interpolating solution), with
#' `nlambdas` controlling the granularity of the hyperparameter grid.
#' @param V Number of folds that the data are partitioned into for \emph{V}-fold
#' cross validation. Defaults to `V = 10`.
#' @param loss_funcs (Optional) A named list of one or more functions, with each
#' defining a loss function to be evaluated on held-out folds during cross
#' validation. By default, `cv_trendfilter()` will automatically compute and
#' return CV error curves for 9 common regression loss functions (see the
#' Details section below). Therefore, the `loss_funcs` argument need only be
#' used in order to define loss functions that are not among these 9 choices.
#'
#' When defining custom loss functions, each function within the named list
#' passed to `loss_funcs` should take three vector arguments --- `y`,
#' `tf_estimate`, and `weights` --- and return a single scalar value for the
#' validation error. For example, if I wanted a CV error curve based on a
#' weighted median of the absolute errors to be computed, I would pass the list
#' below to `loss_funcs`:
#' ```{r, eval = FALSE}
#' MedAE <- function(tf_estimate, y, weights) {
#'   matrixStats::weightedMedian(abs(tf_estimate - y), sqrt(weights))
#' }
#'
#' list(MedAE = MedAE)
#' ```
#' @param fold_ids (Optional) An integer vector defining a custom partition of
#' the data for cross validation. `fold_ids` must have the same length as `x`
#' and `y`, and only contain values in the set `1:V`.
#' @param mc_cores Multi-core computing using the
#' [`parallel`][`parallel::parallel-package`] R package: The number of cores to
#' utilize. Defaults to the number of cross validation folds, `mc_cores = V`.
#' If the value passed to `mc_cores` exceeds the number of cores available on
#' the machine, `mc_cores` is internally updated to
#' `mc_cores = parallel::detectCores()`.
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
#' \item{`obj_tol`}{A stopping threshold for the ADMM algorithm. If the relative
#' change in the algorithm's cost functional between two consecutive steps is
#' less than `obj_tol`, the algorithm terminates. The algorithm's termination
#' can also result from it reaching the maximum tolerable iterations set
#' by the `max_iter` parameter (see below). The `obj_tol` parameter defaults to
#' `obj_tol = 1e-10`. The `cost_functional` vector, returned within the
#' `sure_trendfilter()` output, gives the relative change in the trend filtering
#' cost functional over the algorithm's final iteration, for every candidate
#' hyperparameter value.}
#' \item{`max_iter`}{Maximum number of ADMM iterations that we will tolerate.
#' Defaults to `max_iter = length(y)`. The actual number of iterations performed
#' by the algorithm, for every candidate hyperparameter value, is returned in
#' the `n_iter` vector, within the `cv_trendfilter()` output. If any of the
#' elements of `n_iter` are equal to `max_iter`, the tolerance defined by
#' `obj_tol` has not been attained and `max_iter` may need to be increased.}
#' \item{`thinning`}{Logical. If `thinning = TRUE`, then the data are
#' preprocessed so that a smaller data set is used to fit the trend filtering
#' estimate, which will ease the ADMM algorithm's convergence. This can be
#' very useful when a signal is so well-sampled that very little additional
#' information / predictive accuracy is gained by fitting the trend filtering
#' estimate on the full data set, compared to some subset of it. See the
#' [`cv_trendfilter()`] examples for a case study of this nature. When nothing
#' is passed to `thinning`, the algorithm will automatically detect whether
#' thinning should be applied. This preprocessing procedure is controlled by the
#' `x_tol` parameter below.}
#' \item{`x_tol`}{Controls the automatic detection of when thinning should be
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
#' The following loss functions are automatically computed during cross
#' validation and their CV error curves are returned within the `errors`
#' list of the `'cv_tf'` output object.
#'
#' \enumerate{
#' \item Mean absolute deviations error: \mjsdeqn{\text{MAE}(\lambda) =
#' \frac{1}{n} \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|}
#' \item Weighted mean absolute deviations error:
#' \mjsdeqn{\text{WMAE}(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' \item Mean-squared error: \mjsdeqn{\text{MSE}(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|^2}
#' \item Weighted mean-squared error: \mjsdeqn{\text{WMSE}(\lambda)
#' = \sum_{i=1}^{n}|Y_i - \hat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#' \item log-cosh error: \mjsdeqn{\text{logcosh}(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} \log\left(\cosh\left(Y_i - \hat{f}(x_i; \lambda)\right)\right)}
#' \item Weighted log-cosh error: \mjsdeqn{\text{wlogcosh}(\lambda) =
#' \sum_{i=1}^{n}
#' \log\left(\cosh\left((Y_i - \hat{f}(x_i; \lambda))\sqrt{w_i}\right)\right)}
#' \item Huber loss: \mjsdeqn{\text{Huber}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}L_{\delta}(Y_i; \lambda)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\delta}(Y_i; \lambda) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)| - \delta^2, &
#' $|Y_i - \hat{f}(x_i; \lambda)| > \delta$}}
#' \item Weighted Huber loss: \mjsdeqn{\text{wHuber}(\lambda) =
#' \sum_{i=1}^{n}L_{\delta}(Y_i; \lambda)}
#' \mjsdeqn{\text{where}\;\;\;\;L_{\delta}(Y_i; \lambda) = \cases{
#' |Y_i - \hat{f}(x_i; \lambda)|^2w_i, &
#' $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} \leq \delta$ \cr
#' 2\delta|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} -
#' \delta^2, & $|Y_i - \hat{f}(x_i; \lambda)|\sqrt{w_i} > \delta$}}
#' \item Mean-squared logarithmic error: \mjsdeqn{\text{MSLE}(\lambda) =
#' \frac{1}{n}\sum_{i=1}^{n}
#' \left|\log(Y_i + 1) - \log(\hat{f}(x_i; \lambda) + 1)\right|}
#' }
#' where \mjseqn{w_i:=} `weights[i]`.
#'
#' @return An object of class `'cv_tf'`. This is a list with the following
#' elements:
#' \describe{
#' \item{`lambdas`}{Vector of candidate hyperparameter values (always returned
#' in descending order).}
#' \item{`edfs`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every candidate hyperparameter value in `lambdas`.}
#' \item{`errors`}{A named list of vectors, with each representing the
#' cross validation error curve for a given loss function. The first 9
#' vectors of the list correspond to MAE, WMAE, MSE, WMSE, log-cosh error,
#' weighted log-cosh error, Huber loss, weighted Huber loss, and MSLE. If any
#' custom loss functions were passed to `loss_funcs`, their cross validation
#' curves will follow the first 9.}
#' \item{`se_errors`}{Standard errors for each of the cross validation error
#' curves in `errors`, within a named list of the same structure.}
#' \item{`lambda_min`}{A named vector with length equal to `length(errors)`,
#' containing the hyperparameter value that minimizes the cross validation error
#' curve, for every loss function.}
#' \item{`lambda_1se`}{A named vector with length equal to `length(errors)`,
#' containing the "1-standard-error rule" hyperparameter, for every loss
#' function. The "1-standard-error rule" hyparameter is the largest
#' hyperparameter value (corresponding to the smoothest trend filtering
#' estimate) that has a CV error within one standard error of the minimum CV
#' error. It serves as an Occam's razor-like heuristic. That is, given two
#' models with approximately equal performance, it may be wise to opt for the
#' simpler model, i.e. the model with fewer effective degrees of freedom.}
#' \item{`edf_min`}{A named vector with length equal to `length(errors)`,
#' containing the number of effective degrees of freedom in the trend filtering
#' estimator that minimizes the CV error curve, for every loss function.}
#' \item{`edf_1se`}{A named vector with length equal to `length(errors)`,
#' containing the number of effective degrees of freedom in the
#' "1-standard-error rule" trend filtering estimator, for every type of
#' validation error.}
#' \item{`i_min`}{A named vector with length equal to `length(errors)`,
#' containing the index of `lambdas` that yields the minimum of the CV error
#' curve, for every loss function.}
#' \item{`i_1se`}{A named vector with length equal to `length(errors)`,
#' containing the index of `lambdas` that gives the "1-standard-error rule"
#' hyperparameter value, for every loss function.}
#' \item{`loss_funcs`}{A named list of functions that defines all loss functions
#' evaluated during cross validation.}
#' \item{`cost_functional`}{The relative change in the cost functional over the
#' ADMM algorithm's final iteration, for every candidate hyperparameter in
#' `lambdas`.}
#' \item{`n_iter`}{Total number of iterations taken by the ADMM algorithm, for
#' every candidate hyperparameter in `lambdas`. If an element of `n_iter`
#' is exactly equal to the value set by `optimization_params$max_iter`, then the
#' ADMM algorithm stopped before reaching the tolerance set by `obj_tol`. In
#' these situations, you may need to increase `max_iter` to ensure the trend
#' filtering solution has converged with satisfactory precision.}
#' \item{`V`}{The number of folds the data were split into for cross
#' validation.}
#' \item{`tf_model`}{A list of objects that is used internally by other
#' functions that operate on the `cv_trendfilter()` output.}
#' }
#'
#' @export cv_trendfilter
#'
#' @references
#' \bold{Companion references}
#' \enumerate{
#' \item{Politsch et al. (2020a).
#' [Trend filtering – I. A modern statistical tool for time-domain astronomy and astronomical spectroscopy](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413). \emph{MNRAS}, 492(3), p. 4005-4018.} \cr
#' \item{Politsch et al. (2020b).
#' [Trend Filtering – II. Denoising astronomical signals with varying degrees of smoothness](
#' https://academic.oup.com/mnras/article/492/3/4019/5704414). \emph{MNRAS},
#' 492(3), p. 4019-4032.}}
#'
#' \bold{Cross validation}
#' \enumerate{
#' \item{Hastie, Tibshirani, and Friedman (2009).
#' [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](
#' https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf).
#' 2nd edition. Springer Series in Statistics. (See Sections 7.10 and 7.12)}}
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
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom magrittr %$% %>% %<>%
#' @importFrom tidyr tibble drop_na
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom stats median sd
cv_trendfilter <- function(x,
                           y,
                           weights,
                           k = 2L,
                           nlambdas = 250L,
                           V = 10L,
                           mc_cores = V,
                           loss_funcs,
                           fold_ids,
                           optimization_params) {
  if (missing(x) || is.null(x)) stop("`x` must be passed.")
  if (missing(y) || is.null(y)) stop("`y` must be passed.")
  if (length(x) != length(y)) stop("`x` and `y` must have equal length.")
  if (k < 0L || k != round(k)) stop("`k` must be a nonnegative integer.")
  if (length(y) < k + 2) {
    stop("Insufficient data. Must have `length(y) >= k + 2`.")
  }

  if (k > 2L) {
    stop(
      "Polynomial choices `k > 2` can be algorithmically unstable and their
       performance does not improve upon `k = 2`."
    )
  }

  if (V < 2 || V != round(V)) {
    stop("V must be an integer between 2 and length(x).")
  }

  if (V > 10L) {
    warning(
      "V-fold cross validation with `V > 10` is strongly discouraged for
       trend filtering analyses. `V = 10` is the optimal choice for V-fold
       cross validation in the sense that it yields the best estimates of a
       model's out-of-sample error. And larger choices of `V`, such as
       `V = length(x)` (a.k.a. leave-one-out cross validation), do not have
       their usual computational benefits with trend filtering since it is a
       nonlinear smoother, and the efficiency of LOOCV relies on linearity
       in the regression estimator.",
      call. = FALSE
    )
  }

  if (missing(weights)) weights <- rep_len(1, length(y))
  if (length(weights) == 1) weights <- rep_len(weights, length(y))

  if (!(class(weights) %in% c("numeric", "integer"))) {
    stop(
      "If passed, weights must be a numeric vector with the same length as `x`
       and `y`, or `length(weights) = 1` when the noise is believed to be
       homoskedasatic (i.e. constant variance)."
    )
  }

  if (nlambdas < 0 || nlambdas != round(nlambdas)) {
    stop("`nlambdas` must be a positive integer.")
  } else {
    nlambdas <- nlambdas %>% as.integer()
  }

  mc_cores <- max(c(1, floor(mc_cores)))
  mc_cores <- min(c(detectCores(), V, mc_cores))

  if (mc_cores < V & mc_cores < detectCores() / 2) {
    warning(
      paste(
        "Your machine has", detectCores(), "cores, but you've only configured",
        "`cv_trendfilter()`\nto use", mc_cores, "of them. Letting",
        "`mc_cores = V` can significantly speed up the cross\nvalidation."
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
          stop(paste0(
            "Incorrect input argument structure for the function ",
            "`loss_funcs[[", X, "]]`.\n Each custom loss function ",
            "should have vector input arguments `y`, `tf_estimate`,\n",
            "`weights`, and then compute and return a scalar value for the ",
            "validation error."
          ))
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
    stopifnot(length(fold_ids) == length(x))
    fold_ids <- as.integer(fold_ids)
    if (!all.equal(sort(unique(fold_ids)), 1:V)) {
      stop(
        "`fold_ids` should only contain integer values 1:V, with no empty folds."
      )
    }

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
      warning(paste(
        "Your cross validation folds are imbalanced, beyond what is simply due",
        "to `length(x)` not being divisible by `V`."
      ))
    }
  }

  data_folded <- data_scaled %>%
    mutate(ids = fold_ids) %>%
    group_split(ids, .keep = FALSE)

  lambdas <- get_lambdas(nlambdas, data_scaled, k, thinning, admm_params)

  cv_out <- mclapply(
    1:V,
    FUN = validate_trendfilter,
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
      which(
        errors[[X]] <= errors[[X]][i_min[X]] +
          se_errors[[X]][i_min[X]]
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

  names(lambda_min) <- names(loss_funcs)
  names(lambda_1se) <- names(loss_funcs)
  names(edf_min) <- names(loss_funcs)
  names(edf_1se) <- names(loss_funcs)
  names(errors) <- names(loss_funcs)
  names(i_min) <- names(loss_funcs)
  names(se_errors) <- names(loss_funcs)
  names(i_1se) <- names(loss_funcs)

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
      errors = errors,
      se_errors = se_errors,
      lambda_min = lambda_min,
      lambda_1se = lambda_1se,
      edf_min = edf_min,
      edf_1se = edf_1se,
      i_min = i_min,
      i_1se = i_1se,
      loss_funcs = loss_funcs,
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
                                 loss_funcs,
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

####

# Functions for common validation error functionals

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

#' @importFrom magrittr %>%
Huber <- function(tf_estimate, y, weights, n_stderr = 3) {
  stderr <- sd(y - tf_estimate)
  delta <- n_stderr * stderr
  sapply(X = 1:length(y), FUN = function(X) {
    ifelse(
      abs(y[X] - tf_estimate[X]) <= delta,
      (y[X] - tf_estimate[X])^2,
      2 * delta * mean(abs(y[X] - tf_estimate[X])) - delta^2
    )
  }) %>% sum()
}

#' @importFrom magrittr %>%
wHuber <- function(tf_estimate, y, weights, delta = 3) {
  sapply(X = 1:length(y), FUN = function(X) {
    ifelse(
      abs(y[X] - tf_estimate[X]) * sqrt(weights[X]) <= delta,
      (y[X] - tf_estimate[X])^2 * weights[X],
      2 * delta * abs(y[X] - tf_estimate[X]) * sqrt(weights[X]) - delta^2
    )
  }) %>% sum()
}

MSLE <- function(tf_estimate, y, weights) {
  offset <- min(c(tf_estimate, y, 0))
  mean((log(tf_estimate - offset + 1) - log(y - offset + 1))^2)
}
