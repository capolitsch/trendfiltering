#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' [`cv_trendfilter()`] optimizes the trend filtering hyperparameter via V-fold
#' cross validation on a grid of candidate hyperparameter settings and selects
#' the value that minimizes a user-specified loss metric. See details for when
#' to use [sure_trendfilter()] vs. [`cv_trendfilter()`].
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights (Optional) Weights for the observed outputs, defined as the
#' reciprocal variance of the additive noise that contaminates the signal.
#' `weights` can be passed as a scalar when the noise is expected to have equal
#' variance for all observations. Otherwise, `weights` must have the same length
#' as `x` and `y`.
#' @param k Degree of the piecewise polynomials that make up the trend
#' filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param nlambdas The number of hyperparameter settings to test during
#' validation. When nothing is passed to `lambdas` (highly recommended for
#' general use), the grid is automatically constructed by [`cv_trendfilter()`],
#' with `nlambdas` controlling the granularity of the grid.
#' @param lambdas (Optional) Overrides `nlambdas` if passed. The vector of trend
#' filtering hyperparameter values for the grid search. Use of this argument is
#' discouraged unless you know what you are doing.
#' @param nx_eval Integer. If nothing is passed to `x_eval`, then it is defined
#' as `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @param x_eval (Optional) A grid of inputs to evaluate the optimized trend
#' filtering estimate on. May be ignored, in which case the grid is determined
#' by `nx_eval`.
#' @param V Number of folds the data are partitioned into for the V-fold cross
#' validation. Defaults to `V = 10`.
#' @param validation_functional Loss functional to optimize during cross
#' validation. Some common choices can be used by passing an appropriate string
#' --- one of `c("MSE","MAE","WMSE","WMAE")`, i.e. mean-absolute deviations
#' error, mean-squared error, and their weighted counterparts. Defaults to
#' `validation_functional = "WMAE"`.
#'
#' Alternatively, custom validation loss functionals can be used by instead
#' passing a function to `validation_functional`. The function should take three
#' vector arguments --- `y`, `tf_estimate`, and `weights` --- and return a
#' single scalar value for the validation loss. For example,
#' `validation_functional = "WMAE"` is equivalent to passing the following
#' function:
#' ```{r, eval = F}
#' function(tf_estimate, y, weights){
#'   sum(abs(tf_estimate - y) * sqrt(weights) / sum(sqrt(weights)))
#' }
#' ```
#' @param lambda_choice One of `c("lambda_min","lambda_1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' Defaults to `lambda_choice = "lambda_min"`.
#' \itemize{
#' \item{`"lambda_min"`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`"lambda_1se"`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates. The motivation here is essentially Occam's razor: the two models
#' yield results that are quantitatively very close, so we favor the simpler
#' model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
#' for more details on the ``one-standard-error rule''.}
#' @param mc_cores Multi-core computing using the
#' [`parallel`][`parallel::parallel-package`] package: The number of cores to
#' utilize. Defaults to the number of cores detected.
#' @param optimization_params (Optional) A named list of parameter choices to be
#' passed to the trend filtering ADMM algorithm ([Ramdas and Tibshirani 2016](
#' http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf)). See the
#' [glmgen::trendfilter.control.list()] documentation for full details. No
#' technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering convex
#' optimization. Defaults to `max_iter = 600L`. See the `n_iter` element of the
#' function output for the actual number of iterations taken for every
#' hyperparameter choice in `lambdas`. If any of the elements of `n_iter` are
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
#'
#' @details \loadmathjax Our recommendations for when to use [cv_trendfilter()]
#' vs. [sure_trendfilter()] are shown in the table below.
#'
#' A regularly-sampled data set with some discarded pixels (either sporadically
#' or in large consecutive chunks) is still considered regularly sampled. When
#' the inputs are regularly sampled on a transformed scale, we recommend
#' transforming to that scale and carrying out the full trend filtering analysis
#' on that scale. See the example below for a case when the inputs are evenly
#' sampled on the `log10(x)` scale.
#'
#' | Scenario                                                            |  Hyperparameter optimization  |
#' | :---                                                                |                         :---: |
#' | `x` is irregularly sampled                                          |      [`cv_trendfilter()`]     |
#' | `x` is regularly sampled and reciprocal variances are not available |      [`cv_trendfilter()`]     |
#' | `x` is regularly sampled and reciprocal variances are available     |      [`sure_trendfilter()`]   |
#'
#' The formal definitions of the common validation loss functionals available
#' via the options `validation_functional = c("MSE","MAE","WMSE","WMAE")` are
#' stated below.
#'
#' \mjsdeqn{MSE(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|^2}
#' \mjsdeqn{MAE(\lambda) = \frac{1}{n}
#' \sum_{i=1}^{n} |Y_i - \hat{f}(x_i; \lambda)|}
#' \mjsdeqn{WMSE(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#' \mjsdeqn{WMAE(\lambda) = \sum_{i=1}^{n}
#' |Y_i - \hat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' where \mjseqn{w_i:=}`weights[i]`.
#'
#' If constant weights are passed, or if nothing is passed, then the weighted
#' and unweighted counterparts are equivalent. \cr
#'
#' Briefly stated, weighting helps combat heteroskedasticity (varying levels
#' of uncertainty in the output measurements) and absolute error is less
#' sensitive to outliers than squared error.
#'
#' @return An object of class [`cv_tf`][cv_trendfilter]. This is a list with the
#' following elements:
#' \item{x_eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf_estimate}{Optimized trend filtering estimate, evaluated at
#' `x_eval`.}
#' \item{validation_method}{`paste0(V,"-fold CV")`}
#' \item{validation_functional}{Type of error that validation was performed on.
#' Either one of `c("MSE","MAE","WMSE","WMAE")` or a custom function passed by
#' the user.}
#' \item{V}{The number of folds the data are split into for the V-fold cross
#' validation.}
#' \item{lambdas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{generalization_errors}{Vector of cross validation estimates of the
#' trend filtering generalization error, for each hyperparameter value
#' (ordered corresponding to the descending-ordered `lambdas` vector).}
#' \item{se_errors}{The standard errors of the cross validation errors.
#' These are particularly useful for implementing the
#' ``1-standard-error rule''.}
#' \item{lambda_min}{Hyperparameter value that minimizes the cross validation
#' generalization error curve.}
#' \item{lambda_1se}{Largest hyperparameter value that is within one standard
#' error of the minimum hyperparameter's cross validation error.}
#' \item{lambda_choice}{One of `c("lambda_min", "lambda_1se")`. The choice
#' of hyperparameter that is used for the returned trend filtering estimate
#' evaluation `tf_estimate`.}
#' \item{i_min}{Index of `lambdas` that minimizes the cross validation error.}
#' \item{i_1se}{Index of `lambdas` that gives the largest hyperparameter
#' value that has a cross validation error within 1 standard error of the
#' minimum of the cross validation error curves.}
#' \item{edf_min}{Effective degrees of freedom of the optimized trend
#' filtering estimator.}
#' \item{edf_1se}{Effective degrees of freedom of the 1-stand-error rule
#' trend filtering estimator.}
#' \item{n_iter}{The number of iterations needed for the ADMM algorithm to
#' converge within the given tolerance, for each hyperparameter value. If many
#' of these are exactly equal to `max_iter`, then their solutions have not
#' converged with the tolerance specified by `obj_tol`. In which case, it is
#' often prudent to increase `max_iter`.}
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
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so that a
#' smaller, better conditioned data set is used for fitting.}
#' \item{x_scale, y_scale, data_scaled}{For internal use.}
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
#' \dontrun{
#' cv_tf <- cv_trendfilter(
#'   x = EB$phase,
#'   y = EB$flux,
#'   weights = 1 / EB$std_err^2,
#'   validation_functional = "MAE",
#'   lambdas = exp(seq(20, 7, length = 250)),
#'   optimization_params = list(
#'     max_iter = 5e3,
#'     obj_tol = 1e-6,
#'     thinning = T
#'   )
#' )
#' }
#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom magrittr %$% %>%
#' @importFrom tidyr tibble drop_na
cv_trendfilter <- function(x, y, weights,
                           k = 2L, nlambdas = 250L, lambdas, V = 10L,
                           lambda_choice = c("lambda_min", "lambda_1se"),
                           validation_functional = "WMAE",
                           nx_eval = 1500L, x_eval,
                           mc_cores = parallel::detectCores(),
                           optimization_params) {
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

  if (missing(lambdas)) {
    if (nlambdas < 0 || nlambdas != round(nlambdas)) {
      stop("nlambdas must be a positive integer")
      nlambdas <- nlambdas %>% as.integer()
    }
  } else {
    if (min(lambdas) <= 0L) {
      stop("All specified lambda values must be positive.")
    }
    if (length(lambdas) < 25L) {
      warning("Recommended to provide more candidate hyperparameter values.")
    }
    if (!all(lambdas == sort(lambdas, decreasing = T))) {
      warning("Sorting lambdas to descending order.")
    }
  }

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer")) || nx_eval < 1 ||
      nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
  } else {
    if (any(x_eval < min(x) || x_eval > max(x))) {
      stop("x_eval should all be in range(x).")
    }
  }

  if (mc_cores < detectCores()) {
    warning(
      paste0(
        "Your machine has ", detectCores(), " cores. ",
        "Consider increasing mc_cores to speed up computation."
      )
    )
  }

  if (!(class(validation_functional) %in% c("character", "function"))) {
    stop(
      "validation_functional must either be one of c('WMAE','WMSE','MAE','MSE')
      or a function."
    )
  }

  if (class(validation_functional) == "character") {
    if (!(validation_functional %in% c("MAE", "MSE", "WMAE", "WMSE"))) {
      stop(
        "character options for validation_functional are
        c('MAE','MSE','WMAE','WMSE')"
      )
    }
  }
  if (class(validation_functional) == "function") {
    if (!all(c("y", "tf_estimate", "weights") %in%
      names(formals(validation_functional)))) {
      stop(
        "Incorrect input argument structure for function passed to
        validation_functional."
      )
    }
  }

  if (mc_cores > detectCores()) mc_cores <- detectCores()
  mc_cores <- min(floor(mc_cores), V)
  if (missing(weights)) weights <- rep(1, length(y))
  if (length(weights) == 1) weights <- rep(weights, length(y))
  lambda_choice <- match.arg(lambda_choice)

  x <- x %>% as.double()
  y <- y %>% as.double()
  weights <- weights %>% as.double()
  k <- k %>% as.integer()
  V <- V %>% as.integer()

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter(weights != 0) %>%
    drop_na()
  rm(x, y, weights)

  if (missing(optimization_params)) {
    optimization_params <- list(max_iter = 600L, obj_tol = 1e-10)
  }
  thinning <- optimization_params$thinning
  optimization_params$thinning <- NULL
  admm_params <- do.call(trendfilter.control.list, optimization_params)
  x_scale <- median(diff(data$x))
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

  if (missing(lambdas)) {
    lambdas <- exp(seq(16, -10, length = nlambdas))
  } else {
    lambdas <- lambdas %>%
      as.double() %>%
      sort(decreasing = T)
  }

  if (missing(x_eval)) {
    x_eval <- seq(min(data$x), max(data$x), length = nx_eval)
  } else {
    x_eval <- x_eval %>%
      as.double() %>%
      sort()
  }

  obj <- structure(
    list(
      x_eval = x_eval,
      validation_method = paste0(V, "-fold CV"),
      V = V,
      validation_functional = validation_functional,
      lambdas = lambdas,
      lambda_choice = lambda_choice,
      x = data$x,
      y = data$y,
      weights = data$weights,
      k = k,
      thinning = thinning,
      admm_params = admm_params,
      data_scaled = data_scaled,
      x_scale = x_scale,
      y_scale = y_scale
    ),
    class = c("cv_tf", "list")
  )

  rm(
    V, validation_functional, lambdas, nlambdas, lambda_choice, k, thinning,
    data, nx_eval, admm_params, data_scaled, x_eval, x_scale, y_scale
  )

  cv_out <- matrix(unlist(mclapply(
    1:(obj$V),
    FUN = trendfilter_validate,
    data_folded = data_folded,
    obj = obj,
    mc.cores = mc_cores
  )),
  ncol = obj$V
  )

  errors <- cv_out %>%
    rowMeans() %>%
    as.double()
  obj$i_min <- errors %>%
    which.min() %>%
    min()
  obj$lambda_min <- obj$lambdas[obj$i_min]

  se_errors <- rowSds(cv_out) / sqrt(obj$V) %>% as.double()
  obj$i_1se <- which(errors <= errors[obj$i_min] + se_errors[obj$i_min]) %>%
    min()
  obj$lambda_1se <- obj$lambdas[obj$i_1se]

  if (obj$validation_functional %in% c("MSE", "WMSE")) {
    obj$generalization_errors <- errors * obj$y_scale^2
    obj$se_errors <- se_errors * obj$y_scale^2
  }
  if (obj$validation_functional %in% c("MAE", "WMAE")) {
    obj$generalization_errors <- errors * obj$y_scale
    obj$se_errors <- se_errors * obj$y_scale
  }
  if (class(obj$validation_functional) == "function") {
    obj$generalization_errors <- errors
    obj$se_errors <- se_errors
  }

  out <- obj %$% trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  lambda_pred <- case_when(
    obj$lambda_choice == "lambda_min" ~ obj$lambda_min,
    obj$lambda_choice == "lambda_1se" ~ obj$lambda_1se
  )

  obj$n_iter <- out$iter %>% as.integer()
  obj$edfs <- out$df %>% as.integer()
  obj$edf_min <- out$df[obj$i_min] %>% as.integer()
  obj$edf_1se <- out$df[obj$i_1se] %>% as.integer()

  # Increase the TF solution's algorithmic precision for the optimized estimate
  obj$admm_params$obj_tol <- obj$admm_params$obj_tol * 1e-2

  out <- obj %$% trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  # Return the objective tolerance to its previous setting
  obj$admm_params$obj_tol <- obj$admm_params$obj_tol * 1e2

  obj$data_scaled$fitted_values <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_pred,
    x.new = obj$data_scaled$x
  ) %>%
    as.double()
  obj$data_scaled$residuals <- obj$data_scaled$y - obj$data_scaled$fitted_values
  obj$tf_estimate <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_pred,
    x.new = obj$x_eval / obj$x_scale
  ) * obj$y_scale %>%
    as.double()
  obj$fitted_values <- obj$data_scaled$fitted_values * obj$y_scale
  obj$residuals <- obj$y - obj$fitted_values

  obj <- obj[c(
    "x_eval", "tf_estimate", "validation_method", "validation_functional",
    "V", "lambdas", "edfs", "generalization_errors",
    "se_errors", "lambda_min", "lambda_1se", "lambda_choice", "i_min",
    "i_1se", "edf_min", "edf_1se", "n_iter", "x", "y", "weights",
    "fitted_values", "residuals", "k", "thinning", "admm_params",
    "x_scale", "y_scale", "data_scaled"
  )]
  return(obj)
}

trendfilter_validate <- function(validation_index, data_folded, obj) {
  data_train <- data_folded[-validation_index] %>% bind_rows()
  data_validate <- data_folded[[validation_index]]

  out <- trendfilter(
    x = data_train$x,
    y = data_train$y,
    weights = data_train$weights,
    k = obj$k,
    lambda = obj$lambdas,
    thinning = obj$thinning,
    control = obj$admm_params
  )

  tf_validate_preds <- glmgen:::predict.trendfilter(
    out,
    lambda = obj$lambdas,
    x.new = data_validate$x
  ) %>%
    suppressWarnings()

  if (is.character(obj$validation_functional)) {
    loss_func <- case_when(
      obj$validation_functional == "MSE" ~ list(MSE),
      obj$validation_functional == "MAE" ~ list(MAE),
      obj$validation_functional == "WMSE" ~ list(WMSE),
      obj$validation_functional == "WMAE" ~ list(WMAE)
    )[[1]]

    validation_errors <- apply(
      tf_validate_preds, 2,
      loss_func,
      y = data_validate$y,
      weights = data_validate$weights
    ) %>%
      as.double()
  } else {
    loss_func <- obj$validation_functional

    validation_errors <- apply(
      tf_validate_preds * obj$y_scale, 2,
      loss_func,
      y = data_validate$y * obj$y_scale,
      weights = data_validate$weights / obj$y_scale^2
    ) %>%
      as.double()
  }
}

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
