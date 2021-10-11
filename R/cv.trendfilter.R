#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' `cv.trendfilter` optimizes the trend filtering hyperparameter via V-fold
#' cross validation on a grid of candidate hyperparameter settings and selects
#' the value that minimizes a user-specified loss metric. See details for when
#' to use \code{\link{SURE.trendfilter}} vs. `cv.trendfilter`.
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
#' @param V Number of folds the data are partitioned into for the V-fold cross
#' validation. Defaults to `V = 10`.
#' @param validation.functional Loss functional to optimize during cross
#' validation. Some common choices can be used by passing an appropriate string
#' --- one of `c("MAE","MSE","WMAE","WMSE")`, i.e. mean-absolute deviations
#' error, mean-squared error, and their weighted counterparts. Defaults to
#' `validation.functional = "WMAE"`.
#'
#' Custom validation loss functionals can be used by instead passing a function
#' to `validation.functional`. The function should take three vector arguments
#' --- `y`, `tf.estimate`, and `weights` --- and return a single scalar value
#' for the validation loss. For example, `validation.functional = "WMAE"` is
#' equivalent to passing the following function:
#' ```{r, eval = F}
#' function(tf.estimate, y, weights){
#'   sum(abs(tf.estimate - y) * sqrt(weights) / sum(sqrt(weights)))
#' }
#' ```
#' @param lambda.choice One of `c("lambda.min","lambda.1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' Defaults to `lambda.min`.
#' \itemize{
#' \item{`lambda.min`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`lambda.1se`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates. The motivation here is essentially Occam's razor: the two models
#' yield results that are quantitatively very close, so we favor the simpler
#' model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
#' for more details on the ``one-standard-error rule''.}
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected on the machine.
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
#'
#' @details \loadmathjax Our recommendations for when to use
#' \code{\link{cv.trendfilter}} vs. `SURE.trendfilter`, as well as each of the
#' available settings for `bootstrap.algorithm` are shown in the table below.
#' The corresponding settings that should be used when constructing bootstrap
#' variability bands with `bootstrap.trendfilter` are also shown.
#'
#' A regularly-sampled data set with some discarded pixels (either sporadically
#' or in large consecutive chunks) is still considered regularly sampled. When
#' the inputs are regularly sampled on a transformed scale, we recommend
#' transforming to that scale and carrying out the full trend filtering analysis
#' on that scale. See the example below for a case when the inputs are evenly
#' sampled on the `log10(x)` scale.
#'
#' | Scenario                                                 | Hyperparameter optimization | `bootstrap.algorithm` |
#' | :------------                                            |     ------------:           |         ------------: |
#' | `x` is irregularly sampled                               | Use `cv.trendfilter`        | "nonparametric"       |
#' | `x` is regularly sampled and `weights` are not available | Use `cv.trendfilter`        | "wild"                |
#' | `x` is regularly sampled and `weights` are available     | Use `SURE.trendfilter`      | "parametric"          |
#'
#' The formal definitions of the common validation loss functionals available
#' via the options `validation.functional = c("WMAE","WMSE","MAE","MSE")` are
#' stated below.
#'
#' \mjsdeqn{WMAE(\lambda) = \sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \lambda)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' \mjsdeqn{WMSE(\lambda) = \sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \lambda)|^2\frac{w_i}{\sum_jw_j}}
#' \mjsdeqn{MAE(\lambda) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \lambda)|}
#' \mjsdeqn{MSE(\lambda) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \lambda)|^2}
#' where \mjseqn{w_i} is the \mjseqn{i}th element of the `weights` vector.
#'
#' If constant weights are passed, or if nothing is passed, then the weighted
#' and unweighted counterparts are equivalent. \cr
#'
#' Briefly stated, weighting helps combat heteroskedasticity (varying levels
#' of uncertainty in the output measurements) and absolute error is less
#' sensitive to outliers than squared error.
#'
#' @return An object of class 'cv.trendfilter'. This is a list with the
#' following elements:
#' \item{x.eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf.estimate}{Optimized trend filtering estimate, evaluated at `x.eval`.}
#' \item{validation.method}{\code{paste0(V,"-fold CV")}}
#' \item{V}{The number of folds the data are split into for the V-fold cross
#' validation.}
#' \item{validation.functional}{Type of error that validation was performed on.
#' Either one of `c("WMAE","WMSE","MAE","MSE")` or a custom function passed by
#' the user.}
#' \item{lambdas}{Vector of hyperparameter values evaluated in the grid search
#' (always returned in descending order).}
#' \item{edfs}{Vector of effective degrees of freedom for all trend filtering
#' estimators fit during validation.}
#' \item{generalization.errors}{Vector of cross validation estimates of the
#' trend filtering generalization error, for each hyperparameter value
#' (ordered corresponding to the descending-ordered `lambdas` vector).}
#' \item{se.errors}{The standard errors of the cross validation errors.
#' These are particularly useful for implementing the ``1-standard-error rule''.}
#' \item{lambda.min}{Hyperparameter value that minimizes the cross validation
#' generalization error curve.}
#' \item{lambda.1se}{Largest hyperparameter value that is within one standard
#' error of the minimum hyperparameter's cross validation error.}
#' \item{lambda.choice}{One of `c("lambda.min","lambda.1se")`. The choice
#' of hyperparameter that is used for the returned trend filtering estimate
#' evaluation `tf.estimate`.}
#' \item{i.min}{Index of `lambdas` that minimizes the cross validation error.}
#' \item{i.1se}{Index of `lambdas` that gives the largest hyperparameter
#' value that has a cross validation error within 1 standard error of the
#' minimum of the cross validation error curves.}
#' \item{edf.min}{Effective degrees of freedom of the optimized trend
#' filtering estimator.}
#' \item{edf.1se}{Effective degrees of freedom of the 1-stand-error rule
#' trend filtering estimator.}
#' \item{n.iter}{The number of iterations needed for the ADMM algorithm to
#' converge within the given tolerance, for each hyperparameter value. If many
#' of these are exactly equal to `max_iter`, then their solutions have not
#' converged with the tolerance specified by `obj_tol`. In which case, it is
#' often prudent to increase `max_iter`.}
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
#' @export cv.trendfilter
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
#' Trend Filtering – II. Denoising astronomical signals with varying degrees of
#' smoothness}. \emph{MNRAS}, 492(3), p. 4019-4032.}}
#'
#' \bold{Cross validation}
#' \enumerate{
#' \item{Hastie, Tibshirani, and Friedman (2009).
#' \href{https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf}{
#' The Elements of Statistical Learning: Data Mining, Inference, and
#' Prediction}. 2nd edition. Springer Series in Statistics. (See Sections 7.10
#' and 7.12)} \cr
#' \item{James, Witten, Hastie, and Tibshirani (2013).
#' \href{https://www.statlearning.com/}{An Introduction to Statistical Learning:
#' with Applications in R}. Springer. (See Section 5.1)} \cr
#' \item{Tibshirani (2013).
#' \href{https://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf}{
#' Model selection and validation 2: Model assessment, more cross-validation}.
#' \emph{36-462: Data Mining course notes} (Carnegie Mellon University).}}
#'
#' @seealso \code{\link{SURE.trendfilter}}, \code{\link{bootstrap.trendfilter}}
#'
#' @examples
#' data(eclipsing_binary)
#' head(EB)
#' # |      phase|      flux|  std.err|
#' # |----------:|---------:|--------:|
#' # | -0.4986308| 0.9384845| 0.010160|
#' # | -0.4978067| 0.9295757| 0.010162|
#' # | -0.4957892| 0.9438493| 0.010162|
#'
#' opt <- cv.trendfilter(EB$phase, EB$flux, 1 / EB$std.err^2,
#'   validation.functional = "MAE",
#'   optimization.params = list(max_iter = 5e3, obj_tol = 1e-6, thinning = T)
#' )
#'
#' plot(log(opt$lambdas), opt$generalization.errors, type = "l", lwd = 1.5)
#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom magrittr %$% %>%
#' @importFrom tidyr tibble drop_na
cv.trendfilter <- function(x, y, weights,
                           k = 2L, nlambdas = 250L, lambdas,
                           V = 10L, lambda.choice = c("lambda.min", "lambda.1se"),
                           validation.functional = "WMAE",
                           x.eval, nx.eval = 1500L,
                           mc.cores = detectCores(),
                           optimization.params = list(max_iter = 600L, obj_tol = 1e-10)) {
  if (missing(x) || is.null(x)) stop("x must be passed.")
  if (missing(y) || is.null(y)) stop("y must be passed.")
  if (length(x) != length(y)) stop("x and y must have equal length.")
  if (length(y) < k + 2) stop("Must have >= k + 2 observations.")
  if (k < 0 || k != round(k)) stop("k must be a nonnegative integer.")
  if (k > 2) stop("k > 2 are algorithmically unstable and do not improve upon k = 2.")
  if (V < 2 || V != round(V)) stop("V must be an integer between 2 and length(x).")

  if (!missing(weights)) {
    if (!(length(weights) %in% c(1, length(y))) || !(class(weights) %in% c("numeric", "integer"))) {
      stop("If passed, weights must be numerically-valued with length(weights) = 1 or the same length as x and y.")
    }
  }

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

  if (mc.cores < detectCores()) {
    warning(paste0("Your machine has ", detectCores(), " cores. Consider increasing mc.cores to speed up computation."))
  }

  if (!(class(validation.functional) %in% c("character", "function"))) {
    stop("validation.functional must either be one of c('WMAE','WMSE','MAE','MSE') or a function.")
  }

  if (class(validation.functional) == "character") {
    if (!(validation.functional %in% c("MAE", "MSE", "WMAE", "WMSE"))) {
      stop("character options for validation.functional are c('MAE','MSE','WMAE','WMSE')")
    }
  }
  if (class(validation.functional) == "function") {
    if (!all(c("y", "tf.estimate", "weights") %in% names(formals(validation.functional)))) {
      stop("Incorrect input argument structure for function passed to validation.functional.")
    }
  }

  if (mc.cores > detectCores()) mc.cores <- detectCores()
  mc.cores <- min(floor(mc.cores), V)
  if (missing(weights)) weights <- rep(1, length(y))
  if (length(weights) == 1) weights <- rep(weights, length(y))
  lambda.choice <- match.arg(lambda.choice)

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

  data.folded <- data.scaled %>%
    group_split(sample(rep_len(1:V, nrow(data.scaled))), .keep = FALSE)

  if (missing(lambdas)) {
    lambdas <- exp(seq(16, -10, length = nlambdas))
  } else {
    lambdas <- lambdas %>%
      as.double() %>%
      sort(decreasing = T)
  }

  if (missing(x.eval)) {
    x.eval <- seq(min(data$x), max(data$x), length = nx.eval)
  } else {
    x.eval <- x.eval %>%
      as.double() %>%
      sort()
  }

  obj <- structure(list(
    x.eval = x.eval,
    validation.method = paste0(V, "-fold CV"),
    V = V,
    validation.functional = validation.functional,
    lambdas = lambdas,
    lambda.choice = lambda.choice,
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = k,
    thinning = thinning,
    ADMM.params = ADMM.params,
    data.scaled = data.scaled,
    x.scale = x.scale,
    y.scale = y.scale
  ),
  class = "cv.trendfilter"
  )

  rm(
    V, validation.functional, lambdas, nlambdas, lambda.choice, k, thinning, data,
    nx.eval, ADMM.params, data.scaled, x.eval, x.scale, y.scale
  )

  cv.out <- matrix(unlist(mclapply(1:(obj$V),
    FUN = trendfilter.validate,
    data.folded = data.folded, obj = obj, mc.cores = mc.cores
  )),
  ncol = obj$V
  )

  errors <- cv.out %>%
    rowMeans() %>%
    as.double()
  obj$i.min <- errors %>%
    which.min() %>%
    min()
  obj$lambda.min <- obj$lambdas[obj$i.min]

  se.errors <- rowSds(cv.out) / sqrt(obj$V) %>% as.double()
  obj$i.1se <- which(errors <= errors[obj$i.min] + se.errors[obj$i.min]) %>% min()
  obj$lambda.1se <- obj$lambdas[obj$i.1se]

  if (obj$validation.functional %in% c("MSE", "WMSE")) {
    obj$generalization.errors <- errors * obj$y.scale^2
    obj$se.errors <- se.errors * obj$y.scale^2
  }
  if (obj$validation.functional %in% c("MAE", "WMAE")) {
    obj$generalization.errors <- errors * obj$y.scale
    obj$se.errors <- se.errors * obj$y.scale
  }
  if (class(obj$validation.functional) == "function") {
    obj$generalization.errors <- errors
    obj$se.errors <- se.errors
  }

  out <- obj %$% trendfilter(
    x = data.scaled$x,
    y = data.scaled$y,
    weights = data.scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = ADMM.params
  )

  lambda.pred <- case_when(
    obj$lambda.choice == "lambda.min" ~ obj$lambda.min,
    obj$lambda.choice == "lambda.1se" ~ obj$lambda.1se
  )

  obj$n.iter <- out$iter %>% as.integer()
  obj$edfs <- out$df %>% as.integer()
  obj$edf.min <- out$df[obj$i.min] %>% as.integer()
  obj$edf.1se <- out$df[obj$i.1se] %>% as.integer()

  # Increase the TF solution's algorithmic precision for the optimized estimate
  obj$ADMM.params$obj_tol <- obj$ADMM.params$obj_tol * 1e-2

  out <- obj %$% trendfilter(
    x = data.scaled$x,
    y = data.scaled$y,
    weights = data.scaled$weights,
    lambda = lambdas,
    k = k,
    thinning = thinning,
    control = ADMM.params
  )

  # Return the objective tolerance to its previous setting
  obj$ADMM.params$obj_tol <- obj$ADMM.params$obj_tol * 1e2

  obj$data.scaled$fitted.values <- glmgen:::predict.trendfilter(out,
    lambda = lambda.pred,
    x.new = obj$data.scaled$x
  ) %>%
    as.double()
  obj$data.scaled$residuals <- obj$data.scaled$y - obj$data.scaled$fitted.values
  obj$tf.estimate <- glmgen:::predict.trendfilter(out,
    lambda = lambda.pred,
    x.new = obj$x.eval / obj$x.scale
  ) * obj$y.scale %>%
    as.double()
  obj$fitted.values <- obj$data.scaled$fitted.values * obj$y.scale
  obj$residuals <- obj$y - obj$fitted.values

  obj <- obj[c(
    "x.eval", "tf.estimate", "validation.method", "V",
    "validation.functional", "lambdas", "edfs", "generalization.errors",
    "se.errors", "lambda.min", "lambda.1se", "lambda.choice", "i.min",
    "i.1se", "edf.min", "edf.1se", "n.iter", "x", "y", "weights",
    "fitted.values", "residuals", "k", "thinning", "ADMM.params",
    "x.scale", "y.scale", "data.scaled"
  )]
  return(obj)
}

trendfilter.validate <- function(validation.index, data.folded, obj) {
  data.train <- data.folded[-validation.index] %>% bind_rows()
  data.validate <- data.folded[[validation.index]]

  out <- trendfilter(
    x = data.train$x,
    y = data.train$y,
    weights = data.train$weights,
    k = obj$k,
    lambda = obj$lambdas,
    thinning = obj$thinning,
    control = obj$ADMM.params
  )

  tf.validate.preds <- glmgen:::predict.trendfilter(out, lambda = obj$lambdas, x.new = data.validate$x) %>%
    suppressWarnings()

  if (is.character(obj$validation.functional)) {
    loss.func <- case_when(
      obj$validation.functional == "MSE" ~ list(MSE),
      obj$validation.functional == "MAE" ~ list(MAE),
      obj$validation.functional == "WMSE" ~ list(WMSE),
      obj$validation.functional == "WMAE" ~ list(WMAE)
    )[[1]]

    validation.errors <- apply(tf.validate.preds, 2, loss.func,
      y = data.validate$y,
      weights = data.validate$weights
    ) %>%
      as.double()
  } else {
    loss.func <- obj$validation.functional

    validation.errors <- apply(tf.validate.preds * obj$y.scale, 2, loss.func,
      y = data.validate$y * obj$y.scale,
      weights = data.validate$weights / obj$y.scale^2
    ) %>%
      as.double()
  }
}

MSE <- function(tf.estimate, y, weights) {
  mean((tf.estimate - y)^2)
}

MAE <- function(tf.estimate, y, weights) {
  mean(abs(tf.estimate - y))
}

WMSE <- function(tf.estimate, y, weights) {
  sum((tf.estimate - y)^2 * weights / sum(weights))
}

WMAE <- function(tf.estimate, y, weights) {
  sum(abs(tf.estimate - y) * sqrt(weights) / sum(sqrt(weights)))
}
