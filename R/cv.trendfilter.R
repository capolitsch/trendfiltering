#' Optimize the trend filtering hyperparameter by V-fold cross validation
#'
#' `cv.trendfilter` optimizes the trend filtering hyperparameter via V-fold
#' cross validation on a grid of candidate hyperparameter settings and selects
#' the value that minimizes a user-specified loss metric. See details for when
#' to use \code{\link{`SURE.trendfilter`}} vs. `cv.trendfilter`.
#'
#' @param x Vector of observed values for the input variable.
#' @param y Vector of observed values for the output variable.
#' @param weights Weights for the observed outputs, defined as the reciprocal
#' variance of the error distribution. `weights` can be passed as a scalar when
#' all errors are believed to have equal variance. Otherwise, `weights` must
#' have the same length as `x` and `y`.
#' @param k Degree of the piecewise polynomials that make up the trend
#' filtering estimate. Defaults to `k = 2` (i.e. a piecewise quadratic
#' estimate). Must be one of `k = 0,1,2`. Higher order polynomials are
#' disallowed since their smoothness is indistinguishable from `k = 2` and
#' their use can lead to instability in the convex optimization.
#' @param ngammas The number of hyperparameter settings to test during
#' validation. When `gammas` is left blank (highly recommended for general use),
#' the grid is automatically chosen by `SURE.trendfilter`, with `ngammas`
#' controlling the granularity of the grid.
#' @param gammas Overrides `ngammas` if passed. The vector of trend filtering
#' hyperparameter values for the grid search. Use is discouraged unless you
#' know what you are doing.
#' @param x.eval A grid of inputs to evaluate the optimized trend filtering
#' estimate on. May be ignored, in which case the grid is determined by
#' `nx.eval`.
#' @param nx.eval Integer. If nothing is passed to `x.eval`, then it is defined
#' as `x.eval = seq(min(x), max(x), length = nx.eval)`.
#' @param V The number of folds the data are split into for the V-fold cross
#' validation. Defaults to `V = 10` (recommended).
#' @param loss.metric Type of error to optimize during cross
#' validation. One of `c("MAE","MSE","WMAE","WMSE")`, i.e. mean-absolute
#' deviations error, mean-squared error, and their weighted counterparts.
#' Defaults to `"WMAE"`.
#' @param gamma.choice One of `c("gamma.min","gamma.1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' \itemize{
#' \item{`gamma.min`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`gamma.1se`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates. The motivation here is essentially Occam's razor: the two models
#' yield results that are quantitatively very close, so we favor the simpler
#' model.}
#' @param optimization.params A named list of parameters that contains all
#' parameter choices to be passed to the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the [glmgen::trendfilter.control.list()]
#' documentation for full details.
#' No technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering
#' convex optimization. Defaults to `max_iter = 600L`. Increase this if
#' the trend filtering estimate does not appear to have fully converged to a
#' reasonable estimate of the signal.
#' \item{`obj_tol`}: The tolerance used in the convex optimization stopping
#' criterion; when the relative change in the objective function is less than
#' this value, the algorithm terminates. Decrease this if the trend filtering
#' estimate does not appear to have fully converged to a reasonable estimate of
#' the signal.
#' \item{`thinning`}: Logical. If `TRUE`, then the data are preprocessed so that
#' a smaller, better conditioned data set is used for fitting. When left `NULL`
#' (the default setting), the optimization will automatically detect whether
#' thinning should be applied (i.e. cases in which the numerical fitting
#' algorithm will struggle to converge). This preprocessing procedure is
#' controlled by the `x_tol` argument below.
#' \item{`x_tol`}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then we thin the data.
#' }
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected.
#' @param ... Additional named arguments to be passed to
#' [glmgen::trendfilter.control.list()].
#'
#' @return An object of class 'cv.trendfilter'. This is a list with the
#' following elements:
#' \item{x.eval}{Input grid to evaluate the optimized trend filtering estimate
#' on.}
#' \item{tf.estimate}{Optimized trend filtering estimate, evaluated at `x.eval`.}
#' \item{validation.method}{\code{paste0(V,"-fold CV")}}
#' \item{V}{The number of folds the data are split into for the V-fold cross
#' validation.}
#' \item{loss.metric}{Type of error that validation was performed on.
#' One of \code{c("WMAE","WMSE","MAE","MSE")}.}
#' \item{gammas}{Vector of hyperparameter values tested during validation. This
#' vector will always be returned in descending order, regardless of the
#' ordering provided by the user. The indices `i.min` and `i.1se` correspond to
#' this descending ordering.}
#' \item{errors}{Vector of cross validation errors for the given hyperparameter
#' values.}
#' \item{se.errors}{The standard errors of the cross validation errors.
#' These are particularly useful for implementing the ``1-standard-error rule''.
#' The 1-SE rule favors a smoother trend filtering estimate by, instead of
#' using the hyperparameter that minimizes the CV error, instead uses the
#' largest hyperparameter that has a CV error within 1 standard error of the
#' smallest CV error.}
#' \item{gamma.min}{Hyperparameter value that minimizes the SURE error curve.}
#' \item{gamma.1se}{The largest hyperparameter value that is still within one
#' standard error of the minimum hyperparameter's cross validation error.}
#' \item{gamma.choice}{One of `c("gamma.min","gamma.1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.}
#' \item{edfs}{Vector of effective degrees of freedom for trend filtering
#' estimators fit during validation.}
#' \item{edf.min}{The effective degrees of freedom of the optimally-tuned trend
#' filtering estimator.}
#' \item{edf.1se}{The effective degrees of freedom of the 1-stand-error rule
#' trend filtering estimator.}
#' \item{i.min}{The index of `gammas` that minimizes the cross validation error.}
#' \item{i.1se}{The index of `gammas` that gives the largest hyperparameter
#' value that has a cross validation error within 1 standard error of the
#' minimum of the cross validation error curves.}
#' \item{x}{Vector of observed inputs.}
#' \item{y}{Vector of observed outputs.}
#' \item{weights}{A vector of weights for the observed outputs. These are
#' defined as `weights = 1 / sigmas^2`, where `sigmas` is a vector of
#' standard errors of the uncertainty in the observed outputs.}
#' \item{fitted.values}{The optimized trend filtering estimate of the signal,
#' evaluated at the observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted.values`}
#' \item{k}{The degree of the trend filtering estimator.}
#' \item{optimization.params}{A list of parameters that control the trend
#' filtering convex optimization.}
#' \item{n.iter}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each hyperparameter
#' value. If many of these are exactly equal to `max_iter`, then their
#' solutions have not converged with the tolerance specified by `obj_tol`.
#' In which case, it is often prudent to increase `max_iter`.}
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so
#' that a smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{For internal use.}
#'
#' @details \loadmathjax
#'
#' | Scenario                                                 | Hyperparameter optimization | `bootstrap.algorithm` |
#' | :------------                                            |     ------------:           |         ------------: |
#' | `x` is irregularly sampled                               | Use `cv.trendfilter`        | "nonparametric"       |
#' | `x` is regularly sampled and `weights` are not available | Use `cv.trendfilter`        | "wild"                |
#' | `x` is regularly sampled and `weights` are available     | Use `SURE.trendfilter`      | "parametric"          |
#'
#' Our recommendations for when to use \code{\link{`cv.trendfilter`}} vs.
#' `SURE.trendfilter`, as well as each of the settings for `bootstrap.algorithm`
#' are shown in the table above.
#'
#' A regularly-sampled data set with some discarded pixels (either sporadically
#' or in large consecutive chunks) is still considered regularly sampled. When
#' the inputs are regularly sampled on a transformed scale, we recommend
#' transforming to that scale and carrying out the full trend filtering analysis
#' (using `SURE.trendfilter`) on that scale. See the example below for a case
#' when the inputs are evenly sampled on the `log10(x)` scale.
#'
#'  Below we define the various types of validation error that
#' can be used with `cv.trendfilter` by passing the appropriate string
#' (one of `c("WMAE","WMSE","MAE","MSE")`) to the `loss.metric`
#' argument. For the weighted validation errors, `weights` must be passed.
#' \mjsdeqn{WMAE(\gamma) = \sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' \mjsdeqn{WMSE(\gamma) = \sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|^2\frac{w_i}{\sum_jw_j}}
#' \mjsdeqn{MAE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|}
#' \mjsdeqn{MSE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|^2}
#' where \mjseqn{w_i} is the \mjseqn{i}th element of the `weights` vector.
#'
#' Concisely stated, weighting helps combat heteroskedasticity and
#' absolute error decreases sensitivity to outliers. If `weights = NULL`, then
#' the weighted and unweighted counterparts are equivalent. \cr
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
#' with Applications in R}. Springer. (See Section 5.1; Less technical than
#' ESL)} \cr
#' \item{Tibshirani (2013).
#' \href{https://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf}{
#' Model selection and validation 2: Model assessment, more cross-validation}.
#' \emph{36-462: Data Mining course notes} (Carnegie Mellon University).}}
#'
#' @seealso \code{\link{SURE.trendfilter}}, \code{\link{bootstrap.trendfilter}}
#'
#' @examples
#' data(eclipsing_binary)
#'
#' opt <- cv.trendfilter(EB$phase, EB$flux, 1 / EB$std.err ^ 2,
#'                       loss.metric = "MAE",
#'                       optimization.params = list(max_iter = 5e3, obj_tol = 1e-6, thinning = T))


#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom magrittr %$% %>%
#' @importFrom tidyr tibble drop_na
cv.trendfilter <- function(x, y, weights,
                           k = 2L, ngammas = 250L, gammas,
                           V = 10L, gamma.choice = c("gamma.min","gamma.1se"),
                           loss.metric = c("WMAE","WMSE","MAE","MSE"),
                           x.eval, nx.eval = 1500L,
                           optimization.params = list(max_iter = 600L, obj_tol = 1e-10),
                           mc.cores = detectCores(),
                           ...){

  if ( missing(x) || is.null(x) ) stop("x must be passed.")
  if ( missing(y) || is.null(y) ) stop("y must be passed.")
  if ( length(x) != length(y) ) stop("x and y must have the same length.")

  if ( !is.null(weights) ){
    if ( !(length(weights) %in% c(1,length(y))) ){
      stop("weights must either be have length 1 or length(y), or be NULL.")
    }
  }

  if ( length(y) < k + 2 ){
    stop("y must have length >= k + 2 for kth order trend filtering.")
  }

  if ( k < 0 || k != round(k) ){
    stop("k must be a nonnegative integer.")
  }

  if ( k > 3 ){
    stop(paste0("Large k leads to generally worse conditioning.\n",
                "k = 0,1,2 are the most stable choices."))
  }

  if ( k == 3 ){
    warning(paste0("k = 3 can have poor conditioning...\n",
                   "k = 2 is more stable and visually indistinguishable."))
  }

  if ( !missing(gammas) ){
    if ( min(gammas) < 0L ){
      stop("All specified gamma values must be positive.")
    }
    if ( length(gammas) < 25L ) stop("gammas must have length >= 25.")
  }

  if ( !missing(nx.eval) ){
    if ( nx.eval != round(nx.eval) ) stop("nx.eval must be a positive integer.")
  }else{
    if ( any(x.eval < min(x) || x.eval > max(x)) ){
      stop("x.eval should all be in range(x).")
    }
  }

  if ( mc.cores < detectCores() ){
    warning(paste0("Your machine has ", detectCores(), " cores. Consider increasing mc.cores to speed up computation."))
  }

  if ( mc.cores > detectCores() ) mc.cores <- detectCores()
  if ( length(weights) == 1 ) weights <- rep(weights, length(y))
  if ( length(weights) == 0 ) weights <- rep(1, length(y))

  mc.cores <- min(mc.cores, V)
  gamma.choice <- match.arg(gamma.choice)
  loss.metric <- match.arg(loss.metric)

  data <- tibble(x, y, weights) %>%
    arrange(x) %>%
    filter( weights != 0 ) %>%
    drop_na
  rm(x,y,weights)

  thinning <- optimization.params$thinning
  optimization.params <- trendfilter.control.list(max_iter = optimization.params$max_iter,
                                                  obj_tol = optimization.params$obj_tol,
                                                  ...)
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  optimization.params$x_tol <- optimization.params$x_tol / x.scale

  data.scaled <- data %>%
    mutate(x = x / x.scale,
           y = y / y.scale,
           weights = weights * y.scale ^ 2)

  data.folded <- data.scaled %>%
    group_split( sample( rep_len(1:V, nrow(data.scaled)) ), .keep = FALSE )

  if ( missing(gammas) ){
    gammas <- exp(seq(16, -10, length = ngammas))
  }else{
    gammas <- sort(gammas, decreasing = T)
  }

  if ( !missing(x.eval) ){
    x.eval <- seq(min(data$x), max(data$x), length = nx.eval)
  }else{
    x.eval <- sort(x.eval)
  }

  obj <- structure(list(x.eval = x.eval,
                        validation.method = paste0(V,"-fold CV"),
                        V = V,
                        loss.metric = loss.metric,
                        gammas = gammas,
                        gamma.choice = gamma.choice,
                        x = data$x,
                        y = data$y,
                        weights = data$weights,
                        k = k,
                        thinning = thinning,
                        optimization.params = optimization.params,
                        data.scaled = data.scaled,
                        x.scale = x.scale,
                        y.scale = y.scale),
                   class = "cv.trendfilter"
                   )

  rm(V,loss.metric,gammas,ngammas,gamma.choice,k,thinning,data,nx.eval,
     optimization.params,data.scaled,x.eval,x.scale,y.scale)

  cv.out <- matrix(unlist(mclapply(1:(obj$V), FUN = trendfilter.validate,
                                   data.folded = data.folded,
                                   obj = obj,
                                   mc.cores = mc.cores)
                          ),
                   ncol = obj$V)

  errors <- rowMeans(cv.out) %>% as.double
  se.errors <- rowSds(cv.out) / sqrt(obj$V) %>% as.double
  obj$i.min <- which.min(errors) %>% min
  obj$i.1se <- which(errors <= errors[obj$i.min] + se.errors[obj$i.min]) %>% min
  obj$gamma.min <- obj$gammas[obj$i.min]
  obj$gamma.1se <- obj$gammas[obj$i.1se]

  if ( obj$loss.metric %in% c("MSE","WMSE") ){
    obj$errors <- errors * obj$y.scale ^ 2
    obj$se.errors <- se.errors * obj$y.scale ^ 2
  }
  if ( obj$loss.metric %in% c("MAE","WMAE") ){
    obj$errors <- errors * obj$y.scale
    obj$se.errors <- se.errors * obj$y.scale
  }

  out <- obj %$% trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gammas,
                             k = k,
                             thinning = thinning,
                             control = optimization.params)

  gamma.pred <- case_when(
    obj$gamma.choice == "gamma.min" ~ obj$gamma.min,
    obj$gamma.choice == "gamma.1se" ~ obj$gamma.1se
  )

  obj$n.iter <- out$iter
  obj$edfs <- out$df
  obj$edf.min <- out$df[obj$i.min]
  obj$edf.1se <- out$df[obj$i.1se]

  # Increase the TF solution's algorithmic precision for the optimized estimate
  obj$optimization.params$obj_tol <- obj$optimization.params$obj_tol * 1e-2

  out <- obj %$% trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gammas,
                             k = k,
                             thinning = thinning,
                             control = optimization.params)

  obj$optimization.params$obj_tol <- obj$optimization.params$obj_tol * 1e2

  obj$data.scaled$fitted.values <- glmgen:::predict.trendfilter(out, lambda = gamma.pred,
                                                                x.new = obj$data.scaled$x) %>%
    as.double
  obj$data.scaled$residuals <- obj$data.scaled$y - obj$data.scaled$fitted.values
  obj$tf.estimate <- glmgen:::predict.trendfilter(out, lambda = gamma.pred,
                                                  x.new = obj$x.eval / obj$x.scale) * obj$y.scale %>%
    as.double
  obj$fitted.values <- obj$data.scaled$fitted.values * obj$y.scale
  obj$residuals <- obj$y - obj$fitted.values

  obj <- obj[c("x.eval","tf.estimate","validation.method","V",
               "loss.metric","gammas","gamma.min","gamma.1se",
               "gamma.choice","errors","se.errors","edfs","edf.min","edf.1se",
               "i.min","i.1se","x","y","weights","fitted.values", "residuals",
               "k","thinning","optimization.params","n.iter","x.scale","y.scale",
               "data.scaled")]
  return(obj)
}


trendfilter.validate <- function(validation.index, data.folded, obj){

  data.train <- data.folded[-validation.index] %>% bind_rows
  data.validate <- data.folded[[validation.index]]

  out <- trendfilter(x = data.train$x,
                     y = data.train$y,
                     weights = data.train$weights,
                     k = obj$k,
                     lambda = obj$gammas,
                     thinning = obj$thinning,
                     control = obj$optimization.params)

  tf.validate.preds <- glmgen:::predict.trendfilter(out, lambda = obj$gammas, x.new = data.validate$x) %>%
    suppressWarnings

  if ( obj$loss.metric == "MSE" ){
    validation.error.mat <- mean( (tf.validate.preds - data.validate$y) ^ 2)
  }
  if ( obj$loss.metric == "MAE" ){
    validation.error.mat <- abs(tf.validate.preds - data.validate$y)
  }
  if ( obj$loss.metric == "WMSE" ){
    validation.error.mat <- (tf.validate.preds - data.validate$y) ^ 2 *
      data.validate$weights / sum(data.validate$weights)
  }
  if ( obj$loss.metric == "WMAE" ){
    validation.error.mat <- abs(tf.validate.preds - data.validate$y) *
      sqrt(data.validate$weights) / sum(sqrt(data.validate$weights))
  }

  colMeans(validation.error.mat) %>% as.double
}


MSE <- function(residuals, weights){
  mean( residuals ^ 2 )
}

MAE <- function(residuals, weights){
  mean( abs(residuals) )
}

WMSE <- function(residuals, weights){

}

WMAE <- function(residuals, weights){

}
