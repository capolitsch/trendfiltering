#' Get coefficients from a trendfilter object
#'
#' @param obj
#'   Object of class/subclass '[`trendfilter`][`trendfilter()`]'.
#' @param lambda
#'   One of more hyperparameter values to calculate model coefficients for.
#'   Defaults to `lambda = NULL`, in which case, model coefficients are computed
#'   for every hyperparameter value in `obj$lambda`.
#'
#' @aliases coef.cv_trendfilter coef.sure_trendfilter coef.bootstrap_trendfilter
#' @aliases coefficients.cv_trendfilter coefficients.sure_trendfilter
#' @aliases coefficients.bootstrap_trendfilter
#' @rdname coef.trendfilter
coef.trendfilter <- function(obj, lambda = NULL, ...) {
  if (is.null(lambda)) {
    return(obj$beta)
  }

  stopifnot(is.numeric(lambda))
  stopifnot(min(lambda) >= 0L)
  if (!all(lambda %in% obj$lambda)) {
    stop("`lambda` must only contain values in `obj$lambda`.")
  }

  inds <- match(lambda, obj$lambda)
  obj$beta[, inds, drop = FALSE]
}


#' Get predictions from a trendfilter object
#'
#' @param obj
#'   An object of class/subclass '[`trendfilter`][`trendfilter()`]'.
#' @param lambda
#'   One or more lambda values to evaluate predictions for. Defaults to
#'   `lambda = NULL`, in which case, predictions are computed for every model in
#'   `obj$lambda`.
#' @param x_eval
#'   Vector of inputs where the trend filtering model(s) will be evaluated.
#'   Defaults to `x_eval = NULL`, in which case `x_eval = obj$x`.
#' @param zero_tol
#'   Threshold parameter that controls the point at which a small coefficient
#'   value is set to zero. Defaults to `zero_tol = 1e-6`.
#' @param ...
#'   Additional named arguments. Currently unused.

#' @importFrom glmgen .tf_predict
#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %<>% %>%
#' @importFrom rlang %||%
#' @rdname predict.trendfilter
predict.trendfilter <- function(obj,
                                lambda = NULL,
                                x_eval = NULL,
                                zero_tol = 1e-6,
                                ...) {
  stopifnot(any(class(obj) == "trendfilter"))

  lambda <- lambda %||% obj$lambda

  stopifnot(is.numeric(lambda))
  stopifnot(min(lambda) >= 0L)

  if (!all(lambda %in% obj$lambda)) {
    stop("`lambda` must only contain values in `obj$lambda`.")
  }

  inds <- match(lambda, obj$lambda)
  obj$beta[, inds, drop = FALSE]
  x_eval <- x_eval %||% obj$tf_model$x

  if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
    stop("`x_eval` should all be in `range(x)`.")
  }

  .Call(".tf_predict",
        sX = as.double(object$x),
        sBeta = as.double(co),
        sN = length(object$y),
        sK = as.integer(object$k),
        sX0 = as.double(x.new),
        sN0 = length(x.new),
        sNLambda = length(lambda),
        sFamily = family_cd,
        sZeroTol = as.double(zero_tol),
        PACKAGE = "glmgen")
}


#' Get fitted values from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambda
#'   One or more lambda values to compute fitted values for. Defaults to
#'   `lambda = NULL`, in which case, fitted values are computed for all
#'   hyperparameter values in `obj$lambda`.
#' @param zero_tol
#'   Threshold parameter that controls the point at which a small coefficient
#'   value is set to zero. Defaults to `zero_tol = 1e-6`.
#' @param ...
#'   Additional named arguments. Currently unused.
#'
#' @aliases fitted.values.trendfilter fitted.values.cv_trendfilter
#' @aliases fitted.values.sure_trendfilter fitted.values.bootstrap_trendfilter
#' @aliases fitted.cv_trendfilter fitted.sure_trendfilter
#' @aliases fitted.bootstrap_trendfilter
fitted.trendfilter <- function(obj, lambda = NULL, ...) {
  coef.trendfilter(obj, lambda, ...)
}


#' Get residuals from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambda
#'   One or more lambda values to compute residuals for. Defaults to
#'   `lambda = NULL`, in which case, residuals are computed for all
#'   hyperparameter values in `obj$lambda`.
#' @param zero_tol
#'   Threshold parameter that controls the point at which a small coefficient
#'   value is set to zero. Defaults to `zero_tol = 1e-6`.
#' @param ...
#'   Additional named arguments. Currently unused.
#'
#' @aliases residuals.cv_trendfilter residuals.sure_trendfilter
#' @aliases residuals.bootstrap_trendfilter resids.trendfilter
#' @aliases resids.cv_trendfilter resids.sure_trendfilter
#' @aliases resids.bootstrap_trendfilter
residuals.trendfilter <- function(obj, lambda = NULL, ...) {
  obj$y - fitted.trendfilter(obj, lambda, ...)
}
