#' Get predictions from a trendfilter object
#'
#' Get predictions from a '[`trendfilter`][`trendfilter()`]' object
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
#' @export
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

  if (!is.null(x_eval) && (any(x_eval < min(obj$x) || x_eval > max(obj$x)))) {
    stop("`x_eval` should all be in `range(x)`.")
  }

  x_eval <- (x_eval %||% obj$x) / obj$x_scale

  fitted_values <- fitted(obj, lambda)
  .tf_predict(obj, lambda, x_eval, fitted_values, zero_tol)
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
#' @rdname fitted.trendfilter
#' @export
fitted.trendfilter <- function(obj, lambda = NULL, ...) {
  if (is.null(lambda)) {
    return(obj$fitted_values)
  }

  stopifnot(is.numeric(lambda))
  stopifnot(min(lambda) >= 0L)
  if (!all(lambda %in% obj$lambda)) {
    stop("`lambda` must only contain values in `obj$lambda`.")
  }

  inds <- match(lambda, obj$lambda)
  obj$fitted_values[, inds, drop = FALSE]
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
#' @rdname residuals.trendfilter
#' @export
residuals.trendfilter <- function(obj, lambda = NULL, ...) {
  obj$y - fitted(obj, lambda, ...)
}
