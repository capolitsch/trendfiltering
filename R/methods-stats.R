#' Predict from a trend filtering model
#'
#' Get predictions from a "[`trendfilter`][`trendfilter()`]" object.
#'
#' @param object
#'   An object of class/subclass "[`trendfilter`][`trendfilter()`]".
#' @param ...
#'   Additional named arguments.
#'
#' @details
#' As of now, the additional parameters that may be passed are:
#' \describe{
#' \item{`lambda`}{One or more lambda values to evaluate predictions for.
#' By default, predictions are computed for every model in `object$lambda`.}
#' \item{`x_eval`}{Vector of inputs where the trend filtering model(s) will be
#' evaluated. By default, `x_eval = object$x`.}
#' \item{`zero_tol`}{(For experts) Threshold parameter that controls the point
#' at which a small coefficient value is set to zero. Defaults to
#' `zero_tol = 1e-10`.}
#' }
#'
#' @importFrom glmgen .tf_predict
#' @importFrom rlang %||%
#' @rdname predict.trendfilter
#' @export
predict.trendfilter <- function(object, ...) {
  stopifnot(
    any(class(object) == "trendfilter") & any(class(object) == "trendfiltering")
  )

  extra_args <- list(...)

  if (any(names(extra_args) == "lambda")) {
    lambda <- extra_args$lambda
    extra_args$lambda <- NULL
    stopifnot(is.numeric(lambda))
    stopifnot(min(lambda) >= 0L)
    if (!all(lambda %in% object$lambda)) {
      stop("`lambda` must only contain values in `object$lambda`.")
    }
  } else {
    lambda <- NULL
  }

  lambda_flag <- is.null(lambda)
  lambda <- lambda %||% object$lambda

  if (any(names(extra_args) == "x_eval")) {
    x_eval <- extra_args$x_eval
    extra_args$x_eval <- NULL
    stopifnot(is.numeric(x_eval))
    if (any(x_eval < min(object$x)) | any(x_eval > max(object$x))) {
      warning("One or more elements of `x_eval` are outside `range(x)`.",
              call. = FALSE)
    }
  } else {
    x_eval <- NULL
  }

  x_flag <- is.null(x_eval)
  x_eval <- x_eval %||% object$x

  if (any(names(extra_args) == "zero_tol")) {
    zero_tol <- extra_args$zero_tol
    extra_args$zero_tol <- NULL
    stopifnot(is.numeric(zero_tol))
    stopifnot(zero_tol >= 0L)
  } else {
    zero_tol <- NULL
  }

  if (length(x_eval) == length(object$x)) {
    if (all.equal(x_eval, object$x)) {
      x_flag <- TRUE
    }
  }

  if (x_flag & lambda_flag) {
    return(object$fitted_values)
  }

  if (length(object$lambda) == 1) {
    fitted_values <- as.numeric(object$fitted_values)
  } else {
    inds <- match(lambda, object$lambda)
    fitted_values <- object$fitted_values[, inds]
  }

  if (x_flag) {
    return(fitted_values)
  } else {
    p <- .tf_predict(
      obj = object,
      lambda = lambda,
      x_eval = x_eval,
      fitted_values = matrix(fitted_values, ncol = 1),
      zero_tol = zero_tol
    )

    if (length(object$lambda) == 1) {
      return(p)
    } else{
      return(matrix(p, ncol = length(lambda)))
    }
  }
}


#' Fitted values of a trend filtering model
#'
#' Get fitted values from a "[`trendfilter`][`trendfilter()`]" object.
#'
#' @param object
#'   Object of class "[`trendfilter`][`trendfilter()`]".
#' @param ...
#'   Additional named arguments.
#'
#' @details
#' As of now, the additional named arguments that can be passed are:
#' \describe{
#' \item{`lambda`}{One or more lambda values to compute fitted values for.
#' By default, fitted values are computed for every model in
#' `object$lambda`.}
#' \item{`x_eval`}{Vector of inputs where the trend filtering model(s) will be
#' evaluated. By default, `x_eval = object$x`.}
#' \item{`zero_tol`}{(For experts) Threshold parameter that controls the point
#' at which a small coefficient value is set to zero. Defaults to
#' `zero_tol = 1e-10`.}
#' }
#'
#' @importFrom stats predict
#' @aliases fitted.values.trendfilter fitted.values.cv_trendfilter
#' @aliases fitted.values.sure_trendfilter fitted.values.bootstrap_trendfilter
#' @aliases fitted.cv_trendfilter fitted.sure_trendfilter
#' @aliases fitted.bootstrap_trendfilter
#' @rdname fitted.trendfilter
#' @export
fitted.trendfilter <- function(object, ...) {
  stopifnot(
    any(class(object) == "trendfilter") & any(class(object) == "trendfiltering")
  )

  extra_args <- list(...)

  if (any(names(extra_args) == "lambda")) {
    lambda <- extra_args$lambda
    extra_args$lambda <- NULL
    stopifnot(is.numeric(lambda))
    stopifnot(min(lambda) >= 0L)
    if (!all(lambda %in% object$lambda)) {
      stop("`lambda` must only contain values in `object$lambda`.")
    }
  } else {
    lambda <- NULL
  }

  if (is.null(lambda)) {
    return(object$fitted_values)
  }

  inds <- match(lambda, object$lambda)
  object$fitted_values[, inds]
}


#' Residuals of a trend filtering model
#'
#' Get residuals from a "[`trendfilter`][`trendfilter()`]" object.
#'
#' @param object
#'   Object of class "[`trendfilter`][`trendfilter()`]".
#' @param ...
#'   Additional named arguments.
#'
#' @details
#' As of now, the additional named arguments that may be passed are:
#' \describe{
#' \item{`lambda`}{One or more lambda values to compute model residuals for.
#' By default, residuals are computed for every model in
#' `object$lambda`.}
#' \item{`x_eval`}{Vector of inputs where the trend filtering model(s) will be
#' evaluated. By default, `x_eval = object$x`.}
#' \item{`zero_tol`}{(For experts) Threshold parameter that controls the point
#' at which a small coefficient value is set to zero. Defaults to
#' `zero_tol = 1e-10`.}
#' }
#'
#' @aliases residuals.cv_trendfilter residuals.sure_trendfilter
#' @aliases residuals.bootstrap_trendfilter resids.trendfilter
#' @aliases resids.cv_trendfilter resids.sure_trendfilter
#' @aliases resids.bootstrap_trendfilter
#' @rdname residuals.trendfilter
#' @export
residuals.trendfilter <- function(object, ...) {
  stopifnot(
    any(class(object) == "trendfilter") & any(class(object) == "trendfiltering")
  )

  extra_args <- list(...)

  if (any(names(extra_args) == "lambda")) {
    lambda <- extra_args$lambda
    extra_args$lambda <- NULL
  } else {
    lambda <- NULL
  }

  if (any(names(extra_args) == "x_eval")) {
    x_eval <- extra_args$x_eval
    extra_args$x_eval <- NULL
  } else {
    x_eval <- NULL
  }

  if (any(names(extra_args) == "zero_tol")) {
    zero_tol <- extra_args$zero_tol
    extra_args$lambda <- NULL
  } else {
    zero_tol <- NULL
  }

  object$y - predict(
    object,
    lambda = lambda,
    x_eval = x_eval,
    zero_tol = zero_tol,
    ...
  )
}


#' @importFrom stats weights
#' @noRd
weights.trendfilter <- function(object, ...) {
  stopifnot(
    any(class(object) == "trendfilter") & any(class(object) == "trendfiltering")
  )
  return(object$weights)
}
