#' Predict method for trend filtering fits
#'
#' Predict values based on a [`sure_tf`][sure_trendfilter()] or
#' [`cv_tf`][cv_trendfilter()] object.
#'
#' @param obj An object of class [`sure_tf`][sure_trendfilter()] or
#' [`cv_tf`][cv_trendfilter].
#' @param x_eval (Optional) Overrides `nx_eval` if passed. A grid of inputs to
#' evaluate the optimized trend filtering estimate on.
#' @param nx_eval Integer. The length of the input grid that the optimized
#' trend filtering estimate is evaluated on; i.e. if nothing is passed to
#' `x_eval`, then it is defined as
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#'
#' @return A list with the following elements:
#' \describe{
#' \item{x_eval}{Input grid used to evaluate the optimized trend filtering
#' estimate on.}
#' \item{tf_estimate}{Optimized trend filtering estimate, evaluated at `x_eval`.}
#' \item{fitted_values}{Optimized trend filtering estimate, evaluated at the
#' observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted_values`}
#' }
#'
#' @export

#' @importFrom glmgen trendfilter
#' @importFrom dplyr mutate
#' @importFrom magrittr %<>% %>% %$%
predict.sure_tf <- function(obj, lambda, x_eval, nx_eval) {
  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
  } else {
    if (any(x_eval < min(obj$x) || x_eval > max(obj$x))) {
      stop("x_eval should all be in range(x).")
    }
  }

  if (missing(x_eval)) {
    x_eval <- seq(min(obj$x), max(obj$x), length = nx_eval)
  } else {
    x_eval %<>%
      as.double() %>%
      sort()
  }

  out <- obj %$% trendfilter(
    data_scaled$x,
    data_scaled$y,
    data_scaled$weights,
    lambda = lambda_pred,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  obj$tf_estimate <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_pred,
    x.new = obj$x_eval / obj$x_scale
  ) * obj$y_scale %>%
    as.double()

  obj$data_scaled$fitted_values <- glmgen:::predict.trendfilter(
    out,
    lambda = lambda_pred,
    x.new = obj$data_scaled$x
  ) %>%
    as.double()

  obj$data_scaled %<>% mutate(residuals = y - fitted_values)
  obj$fitted_values <- obj$data_scaled$fitted_values * obj$y_scale
  obj$residuals <- obj$data_scaled$residuals * obj$y_scale^2
}
