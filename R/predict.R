#' Predict values from a trend filtering object
#'
#' Evaluate an optimized trend filtering object on a grid of input values.
#' The predict function can be called on an object of class
#' [`"cv_tf"`][cv_trendfilter()] or [`"sure_tf"`][sure_trendfilter()], and the
#' arguments `validation_error_metric` (for `"cv_tf"` objects only) and
#' `lambda_choice` should be used to specify the desired method for optimizing
#' the trend filtering hyperparameter.
#'
#' @param obj An object of class [`"cv_tf"`][cv_trendfilter()] or
#' [`"sure_tf"`][sure_trendfilter()].
#' @param validation_error_metric (For class `"cv_tf"` only) A string or
#' index specifying which cross validation error curve stored within the
#' `cv_tf` object will be used to choose the trend filtering hyperparameter.
#' The first 4 options are `"WMAE"`, `"WMSE"`, `"MAE"`, and `"MSE"`.
#' Therefore, for example, one could select the CV curve based on mean absolute
#' deviations error by passing either
#' `validation_error_metric = "MAE"` or `validation_error_metric = 3`. If the
#' user passed custom error functions to [`cv_trendfilter()`], then these will
#' arise as additional options that follow the first 4, and their string
#' designations can be seen by running `names(obj$validation_error_funcs)`.
#' @param lambda_choice One of `c("lambda_min","lambda_1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' Defaults to `lambda_choice = "lambda_min"`.
#' \itemize{
#' \item{`"lambda_min"`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`"lambda_1se"`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates.
#' }
#' @param x_eval (Optional) A grid of inputs to evaluate the optimized trend
#' filtering estimate on.
#' @param nx_eval Integer. The length of the input grid that the optimized
#' trend filtering estimate is evaluated on; i.e. if nothing is passed to
#' `x_eval`, then it is defined as
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @details
#' The motivation for using `lambda_choice = "lambda_1se"` is essentially
#' Occam's razor: the two models yield results that are quantitatively very
#' close, so we favor the simpler model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
#' for more details on the "one-standard-error rule".
#'
#' @return The optimized trend filtering estimate, evaluated at `x_eval`.
#' @rdname predict_trendfilter
#'
#' @seealso [cv_trendfilter()], [sure_trendfilter()]
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
#'
#' tf_preds <- predict(cv_tf,
#'   validation_error_metric = "MAE",
#'   lambda_choice = "lambda_1se"
#' )
#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %>%
#' @export
predict.cv_tf <- function(obj,
                          validation_error_metric = "WMAE",
                          lambda_choice = c("lambda_min", "lambda_1se"),
                          x_eval,
                          nx_eval = 1500L) {
  stopifnot(any(class(obj) == "cv_tf"))
  stopifnot(
    class(validation_error_metric) %in% c("character", "integer", "numeric")
  )

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  } else {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  if (is.character(validation_error_metric)) {
    stopifnot(validation_error_metric %in% names(obj$validation_error_funcs))
  }

  if (is.double(validation_error_metric) | is.integer(validation_error_metric)) {
    if (validation_error_metric != round(validation_error_metric)) {
      validation_error_metric <- which.min(
        abs(validation_error_metric - 1:length(obj$i_min))
      )

      warning(cat(
        paste0(
          "validation_error_metric should either be one of c('",
          paste(names(obj$i_min), collapse = "', '"),
          "'), or an index in 1:", length(obj$i_min),
          ".\nChoosing the closest index option: ", validation_error_metric,
          " ('", names(obj$i_min)[validation_error_metric], "')."
        )
      ))
    }
  }

  lambda_choice <- match.arg(lambda_choice)

  lambda_pred <- case_when(
    lambda_choice == "lambda_min" ~ obj$lambda_min[validation_error_metric],
    lambda_choice == "lambda_1se" ~ obj$lambda_1se[validation_error_metric]
  )

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      obj$tf_model$model_fit,
      lambda = lambda_pred,
      x.new = x_eval / obj$tf_model$x_scale
    )
  ) * obj$tf_model$y_scale

  tibble(x = x_eval, y = tf_estimate)
}


#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %>%
#' @rdname predict_trendfilter
#' @export
predict.sure_tf <- function(obj,
                            lambda_choice = c("lambda_min", "lambda_1se"),
                            x_eval,
                            nx_eval = 250L) {
  stopifnot(any(class(obj) == "sure_tf"))

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  } else {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  lambda_choice <- match.arg(lambda_choice)

  lambda_pred <- case_when(
    lambda_choice == "lambda_min" ~ obj$lambda_min,
    lambda_choice == "lambda_1se" ~ obj$lambda_1se
  )

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      obj$tf_model$model_fit,
      lambda = lambda_pred,
      x.new = x_eval / obj$tf_model$x_scale
    )
  ) * obj$tf_model$y_scale

  tibble(x = x_eval, y = tf_estimate)
}
