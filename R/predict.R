#' Evaluate an optimized trend filtering estimate
#'
#' Evaluate an optimized trend filtering object on a grid of input values. The
#' predict function can be called on an object of class
#' '[`cv_tf`][cv_trendfilter()]' or '[`sure_tf`][sure_trendfilter()]', and the
#' arguments `loss_func` (for `'cv_tf'` objects only) and
#' `lambda_choice` should be used to specify the desired method for optimizing
#' the trend filtering hyperparameter.
#'
#' @param obj An object of class '[`cv_tf`][cv_trendfilter()]' or
#' '[`sure_tf`][sure_trendfilter()]'.
#' @param loss_func (For class `'cv_tf'` only) A string or index specifying
#' which cross validation error curve stored within the `cv_tf` object will be
#' used to choose the trend filtering hyperparameter. The first 4 options are
#' `"WMAE"`, `"WMSE"`, `"MAE"`, and `"MSE"`. Therefore, for example, one could
#' select the CV curve based on mean absolute  deviations error by passing
#' either `loss_func = "MAE"` or `loss_func = 3`. If the user passed custom
#' error functions to [`cv_trendfilter()`], then these will be available as
#' additional options following the first 4, and their string designations can
#' be seen by running `names(obj$cv_loss_funcs)`.
#' @param lambda_choice One of `c("lambda_min", "lambda_1se")`. The choice of
#' hyperparameter that is used for optimized trend filtering estimate. Defaults
#' to `lambda_choice = "lambda_min"`.
#' \itemize{
#' \item{`"lambda_min"`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`"lambda_1se"`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates.
#' }
#' @param x_eval A grid of inputs to evaluate the optimized trend filtering
#' estimate on. Defaults to the observed values `x` when nothing is passed.
#' @param nx_eval Integer. If passed, overrides `x_eval` with
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @details
#' The motivation for using `lambda_choice = "lambda_1se"` is essentially
#' Occam's razor: the two models yield results that are quantitatively very
#' close, so we favor the simpler model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/Papers/ESLII.pdf) for more details on the
#' "one-standard-error rule".
#'
#' @return An object of class `'pred_tf'`. This is a list with the following
#' elements:
#' \describe{
#' \item{x_eval}{}
#' \item{tf_estimate}{}
#' \item{lambdas}{}
#' \item{edfs}{}
#' \item{loss_func}{}
#' \item{validation_errors}{}
#' \item{se_validation_errors}{}
#' \item{lambda_opt}{}
#' \item{edf_opt}{}
#' \item{i_opt}{}
#' \item{tf_model}{}
#' }
#'
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
#' pred_tf <- predict(
#'   cv_tf,
#'   loss_func = "MAE",
#'   lambda_choice = "lambda_1se",
#'   nx_eval = 1500L
#' )
#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %<>% %>%
#' @export
predict.cv_tf <- function(obj,
                          loss_func = "WMAE",
                          lambda_choice = c("lambda_min", "lambda_1se"),
                          x_eval,
                          nx_eval) {
  stopifnot(any(class(obj) == "cv_tf"))
  stopifnot(
    class(loss_func) %in% c("character", "integer", "numeric")
  )

  if (is.character(loss_func)) {
    stopifnot(loss_func %in% names(obj$validation_errors))
  }

  if (is.double(loss_func) | is.integer(loss_func)) {
    if (loss_func != round(loss_func)) {
      loss_func <- which.min(
        abs(loss_func - 1:length(obj$i_min))
      )

      warning(
        cat(paste0(
          "loss_func should either be one of c('",
          paste(names(obj$i_min), collapse = "', '"),
          "'), or an index in 1:", length(obj$i_min),
          ".\nChoosing the closest index option: ", loss_func,
          " ('", names(obj$i_min)[loss_func], "')."
        )),
        call. = FALSE
      )
    }
  }

  if (missing(x_eval) & missing(nx_eval)) {
    x_eval <- obj$tf_model$x %>%
      as.double() %>%
      sort()
  }

  if (!missing(x_eval) & missing(nx_eval)) {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  if (!missing(nx_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  }

  lambda_choice <- match.arg(lambda_choice)

  i_opt <- case_when(
    lambda_choice == "lambda_min" ~ obj$i_min[loss_func],
    lambda_choice == "lambda_1se" ~ obj$i_1se[loss_func]
  )

  names(i_opt) <- NULL

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      obj$tf_model$model_fit,
      lambda = obj$lambdas[i_opt],
      x.new = x_eval / obj$tf_model$x_scale
    )
  ) * obj$tf_model$y_scale

  structure(
    list(
      x_eval = x_eval,
      tf_estimate = tf_estimate,
      lambdas = obj$lambdas,
      edfs = obj$edfs,
      loss_func = loss_func,
      validation_error_func = obj$validation_error_funcs[[loss_func]],
      validation_errors = obj$validation_errors[[loss_func]],
      se_validation_errors = obj$se_validation_errors[[loss_func]],
      lambda_opt = obj$lambdas[i_opt],
      edf_opt = obj$edfs[i_opt],
      i_opt = i_opt,
      V = obj$V,
      tf_model = obj$tf_model
    ),
    class = c("pred_tf", "list")
  )
}


#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %<>% %>%
#' @rdname predict_trendfilter
#' @export
predict.sure_tf <- function(obj,
                            lambda_choice = c("lambda_min", "lambda_1se"),
                            x_eval,
                            nx_eval) {
  stopifnot(any(class(obj) == "sure_tf"))

  if (missing(x_eval) & missing(nx_eval)) {
    x_eval <- obj$tf_model$x %>%
      as.double() %>%
      sort()
  }

  if (!missing(x_eval) & missing(nx_eval)) {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  if (!missing(nx_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  }

  lambda_choice <- match.arg(lambda_choice)

  i_opt <- case_when(
    lambda_choice == "lambda_min" ~ obj$i_min,
    lambda_choice == "lambda_1se" ~ obj$i_1se
  )

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      obj$tf_model$model_fit,
      lambda = obj$lambdas[i_opt],
      x.new = x_eval / obj$tf_model$x_scale
    )
  ) * obj$tf_model$y_scale

  structure(
    list(
      x_eval = x_eval,
      tf_estimate = tf_estimate,
      lambdas = obj$lambdas,
      edfs = obj$edfs,
      validation_errors = obj$validation_errors,
      se_validation_errors = obj$se_validation_errors,
      lambda_opt = obj$lambdas[i_opt],
      edf_opt = obj$edfs[i_opt],
      i_opt = i_opt,
      tf_model = obj$tf_model
    ),
    class = c("pred_tf", "list")
  )
}
