#' Get coefficients from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambdas
#'   (Optional) Vector of lambda values to calculate coefficients at. If
#'   missing, will use break points in the fit.
#'
#' @aliases coef.cv_trendfilter coef.sure_trendfilter
#' @export
coef.trendfilter <- function(obj, lambdas = NULL) {
  if (is.null(lambdas)) {
    return(obj$beta)
  }

  # If all lambdas are equal to some computed lambda, return coefficients from
  # `obj$beta`
  if (all(!is.na(index <- match(lambdas, obj$lambdas)))) {
    return(obj$beta[, index, drop = FALSE])
  }

  if (min(lambdas) < 0) stop("All specified lambda values must be nonnegative.")
  if (min(lambdas) < min(obj$lambdas) | max(lambdas) > max(obj$lambdas)) {
    stop("Cannot predict lambda outside the range used when fitting.")
  }

  # If here, need to interpolate `lambdas`
  o <- order(lambdas, decreasing = TRUE)
  o2 <- order(obj$lambdas, decreasing = TRUE)
  lambdas <- lambdas[o]
  knots <- obj$lambdas[o2]
  k <- length(lambdas)
  mat <- matrix(rep(knots, each = k), nrow = k)
  b <- lambdas >= mat
  blo <- max.col(b, ties.method = "first")
  bhi <- pmax(blo - 1, 1)
  i <- bhi == blo
  p <- numeric(k)
  p[i] <- 0
  p[!i] <- ((lambdas - knots[blo]) / (knots[bhi] - knots[blo]))[!i]

  betas <- obj$beta[, o2, drop = FALSE]
  beta <- t((1 - p) * t(betas[, blo, drop = FALSE]) +
    p * t(betas[, bhi, drop = FALSE]))
  colnames(beta) <- as.character(round(lambdas, 3))

  beta[, order(o), drop = FALSE]
}


#' Get predictions from a trendfilter object
#'
#' @param obj
#'   output of [`trendfilter()`]
#' @param lambdas
#'   vector of lambda values to calculate coefficients
#'   at. If missing, will use break points in the fit.
#' @param x_eval
#'   vector of new x points. Set to NULL (the default) to use the
#'   original locations.
#' @param zero_tol
#'   numerical tolerance parameter, for determining whether a
#    coefficient should be rounded to zero
#' @param ...
#'   optional, currently unused, arguments

#' @importFrom glmgen .tf_predict
#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %<>% %>%
#' @aliases predict.cv_trendfilter predict.sure_trendfilter
#' @export
predict.trendfilter <- function(obj,
                                lambdas,
                                x_eval = NULL,
                                zero_tol = 1e-6,
                                ...) {
  stopifnot(any(class(obj) == "trendfilter"))
  stopifnot(class(loss_func) %in% c("character", "numeric"))

  if (is.character(loss_func)) {
    stopifnot(loss_func %in% names(obj$errors))
  } else {
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

  if (is.null(x_eval)) {
    x_eval <- obj$tf_model$x %>%
      as.double() %>%
      sort()
  } else {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("`x_eval` should all be in `range(x)`.")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  extra_args <- list(...)

  i_opt <- case_when(
    lambda == "lambda_min" ~ obj$i_min[loss_func],
    lambda == "lambda_1se" ~ obj$i_1se[loss_func]
  )

  names(i_opt) <- NULL
  lambda <- obj$lambdas[i_opt]
  coefs <- coef(obj, lambda)
}


#' Get fitted values from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambdas
#'   (Optional) Vector of lambda values to calculate fitted values at. If
#'   missing, will use break points in the fit.
#'
#' @aliases fitted.values.trendfilter fitted.values.cv_trendfilter fitted.values.sure_trendfilter fitted.cv_trendfilter fitted.sure_trendfilter
#' @export
fitted.trendfilter <- function(obj, lambdas = NULL) {

}


#' Get residuals from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambdas
#'   (Optional) Vector of lambda values to calculate residuals at. If
#'   missing, will use break points in the fit.
#'
#' @aliases residuals.cv_trendfilter residuals.sure_trendfilter resids.trendfilter resids.cv_trendfilter resids.sure_trendfilter
#' @export
residuals.trendfilter <- function(obj, lambdas = NULL) {

}

