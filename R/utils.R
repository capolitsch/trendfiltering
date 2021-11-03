#' Utility functions

#' @importFrom glmgen trendfilter.control.list
#' @noRd
get_optimization_params <- function(optimization_params = NULL, n) {
  stopifnot(class(optimization_params) %in% c("NULL", "list"))

  if (is.null(optimization_params)) {
    optimization_params <- list(
      max_iter = n,
      obj_tol = 1e-10
    )

    thinning <- NULL
  } else {
    stopifnot(all(names(optimization_params) %in%
      c(names(formals(trendfilter.control.list)), "thinning")))

    if (!("max_iter" %in% names(optimization_params))) {
      optimization_params$max_iter <- n
    }
    if (!("obj_tol" %in% names(optimization_params))) {
      optimization_params$obj_tol <- 1e-10
    }
    if (!("thinning" %in% names(optimization_params))) {
      thinning <- NULL
    } else {
      thinning <- optimization_params$thinning
      optimization_params$thinning <- NULL
    }
  }

  return(list(optimization_params = optimization_params, thinning = thinning))
}


#' @importFrom glmgen trendfilter
#' @importFrom dplyr last %>%
#' @importFrom stats approx
#' @noRd
get_lambdas <- function(obj) {
  nlambdas_start <- ifelse(obj$nlambdas >= 150, 100, 50)

  out <- trendfilter(
    x = obj$data_scaled$x,
    y = obj$data_scaled$y,
    weights = obj$data_scaled$weights,
    lambda.min.ratio = 1e-16,
    nlambda = nlambdas_start,
    k = obj$k,
    thinning = obj$thinning,
    control = obj$admm_params
  )

  lambdas_start <- out$lambda
  edfs_start <- out$df

  if (any(out$df == length(obj$x))) {
    inds <- which(out$df == length(obj$x))[-1]
    if (length(inds) > 0) {
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  if (any(out$df <= obj$k + 2)) {
    inds <- which(out$df <= obj$k + 2)
    if (length(inds) > 1) {
      inds <- inds[-last(inds)]
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  obj$lambdas <- c(
    lambdas_start,
    approx(
      x = edfs_start,
      y = log(lambdas_start),
      xout = seq(
        min(edfs_start),
        max(edfs_start),
        length = obj$nlambdas - length(lambdas_start) + 2
      )[-c(1, obj$nlambdas - length(lambdas_start) + 2)]
    )[["y"]] %>%
      suppressWarnings() %>%
      exp()
  ) %>%
    unique() %>%
    sort(decreasing = TRUE)

  return(obj)
}
