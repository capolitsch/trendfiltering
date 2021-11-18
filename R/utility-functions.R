#' Utility functions for experts only
#' @useDynLib glmgen tf_R
#' @importFrom dplyr last
#' @noRd
make_lambda_grid_edf_spacing <- function(x,
                                         y,
                                         weights,
                                         admm_params,
                                         nlambdas,
                                         lambda_min_ratio = 1e-16,
                                         k = 2L) {
  nlambdas_start <- ifelse(nlambdas >= 150, 100, 50)
  n <- length(y)

  tf_out <- .Call("tf_R",
    sX = x,
    sY = y,
    sW = weights,
    sN = n,
    sK = k,
    sFamily = 0L,
    sMethod = 0L,
    sBeta0 = NULL,
    sLamFlag = 0L,
    sLambda = rep(0, nlambdas),
    sNlambda = nlambdas,
    sLambdaMinRatio = lambda_min_ratio,
    sVerbose = 0L,
    sControl = admm_params,
    PACKAGE = "glmgen"
  )

  lambdas_start <- tf_out$lambda
  edfs_start <- tf_out$df

  if (any(edfs_start >= n - k - 1L)) {
    inds <- which(edfs_start >= n - k - 1L)[-1]
    if (length(inds) > 0L) {
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  if (any(edfs_start <= k + 1L)) {
    inds <- rev(rev(which(edfs_start <= k + 1L))[-1])
    if (length(inds) > 0L) {
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  approx(
    x = edfs_start,
    y = log(lambdas_start),
    xout = seq(
      min(edfs_start),
      max(edfs_start),
      length = nlambdas - length(lambdas_start) + 2
    )[-c(1, nlambdas - length(lambdas_start) + 2)]
  )[["y"]] %>%
    suppressWarnings() %>%
    exp() %>%
    unique() %>%
    sort(decreasing = TRUE)
}


#' @importFrom glmgen trendfilter
#' @importFrom dplyr last %>%
#' @importFrom stats approx
#' @noRd
get_lambdas <- function(nlambdas, data, k, thinning, admm_params) {
  nlambdas_start <- ifelse(nlambdas >= 150, 100, 50)

  out <- trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    lambda.min.ratio = 1e-16,
    nlambda = nlambdas_start,
    k = k,
    thinning = thinning,
    control = admm_params
  )

  lambdas_start <- out$lambda
  edfs_start <- out$df

  if (any(out$df == nrow(data))) {
    inds <- which(out$df == nrow(data))[-1]
    if (length(inds) > 0) {
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  if (any(out$df <= k + 2)) {
    inds <- which(out$df <= k + 2)
    if (length(inds) > 1) {
      inds <- inds[-last(inds)]
      lambdas_start <- lambdas_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  c(
    lambdas_start,
    approx(
      x = edfs_start,
      y = log(lambdas_start),
      xout = seq(
        min(edfs_start),
        max(edfs_start),
        length = nlambdas - length(lambdas_start) + 2
      )[-c(1, nlambdas - length(lambdas_start) + 2)]
    )[["y"]] %>%
      suppressWarnings() %>%
      exp()
  ) %>%
    unique() %>%
    sort(decreasing = TRUE)
}
