#' Utility functions for internal/expert use only

#' @noRd
get_admm_params <- function(obj_tol = 1e-10, max_iter = 500) {
  list(
    obj_tol = as.double(obj_tol),
    max_iter = as.double(max_iter),
    x_tol = as.double(1e-6),
    rho = as.double(1),
    obj_tol_newton = as.double(1e-5),
    max_iter_newton = as.double(50),
    alpha_ls = as.double(0.5),
    gamma_ls = as.double(0.8),
    max_iter_ls = as.double(30),
    tridiag = as.double(0)
  )
}


#' @importFrom glmgen .tf_fit
#' @importFrom dplyr last
#' @importFrom magrittr %>%
#' @importFrom stats approx
#' @noRd
get_lambda_grid_edf_spacing <- function(df,
                                        admm_params,
                                        nlambda,
                                        k = 2L,
                                        lambda_min_ratio = 1e-16) {
  nlambda_start <- ifelse(nlambda >= 150, 100, 50)

  tf_out <- .Call("tf_R",
    sX = as.double(df$x),
    sY = as.double(df$y),
    sW = as.double(df$weights),
    sN = nrow(df),
    sK = as.integer(k),
    sFamily = 0L,
    sMethod = 0L,
    sBeta0 = NULL,
    sLamFlag = 0L,
    sLambda = as.double(rep(0, nlambda)),
    sNlambda = as.integer(nlambda),
    sLambdaMinRatio = as.double(lambda_min_ratio),
    sVerbose = 0L,
    sControl = admm_params,
    PACKAGE = "glmgen"
  )

  nlambda_start <- tf_out$lambda
  edfs_start <- tf_out$df

  if (any(edfs_start >= nrow(df) - k - 1L)) {
    inds <- which(edfs_start >= nrow(df) - k - 1L)[-1]
    if (length(inds) > 0L) {
      nlambda_start <- nlambda_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  if (any(edfs_start <= k + 1L)) {
    inds <- rev(rev(which(edfs_start <= k + 1L))[-1])
    if (length(inds) > 0L) {
      nlambda_start <- nlambda_start[-inds]
      edfs_start <- edfs_start[-inds]
    }
  }

  approx(
    x = edfs_start,
    y = log(nlambda_start),
    xout = seq(
      min(edfs_start),
      max(edfs_start),
      length = nlambda - length(nlambda_start) + 2
    )[-c(1, nlambda - length(nlambda_start) + 2)]
  )[["y"]] %>%
    suppressWarnings() %>%
    exp() %>%
    unique.default() %>%
    sort.default(decreasing = TRUE)
}
