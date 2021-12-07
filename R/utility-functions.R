#' Utility functions for internal/expert use only

#' @noRd
#' @importFrom rlang %||%
get_admm_params <- function(obj_tol = NULL, max_iter = NULL) {
  obj_tol <- obj_tol %||% 1e-10
  max_iter <- obj_tol %||% 200
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
get_lambda_grid_edf_spacing <- function(data,
                                        admm_params,
                                        nlambda,
                                        k = 2L,
                                        lambda_min_ratio = 1e-16,
                                        ...) {
  nlambda_start <- ifelse(nlambda >= 150, 100, 50)
  n <- nrow(data)

  tf_out <- .tf_fit(
    data$x,
    data$y,
    data$weights,
    k = 2L,
    admm_params,
    nlambda = nlambda_start,
    lambda_min_ratio = lambda_min_ratio,
    ...
  )

  nlambda_start <- tf_out$lambda
  edf_start <- tf_out$df

  if (any(edf_start >= nrow(data) - k - 1L)) {
    inds <- which(edf_start >= nrow(data) - k - 1L)[-1]
    if (length(inds) > 0L) {
      nlambda_start <- nlambda_start[-inds]
      edf_start <- edf_start[-inds]
    }
  }

  if (any(edf_start <= k + 1L)) {
    inds <- rev(rev(which(edf_start <= k + 1L))[-1])
    if (length(inds) > 0L) {
      nlambda_start <- nlambda_start[-inds]
      edf_start <- edf_start[-inds]
    }
  }

  approx(
    x = edf_start,
    y = log(nlambda_start),
    xout = seq(
      min(edf_start),
      max(edf_start),
      length = nlambda - length(nlambda_start) + 2
    )[-c(1, nlambda - length(nlambda_start) + 2)]
  )[["y"]] %>%
    suppressWarnings() %>%
    exp() %>%
    unique.default() %>%
    sort.default(decreasing = TRUE)
}
