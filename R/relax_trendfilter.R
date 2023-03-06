#' Fit a "relaxed" trend filtering model
#'
#' Fit a relaxed trend filtering model by computing a weighted average of the
#' standard trend filtering estimate and an (unpenalized) spline that shares the
#' same knot set. Generic functions such as [`predict()`],
#' [`fitted()`], and [`residuals()`] may be called on the output. The weighted
#' average gives rise to a second model hyperparameter that requires tuning.
#'
#' @param obj
#'   An object of class "[`trendfilter`][trendfilter()]" or
#'   "[`cv_trendfilter`][cv_trendfilter()]". Defaults to `obj = NULL`, in which
#'   case, the rest of the arguments below should be used to supply the data,
#'   model parameters, etc.
#' @param alpha
#'   Weighting parameter
#' @param ...
#'   If `obj == NULL`, additional named arguments to be passed to
#'   [`trendfilter()`].
#'
#' @return An object of class "`relax_trendfilter`". Generic functions such as
#' [`predict()`], [`fitted()`], and [`residuals()`] may be called on any object
#' of class `relax_trendfilter`". A "`relax_trendfilter`" object is a
#' list with the following elements:
#'
relax_trendfilter <- function(obj, alpha, ...) {

  d <- d_mat(k = 2, x)
  p <- predict(fit, x_eval = x)
  knots <- x[order(abs(D %*% p), decreasing = T)[1:(fit$edf - k - 1)] + 1]
  basis <- bs(d$phase, knots = knots, degree = k, intercept = F)
  basis.eval <- bs(x.grid, knots = knots, degree = k, intercept = F)
  y.hat <- basis.eval %*% solve(t(basis) %*% basis) %*% t(basis) %*% d$tmp2
  relaxed.TF <- TF.pred * alpha + y.hat * (1-alpha)

  pred.out <- data.frame(tf.pred = TF.pred, reg.spline = y.hat, relaxed.tf = relaxed.TF)
  names(pred.out) <- c("tf.pred","reg.spline","relaxed.tf")
  out.list <- list(preds = pred.out, knots = knots)

  if ("edf_radius" %in% names(extra_args)) {
    edf_radius <- extra_args$edf_radius
    extra_args$edf_radius <- NULL
  } else {
    edf_radius <- 5
  }


  lambda_grid <- c(
    obj$lambda[
      max(i_opt - edf_radius, 1):min(i_opt + edf_radius, length(obj$edf))
    ],
    edf_opt
  ) %>%
    unique.default() %>%
    sort.default(decreasing = TRUE)


  if ("edf_tol" %in% names(extra_args)) {
    edf_tol <- extra_args$edf_tol
    extra_args$edf_tol <- NULL
  } else {
    edf_tol <- 0.3
  }


  if ((abs(edf_opt - edf_boot) / edf_opt) > edf_tol) {
    return(
      bootstrap_parallel(
        b = 1,
        dat_scaled,
        k,
        admm_params,
        edf_opt,
        lambda_grid,
        sampler,
        x_eval,
        edf_tol,
        zero_tol,
        scale
      )
    )
  }
