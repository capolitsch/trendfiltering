#' (Not yet available) Fit a "relaxed" trend filtering model by computing a
#' weighted average of the standard trend filtering estimate and an
#' (unpenalized) spline that shares the same knot set. The weighted
#' average gives rise to a second model hyperparameter that requires tuning.
#'
#' Fit a relaxed trend filtering model. Generic functions such as [`predict()`],
#' [`fitted()`], and [`residuals()`] may be called on the output.
