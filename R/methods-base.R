#' Print the output of a trendfilter object
#'
#' @param obj
#'   Object of class [`trendfilter`][trendfilter()].
#'
#' @export
print.trendfilter <- function(obj) {
  cat("\nCall:\n")
  dput(obj$call)
  cat("\nOutput:\n")
  cat(paste(
    class(obj), "model with", length(obj$lambdas), "values of lambda.", "\n\n"
    ))
}


#' Summarize a trendfilter object
#'
#' @param obj
#'   Object of class [`trendfilter`][trendfilter()].
#'
#' @export
summary.trendfilter <- function(obj) {
  df <- apply(obj$beta != 0, 2, sum)
  rss <- colSums((obj$y - predict(obj, type = "response"))^2)
  mat <- cbind(df, obj$lambdas, rss)
  rownames(mat) <- rep("", nrow(mat))
  colnames(mat) <- c("df", "lambdas", "rss")
  class(mat) <- "summary.trendfilter"
  mat
}


#' Print the output of a trendfilter summary object
#'
#' @param obj
#'   Object of class [`trendfilter`][trendfilter()].
#'
#' @export
print.summary.trendfilter <- function(obj) {
  class(obj) <- "matrix"
  print(obj, digits = 4, print.gap = 3)
}
