#' Construct pointwise variability bands via a bootstrap
#'
#' @param obj An object of class '\link{SURE.trendfilter}' or
#' '\link{cv.trendfilter}'.
#' @param level The level of the pointwise variability bands. Defaults to
#' `level = 0.95`.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100`.
#' @param bootstrap.algorithm A string specifying which variation of the
#' bootstrap to use. One of `c("nonparametric","parametric","wild")`. See
#' details below for recommendations on when each option is appropriate.
#' @param return.ensemble \loadmathjax Logical. If `TRUE`, the full trend
#' filtering bootstrap ensemble is returned as an \mjseqn{n \times B} matrix,
#' less any columns from post-hoc pruning (see `prune` below). Defaults to
#' `return.ensemble = FALSE` to save memory.
#' @param prune Logical. If `TRUE`, then the trend filtering bootstrap
#' ensemble is examined for rare instances in which the optimization has
#' stopped at zero knots (likely erroneously), and removes them from the
#' ensemble that is used to compute the variability bands. Defaults to `TRUE`.
#' Do not change this unless you know what you are doing!
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected.
#' @param seed Random number seed (for reproducible results).
#'
#' @details Our recommendations for when to use \code{\link{cv.trendfilter}} vs.
#' `SURE.trendfilter`, as well as each of the available settings for
#' `bootstrap.algorithm` are shown in the table below. See
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{Politsch et al. (2020a)}
#' for more details.
#'
#' | Scenario                                                          | Hyperparameter optimization | `bootstrap.algorithm` |
#' | :------------                                                     |     ------------:           |         ------------: |
#' | `x` is irregularly sampled                                        | `cv.trendfilter`            | "nonparametric"       |
#' | `x` is regularly sampled & reciprocal variances are not available | `cv.trendfilter`            | "wild"                |
#' | `x` is regularly sampled & reciprocal variances are available     | `SURE.trendfilter`          | "parametric"          |
#'
#' @return An object of class 'bootstrap.trendfilter'. This is a comprehensive
#' list containing all of the analysis important information, data, and
#' results:
#' \item{tf.standard.errors}{The standard errors of the optimized trend
#' filtering point estimator.}
#' \item{bootstrap.lower.band}{Vector of lower bounds for the
#' pointwise variability bands, evaluated on `x.eval`.}
#' \item{bootstrap.upper.band}{Vector of upper bounds for the
#' pointwise variability bands, evaluated on `x.eval`.}
#' \item{bootstrap.algorithm}{A string specifying which variation of the
#' bootstrap was used to obtain the variability bands.}
#' \item{level}{The level of the pointwise variability bands.}
#' \item{B}{The number of bootstrap samples used to estimate the pointwise
#' variability bands.}
#' \item{tf.bootstrap.ensemble}{If `return.ensemble = TRUE`, the
#' full trend filtering bootstrap ensemble as an \mjseqn{n \times B} matrix,
#' less any columns from post-hoc pruning (if `prune = TRUE`). Else, this will
#' return `NULL`.}
#' \item{edf.boots}{An integer vector of the estimated number of effective
#' degrees of freedom of each trend filtering bootstrap estimate. These should
#' all be relatively close to `edf.min` (below).}
#' \item{prune}{Logical. If `TRUE`, then the trend filtering bootstrap
#' ensemble is examined for rare instances in which the optimization has
#' stopped at zero knots (likely erroneously), and removes them from the
#' ensemble.}
#' \item{n.pruned}{The number of poorly-converged bootstrap trend filtering
#' estimates pruned from the ensemble.}
#' \item{n.iter.boots}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each bootstrap trend
#' filtering estimate.}
#' \item{...}{Named elements inherited from `obj` --- an object either of class
#' '\link{SURE.trendfilter}' or '\link{cv.trendfilter}'. See the relevant
#' function documentation for details.}
#'
#' @export bootstrap.trendfilter
#'
#' @author
#' \bold{Collin A. Politsch, Ph.D.} \cr
#' Email: collinpolitsch@@gmail.com \cr
#' Website: [collinpolitsch.com](https://collinpolitsch.com/) \cr
#' GitHub: [github.com/capolitsch](https://github.com/capolitsch/) \cr \cr
#'
#' @references
#' \bold{Companion references}
#' \enumerate{
#' \item{Politsch et al. (2020a).
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Trend filtering – I. A modern statistical tool for time-domain astronomy and
#' astronomical spectroscopy}. \emph{MNRAS}, 492(3), p. 4005-4018.} \cr
#' \item{Politsch et al. (2020b).
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' Trend Filtering – II. Denoising astronomical signals with varying degrees of
#' smoothness}. \emph{MNRAS}, 492(3), p. 4019-4032.}}
#'
#' \bold{The Bootstrap and variations}
#' \enumerate{
#' \item{Efron and Tibshirani (1986).
#' \href{https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full}{
#' Bootstrap Methods for Standard Errors, Confidence Intervals, and Other
#' Measures of Statistical Accuracy}.
#' \emph{Statistical Science}, 1(1), p. 54-75.} \cr
#' \item{Mammen (1993).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-21/issue-1/Bootstrap-and-Wild-Bootstrap-for-High-Dimensional-Linear-Models/10.1214/aos/1176349025.full}{
#' Bootstrap and Wild Bootstrap for High Dimensional Linear Models}. \emph{The
#' Annals of Statistics}, 21(1), p. 255-285.} \cr
#' \item{Efron (1979).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full}{
#' Bootstrap Methods: Another Look at the Jackknife}.
#' \emph{The Annals of Statistics}, 7(1), p. 1-26.}}
#'
#' @seealso \code{\link{SURE.trendfilter}}, \code{\link{cv.trendfilter}}
#'
#' @examples
#' \dontrun{
#' data(quasar_spectrum)
#' head(spec)
#'
#' SURE.obj <- SURE.trendfilter(spec$log10.wavelength, spec$flux, spec$weights)
#' opt <- bootstrap.trendfilter(SURE.obj, bootstrap.algorithm = "parametric")
#' }
#' @importFrom glmgen trendfilter
#' @importFrom dplyr %>% mutate case_when select n
#' @importFrom tidyr tibble
#' @importFrom parallel mclapply detectCores
#' @importFrom stats quantile rnorm
bootstrap.trendfilter <- function(obj,
                                  bootstrap.algorithm, level = 0.95, B = 100L,
                                  return.ensemble = FALSE, prune = TRUE,
                                  mc.cores = detectCores(), seed = 1) {
  stopifnot(class(obj) %in% c("SURE.trendfilter", "cv.trendfilter"))
  stopifnot(is.double(level) & level > 0 & level < 1)
  stopifnot(B >= 10)

  if (!prune) warning("I hope you know what you are doing!")

  if (mc.cores < detectCores()) {
    warning(paste0("Your machine has ", detectCores(), " cores. Consider increasing mc.cores to speed up computation."))
  }

  if (!missing(seed)) {
    RNGkind("L'Ecuyer-CMRG")
    set.seed(seed)
  }

  if (mc.cores > detectCores()) mc.cores <- detectCores()
  mc.cores <- min(floor(mc.cores), B)

  sampler <- case_when(
    bootstrap.algorithm == "nonparametric" ~ list(nonparametric.resampler),
    bootstrap.algorithm == "parametric" ~ list(parametric.sampler),
    bootstrap.algorithm == "wild" ~ list(wild.sampler)
  )[[1]]

  obj$prune <- prune
  par.out <- mclapply(1:B, tf.estimator,
    data = sampler(obj$data.scaled),
    obj = obj, mode = "edf", mc.cores = mc.cores
  )

  tf.boot.ensemble <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["tf.estimate"]]) %>%
    unlist() %>%
    matrix(nrow = length(obj$x.eval))

  obj$edf.boots <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["edf"]]) %>%
    unlist() %>%
    as.integer()
  obj$n.iter.boots <- lapply(X = 1:B, FUN = function(X) par.out[[X]][["n.iter"]]) %>%
    unlist() %>%
    as.integer()

  obj$n.pruned <- (B - ncol(tf.boot.ensemble)) %>% as.integer()
  obj$tf.standard.errors <- apply(tf.boot.ensemble, 1, sd)
  obj$bootstrap.lower.band <- apply(tf.boot.ensemble, 1, quantile, probs = (1 - level) / 2)
  obj$bootstrap.upper.band <- apply(tf.boot.ensemble, 1, quantile, probs = 1 - (1 - level) / 2)
  obj <- c(obj, list(bootstrap.algorithm = bootstrap.algorithm, level = level, B = B))

  if (return.ensemble) {
    obj$tf.bootstrap.ensemble <- tf.boot.ensemble
  } else {
    obj <- c(obj, list(tf.bootstrap.ensemble = NULL))
  }

  obj <- obj[c(
    "x.eval", "tf.estimate", "tf.standard.errors", "bootstrap.lower.band",
    "bootstrap.upper.band", "bootstrap.algorithm", "level", "B",
    "edf.boots", "tf.bootstrap.ensemble", "prune", "n.pruned", "x", "y",
    "weights", "fitted.values", "residuals", "k", "lambdas", "lambda.min",
    "edfs", "edf.min", "i.min", "validation.method", "generalization.errors",
    "ADMM.params", "n.iter", "n.iter.boots", "x.scale", "y.scale",
    "data.scaled"
  )]
  class(obj) <- "bootstrap.trendfilter"
  return(obj)
}

#' @importFrom glmgen trendfilter
tf.estimator <- function(b, data, obj, mode = "lambda") {
  if (mode == "edf") {
    tf.fit <- trendfilter(
      x = data$x,
      y = data$y,
      weights = data$weights,
      k = obj$k,
      lambda = obj$lambdas,
      thinning = obj$thinning,
      control = obj$ADMM.params
    )

    i.min <- which.min(abs(tf.fit$df - obj$edf.min))
    lambda.min <- obj$lambdas[i.min]
    edf.min <- tf.fit$df[i.min]
    n.iter <- tf.fit$iter[i.min]

    if (obj$prune && edf.min <= 2) {
      return(list(tf.estimate = integer(0), df = NA, n.iter = NA))
    }
  }

  if (mode == "lambda") {
    tf.fit <- trendfilter(
      x = data$x,
      y = data$y,
      weights = data$weights,
      k = obj$k,
      lambda = obj$lambda.min,
      thinning = obj$thinning,
      control = obj$ADMM.params
    )

    lambda.min <- obj$lambda.min
    edf.min <- tf.fit$df
    n.iter <- tf.fit$iter
  }

  tf.estimate <- glmgen:::predict.trendfilter(
    object = tf.fit,
    x.new = obj$x.eval / obj$x.scale,
    lambda = lambda.min
  ) %>%
    as.double()

  return(list(tf.estimate = tf.estimate * obj$y.scale, edf = edf.min, n.iter = n.iter))
}


#' @importFrom dplyr %>% mutate n
parametric.sampler <- function(data) {
  data %>% mutate(y = fitted.values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @importFrom dplyr %>% slice_sample n
nonparametric.resampler <- function(data) {
  data %>% slice_sample(n = n(), replace = TRUE)
}


#' @importFrom dplyr %>% mutate n
wild.sampler <- function(data) {
  data %>% mutate(y = fitted.values + residuals *
    sample(
      x = c(
        (1 + sqrt(5)) / 2,
        (1 - sqrt(5)) / 2
      ),
      size = n(), replace = TRUE,
      prob = c(
        (1 + sqrt(5)) / (2 * sqrt(5)),
        (sqrt(5) - 1) / (2 * sqrt(5))
      )
    ))
}
