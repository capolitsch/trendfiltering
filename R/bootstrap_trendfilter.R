#' Construct pointwise variability bands via a bootstrap
#'
#' \loadmathjax See [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for details.
#'
#' @param obj An object of class [`sure_tf`][sure_trendfilter()] or
#' [`cv_tf`][cv_trendfilter].
#' @param lambda_choice One of `c("lambda_min","lambda_1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' Defaults to `lambda_choice = "lambda_min"`.
#' \itemize{
#' \item{`"lambda_min"`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`"lambda_1se"`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates. The motivation here is essentially Occam's razor: the two models
#' yield results that are quantitatively very close, so we favor the simpler
#' model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
#' for more details on the "one-standard-error rule".}
#' @param level The level of the pointwise variability bands. Defaults to
#' `level = 0.95`.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100`.
#' @param bootstrap_algorithm A string specifying which variation of the
#' bootstrap to use. One of `c("nonparametric","parametric","wild")`. See
#' details below for recommendations on when each option is appropriate.
#' @param x_eval (Optional) Overrides `nx_eval` if passed. A grid of inputs to
#' evaluate the optimized trend filtering estimate on.
#' @param nx_eval Integer. The length of the input grid that the optimized
#' trend filtering estimate is evaluated on; i.e. if nothing is passed to
#' `x_eval`, then it is defined as
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @param return_ensemble Logical. If `TRUE`, the full trend filtering bootstrap
#' ensemble is returned as an \mjseqn{n \times B} matrix, less any columns from
#' post-hoc pruning (see `prune` below). Defaults to `return_ensemble = FALSE`
#' to save memory.
#' @param prune Logical. If `TRUE`, then the trend filtering bootstrap ensemble
#' is examined for rare instances in which the optimization has stopped at zero
#' knots (likely erroneously), and removes them from the ensemble that is used
#' to compute the variability bands. Defaults to `prune = TRUE`. Do not change
#' this unless you know what you are doing!
#' @param mc_cores Multi-core computing using the
#' [`parallel`][`parallel::parallel-package`] package: The number of cores to
#' utilize. Defaults to the number of cores detected, minus 4.
#'
#' @details Our recommendations for when to use each of the possible settings
#' for the `bootstrap_algorithm` argument are shown in the table below. See
#' [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for more details.
#'
#' | Scenario                                                         |        Uncertainty quantification      |
#' | :---                                                             |                   :---                 |
#' | `x` is unevenly sampled                                          | `bootstrap_algorithm = "nonparametric"`|
#' | `x` is evenly sampled and reciprocal variances are not available | `bootstrap_algorithm = "wild"`         |
#' | `x` is evenly sampled and reciprocal variances are available     | `bootstrap_algorithm = "parametric"`   |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in large consecutive chunks) is still considered to
#' be evenly sampled. When the inputs are evenly sampled on a transformed scale,
#' we recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See the example below for a case when the
#' inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class `bootstrap_tf`. This is a comprehensive
#' list containing all of the analysis' important information, data, and
#' results:
#' \describe{
#' \item{bootstrap_lower_band}{Vector of lower bounds for the pointwise
#' variability bands, evaluated on `x_eval`.}
#' \item{bootstrap_upper_band}{Vector of upper bounds for the pointwise
#' variability bands, evaluated on `x_eval`.}
#' \item{bootstrap_algorithm}{A string specifying which variation of the
#' bootstrap was used to obtain the variability bands.}
#' \item{level}{The level of the pointwise variability bands.}
#' \item{B}{The number of bootstrap samples used to estimate the pointwise
#' variability bands.}
#' \item{tf_bootstrap_ensemble}{If `return_ensemble = TRUE`, the full trend
#' filtering bootstrap ensemble as an \mjseqn{n \times B} matrix, less any
#' columns from post-hoc pruning (if `prune = TRUE`). Else, this will return
#' `NULL`.}
#' \item{edf_boots}{An integer vector of the estimated number of effective
#' degrees of freedom of each trend filtering bootstrap estimate. These should
#' all be relatively close to `edf_min`.}
#' \item{prune}{Logical. If `TRUE`, then the trend filtering bootstrap
#' ensemble is examined for rare instances in which the optimization has
#' stopped at zero knots (likely erroneously), and removes them from the
#' ensemble.}
#' \item{n_pruned}{The number of poorly-converged bootstrap trend filtering
#' estimates pruned from the ensemble.}
#' \item{n_iter_boots}{Vector of the number of iterations taken by the ADMM
#' algorithm before reaching a stopping criterion, for each bootstrap trend
#' filtering estimate.}
#' \item{...}{Named elements inherited from `obj` --- an object either of class
#' [`sure_tf`][sure_trendfilter] or [`cv_tf`][cv_trendfilter].}
#' }
#'
#' @export bootstrap_trendfilter
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
#' @seealso [sure_trendfilter()], [cv_trendfilter()]
#'
#' @examples
#' data(quasar_spectrum)
#' head(spec)
#'
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' opt_tf <- bootstrap_trendfilter(sure_tf, bootstrap_algorithm = "parametric")
#' @importFrom glmgen trendfilter
#' @importFrom dplyr %>% mutate case_when select n
#' @importFrom tidyr tibble
#' @importFrom parallel mclapply detectCores
#' @importFrom stats quantile rnorm
bootstrap_trendfilter <- function(obj, lambda_choice, x_eval, nx_eval,
                                  bootstrap_algorithm, level = 0.95, B = 100L,
                                  return_ensemble = FALSE, prune = TRUE,
                                  mc_cores = parallel::detectCores() - 4) {
  stopifnot(any(class(obj) %in% c("sure_tf", "cv_tf")))
  stopifnot(is.double(level) & level > 0 & level < 1)
  stopifnot(B >= 10)

  if (mc_cores < detectCores() / 2) {
    warning(paste0(
      "Your machine has ", detectCores(), " cores.\n",
      "Consider increasing mc_cores to speed up computation."
    ))
  }

  if (mc_cores > detectCores()) mc_cores <- detectCores()
  mc_cores <- min(floor(mc_cores), B)

  sampler <- case_when(
    bootstrap_algorithm == "nonparametric" ~ list(nonparametric_resampler),
    bootstrap_algorithm == "parametric" ~ list(parametric_sampler),
    bootstrap_algorithm == "wild" ~ list(wild_sampler)
  )[[1]]

  obj$prune <- prune
  par_out <- mclapply(
    1:B,
    tf_parallel,
    data = sampler(obj$data_scaled),
    obj = obj,
    mode = "edf",
    mc.cores = mc_cores
  )

  tf_boot_ensemble <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["tf_estimate"]]
  ) %>%
    unlist() %>%
    matrix(nrow = length(obj$x_eval))

  obj$edf_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["edf"]]
  ) %>%
    unlist() %>%
    as.integer()

  obj$n_iter_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["n_iter"]]
  ) %>%
    unlist() %>%
    as.integer()

  obj$n_pruned <- (B - ncol(tf_boot_ensemble)) %>% as.integer()

  obj$tf_standard_errors <- apply(
    tf_boot_ensemble,
    1,
    sd
  )
  obj$bootstrap_lower_band <- apply(
    tf_boot_ensemble,
    1,
    quantile,
    probs = (1 - level) / 2
  )
  obj$bootstrap_upper_band <- apply(
    tf_boot_ensemble,
    1,
    quantile,
    probs = 1 - (1 - level) / 2
  )

  obj <- c(
    obj,
    list(
      bootstrap_algorithm = bootstrap_algorithm,
      level = level,
      B = B
    )
  )

  if (return_ensemble) {
    obj$tf_bootstrap_ensemble <- tf_boot_ensemble
  } else {
    obj <- c(obj, list(tf_bootstrap_ensemble = NULL))
  }

  obj <- obj[c(
    "x_eval", "tf_estimate", "tf_standard_errors", "bootstrap_lower_band",
    "bootstrap_upper_band", "bootstrap_algorithm", "level", "B",
    "edf_boots", "tf_bootstrap_ensemble", "prune", "n_pruned", "x", "y",
    "weights", "fitted_values", "residuals", "k", "lambdas", "lambda_min",
    "edfs", "edf_min", "i_min", "validation_method", "generalization_errors",
    "admm_params", "n_iter", "n_iter_boots", "x_scale", "y_scale",
    "data_scaled"
  )]
  class(obj) <- c("bootstrap_tf", "list")
  return(obj)
}


#' @noRd
#' @importFrom glmgen trendfilter
tf_parallel <- function(b, data, obj, mode = "lambda") {
  tf_fit <- trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = obj$k,
    lambda = obj$lambda_min,
    thinning = obj$thinning,
    control = obj$admm_params
  )

  if (mode == "edf") {
    i_min <- which.min(abs(tf_fit$df - obj$edf_min))
    lambda_min <- obj$lambdas[i_min]
    edf_min <- tf_fit$df[i_min]
    n_iter <- tf_fit$iter[i_min]

    if (obj$prune && edf_min <= 4) {
      return(list(tf_estimate = integer(0), df = NA, n_iter = NA))
    }
  }

  if (mode == "lambda") {
    lambda_min <- obj$lambda_min
    edf_min <- tf_fit$df
    n_iter <- tf_fit$iter
  }

  tf_estimate <- glmgen:::predict.trendfilter(
    object = tf_fit,
    x.new = obj$x_eval / obj$x_scale,
    lambda = lambda_min
  ) %>%
    as.double()

  return(
    list(
      tf_estimate = tf_estimate * obj$y_scale,
      edf = edf_min,
      n_iter = n_iter
    )
  )
}


####

# Bootstrap sampling/resampling functions

#' @noRd
#' @importFrom dplyr %>% mutate n
parametric_sampler <- function(data) {
  data %>% mutate(y = fitted_values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @noRd
#' @importFrom dplyr %>% slice_sample n
nonparametric_resampler <- function(data) {
  data %>% slice_sample(n = n(), replace = TRUE)
}


#' @noRd
#' @importFrom dplyr %>% mutate n
wild_sampler <- function(data) {
  data %>% mutate(y = fitted_values + residuals *
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
