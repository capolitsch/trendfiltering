#' Construct pointwise variability bands via a tailored bootstrap algorithm
#'
#' \loadmathjax Generate a bootstrap ensemble of trend filtering estimates in
#' order to quantify the uncertainty in the optimized estimate. One of three
#' possible bootstrap algorithms should be chosen according to the criteria in
#' the details section below. See [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for the technical
#' details of each bootstrap algorithm. Pointwise variability bands are then
#' obtained by passing the `bootstrap_trendfilter()` output and a desired level
#' (e.g. `level = 0.95`) to [`vbands()`].
#'
#' @param obj An object of class '[`pred_tf`][predict_trendfilter()]'.
#' @param bootstrap_algorithm A string specifying which variation of the
#' bootstrap to use. One of `c("nonparametric", "parametric", "wild")`. See
#' details section below for when each option is appropriate.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100L`.
#' @param mc_cores Multi-core computing using the
#' [`parallel`][`parallel::parallel-package`] package: The number of cores to
#' utilize. Defaults to the number of cores detected, minus 4.
#'
#' @details Our recommendations for when to use each of the possible settings
#' for the `bootstrap_algorithm` argument are shown in the table below. See
#' [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for more details.
#'
#' | Scenario                                                                  |        Uncertainty quantification      |
#' | :---                                                                      |                   :---                 |
#' | `x` is unevenly sampled                                                   | `bootstrap_algorithm = "nonparametric"`|
#' | `x` is evenly sampled and measurement variances for `y` are available     | `bootstrap_algorithm = "parametric"`   |
#' | `x` is evenly sampled and measurement variances for `y` are not available | `bootstrap_algorithm = "wild"`         |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in large consecutive chunks) is still considered to
#' be evenly sampled. When the inputs are evenly sampled on a transformed scale,
#' we recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See Example 2 below for a case when the
#' inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class '`bootstrap_tf`'. This is a list with the
#' following elements
#' \describe{
#' \item{x_eval}{Input grid that each bootstrap trend filtering estimate was
#' evaluated on.}
#' \item{ensemble}{The full trend filtering bootstrap ensemble as an
#' \mjseqn{n \times B} matrix.}
#' \item{edf_boots}{Vector of the estimated number of effective degrees of
#' freedom of each trend filtering bootstrap estimate. These should all be
#' relatively close to `obj$edf_opt`.}
#' \item{n_iter_boots}{Vector of the number of iterations taken by the ADMM
#' algorithm before reaching a stopping criterion, for each bootstrap estimate.}
#' \item{lambda_boots}{Vector of the hyperparameter values used for each
#' bootstrap fit. In general, these are not all equal because our bootstrap
#' implementation instead seeks to hold the number of effective degrees of
#' freedom constant across all bootstrap estimates.}
#' \item{bootstrap_algorithm}{A string specifying which variation of the
#' bootstrap was used to generate the ensemble.}
#' }
#'
#' @references
#' \bold{Companion references}
#' \enumerate{
#' \item{Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#' for time-domain astronomy and astronomical spectroscopy. \emph{MNRAS},
#' 492(3), p. 4005-4018.
#' [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#' [[arXiv](https://arxiv.org/abs/1908.07151)]
#' [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].} \cr
#' \item{Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#' signals with varying degrees of smoothness.
#' https://academic.oup.com/mnras/article/492/3/4019/5704414. \emph{MNRAS},
#' 492(3), p. 4019-4032.
#' [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#' [[arXiv](https://arxiv.org/abs/2001.03552)]
#' [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].}}
#'
#' \bold{The Bootstrap and variations}
#' \enumerate{
#' \item{Efron and Tibshirani (1986).
#' [Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy](
#' https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full).
#' \emph{Statistical Science}, 1(1), p. 54-75.} \cr
#' \item{Mammen (1993).
#' [Bootstrap and Wild Bootstrap for High Dimensional Linear Models](
#' https://projecteuclid.org/journals/annals-of-statistics/volume-21/issue-1/Bootstrap-and-Wild-Bootstrap-for-High-Dimensional-Linear-Models/10.1214/aos/1176349025.full).
#' \emph{The Annals of Statistics}, 21(1), p. 255-285.} \cr
#' \item{Efron (1979).
#' [Bootstrap Methods: Another Look at the Jackknife](
#' https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full).
#' \emph{The Annals of Statistics}, 7(1), p. 1-26.}}
#'
#' @seealso [cv_trendfilter()], [sure_trendfilter()]
#'
#' @examples
#' # Example 1: Phase-folded light curve of an eclipsing binary star system
#' #
#' # The apparent brightness over time of a star system that has two suns
#' # that regularly eclipse one another from our vantage point on Earth. Here,
#' # the time series is stacked according to the orbital period of the binary
#' # system, with the primary eclipse occuring at `phase = 0` and the input
#' # domain ranging from -0.5 to 0.5.
#'
#' data(eclipsing_binary)
#' head(EB)
#'
#' cv_tf <- cv_trendfilter(
#'   x = EB$phase,
#'   y = EB$flux,
#'   weights = 1 / EB$std_err^2,
#'   optimization_params = list(
#'     max_iter = 1e4,
#'     obj_tol = 1e-6,
#'     thinning = TRUE
#'   )
#' )
#'
#' pred_tf <- predict(
#'   cv_tf,
#'   loss_func = "MAE",
#'   lambda_choice = "lambda_1se",
#'   nx_eval = 1500L
#' )
#'
#' boot_tf <- bootstrap_trendfilter(pred_tf, "nonparametric")
#' bands <- vbands(boot_tf)
#'
#'
#' # Example 2: The "Lyman-alpha forest" in the spectrum of a distant quasar
#'
#' data(quasar_spectrum)
#' head(spec)
#'
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' pred_tf <- predict(sure_tf)
#'
#' boot_tf <- bootstrap_trendfilter(pred_tf, "parametric")
#' bands <- vbands(boot_tf)
#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom parallel mclapply detectCores
#' @rdname bootstrap_trendfilter
#' @export
bootstrap_trendfilter <- function(obj,
                                  bootstrap_algorithm = c("nonparametric", "parametric", "wild"),
                                  B = 100L,
                                  mc_cores = parallel::detectCores() - 4) {
  stopifnot(any(class(obj) == "pred_tf"))
  stopifnot(B >= 20)
  bootstrap_algorithm <- match.arg(bootstrap_algorithm)

  mc_cores <- max(c(1, floor(mc_cores)))
  mc_cores <- min(c(detectCores(), B, mc_cores))

  if (mc_cores < detectCores() / 2) {
    warning(
      cat(paste0(
        "Your machine has ", detectCores(), " cores.\n Consider increasing",
        "mc_cores to speed up computation."
      )),
      call. = FALSE
    )
  }

  if (bootstrap_algorithm != "nonparametric") {
    obj$tf_model$data_scaled %<>% mutate(
      fitted_values = glmgen:::predict.trendfilter(
        obj$tf_model$model_fit,
        lambda = obj$lambda_opt,
        x.new = obj$tf_model$data_scaled$x
      ) %>%
        as.double(),
      residuals = y - fitted_values
    )
  }

  sampler <- case_when(
    bootstrap_algorithm == "nonparametric" ~ list(nonparametric_resampler),
    bootstrap_algorithm == "parametric" ~ list(parametric_sampler),
    bootstrap_algorithm == "wild" ~ list(wild_sampler)
  )[[1]]

  par_out <- mclapply(
    1:B,
    bootstrap_parallel,
    obj = obj,
    sampler = sampler,
    mc.cores = mc_cores
  )

  ensemble <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["tf_estimate"]]
  ) %>%
    unlist(labels = FALSE) %>%
    matrix(nrow = length(obj$x_eval))

  edf_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["edf"]]
  ) %>%
    unlist(labels = FALSE) %>%
    as.integer()

  n_iter_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["n_iter"]]
  ) %>%
    unlist(labels = FALSE) %>%
    as.integer()

  lambda_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["lambda"]]
  ) %>%
    unlist(labels = FALSE)

  structure(
    list(
      x_eval = obj$x_eval,
      ensemble = ensemble,
      edf_boots = edf_boots,
      n_iter_boots = n_iter_boots,
      lambda_boots = lambda_boots,
      bootstrap_algorithm = bootstrap_algorithm
    ),
    class = c("boot_tf", "list")
  )
}


#' @noRd
#' @importFrom dplyr case_when
bootstrap_parallel <- function(b, obj, sampler) {
  data <- sampler(obj$tf_model$data_scaled)
  lambdas <- obj$lambdas[max(obj$i_opt - 10, 1):min(obj$i_opt + 10)]

  tf_fit <- trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = obj$tf_model$k,
    lambda = lambdas,
    thinning = obj$tf_model$thinning,
    control = obj$tf_model$admm_params
  )

  i_min <- which.min(abs(tf_fit$df - obj$edf_opt))
  edf <- tf_fit$df[i_min]
  n_iter <- tf_fit$iter[i_min]
  lambda <- lambdas[i_min]

  if (min(abs(tf_fit$df - obj$edf_opt)) / obj$edf_opt > 0.2) {
    return(bootstrap_parallel(1, obj, sampler))
  }

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      object = tf_fit,
      x.new = obj$x_eval / obj$tf_model$x_scale,
      lambda = lambda
    )
  ) * obj$tf_model$y_scale

  list(
    tf_estimate = tf_estimate,
    edf = edf,
    n_iter = n_iter,
    lambda = lambda
  )
}


###########

#' Bootstrap sampling/resampling functions
#'
#' @param data Tibble / data frame with minimal column set: `x` and `y` (for all
#' samplers), `weights` and `fitted.values` (for `parametric.sampler`), and
#' `residuals` (for `wild.sampler`).
#'
#' @return Bootstrap sample returned in the same format as the input tibble /
#' data frame.


#' @importFrom dplyr %>% mutate n
#' @importFrom stats rnorm
#' @noRd
parametric_sampler <- function(data) {
  data %>% mutate(y = fitted_values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @importFrom dplyr %>% slice_sample n
#' @noRd
nonparametric_resampler <- function(data) {
  data %>% slice_sample(n = nrow(data), replace = TRUE)
}


#' @importFrom dplyr %>% mutate n
#' @noRd
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
