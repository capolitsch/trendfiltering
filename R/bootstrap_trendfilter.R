#' Construct pointwise variability bands via a tailored bootstrap algorithm
#'
#' Generate a bootstrap ensemble of trend filtering estimates in order to
#' quantify the uncertainty in the optimized estimate. One of three possible
#' bootstrap algorithms should be chosen according to the criteria in the
#' Details section below. Pointwise variability bands are then obtained by
#' passing the '`bootstrap_trendfilter`' object to [`vbands()`], along with the
#' desired level (e.g. `level = 0.95`) .
#'
#' @param obj
#'   An object of class '[`cv_trendfilter`][cv_trendfilter()]' or
#'   '[`sure_trendfilter`][sure_trendfilter()]'.
#' @param algorithm
#'   A string specifying which variation of the bootstrap to use. One of
#'   `c("nonparametric", "parametric", "wild")`. See Details section below for
#'   guidelines on when each choice should be used.
#' @param B
#'   The number of bootstrap samples used to estimate the pointwise variability
#'   bands. Defaults to `B = 100L`.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to the number
#'   of cores detected, minus 4.
#'
#' @details Our recommendations for when to use each of the possible settings
#' for the `algorithm` argument are shown in the table below. See
#' [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for more details.
#'
#' | Scenario                                                                  |  Uncertainty quantification  |
#' | :---                                                                      |                   :---       |
#' | `x` is unevenly sampled                                                   | `algorithm = "nonparametric"`|
#' | `x` is evenly sampled and measurement variances for `y` are available     | `algorithm = "parametric"`   |
#' | `x` is evenly sampled and measurement variances for `y` are not available | `algorithm = "wild"`         |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in large consecutive chunks) is still considered to
#' be evenly sampled. When the inputs are evenly sampled on a transformed scale,
#' we recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See Example 2 below for a case when the
#' inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class '`bootstrap_trendfilter`' and subclass
#' '[`trendfilter`][trendfilter()]'. This is a list with the following elements:
#' \describe{
#' \item{x_eval}{Input grid that each bootstrap trend filtering estimate was
#' evaluated on.}
#' \item{ensemble}{The full trend filtering bootstrap ensemble as a matrix with
#' `length(x_eval)` rows and `B` columns.}
#' \item{edf_boots}{Vector of the estimated number of effective degrees of
#' freedom of each trend filtering bootstrap estimate. These should all be
#' relatively close to `obj$edf_opt`.}
#' \item{n_iter_boots}{Vector of the number of iterations taken by the ADMM
#' algorithm before reaching a stopping criterion, for each bootstrap estimate.}
#' \item{lambda_boots}{Vector of the hyperparameter values used for each
#' bootstrap fit. In general, these are not all equal because our bootstrap
#' implementation instead seeks to hold the number of effective degrees of
#' freedom constant across all bootstrap estimates.}
#' \item{algorithm}{A string specifying which variation of the bootstrap was
#' used to generate the ensemble.}
#' }
#'
#' @references
#' 1. Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#'    for time-domain astronomy and astronomical spectroscopy. *MNRAS*, 492(3),
#'    p. 4005-4018.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#'    [[arXiv](https://arxiv.org/abs/1908.07151)].
#' 2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#'    signals with varying degrees of smoothness. *MNRAS*, 492(3), p. 4019-4032.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)]
#'    [[arXiv](https://arxiv.org/abs/2001.03552)].
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
#'   max_iter = 1e4,
#'   obj_tol = 1e-6
#' )
#'
#' boot_tf <- bootstrap_trendfilter(cv_tf, algorithm = "nonparametric")
#' bands <- vbands(boot_tf)
#'
#' # Example 2: The "Lyman-alpha forest" in the spectrum of a distant quasar
#'
#' data(quasar_spectrum)
#' head(spec)
#'
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' pred_tf <- predict(sure_tf)
#'
#' boot_tf <- bootstrap_trendfilter(pred_tf, algorithm = "parametric")
#' bands <- vbands(boot_tf)
#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom parallel mclapply detectCores
#' @export
bootstrap_trendfilter <- function(obj,
                                  algorithm = NULL,
                                  B = 100L,
                                  mc_cores = parallel::detectCores() - 4) {
  stopifnot(B >= 20)

  mc_cores <- min(c(detectCores(), B, max(c(1, floor(mc_cores)))))

  if (mc_cores < detectCores() / 2) {
    warning(
      cat(paste0(
        "Your machine has ", detectCores(), " cores.\n Consider increasing",
        "mc_cores to speed up computation."
      )),
      call. = FALSE
    )
  }

  if (algorithm != "nonparametric") {
    obj$model_obj$df_scaled %<>% mutate(
      fitted_values = glmgen:::predict.trendfilter(
        obj$model_obj$model_fit,
        lambda = obj$lambda_opt,
        x.new = obj$model_obj$data_scaled$x
      ) %>%
        as.double(),
      residuals = y - fitted_values
    )
  }

  sampler <- case_when(
    algorithm == "nonparametric" ~ list(nonparametric_resampler),
    algorithm == "parametric" ~ list(parametric_sampler),
    algorithm == "wild" ~ list(wild_sampler)
  )[[1]]

  par_out <- mclapply(
    1:B,
    bootstrap_parallel,
    obj = obj,
    sampler = sampler,
    mc.cores = mc_cores
  )

  ensemble <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["tf_estimate"]]
  ) %>%
    unlist(labels = FALSE) %>%
    matrix(nrow = length(obj$x_eval))

  edf_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["edf"]]
  ) %>%
    unlist(labels = FALSE) %>%
    as.integer()

  n_iter_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["n_iter"]]
  ) %>%
    unlist(labels = FALSE) %>%
    as.integer()

  lambdas_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["lambdas"]]
  ) %>%
    unlist(labels = FALSE)

  invisible(
    structure(
      list(
        x_eval = obj$x_eval,
        ensemble = ensemble,
        edf_boots = edf_boots,
        n_iter_boots = n_iter_boots,
        lambda_boots = lambda_boots,
        algorithm = algorithm
      ),
      class = c("bootstrap_trendfilter", "trendfilter", "trendfiltering")
    )
  )
}


#' @noRd
#' @importFrom dplyr case_when
bootstrap_parallel <- function(b, obj, sampler) {
  data <- sampler(obj$model_obj$data_scaled)
  lambdas <- obj$lambdas[max(obj$i_opt - 10, 1):min(obj$i_opt + 10)]

  tf_fit <- .trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = obj$model_obj$k,
    lambdas = lambdas,
    obj_tol = obj$model_obj$admm_params$obj_tol,
    max_iter = obj$model_obj$admm_params$max_iter
  )

  i_min <- which.min(abs(tf_fit$df - obj$edf_opt))
  edf <- tf_fit$df[i_min]
  n_iter <- tf_fit$iter[i_min]
  lambdas <- lambdas[i_min]

  if (min(abs(tf_fit$df - obj$edf_opt)) / obj$edf_opt > 0.2) {
    return(bootstrap_parallel(1, obj, sampler))
  }

  # tf_estimate <- as.numeric(
  #  glmgen:::predict.trendfilter(
  #    object = tf_fit,
  #    x.new = obj$x_eval / obj$model_obj$x_scale,
  #    lambda = lambdas
  #  )
  # ) * obj$model_obj$y_scale

  # list(
  #  tf_estimate = tf_estimate,
  #  edf = edf,
  #  n_iter = n_iter,
  #  lambdas = lambdas
  # )
}


###########

#' Bootstrap sampling/resampling functions
#'
#' @param df A tibble or data frame with minimal column set: `x` and `y` (for
#' all samplers), `weights` and `fitted.values` (for `parametric.sampler`), and
#' `residuals` (for `wild.sampler`).
#'
#' @return Bootstrap sample returned in the same format as `df`.


#' @importFrom dplyr mutate n
#' @importFrom magrittr %>%
#' @importFrom stats rnorm
#' @noRd
parametric_sampler <- function(df) {
  df %>% mutate(y = fitted_values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @importFrom dplyr slice_sample n
#' @importFrom magrittr %>%
#' @noRd
nonparametric_resampler <- function(df) {
  df %>% slice_sample(n = nrow(df), replace = TRUE)
}


#' @importFrom dplyr mutate n
#' @importFrom magrittr %>%
#' @noRd
wild_sampler <- function(df) {
  df %>% mutate(y = fitted_values + residuals *
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
