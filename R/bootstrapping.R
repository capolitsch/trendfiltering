#' Construct pointwise variability bands via a tailored bootstrap algorithm
#'
#' Generate a bootstrap ensemble of trend filtering estimates in order to
#' quantify the uncertainty in the optimized estimate. One of three possible
#' bootstrap algorithms should be chosen according to the criteria in the
#' *Details* section below. Pointwise variability bands are then obtained by
#' passing the '`bootstrap_trendfilter`' object to [`vbands()`], along with the
#' desired level (e.g. `level = 0.95`) .
#'
#' @param obj
#'   An object of class '[`cv_trendfilter`][cv_trendfilter()]' or
#'   '[`sure_trendfilter`][sure_trendfilter()]'.
#' @param algorithm
#'   A string specifying which variation of the bootstrap to use. One of
#'   `c("nonparametric", "parametric", "wild")`. See *Details* section below for
#'   guidelines on when each choice should be used.
#' @param B
#'   The number of bootstrap samples used to estimate the pointwise variability
#'   bands. Defaults to `B = 100L`.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to the number
#'   of cores detected, minus 4.
#' @param ...
#'   Additional named arguments. Currently unused.
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
#' '[`trendfilter`][trendfilter()]'. This is a list with the elements below,
#' as well as all elements from `obj`.
#' \describe{
#' \item{x_eval}{Input grid that each bootstrap trend filtering estimate was
#' evaluated on.}
#' \item{ensemble}{The full trend filtering bootstrap ensemble as a matrix with
#' `length(x_eval)` rows and `B` columns.}
#' \item{edf_boots}{Vector of the estimated number of effective degrees of
#' freedom of each trend filtering bootstrap estimate.}
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
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' cv_tf <- cv_trendfilter(x, y, weights, max_iter = 1e4, obj_tol = 1e-6)
#' boot_tf <- bootstrap_trendfilter(cv_tf, algorithm = "nonparametric", lambda = cv_tf$lambda_min["MAE"])
#'
#'
#' # Example 2: The "Lyman-alpha forest" in the spectrum of a distant quasar
#'
#' data("quasar_spectrum")
#' head(quasar_spectrum)
#'
#' x <- quasar_spectrum$log10_wavelength
#' y <- quasar_spectrum$flux
#' weights <- quasar_spectrum$weights
#'
#' sure_tf <- sure_trendfilter(x, y, weights)
#' boot_tf <- bootstrap_trendfilter(sure_tf, algorithm = "parametric", lambda = sure_tf$lambda_min)

#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom rlang %||%
#' @importFrom parallel mclapply detectCores
#' @importFrom stats residuals fitted.values
#' @export
bootstrap_trendfilter <- function(obj,
                                  algorithm = c("nonparametric","parametric","wild"),
                                  B = 100L,
                                  x_eval = NULL,
                                  lambda = NULL,
                                  mc_cores = parallel::detectCores() - 4,
                                  ...) {
  stopifnot(
    any(class(obj) == "cv_trendfilter") || any(class(obj) == "sure_trendfilter")
  )
  stopifnot(B >= 20)

  boot.call <- match.call
  extra_args <- list(...)
  algorithm <- match.arg(algorithm)
  lambda <- lambda %||% obj$lambda_min["MAE"]
  x_eval <- x_eval %||% obj$x

  stopifnot(is.numeric(lambda))
  if (length(lambda) > 1) {
    stop("`lambda` must be of length 1 for bootstrap_trendfilter().")
  }
  stopifnot(lambda >= 0L)

  if (!(lambda %in% obj$lambda)) {
    stop("`lambda` must be in `obj$lambda`.")
  }

  stopifnot(is.numeric(x_eval))
  stopifnot(any(x_eval >= min(obj$x) || x_eval <= max(obj$x)))

  if (!is.null(x_eval) && (any(x_eval < min(obj$x) || x_eval > max(obj$x)))) {
    warning("One or more elements of `x_eval` are outside `range(x)`.",
            call. = FALSE)
  }

  stopifnot(
    is.numeric(mc_cores) && length(mc_cores) == 1 && round(mc_cores) == mc_cores
  )
  mc_cores <- min(detectCores(), B, max(c(1, floor(mc_cores)))) %>% as.integer()

  if (mc_cores < detectCores() / 2) {
    warning(
      cat(paste0(
        "Your machine has ", detectCores(), " cores.\n Consider increasing",
        "mc_cores to speed up computation."
      )),
      call. = FALSE
    )
  }

  i_opt <- match(lambda, obj$lambda)
  edf_opt <- obj$edf[i_opt]

  data_scaled <- tibble(
    x = obj$x / obj$scale["x"],
    y = obj$y / obj$scale["y"],
    weights = obj$weights * obj$scale["y"]^2,
    fitted_values = fitted(obj, lambda = lambda),
    residuals = residuals(obj, lambda = lambda)
  )

  sampler <- case_when(
    algorithm == "nonparametric" ~ list(nonparametric_resampler),
    algorithm == "parametric" ~ list(parametric_sampler),
    algorithm == "wild" ~ list(wild_sampler)
  )[[1]]

  if ("lambda_radius" %in% names(extra_args)) {
    lambda_radius <- extra_args$lambda_radius
  } else {
    lambda_radius <- 7
  }

  lambda_grid <- obj$lambda[
    max(i_opt - lambda_radius, 1):min(i_opt + lambda_radius, length(obj$lambda))
  ]

  if ("zero_tol" %in% names(extra_args)) {
    zero_tol <- extra_args$zero_tol
  } else {
    zero_tol <- 1e-6
  }

  par_args <- list(
    data = data_scaled,
    k = obj$k,
    admm_params = obj$admm_params,
    edf_opt = edf_opt,
    lambda_grid = lambda_grid,
    sampler = sampler,
    x_eval = x_eval / obj$scale["x"],
    zero_tol = zero_tol
  )

  par_out <- mclapply(
    1:B,
    bootstrap_parallel,
    par_args,
    mc.cores = mc_cores
  )

  ensemble <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["tf_estimate_boot"]]
  ) %>%
    unlist(use.names = FALSE) %>%
    matrix(nrow = length(x_eval))

  edf_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["edf_boot"]]
  ) %>%
    unlist(use.names = FALSE) %>%
    as.integer()

  n_iter_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["n_iter_boot"]]
  ) %>%
    unlist(use.names = FALSE) %>%
    as.integer()

  lambda_boots <- lapply(
    1:B,
    FUN = function(X) par_out[[X]][["lambda_boot"]]
  ) %>%
    unlist(use.names = FALSE)

  invisible(
    structure(
      list(
        x_eval = x_eval,
        ensemble = ensemble,
        algorithm = algorithm,
        edf_boots = edf_boots,
        n_iter_boots = n_iter_boots,
        lambda_boots = lambda_boots,
        x = obj$x,
        y = obj$y,
        lambda = obj$lambda,
        k = obj$k,
        fitted_values = obj$fitted_values,
        scale = obj$scale,
        call = boot.call
      ),
      class = c("bootstrap_trendfilter", "trendfilter", "trendfiltering")
    )
  )
}


#' @noRd
#' @importFrom glmgen .tf_fit
#' @importFrom mvbutils extract.named
bootstrap_parallel <- function(b, par_args) {
  extract.named(par.args)
  data <- sampler(data)

  fit <- .tf_fit(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = k,
    lambda = lambda_grid,
    admm_params = admm_params
  )

  i_min <- which.min(abs(fit$df - edf_opt))
  edf_boot <- fit$df[i_min]

  if (min(abs(edf_boot - edf_opt)) / edf_opt > 0.1) {
    return(
      bootstrap_parallel(
        b = 1,
        data = data,
        k = k,
        admm_params = admm_params,
        edf_opt = edf_opt,
        lambda_grid = lambda_grid,
        sampler = sampler,
        x_eval = x_eval
      )
    )
  }

  n_iter_boot <- fit$iter[i_min]
  lambda_boot <- lambda_grid[i_min]

  fit <- .trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = k,
    lambda = lambda_boot,
    obj_tol = admm_params$obj_tol,
    max_iter = admm_params$max_iter
  )

  tf_estimate_boot <- predict(fit, x_eval = x_eval, zero_tol = zero_tol)

  list(
    tf_estimate_boot = tf_estimate_boot,
    edf_boot = edf_boot,
    lambda_boot = lambda_boot,
    n_iter_boot = n_iter_boot
  )
}


#' Bootstrap sampling/resampling functions
#'
#' @param data A tibble or data frame with minimal column set: `x` and `y` (for
#' all samplers), `weights` and `fitted.values` (for `parametric.sampler`), and
#' `residuals` (for `wild.sampler`).
#'
#' @return Bootstrap sample returned in the same format as `data`.


#' @importFrom dplyr mutate n
#' @importFrom magrittr %>%
#' @importFrom stats rnorm
#' @noRd
parametric_sampler <- function(data) {
  data %>% mutate(y = fitted_values + rnorm(n = n(), sd = 1 / sqrt(weights)))
}


#' @importFrom dplyr slice_sample n
#' @importFrom magrittr %>%
#' @noRd
nonparametric_resampler <- function(data) {
  data %>% slice_sample(n = nrow(data), replace = TRUE)
}


#' @importFrom dplyr mutate n
#' @importFrom magrittr %>%
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


#' Compute bootstrap variability bands
#'
#' Compute variability bands for the optimized trend filtering estimate via the
#' sample quantiles of the bootstrap ensemble generated by
#' [`bootstrap_trendfilter()`].
#'
#' @param obj
#'   A '[`bootstrap_trendfilter`][bootstrap_trendfilter()]' object.
#' @param level
#'   The level of the pointwise variability bands. Defaults to `level = 0.95`.
#' @return A tibble with column set `c("x","lower_band","upper_band")`.
#'
#' @seealso [`bootstrap_trendfilter()`]
#'
#' @references
#' 1. Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#'    for time-domain astronomy and astronomical spectroscopy. *MNRAS*, 492(3),
#'    p. 4005-4018.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)]
#'    [[arXiv](https://arxiv.org/abs/1908.07151)]
#'    [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].
#' 2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
#'    signals with varying degrees of smoothness. *MNRAS*, 492(3), p. 4019-4032.
#'    [[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)]
#'    [[arXiv](https://arxiv.org/abs/2001.03552)]
#'    [[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].
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
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' cv_tf <- cv_trendfilter(x, y, weights, max_iter = 1e4, obj_tol = 1e-6)
#'
#' boot_tf <- bootstrap_trendfilter(cv_tf, "nonparametric")
#' bands <- vbands(boot_tf)
#'
#'
#' # Example 2: The "Lyman-alpha forest" in the spectrum of a distant quasar
#'
#' data("quasar_spectrum")
#' head(quasar_spectrum)
#'
#' x <- quasar_spectrum$log10_wavelength
#' y <- quasar_spectrum$flux
#' weights <- quasar_spectrum$weights
#'
#' sure_tf <- sure_trendfilter(x, y, weights)
#' boot_tf <- bootstrap_trendfilter(sure_tf, "parametric")

#' @importFrom dplyr tibble
#' @export
vbands <- function(obj, level = 0.95) {
  stopifnot(level > 0 & level < 1)

  lower_band <- apply(
    obj$ensemble,
    1,
    quantile,
    probs = (1 - level) / 2
  )

  upper_band <- apply(
    obj$ensemble,
    1,
    quantile,
    probs = 1 - (1 - level) / 2
  )

  tibble(
    x_eval = obj$x_eval,
    lower_band = lower_band,
    upper_band = upper_band
  )
}
