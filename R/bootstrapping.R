#' Construct pointwise variability bands via a bootstrap algorithm that's
#' tailored to the observed data
#'
#' Generate a bootstrap ensemble of trend filtering estimates in order to
#' quantify the uncertainty in the optimized trend filtering estimate. One of
#' three possible bootstrap algorithms should be chosen according to the
#' criteria summarized in the **Details** section below. Pointwise variability
#' bands are then obtained by passing the '`bootstrap_trendfilter`' object to
#' [`vbands()`], along with the desired level (e.g. `level = 0.95`).
#' Bootstrapping trend filtering estimators tends to yield more accurate
#' uncertainties when, for each bootstrap estimate, we fix the number of
#' effective degrees of freedom, `edf` (a reparametrization of the
#' hyperparameter `lambda`), instead of fixing `lambda` itself. Thus,
#' `bootstrap_trendfilter()` has an `edf` argument instead of `lambda`. See
#' the `edf` argument description and **Examples** section for guidance on how
#' `edf` can be chosen.
#'
#' @param obj
#'   An object of class '[`cv_trendfilter`][cv_trendfilter()]' or
#'   '[`sure_trendfilter`][sure_trendfilter()]'.
#' @param algorithm
#'   A string specifying which variation of the bootstrap to use. One of
#'   `c("nonparametric", "parametric", "wild")`. See **Details** section below
#'   for guidelines on when each choice should be used.
#' @param B
#'   The number of bootstrap samples to be drawn to generate the trend filtering
#'   ensemble. Defaults to `B = 100L` (larger values encouraged if the
#'   computational cost is acceptable).
#' @param edf
#'   The desired number of effective degrees of freedom in each bootstrap
#'   estimate. When `obj` is of class
#'   '[`sure_trendfilter`][sure_trendfilter()]', `edf = obj$edf_min` and
#'   `edf = obj$edf_1se` are advisible options. When `obj` is of class
#'   '[`cv_trendfilter`][cv_trendfilter()]', any element of the (now vectors)
#'   `obj$edf_min` and `obj$edf_1se` may be a reasonable choice. Defaults to
#'   `edf = obj$edf_min["MAE"]`.
#' @param mc_cores
#'   Number of cores to utilize for parallel computing. Defaults to the number
#'   of cores detected, minus 4.
#' @param ...
#'   Additional named arguments. Currently only a few experimental arguments
#'   may be passed by experts.
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
#' '[`trendfilter`][trendfilter()]'. Generic functions such as [`predict()`],
#' [`fitted.values()`], and [`residuals()`] may also be called on
#' `bootstrap_trendfilter()` objects, with the same effect as if they were
#' called on the `obj` argument originally passed to `bootstrap_trendfilter()`.
#' A `bootstrap_trendfilter` object is a list containing the follow elements:
#' \describe{
#' \item{`x_eval`}{Input grid that each bootstrap trend filtering estimate was
#' evaluated on.}
#' \item{`ensemble`}{The full trend filtering bootstrap ensemble as a matrix
#' with `length(x_eval)` rows and `B` columns.}
#' \item{`algorithm`}{String specifying which variation of the bootstrap was
#' used to generate the ensemble.}
#' \item{`edf_opt`}{Number of effective degrees of freedom that each bootstrap
#' trend filtering fit should approximately possess in our fixed-edf bootstrap
#' procedure. Identical to the value passed to `edf`, or its default.}
#' \item{`edf_boots`}{Vector of the estimated number of effective degrees of
#' freedom of each trend filtering bootstrap estimate. These are unlikely to
#' all be exactly equal to `edf_opt`, but should be relatively close.}
#' \item{`n_iter_boots`}{Vector of the number of iterations taken by the ADMM
#' algorithm before reaching a stopping criterion, for each bootstrap estimate.}
#' \item{`lambda_boots`}{Vector of the hyperparameter values used for each
#' bootstrap fit. In general, these are not all equal because our bootstrap
#' implementation instead seeks to hold the number of effective degrees of
#' freedom constant across all bootstrap estimates.}
#' \item{`lambda`}{Vector of the original grid of candidate hyperparameter
#' values, inherited from `obj`.}
#' \item{`edf`}{Number of effective degrees of freedom in the trend filtering
#' estimator, for every hyperparameter value in `lambda`.}
#' \item{`fitted_values`}{Fitted values of all trend filtering point
#' estimates with hyperparameter values in `lambda`, inherited from `obj`.}
#' \item{`x`}{Vector of observed values for the input variable, inherited from
#' `obj`.}
#' \item{`y`}{Vector of observed values for the output variable, inherited from
#' `obj`.}
#' \item{`weights`}{Vector of weights for the observed outputs, inherited from
#' `obj`.}
#' \item{`k`}{Degree of the trend filtering point estimate (and bootstrap
#' estimates), inherited from `obj`.}
#' \item{`call`}{The function call.}
#' \item{`scale`}{For internal use.}
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
#'
#' boot_tf <- bootstrap_trendfilter(
#'   obj = cv_tf,
#'   algorithm = "nonparametric",
#'   edf = cv_tf$edf_min["MAE"]
#' )
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
#' boot_tf <- bootstrap_trendfilter(
#'   obj = sure_tf,
#'   algorithm = "parametric",
#'   edf = sure_tf$edf_min
#' )

#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom rlang %||%
#' @importFrom parallel mclapply detectCores
#' @importFrom stats residuals fitted.values approx
#' @export
bootstrap_trendfilter <- function(obj,
                                  algorithm = c("nonparametric","parametric","wild"),
                                  B = 100L,
                                  x_eval = NULL,
                                  edf = NULL,
                                  mc_cores = parallel::detectCores() - 4,
                                  ...) {
  stopifnot(B >= 20)
  stopifnot(any(class(obj) == "trendfiltering"))
  stopifnot(
    any(class(obj) == "cv_trendfilter") || any(class(obj) == "sure_trendfilter")
  )

  boot.call <- match.call
  extra_args <- list(...)
  algorithm <- match.arg(algorithm)

  if (is.null(edf)) {
    if (any(class(obj) == "cv_trendfilter")) {
      if (sd(obj$weights) > 0L) {
        edf_opt <- median(obj$edf_min[c(2,4,6,8,9)])
      } else {
        edf_opt <- median(obj$edf_min[c(1,3,5,7,9)])
      }
    } else {
        edf_opt <- obj$edf_min
    }
  } else {
    edf_opt <- edf
    stopifnot(is.numeric(edf_opt))
    if (length(edf_opt) > 1) {
      stop("`edf` must be of length 1.")
    }
    if (edf_opt < obj$k + 1 || edf_opt > length(obj$x)) {
      stop(
        "`edf` must be greater than `k + 1` and less than n. See ",
        "`obj$edf_min` and `obj$edf_1se` for reasonable choices."
      )
    }
  }

  if (is.null(x_eval)) x_flag <- TRUE
  x_eval <- x_eval %||% obj$x

  stopifnot(is.numeric(x_eval))
  if (any(x_eval < min(obj$x) || x_eval > max(obj$x))) {
    stop("One of more values in `x_eval` is outside the observed `x` range.")
  }

  stopifnot(
    is.numeric(mc_cores) && length(mc_cores) == 1 && round(mc_cores) == mc_cores
  )

  mc_cores <- min(
    B,
    detectCores(),
    max(1, floor(mc_cores))
  ) %>%
    as.integer()

  if (mc_cores < detectCores() / 2 && mc_cores < B) {
    warning(
      "Your machine has ", detectCores(), " cores.\n Consider increasing ",
      "`mc_cores` to speed up computation."
    )
  }

  if ("edf_radius" %in% names(extra_args)) {
    edf_radius <- extra_args$edf_radius
    extra_args$edf_radius <- NULL
  } else {
    edf_radius <- 5
  }

  i_opt <- which.min(abs(obj$edf - edf_opt))

  lambda_grid <- c(
    obj$lambda[
      max(i_opt - edf_radius, 1):min(i_opt + edf_radius, length(obj$edf))
    ],
    edf_opt
  ) %>%
    unique() %>%
    sort(decreasing = TRUE)

  if ("edf_tol" %in% names(extra_args)) {
    edf_tol <- extra_args$edf_tol
    extra_args$edf_tol <- NULL
  } else {
    edf_tol <- 0.3
  }

  if ("zero_tol" %in% names(extra_args)) {
    zero_tol <- extra_args$zero_tol
    extra_args$zero_tol <- NULL
  } else {
    zero_tol <- 1e-10
  }

  sampler <- case_when(
    algorithm == "nonparametric" ~ list(nonparametric_resampler),
    algorithm == "parametric" ~ list(parametric_sampler),
    algorithm == "wild" ~ list(wild_sampler)
  )[[1]]

  data_scaled <- tibble(
    x = obj$x / obj$scale["x"],
    y = obj$y / obj$scale["y"],
    weights = obj$weights * obj$scale["y"]^2
  )

  lambda_opt <- obj$lambda[i_opt]

  if (algorithm == "parametric") {
    data_scaled %<>% mutate(
      fitted_values = fitted(obj, lambda = lambda_opt) / obj$scale["y"]
    )
  }

  if (algorithm == "wild") {
    data_scaled %<>% mutate(
      fitted_values = fitted(obj, lambda = lambda_opt) / obj$scale["y"],
      residuals = residuals(obj, lambda = lambda_opt) / obj$scale["y"]
    )
  }

  par_out <- mclapply(
    1:B,
    bootstrap_parallel,
    data_scaled = data_scaled,
    k = obj$k,
    admm_params = obj$admm_params,
    edf_opt = edf_opt,
    lambda_grid = lambda_grid,
    sampler = sampler,
    x_eval = x_eval / obj$scale["x"],
    edf_tol = edf_tol,
    zero_tol = zero_tol,
    scale = obj$scale,
    mc.cores = mc_cores
  )

  ensemble <- sapply(
    1:B,
    FUN = function(X) par_out[[X]][["tf_estimate_boot"]]
  )

  edf_boots <- sapply(
    1:B,
    FUN = function(X) par_out[[X]][["edf_boot"]]
  ) %>%
    as.integer()

  n_iter_boots <- sapply(
    1:B,
    FUN = function(X) par_out[[X]][["n_iter_boot"]]
  ) %>%
    as.integer()

  lambda_boots <- sapply(
    1:B,
    FUN = function(X) par_out[[X]][["lambda_boot"]]
  )

  invisible(
    structure(
      list(
        x_eval = x_eval,
        ensemble = ensemble,
        algorithm = algorithm,
        edf_opt = edf_opt,
        edf_min = obj$edf_min,
        edf_1se = obj$edf_1se,
        lambda_min = obj$lambda_min,
        lambda_1se = obj$lambda_1se,
        i_min = obj$i_min,
        i_1se = obj$i_1se,
        edf_boots = edf_boots,
        n_iter_boots = n_iter_boots,
        lambda_boots = lambda_boots,
        lambda = obj$lambda,
        edf = obj$edf,
        fitted_values = obj$fitted_values,
        x = obj$x,
        y = obj$y,
        weights = obj$weights,
        k = obj$k,
        call = boot.call,
        scale = obj$scale
      ),
      class = c("bootstrap_trendfilter", "trendfilter", "trendfiltering")
    )
  )
}


#' @noRd
#' @importFrom glmgen .tf_fit .tf_boot
bootstrap_parallel <- function(b,
                               data_scaled,
                               k,
                               admm_params,
                               edf_opt,
                               lambda_grid,
                               sampler,
                               x_eval,
                               edf_tol,
                               zero_tol,
                               scale) {
  data_scaled <- sampler(data_scaled)

  fit <- .trendfilter(
    x = data_scaled$x,
    y = data_scaled$y,
    weights = data_scaled$weights,
    k = k,
    lambda = lambda_grid,
    obj_tol = admm_params$obj_tol,
    max_iter = admm_params$max_iter,
    scaling = FALSE
  )

  i_min <- which.min(abs(fit$edf - edf_opt))[1]
  edf_boot <- fit$edf[i_min]

  if ((abs(edf_opt - edf_boot) / edf_opt) > edf_tol) {
    return(
      bootstrap_parallel(
        b = 1,
        data_scaled,
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

  n_iter_boot <- fit$n_iter[i_min]
  lambda_boot <- lambda_grid[i_min]

  tf_estimate_boot <- predict(
    fit,
    lambda = lambda_boot,
    x_eval = x_eval,
    zero_tol = zero_tol
  ) * scale["y"]

  tf_estimate_boot %<>% as.numeric()
  names(tf_estimate_boot) <- NULL

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
#' boot_tf <- bootstrap_trendfilter(
#'   obj = cv_tf,
#'   algorithm = "nonparametric",
#'   edf = cv_tf$edf_min["MAE"]
#' )
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
#' boot_tf <- bootstrap_trendfilter(
#'   obj = sure_tf,
#'   algorithm = "parametric",
#'   edf = sure_tf$edf_min
#' )
#' bands <- vbands(boot_tf)

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
