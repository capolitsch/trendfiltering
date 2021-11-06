#' Construct pointwise variability bands via a bootstrap
#'
#' \loadmathjax Generate a bootstrap ensemble of trend filtering estimates in
#' order to quantify the uncertainty in the optimized estimate.
#' One of three possible bootstrap algorithms should be chosen according to the
#' criteria in the details section below. See [Politsch et al. (2020a)](
#' https://academic.oup.com/mnras/article/492/3/4005/5704413) for the technical
#' details of each bootstrap algorithm. Pointwise variability bands are then
#' obtained by passing the `bootstrap_trendfilter()` output and a desired level
#' (e.g. `level = 0.95`) to [`vbands()`].
#'
#' @param obj An object of class [`'sure_tf'`][sure_trendfilter()] or
#' [`'cv_tf'`][cv_trendfilter].
#' @param validation_error_metric (For object class `'cv_tf'` only) A string
#' specifying which cross validation error curve to optimize with respect to.
#' The options can be returned by running `names(obj$validation_errors)`, and
#' they can also be specified by passing their index.
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
#' @param bootstrap_algorithm A string specifying which variation of the
#' bootstrap to use. One of `c("nonparametric","parametric","wild")`. See
#' details below for recommendations on when each option is appropriate.
#' @param B The number of bootstrap samples used to estimate the pointwise
#' variability bands. Defaults to `B = 100`.
#' @param x_eval (Optional) A grid of inputs to evaluate the bootstrap trend
#' filtering estimates on. This should be the same as the grid you want the
#' variability bands to be evaluated on.
#' @param nx_eval Integer. The length of the input grid that the optimized
#' trend filtering estimate is evaluated on; i.e. if nothing is passed to
#' `x_eval`, then it is defined as
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
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
#' | `x` is evenly sampled and reciprocal variances are available     | `bootstrap_algorithm = "parametric"`   |
#' | `x` is evenly sampled and reciprocal variances are not available | `bootstrap_algorithm = "wild"`         |
#'
#' For our purposes, an evenly sampled data set with some discarded pixels
#' (either sporadically or in large consecutive chunks) is still considered to
#' be evenly sampled. When the inputs are evenly sampled on a transformed scale,
#' we recommend transforming to that scale and carrying out the full trend
#' filtering analysis on that scale. See the example below for a case when the
#' inputs are evenly sampled on the `log10(x)` scale.
#'
#' @return An object of class `'bootstrap_tf'`. This is a list with the
#' following elements
#' \describe{
#' \item{ensemble}{The full trend filtering bootstrap ensemble as an
#' \mjseqn{n \times B} matrix.}
#' \item{edf_boots}{Vector of the estimated number of effective degrees of
#' freedom of each trend filtering bootstrap estimate. These should all be
#' relatively close to `obj$edf_min`.}
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
#' @seealso [cv_trendfilter()], [sure_trendfilter()]
#'
#' @examples
#' data(quasar_spectrum)
#' head(spec)
#'
#' sure_tf <- sure_trendfilter(spec$log10_wavelength, spec$flux, spec$weights)
#' boot_tf <- bootstrap_trendfilter(sure_tf, bootstrap_algorithm = "parametric")
#' bands <- vbands(boot_tf)
#' @importFrom magrittr %>% %<>%
#' @importFrom dplyr case_when
#' @importFrom parallel detectCores
#' @rdname bootstrap_trendfilter
#' @export
bootstrap_trendfilter <- function(obj,
                                  validation_error_metric,
                                  lambda_choice = c("lambda_min", "lambda_1se"),
                                  bootstrap_algorithm = c("nonparametric", "parametric", "wild"),
                                  B = 100L,
                                  x_eval,
                                  nx_eval = 1500L,
                                  mc_cores = parallel::detectCores() - 4) {
  UseMethod("bootstrap_trendfilter")
}


#' @importFrom glmgen trendfilter
#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom parallel mclapply detectCores
#' @rdname bootstrap_trendfilter
#' @export
bootstrap_trendfilter.cv_tf <- function(obj,
                                        validation_error_metric,
                                        lambda_choice = c("lambda_min", "lambda_1se"),
                                        bootstrap_algorithm = c("nonparametric", "parametric", "wild"),
                                        B = 100L,
                                        x_eval,
                                        nx_eval = 1500L,
                                        mc_cores = parallel::detectCores() - 4) {
  stopifnot(any(class(obj) %in% "cv_tf"))
  stopifnot(B >= 10)
  stopifnot(
    class(validation_error_metric) %in% c("character", "integer", "numeric")
  )

  if (is.character(validation_error_metric)) {
    stopifnot(validation_error_metric %in% names(obj$validation_errors))
  }

  if (is.double(validation_error_metric) | is.integer(validation_error_metric)) {
    if (validation_error_metric != round(validation_error_metric)) {
      validation_error_metric <- which.min(
        abs(validation_error_metric - 1:length(obj$i_min))
      )

      warning(
        cat(paste0(
          "validation_error_metric should either be one of c('",
          paste(names(obj$i_min), collapse = "', '"),
          "'), or an index in 1:", length(obj$i_min),
          ".\nChoosing the closest index option: ", validation_error_metric,
          " ('", names(obj$i_min)[validation_error_metric], "')."
        )),
        call. = FALSE
      )
    }
  }

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  } else {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  if (mc_cores < detectCores() / 2) {
    warning(
      cat(paste0(
        "Your machine has ", detectCores(), " cores.\n Consider increasing",
        "mc_cores to speed up computation."
      )),
      call. = FALSE
    )
  }

  if (mc_cores > detectCores()) mc_cores <- detectCores()
  if (mc_cores < 1) mc_cores <- 1
  mc_cores <- min(floor(mc_cores), B)
  lambda_choice <- match.arg(lambda_choice)

  if (bootstrap_algorithm != "nonparametric") {
    lambda_opt <- case_when(
      lambda_choice == "lambda_min" ~ obj$lambda_min,
      lambda_choice == "lambda_1se" ~ obj$lambda_1se
    )

    obj$tf_model$data_scaled %<>% mutate(
      fitted_values = glmgen:::predict.trendfilter(
        obj$tf_model$model_fit,
        lambda = lambda_opt,
        x.new = obj$tf_model$model_fit$x
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
    bootstrap_parallel_cv,
    obj = obj,
    sampler = sampler,
    lambda_choice = lambda_choice,
    validation_error_metric = validation_error_metric,
    x_eval = x_eval,
    mc.cores = mc_cores
  )

  ensemble <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["tf_estimate"]]
  ) %>%
    unlist() %>%
    matrix(nrow = length(x_eval))

  edf_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["edf"]]
  ) %>%
    unlist() %>%
    as.integer()

  n_iter_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["n_iter"]]
  ) %>%
    unlist() %>%
    as.integer()

  lambda_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["lambda"]]
  ) %>%
    unlist()

  structure(
    list(
      x_eval = x_eval,
      ensemble = ensemble,
      edf_boots = edf_boots,
      n_iter_boots = n_iter_boots,
      lambda_boots = lambda_boots,
      bootstrap_algorithm = bootstrap_algorithm
    ),
    class = c("bootstrap_tf", "list")
  )
}


#' @noRd
#' @importFrom dplyr case_when
#' @importFrom glmgen trendfilter
bootstrap_parallel_cv <- function(b,
                                  obj,
                                  sampler,
                                  lambda_choice,
                                  validation_error_metric,
                                  x_eval) {
  data <- sampler(obj$tf_model$data_scaled)

  index <- case_when(
    lambda_choice == "lambda_min" ~ obj$i_min[validation_error_metric],
    lambda_choice == "lambda_1se" ~ obj$i_1se[validation_error_metric]
  )

  lambdas <- obj$lambdas[max(index - 10, 1):min(index + 10)]

  tf_fit <- trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = obj$tf_model$k,
    lambda = lambdas,
    thinning = obj$tf_model$thinning,
    control = obj$tf_model$admm_params
  )

  edf <- case_when(
    lambda_choice == "lambda_min" ~ obj$edf_min[validation_error_metric],
    lambda_choice == "lambda_1se" ~ obj$edf_1se[validation_error_metric]
  )

  i_min <- which.min(abs(tf_fit$df - edf))
  edf <- tf_fit$df[i_min]
  n_iter <- tf_fit$iter[i_min]
  lambda <- lambdas[i_min]

  if (edf <= 5 | min(abs(tf_fit$df - edf)) / edf - 1 > 0.2) {
    bootstrap_parallel_cv(
      1,
      obj,
      sampler,
      lambda_choice,
      validation_error_metric,
      x_eval
    )
  }

  tf_estimate <- glmgen:::predict.trendfilter(
    object = tf_fit,
    x.new = x_eval / obj$tf_model$x_scale,
    lambda = lambda
  ) %>%
    as.double()

  list(
    tf_estimate = tf_estimate * obj$tf_model$y_scale,
    edf = edf,
    n_iter = n_iter,
    lambda = lambda
  )
}


#' @importFrom glmgen trendfilter
#' @importFrom dplyr case_when mutate
#' @importFrom magrittr %>% %<>%
#' @importFrom parallel mclapply detectCores
#' @rdname bootstrap_trendfilter
#' @export
bootstrap_trendfilter.sure_tf <- function(obj,
                                          lambda_choice = c("lambda_min", "lambda_1se"),
                                          bootstrap_algorithm = c("nonparametric", "parametric", "wild"),
                                          B = 100L,
                                          x_eval,
                                          nx_eval = 1500L,
                                          mc_cores = parallel::detectCores() - 4) {
  stopifnot(any(class(obj) %in% "sure_tf"))
  stopifnot(B >= 10)

  if (missing(x_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  } else {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  if (mc_cores < detectCores() / 2) {
    warning(
      cat(paste0(
        "Your machine has ", detectCores(), " cores.\n Consider increasing",
        "mc_cores to speed up computation."
      )),
      call. = FALSE
    )
  }

  if (mc_cores > detectCores()) mc_cores <- detectCores()
  if (mc_cores < 1) mc_cores <- 1
  mc_cores <- min(floor(mc_cores), B)
  lambda_choice <- match.arg(lambda_choice)

  if (bootstrap_algorithm != "nonparametric") {
    lambda_opt <- case_when(
      lambda_choice == "lambda_min" ~ obj$lambda_min,
      lambda_choice == "lambda_1se" ~ obj$lambda_1se
    )

    obj$tf_model$data_scaled %<>% mutate(
      fitted_values = glmgen:::predict.trendfilter(
        obj$tf_model$model_fit,
        lambda = lambda_opt,
        x.new = obj$tf_model$model_fit$x
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
    bootstrap_parallel_sure,
    obj = obj,
    sampler = sampler,
    lambda_choice = lambda_choice,
    x_eval = x_eval,
    mc.cores = mc_cores
  )

  ensemble <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["tf_estimate"]]
  ) %>%
    unlist() %>%
    matrix(nrow = length(x_eval))

  edf_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["edf"]]
  ) %>%
    unlist() %>%
    as.integer()

  n_iter_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["n_iter"]]
  ) %>%
    unlist() %>%
    as.integer()

  lambda_boots <- lapply(
    X = 1:B,
    FUN = function(X) par_out[[X]][["lambda"]]
  ) %>%
    unlist()

  structure(
    list(
      x_eval = x_eval,
      ensemble = ensemble,
      edf_boots = edf_boots,
      n_iter_boots = n_iter_boots,
      lambda_boots = lambda_boots,
      bootstrap_algorithm = bootstrap_algorithm
    ),
    class = c("bootstrap_tf", "list")
  )
}


#' @noRd
#' @importFrom glmgen trendfilter
bootstrap_parallel_sure <- function(b,
                                    obj,
                                    sampler,
                                    lambda_choice,
                                    x_eval) {
  data <- sampler(obj$tf_model$data_scaled)

  index <- case_when(
    lambda_choice == "lambda_min" ~ obj$i_min,
    lambda_choice == "lambda_1se" ~ obj$i_1se
  )

  lambdas <- obj$lambdas[max(index - 10, 1):min(index + 10)]

  tf_fit <- trendfilter(
    x = data$x,
    y = data$y,
    weights = data$weights,
    k = obj$tf_model$k,
    lambda = lambdas,
    thinning = obj$tf_model$thinning,
    control = obj$tf_model$admm_params
  )

  edf <- case_when(
    lambda_choice == "lambda_min" ~ obj$edf_min,
    lambda_choice == "lambda_1se" ~ obj$edf_1se
  )

  i_min <- which.min(abs(tf_fit$df - edf))
  edf <- tf_fit$df[i_min]
  n_iter <- tf_fit$iter[i_min]
  lambda <- lambdas[i_min]

  if (edf <= 5 | min(abs(tf_fit$df - edf)) / edf - 1 > 0.2) {
    bootstrap_parallel_sure(
      1,
      obj,
      sampler,
      lambda_choice,
      x_eval
    )
  }

  tf_estimate <- glmgen:::predict.trendfilter(
    object = tf_fit,
    x.new = x_eval / obj$tf_model$x_scale,
    lambda = lambda
  ) %>%
    as.double()

  list(
    tf_estimate = tf_estimate * obj$tf_model$y_scale,
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
  data %>% slice_sample(n = n(), replace = TRUE)
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
