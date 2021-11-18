#' Evaluate an optimized trend filtering estimate
#'
#' Evaluate an optimized trend filtering object on a grid of input values. The
#' `lambda` argument (as well as the `loss_func` argument for
#' '[`cv.trendfilter`][`cv.trendfilter()`]' objects) should be used to specify
#' the desired method for optimizing the trend filtering hyperparameter.
#'
#' @param obj An object of class '[`cv.trendfilter`][`cv.trendfilter()`]' or
#' '[`sure.trendfilter`][`sure.trendfilter()`]'.
#' @param loss_func (For class '[`cv.trendfilter`][`cv.trendfilter()`]' only) A
#' string or index specifying which cross validation error curve stored within
#' the '[`cv.trendfilter`][`cv.trendfilter()`]' object will be used to optimize
#' the trend filtering hyperparameter. Run `names(obj$loss_funcs)` to see the
#' available options. Defaults to `loss_func = "WMAE"`.
#' @param lambda One of `c("lambda_min", "lambda_1se")`. The choice of
#' hyperparameter that is used for optimized trend filtering estimate. Defaults
#' to `lambda = "lambda_min"`.
#' \itemize{
#' \item{`"lambda_min"`}: The hyperparameter value that minimizes the cross
#' validation error curve specified by `loss_func`.
#' \item{`"lambda_1se"`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering
#' estimates.
#' }
#' @param x_eval A grid of inputs to evaluate the optimized trend filtering
#' estimate on. Defaults to the observed values `x` when nothing is passed.
#' @param nx_eval Integer. If passed, overrides `x_eval` with
#' `x_eval = seq(min(x), max(x), length = nx_eval)`.
#' @details
#' The motivation for using `lambda = "lambda_1se"` is essentially
#' Occam's razor: the two models yield results that are quantitatively very
#' close, so we favor the simpler model. See Section 7.10 of
#' [Hastie, Tibshirani, and Friedman (2009)](
#' https://web.stanford.edu/~hastie/Papers/ESLII.pdf) for more details on the
#' "one-standard-error rule".
#'
#' @return
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
#' @seealso [`cv.trendfilter()`], [`sure.trendfilter()`]
#'
#' @examples
#' data("eclipsing_binary")
#' head(eclipsing_binary)
#'
#' x <- eclipsing_binary$phase
#' y <- eclipsing_binary$flux
#' weights <- 1 / eclipsing_binary$std_err^2
#'
#' cv_tf <- cv.trendfilter(x, y, weights,
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
#'   lambda = "lambda_1se",
#'   nx_eval = 1500L
#' )
#' @importFrom dplyr case_when tibble
#' @importFrom magrittr %<>% %>%
#' @rdname predict.trendfilter
#' @export
predict.trendfilter <- function(obj,
                                loss_func = "WMAE",
                                lambda = "lambda_min",
                                x_eval,
                                nx_eval,
                                zero_tol = 1e-6,
                                ...) {
  stopifnot(any(class(obj) == "cv.trendfilter"))
  stopifnot(
    class(loss_func) %in% c("character", "integer", "numeric")
  )

  if (is.character(loss_func)) {
    stopifnot(loss_func %in% names(obj$errors))
  }

  if (is.double(loss_func) | is.integer(loss_func)) {
    if (loss_func != round(loss_func)) {
      loss_func <- which.min(
        abs(loss_func - 1:length(obj$i_min))
      )

      warning(
        cat(paste0(
          "loss_func should either be one of c('",
          paste(names(obj$i_min), collapse = "', '"),
          "'), or an index in 1:", length(obj$i_min),
          ".\nChoosing the closest index option: ", loss_func,
          " ('", names(obj$i_min)[loss_func], "')."
        )),
        call. = FALSE
      )
    }
  }

  if (missing(x_eval) & missing(nx_eval)) {
    x_eval <- obj$tf_model$x %>%
      as.double() %>%
      sort()
  }

  if (!missing(x_eval) & missing(nx_eval)) {
    if (any(x_eval < min(obj$tf_model$x) || x_eval > max(obj$tf_model$x))) {
      stop("x_eval should all be in range(x).")
    }
    x_eval %<>%
      as.double() %>%
      sort()
  }

  extra_args <- list(...)

  if (!missing(nx_eval)) {
    if (!(class(nx_eval) %in% c("numeric", "integer"))) {
      stop("nx_eval must be a positive integer.")
    }
    if (nx_eval < 1 || nx_eval != round(nx_eval)) {
      stop("nx_eval must be a positive integer.")
    }
    x_eval <- seq(min(obj$tf_model$x), max(obj$tf_model$x), length = nx_eval)
  }

  i_opt <- case_when(
    lambda == "lambda_min" ~ obj$i_min[loss_func],
    lambda == "lambda_1se" ~ obj$i_1se[loss_func]
  )

  names(i_opt) <- NULL

  tf_estimate <- as.numeric(
    glmgen:::predict.trendfilter(
      obj$tf_model$model_fit,
      lambda = obj$lambdas[i_opt],
      x.new = x_eval / obj$tf_model$x_scale
    )
  ) * obj$tf_model$y_scale

  if (is.null(x.new)) x.new <- obj$x
  if (is.null(lambdas)) lambdas <- obj$lambdas

  co <- coef(obj, lambdas)

  tf_pred <- .Call("tf_predict_R",
                   sX = as.double(obj$x),
                   sBeta = as.double(co),
                   sN = length(obj$y),
                   sK = as.integer(obj$k),
                   sX0 = as.double(x.new),
                   sN0 = length(x.new),
                   sNLambda = length(lambdas),
                   sFamily = 0L,
                   sZeroTol = as.double(zero_tol),
                   PACKAGE = "glmgen"
  )

  matrix(tf_pred, ncol = ncol(co), dimnames = list(NULL, colnames(co)))

  structure(
    list(
      x_eval = x_eval,
      tf_estimate = tf_estimate,
      lambdas = obj$lambdas,
      edfs = obj$edfs,
      errors = obj$errors[[loss_func]],
      se_errors = obj$se_errors[[loss_func]],
      loss_func = loss_func,
      lambda_opt = obj$lambdas[i_opt],
      edf_opt = obj$edfs[i_opt],
      i_opt = i_opt,
      tf_model = obj$tf_model
    ),
    class = c("cv.trendfilter", "trendfilter", "list")
  )
}


#' Get coefficients from a trendfilter object
#'
#' @param obj
#'   Object of class `trendfilter`.
#' @param lambdas
#'   (Optional) Vector of lambda values to calculate coefficients at. If
#'   missing, will use break points in the fit.
#'
#' @export
coef.trendfilter <- function(obj, lambdas = NULL) {
  # If no lambdas given, just return beta
  if (is.null(lambdas)) {
    return(obj$beta)
  }

  # If all lambdas are equal to some computed lambda, just
  # return propely transformed version of `obj$beta`
  if (all(!is.na(index <- match(lambdas, obj$lambdas)))) {
    return(obj$beta[, index, drop = FALSE])
  }

  if (min(lambdas) < 0) stop("All specified lambda values must be nonnegative.")
  if (min(lambdas) < min(obj$lambdas) | max(lambdas) > max(obj$lambdas)) {
    stop("Cannot predict lambda outside the range used when fitting.")
  }

  # If here, need to interpolate `lambdas`
  o <- order(lambdas, decreasing = TRUE)
  o2 <- order(obj$lambdas, decreasing = TRUE)
  lambdas <- lambdas[o]
  knots <- obj$lambdas[o2]
  k <- length(lambdas)
  mat <- matrix(rep(knots, each = k), nrow = k)
  b <- lambdas >= mat
  blo <- max.col(b, ties.method = "first")
  bhi <- pmax(blo - 1, 1)
  i <- bhi == blo
  p <- numeric(k)
  p[i] <- 0
  p[!i] <- ((lambdas - knots[blo]) / (knots[bhi] - knots[blo]))[!i]

  betas <- obj$beta[, o2, drop = FALSE]
  beta <- t((1 - p) * t(betas[, blo, drop = FALSE]) +
              p * t(betas[, bhi, drop = FALSE]))
  colnames(beta) <- as.character(round(lambdas, 3))

  beta[, order(o), drop = FALSE]
}
