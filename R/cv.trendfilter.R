#' Optimize the trend filtering hyperparameter by V-fold cross validation
#' 
#' `cv.trendfilter` optimizes the trend filtering hyperparameter 
#' by performing V-fold cross validation on a vector, `gammas`, of candidate 
#' hyperparameter values. The full generalization error curve and the optimized
#' trend filtering estimate are then returned within a list that also includes 
#' a detailed summary of the analysis. One of `c("gamma.min", "gamma.1se")`.
#' 
#' @param x The vector of observed values of the input variable (a.k.a. the 
#' predictor, covariate, explanatory variable, regressor, independent variable, 
#' control variable, etc.)
#' @param y The vector of observed values of the output variable (a.k.a. the
#' response, target, outcome, regressand, dependent variable, etc.).
#' @param weights A vector of weights for the observed outputs, defined as the
#' reciprocal of the variance of the error distribution. That is, 
#' `weights = 1 / sigmas^2`, where `sigmas` is a vector of standard errors of
#' the uncertainty in the observed outputs. `weights` should either have length
#' equal to 1 (corresponding to an error distribution with a constant variance)
#' or length equal to `length(y)` (i.e. heteroskedastic errors). 
#' @param k The degree of the trend filtering estimator. More precisely, with
#' the trend filtering estimator defined as a piecewise function of polynomials
#' smoothly connected at a set of "knots", `k` controls the degree of the
#' polynomials that build up the trend filtering estimator. Defaults to `k = 2`
#' (i.e. a piecewise quadratic estimate). Must be one of `k = 0,1,2,3`. However,
#' `k = 3` is discouraged due to algorithmic instability, and `k = 2` typically
#' gives a visually indistinguishable estimate anyway.
#' @param ngammas Integer. The number of trend filtering hyperparameter values 
#' to run the grid search over. In this default case, the hyperparameter values
#' are automatically chosen by `SURE.trendfilter` and `ngammas` simply controls
#' the granularity of the grid.
#' @param gammas Overrides `ngammas` if passed. A vector of trend filtering
#' hyperparameter values to run the grid search over. It is advisable to let the
#' vector be equally-spaced in log-space and passed to `SURE.trendfilter` in
#' descending order. The function output will contain the sorted hyperparameter
#' vector regardless of the user-supplied ordering, and all related output
#' objects (e.g. the `errors` vector) will correspond to this descending
#' ordering. It's best to leave this argument alone unless you know what you
#' are doing.
#' @param V The number of folds the data are split into for the V-fold cross
#' validation. Defaults to `V = 10` (recommended).
#' @param validation.error.type Type of error to optimize during cross
#' validation. One of `c("WMAE","WMSE","MAE","MSE")`, i.e. mean-absolute 
#' deviations error, mean-squared error, and their weighted counterparts. 
#' If `weights = NULL`, then the weighted and unweighted counterparts are
#' equivalent. In short, weighting helps combat heteroskedasticity and absolute
#' error decreases sensitivity to outliers. Defaults to `"WMAE"`.
#' @param x.eval A grid of inputs to evaluate the optimized trend filtering 
#' estimate on. Defaults to the observed inputs, `x`.
#' @param nx.eval Integer. If passed, overrides `x.eval` with
#' `seq(min(x), max(x), length = nx.eval)`.
#' @param gamma.choice One of `c("gamma.min","gamma.1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.
#' \itemize{
#' \item{`gamma.min`}: The hyperparameter value that minimizes the cross
#' validation error curve.
#' \item{`gamma.1se`}: The largest hyperparameter value with a cross
#' validation error within 1 standard error of the minimum cross validation 
#' error. This choice therefore favors simpler (i.e. smoother) trend filtering 
#' estimates. The motivation here is essentially Occam's razor: the two models
#' yield results that are quantitatively very close, so we favor the simpler
#' model.}
#' @param optimization.params A named list of parameters that contains all
#' parameter choices to be passed to the trend filtering ADMM algorithm
#' (\href{http://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf}{Ramdas and
#' Tibshirani 2016}). See the 
#' \code{\link[glmgen::trendfilter.control.list]{glmgen::trendfilter.control.list}}
#' documentation for full details. 
#' No technical understanding of the ADMM algorithm is needed and the default
#' parameter choices will almost always suffice. However, the following
#' parameters may require some adjustments to ensure that your trend filtering
#' estimate has sufficiently converged:
#' \enumerate{
#' \item{`max_iter`}: Maximum iterations allowed for the trend filtering 
#' convex optimization. Defaults to `max_iter = 600L`. Increase this if 
#' the trend filtering estimate does not appear to have fully converged to a 
#' reasonable estimate of the signal.
#' \item{`obj_tol`}: The tolerance used in the convex optimization stopping 
#' criterion; when the relative change in the objective function is less than 
#' this value, the algorithm terminates. Decrease this if the trend filtering 
#' estimate does not appear to have fully converged to a reasonable estimate of 
#' the signal.
#' \item{`thinning`}: Logical. If `TRUE`, then the data are preprocessed so that 
#' a smaller, better conditioned data set is used for fitting. When left `NULL`
#' (the default setting), the optimization will automatically detect whether
#' thinning should be applied (i.e. cases in which the numerical fitting
#' algorithm will struggle to converge). This preprocessing procedure is
#' controlled by the `x_tol` argument below.
#' \item{`x_tol`}: Controls the automatic detection of when thinning should be
#' applied to the data. If we make bins of size `x_tol` and find at least two
#' elements of `x` that fall into the same bin, then we thin the data.
#' }
#' @param mc.cores Parallel computing: The number of cores to utilize. Defaults
#' to the number of cores detected.
#' @param ... Additional named arguments to be passed to 
#' \code{\link[glmgen::trendfilter.control.list]{glmgen::trendfilter.control.list}}.
#' 
#' @return An object of class 'cv.trendfilter'. This is a list with the 
#' following elements:
#' \item{x.eval}{The grid of inputs the optimized trend filtering estimate was 
#' evaluated on.}
#' \item{tf.estimate}{The optimized trend filtering estimate of the signal, 
#' evaluated on \code{x.eval}.}
#' \item{validation.method}{\code{paste0(V,"-fold CV")}}
#' \item{V}{The number of folds the data are split into for the V-fold cross
#' validation.}
#' \item{validation.error.type}{Type of error that validation was performed on. 
#' One of \code{c("WMAE","WMSE","MAE","MSE")}.}
#' \item{gammas}{Vector of hyperparameter values tested during validation. This
#' vector will always be returned in descending order, regardless of the 
#' ordering provided by the user. The indices `i.min` and `i.1se` correspond to
#' this descending ordering.}
#' \item{errors}{Vector of cross validation errors for the given hyperparameter 
#' values.}
#' \item{se.errors}{The standard errors of the cross validation errors.
#' These are particularly useful for implementing the ``1-standard-error rule''. 
#' The 1-SE rule favors a smoother trend filtering estimate by, instead of 
#' using the hyperparameter that minimizes the CV error, instead uses the 
#' largest hyperparameter that has a CV error within 1 standard error of the
#' smallest CV error.}
#' \item{gamma.min}{Hyperparameter value that minimizes the SURE error curve.}
#' \item{gamma.1se}{The largest hyperparameter value that is still within one
#' standard error of the minimum hyperparameter's cross validation error.}
#' \item{gamma.choice}{One of `c("gamma.min","gamma.1se")`. The choice
#' of hyperparameter that is used for optimized trend filtering estimate.}
#' \item{edfs}{Vector of effective degrees of freedom for trend filtering
#' estimators fit during validation.}
#' \item{edf.min}{The effective degrees of freedom of the optimally-tuned trend 
#' filtering estimator.}
#' \item{edf.1se}{The effective degrees of freedom of the 1-stand-error rule
#' trend filtering estimator.}
#' \item{i.min}{The index of `gammas` that minimizes the cross validation error.}
#' \item{i.1se}{The index of `gammas` that gives the largest hyperparameter
#' value that has a cross validation error within 1 standard error of the 
#' minimum of the cross validation error curves.}
#' \item{x}{The vector of the observed inputs.}
#' \item{y}{The vector of the observed outputs.}
#' \item{weights}{A vector of weights for the observed outputs. These are
#' defined as `weights = 1 / sigmas^2`, where `sigmas` is a vector of 
#' standard errors of the uncertainty in the observed outputs.}
#' \item{fitted.values}{The optimized trend filtering estimate of the signal, 
#' evaluated at the observed inputs `x`.}
#' \item{residuals}{`residuals = y - fitted.values`}
#' \item{k}{The degree of the trend filtering estimator.}
#' \item{optimization.params}{A list of parameters that control the trend
#' filtering convex optimization.}
#' \item{n.iter}{Vector of the number of iterations needed for the ADMM
#' algorithm to converge within the given tolerance, for each hyperparameter
#' value. If many of these are exactly equal to `max_iter`, then their
#' solutions have not converged with the tolerance specified by `obj_tol`.
#' In which case, it is often prudent to increase `max_iter`.}
#' \item{thinning}{Logical. If `TRUE`, then the data are preprocessed so 
#' that a smaller, better conditioned data set is used for fitting.}
#' \item{x.scale, y.scale, data.scaled}{For internal use.}
#' 
#' @details @noMd \loadmathjax Suppose we observe noisy measurements of a response
#' variable of interest (e.g., flux, magnitude, photon counts) according to the
#' data generating process (DGP)
#' \mjdeqn{f(t_i) = f_0(t_i) + \epsilon_i,  \hfill t_1,\dots,t_n\in(a,b),}{ascii}
#' where \mjeqn{f(t_i)}{ascii} is a noisy measurement of the signal
#' \mjeqn{f_0(t_i)}{ascii}, and \mjeqn{\mathbb{E}[\epsilon_i] = 0}{ascii}.
#' Further, let \mjeqn{\sigma_{i}^{2} = \text{Var}(\epsilon_{i})}{ascii}.
#' The random-input mean-squared prediction error (MSPE) is given by
#' \mjdeqn{\widetilde{R}(\gamma) = \mathbb{E}\left\[\left(f(t) - \widehat{f}_{0}(t;\gamma)\right)^{2}\right\],}{ascii}
#' where \mjeqn{t}{ascii} is considered to be a random component of the DGP with
#' a marginal probability density \mjeqn{p_t(t)}{ascii} supported on the
#' observed input interval. The theoretically optimal choice of
#' \mjeqn{\gamma}{ascii} is defined as the minimizer of this error.
#' \deqn{WMAE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|\frac{\sqrt{w_i}}{\sum_j\sqrt{w_j}}}
#' \deqn{WMSE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|^2\frac{w_i}{\sum_jw_j}}
#' \mjsdeqn{MAE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|}
#' \mjsdeqn{MSE(\gamma) = \frac{1}{n}\sum_{i=1}^{n} |Y_i - \widehat{f}(x_i; \gamma)|^2}
#' where \mjeqn{\widehat{f}(x_i; \gamma)}{ascii} is the trend filtering 
#' estimate with hyperparameter \mjeqn{\gamma}{ascii}, evaluated at 
#' \mjeqn{x_i}{ascii}.
#' 
#' @export cv.trendfilter
#' 
#' @author \cr
#' \strong{Collin A. Politsch, Ph.D.}
#' ---
#' Email: collinpolitsch@@gmail.com \cr
#' Website: [collinpolitsch.com](https://collinpolitsch.com/) \cr
#' GitHub: [github.com/capolitsch](https://github.com/capolitsch/) \cr \cr
#' 
#' @references 
#' \strong{Companion references} 
#' \enumerate{
#' \item{\href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#' for time-domain astronomy and astronomical spectroscopy. \emph{Monthly 
#' Notices of the Royal Astronomical Society}, 492(3), p. 4005-4018.}} \cr
#' \item{\href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical 
#' signals with varying degrees of smoothness. \emph{Monthly Notices of the 
#' Royal Astronomical Society}, 492(3), p. 4019-4032.}}}
#' 
#' \strong{Cross validation}
#' \enumerate{
#' \item \href{https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf}{
#' Hastie, Tibshirani, and Friedman (2009). The Elements of Statistical 
#' Learning: Data Mining, Inference, and Prediction. 2nd edition. Springer 
#' Series in Statistics. (See Sections 7.10 and 7.12)} \cr
#' \item \href{https://www.statlearning.com/}{
#' James, Witten, Hastie, and Tibshirani (2013). An Introduction to 
#' Statistical Learning : with Applications in R. Springer. (See 
#' Section 5.1; Less technical than ESL)} \cr
#' \item \href{https://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf}{
#' Tibshirani (2013). Model selection and validation 2: Model
#' assessment, more cross-validation. \emph{36-462: Data Mining course notes} 
#' (Carnegie Mellon).}}
#' 
#' @seealso \code{\link{SURE.trendfilter}}, \code{\link{bootstrap.trendfilter}}
#' 
#' @examples 
#' #######################################################################
#' ###  Phase-folded light curve of an eclipsing binary star system   ####
#' #######################################################################
#' # A binary star system is a pair of closely-separated stars that move
#' # in an orbit around a common center of mass. When a binary star system 
#' # is oriented in such a way that the stars eclipse one another from our 
#' # vantage point on Earth, we call it an 'eclipsing binary (EB) star 
#' # system'. From our perspective, the total brightness of an EB dips 
#' # periodically over time due to the stars eclipsing one another. And 
#' # the shape of the brightness curve is consistent within each period
#' # of the orbit. In order to learn about the physics of these EBs,
#' # astronomers 'phase-fold' the brightness curve so that all the orbital 
#' # periods are stacked on top of one another in a plot of the EB's phase 
#' # vs. its apparent brightness, and then find a 'best-fitting' model
#' # for the phase-folded curve. Here, we use trend filtering to fit an
#' # optimal phase-folded model for an EB.
#' 
#' data(eclipsing_binary)
#' 
#' # head(df)
#' #
#' # |      phase|      flux|  std.err|
#' # |----------:|---------:|--------:|
#' # | -0.4986308| 0.9384845| 0.010160|
#' # | -0.4978067| 0.9295757| 0.010162|
#' # | -0.4957892| 0.9438493| 0.010162|
#' 
#' gamma.grid <- exp( seq(7, 16, length = 150) )
#' 
#' cv.out <- cv.trendfilter(x = df$phase, 
#'                          y = df$flux, 
#'                          weights = 1 / df$std.err ^ 2,
#'                          gammas = gamma.grid,
#'                          validation.error.type = "MAE",
#'                          optimization.params = list(max_iter = 5e3, obj_tol = 1e-6, thinning = T))
#' 
#' # Plot the results
#' 
#' par(mfrow = c(2,1), mar = c(5,4,2.5,1) + 0.1)
#' plot(log(cv.out$gammas), cv.out$errors, main = "CV error curve", 
#'      xlab = "log(gamma)", ylab = "CV error")
#' segments(x0 = log(cv.out$gammas), x1 = log(cv.out$gammas), 
#'          y0 = cv.out$errors - cv.out$se.errors, 
#'          y1 = cv.out$errors + cv.out$se.errors)
#' abline(v = log(cv.out$gamma.min), lty = 2, col = "blue3")
#' text(x = log(cv.out$gamma.min), y = par("usr")[4], 
#'      labels = "optimal gamma", pos = 1, col = "blue3")
#' plot(df$phase, df$flux, cex = 0.15, xlab = "Phase", ylab = "Flux",
#'      main = "Eclipsing binary phase-folded light curve")
#' segments(x0 = df$phase, x1 = df$phase, 
#'          y0 = df$flux - df$std.err, y1 = df$flux + df$std.err, 
#'          lwd = 0.25)
#' lines(cv.out$x.eval, cv.out$tf.estimate, col = "orange", lwd = 2.5)


#' @importFrom dplyr mutate arrange case_when group_split bind_rows
#' @importFrom glmgen trendfilter trendfilter.control.list
#' @importFrom parallel mclapply detectCores
#' @importFrom matrixStats rowSds
#' @importFrom magrittr %$% %>%
#' @importFrom tidyr tibble drop_na
cv.trendfilter <- function(x, y, weights = NULL, 
                           V = 10L,
                           ngammas = 250L,
                           gammas,
                           x.eval = x,
                           nx.eval,
                           k = 2L,
                           gamma.choice = c("gamma.min","gamma.1se"),
                           validation.error.type = c("WMAE","WMSE","MAE","MSE"),
                           optimization.params = list(max_iter = 600L, obj_tol = 1e-10, thinning = NULL),
                           mc.cores = detectCores(),
                           ...){
  
  if ( missing(x) || is.null(x) ) stop("x must be passed.")
  if ( missing(y) || is.null(y) ) stop("y must be passed.")
  if ( length(x) != length(y) ) stop("x and y must have the same length.")
  
  if ( !is.null(weights) ){
    if ( !(length(weights) %in% c(1,length(y))) ){
      stop("weights must either be have length 1 or length(y), or be NULL.")
    }
  }
  
  if ( length(y) < k + 2 ){
    stop("y must have length >= k + 2 for kth order trend filtering.")
  }
  
  if ( k < 0 || k != round(k) ){
    stop("k must be a nonnegative integer. `k = 2` recommended")
  }
  
  if ( k > 3 ){
    stop(paste0("Large k leads to generally worse conditioning.\n", 
                "k = 0,1,2 are the most stable choices."))
  }
  
  if ( k == 3 ){
    warning(paste0("`k = 3` can have poor conditioning...\n", 
                   "`k = 2` is more stable and visually indistinguishable."))
  }
  
  if ( is.null(gammas) && (ngammas != round(ngammas) || ngammas < 25L) ){
    stop("ngammas must be a positive integer >= 25.")
  }
  
  if ( !is.null(gammas) ){
    if ( min(gammas) < 0L ){
      stop("All specified gamma values must be positive.")
    }
    if ( length(gammas) < 25L ) stop("gammas must have length >= 25.")
  }
  
  if ( is.null(x.eval) ){
    if ( nx.eval != round(nx.eval) ) stop("nx.eval must be a positive integer.")
  }else{
    if ( any(x.eval < min(x) || x.eval > max(x)) ){
      stop("x.eval should all be in range(x).")
    }
  }
  
  if ( mc.cores < detectCores() ){
    warning(paste0("Your machine has ", detectCores(), " cores. Consider increasing `mc.cores` to speed up computation."))
  }
  
  if ( mc.cores > detectCores() ){
    warning(paste0("Your machine only has ", detectCores(), " cores. Adjusting `mc.cores` accordingly."))
    mc.cores <- detectCores()
  }
  
  if ( length(weights) == 1 ){
    weights <- rep(weights, length(y))
  }
  
  if ( length(weights) == 0 ){
    weights <- rep(1, length(y))
  }
  
  if ( is.null(gammas) ){
    gammas <- exp(seq(16, -10, length = ngammas))
  }else{
    gammas <- sort(gammas, decreasing = TRUE)
  }
  
  mc.cores <- min(mc.cores, V)
  gamma.choice <- match.arg(gamma.choice)
  validation.error.type <- match.arg(validation.error.type)
  
  data <- tibble(x, y, weights) %>% 
    arrange(x) %>% 
    filter( weights != 0 ) %>%
    drop_na 
  rm(x,y,weights)
  
  thinning <- optimization.params$thinning
  optimization.params <- trendfilter.control.list(max_iter = optimization.params$max_iter,
                                                  obj_tol = optimization.params$obj_tol,
                                                  ...)
  x.scale <- median(diff(data$x))
  y.scale <- median(abs(data$y)) / 10
  optimization.params$x_tol <- optimization.params$x_tol / x.scale
  
  data.scaled <- data %>%
    mutate(x = x / x.scale,
           y = y / y.scale,
           weights = weights * y.scale ^ 2)
  
  data.folded <- data.scaled %>% 
    group_split( sample( rep_len(1:V, nrow(data.scaled)) ), .keep = FALSE )
  
  if ( is.null(x.eval) ){
    x.eval <- seq(min(data$x), max(data$x), length = nx.eval)
  }else{
    x.eval <- sort(x.eval)
  }
  
  obj <- structure(list(x.eval = x.eval,
                        validation.method = paste0(V,"-fold CV"),
                        V = as.integer(V),
                        validation.error.type = validation.error.type,
                        gammas = gammas, 
                        gamma.choice = gamma.choice,
                        x = data$x,
                        y = data$y,
                        weights = data$weights,
                        k = as.integer(k),
                        thinning = thinning,
                        optimization.params = optimization.params,
                        data.scaled = data.scaled,
                        x.scale = x.scale, 
                        y.scale = y.scale),
                   class = "cv.trendfilter"
                   )
  
  rm(V,validation.error.type,gammas,ngammas,gamma.choice,k,thinning,data,nx.eval,
     optimization.params,data.scaled,x.eval,x.scale,y.scale)
  
  cv.out <- matrix(unlist(mclapply(1:(obj$V), FUN = trendfilter.validate, 
                                   data.folded = data.folded, 
                                   obj = obj, 
                                   mc.cores = mc.cores)
                          ),
                   ncol = obj$V)
  
  errors <- rowMeans(cv.out)
  se.errors <- rowSds(cv.out) / sqrt(obj$V)
  obj$i.min <- which.min(errors)
  obj$i.1se <- which(errors <= errors[obj$i.min] + se.errors[obj$i.min]) %>% min
  obj$gamma.min <- obj$gammas[obj$i.min]
  obj$gamma.1se <- obj$gammas[obj$i.1se]
  
  if ( obj$validation.error.type %in% c("MSE","WMSE") ){
    obj$errors <- errors * obj$y.scale ^ 2
    obj$se.errors <- se.errors * obj$y.scale ^ 2
  }
  if ( obj$validation.error.type %in% c("MAE","WMAE") ){
    obj$errors <- errors * obj$y.scale
    obj$se.errors <- se.errors * obj$y.scale
  }
  
  out <- obj %$% trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gammas,
                             k = k,
                             thinning = thinning,
                             control = optimization.params)
  
  gamma.pred <- case_when(
    obj$gamma.choice == "gamma.min" ~ obj$gamma.min,
    obj$gamma.choice == "gamma.1se" ~ obj$gamma.1se
  )
  
  obj$n.iter <- out$iter
  obj$edfs <- out$df
  obj$edf.min <- out$df[obj$i.min]
  obj$edf.1se <- out$df[obj$i.1se]
  
  # Increase the algorithmic precision for the optimized TF estimate
  obj$optimization.params$obj_tol <- obj$optimization.params$obj_tol * 1e-2
  
  out <- obj %$% trendfilter(x = data.scaled$x,
                             y = data.scaled$y,
                             weights = data.scaled$weights,
                             lambda = gammas,
                             k = k,
                             thinning = thinning,
                             control = optimization.params)
  
  optimization.params$obj_tol <- optimization.params$obj_tol * 1e2
  
  obj$data.scaled$fitted.values <- glmgen:::predict.trendfilter(out, lambda = gamma.pred,
                                                                x.new = obj$data.scaled$x) %>% 
    as.numeric
  obj$data.scaled$residuals <- obj$data.scaled$y - obj$data.scaled$fitted.values
  obj$tf.estimate <- glmgen:::predict.trendfilter(out, lambda = gamma.pred,
                                                  x.new = obj$x.eval / obj$x.scale) * obj$y.scale %>%
    as.numeric
  obj$fitted.values <- as.numeric( obj$data.scaled$fitted.values * obj$y.scale )
  obj$residuals <- obj$y - obj$fitted.values
  
  obj <- obj[c("x.eval","tf.estimate","validation.method","V",
               "validation.error.type","gammas","gamma.min","gamma.1se",
               "gamma.choice","errors","se.errors","edfs","edf.min","edf.1se",
               "i.min","i.1se","x","y","weights","fitted.values", "residuals",
               "k","thinning","optimization.params","n.iter","x.scale","y.scale",
               "data.scaled")]
  return(obj)
}

trendfilter.validate <- function(validation.index, data.folded, obj){
  
  data.train <- data.folded[-validation.index] %>% bind_rows
  data.validate <- data.folded[[validation.index]]
  
  out <- trendfilter(x = data.train$x,
                     y = data.train$y,
                     weights = data.train$weights,
                     k = obj$k,
                     lambda = obj$gammas,
                     thinning = obj$thinning,
                     control = obj$optimization.params)
  
  tf.validate.preds <- glmgen:::predict.trendfilter(out, lambda = obj$gammas, x.new = data.validate$x) %>% 
    suppressWarnings
  
  if ( obj$validation.error.type == "MSE" ){
    validation.error.mat <- (tf.validate.preds - data.validate$y) ^ 2
  }
  if ( obj$validation.error.type == "MAE" ){
    validation.error.mat <- abs(tf.validate.preds - data.validate$y)
  }
  if ( obj$validation.error.type == "WMSE" ){
    validation.error.mat <- (tf.validate.preds - data.validate$y) ^ 2 * 
      (data.validate$weights) / sum((data.validate$weights))
  }
  if ( obj$validation.error.type == "WMAE" ){
    validation.error.mat <- abs(tf.validate.preds - data.validate$y) * 
      sqrt(data.validate$weights) / sum(sqrt(data.validate$weights))
  }
  
  colMeans(validation.error.mat) %>% as.numeric
}
