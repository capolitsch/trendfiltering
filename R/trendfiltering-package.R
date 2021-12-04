#' @aliases trendfiltering
#'
#' @references:
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
#' @section Trend filtering with Stein's unbiased risk estimate:
#'
#'   \loadmathjax Here we describe the general motivation for optimizing a trend
#'   filtering estimator with respect to Stein's unbiased risk estimate. See
#'   [Politsch et al. (2020a)](
#'   https://academic.oup.com/mnras/article/492/3/4005/5704413) for more
#'   details.
#'
#'   Suppose we observe noisy measurements of an output variable of interest
#'   (e.g. flux, magnitude, photon counts) according to the data generating
#'   process
#'   \mjsdeqn{y_i = f(x_i) + \epsilon_i, \quad\quad x_1,\dots,x_n\in(a,b),}
#'   where \mjseqn{y_i} is a noisy observation of a signal \mjseqn{f(x_i)} and
#'   the \mjseqn{\epsilon_i} have mean zero with variance
#'   \mjseqn{\sigma_{i}^{2} = Var(\epsilon_{i})}. Let
#'   \mjseqn{\hat{f}(\cdot\;; \lambda)} denote the trend filtering estimator of
#'   order \mjseqn{k} with tuneable hyperparameter \mjseqn{\lambda}. The
#'   fixed-input mean-squared prediction error (MSPE) of the estimator
#'   \mjseqn{\hat{f}} is defined as
#'   \mjsdeqn{R(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\;E\left\[\left(y_i -
#'   \hat{f}(x_{i};\lambda)\right)^2\;|\;x_{1},\dots,x_{n}\right\]}
#'   \mjsdeqn{= \frac{1}{n}\sum_{i=1}^{n}\left(E\left\[\left(f(x_i) -
#'   \hat{f}(x_i;\lambda)\right)^2\;|\;x_1,\dots,x_n\right\] +
#'   \sigma_i^2\right).}
#'
#'   Stein's unbiased risk estimate (SURE) provides an unbiased estimate of the
#'   fixed-input MSPE via the following formula:
#'   \mjsdeqn{\hat{R}(\lambda) =
#'   \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{f}(x_i; \lambda)\big)^2 +
#'   \frac{2\overline{\sigma}^{2}df(\hat{f})}{n},} where
#'   \mjseqn{\overline{\sigma}^{2} = n^{-1}\sum_{i=1}^{n} \sigma_i^2} and
#'   \mjseqn{df(\hat{f})} is the effective degrees of freedom of the trend
#'   filtering estimator (with a fixed choice of hyperparameter). The
#'   generalized lasso results of [Tibshirani and Taylor (2012)](
#'   https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full)
#'   provide the following formula for the effective degrees of freedom of a
#'   trend filtering estimator (with a fixed hyperparameter choice):
#'   \mjsdeqn{df(\hat{f}) = E\left\[K(\hat{f})\right\] + k + 1},
#'   where \mjseqn{K(\hat{f})} is the number of knots in \mjseqn{\hat{f}}. The
#'   optimal hyperparameter value is then defined as
#'   \mjsdeqn{\hat{\lambda} = \arg\min_{\lambda}
#'   \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{f}(x_i; \lambda)\big)^2 +
#'   \frac{2\hat{\overline{\sigma}}^{2}\hat{df}(\hat{f})}{n},}
#'   where \mjseqn{\hat{df}} is the estimate for the effective degrees of
#'   freedom that is obtained by replacing the expectation with the observed
#'   number of knots, and \mjseqn{\hat{\overline{\sigma}}^2} is an estimate of
#'   \mjseqn{\overline{\sigma}^2}. We define \mjseqn{\overline{\sigma}^2} as
#'   `mean(1 / weights)`, so `weights` must be passed in order to use
#'   `sure.trendfilter()`. If a reliable estimate of
#'   \mjseqn{\overline{\sigma}^2} is not available a priori, a data-driven
#'   estimate can be constructed, e.g. see [Wasserman (2004)](
#'   https://link.springer.com/book/10.1007/978-0-387-21736-9) or
#'   [Hastie, Tibshirani, Friedman (2009)](
#'   https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf).
NULL
