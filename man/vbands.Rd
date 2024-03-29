% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/uncertainty_quantification.R
\name{vbands}
\alias{vbands}
\title{Compute bootstrap variability bands}
\usage{
vbands(obj, level = 0.95)
}
\arguments{
\item{obj}{A "\code{\link[=bootstrap_trendfilter]{bootstrap_trendfilter}}" object.}

\item{level}{The level of the pointwise variability bands. Defaults to \code{level = 0.95}.}
}
\value{
A tibble with column set \code{c("x","lower_band","upper_band")}.
}
\description{
Compute variability bands for the optimized trend filtering estimate via the
sample quantiles of the bootstrap ensemble generated by
\code{\link[=bootstrap_trendfilter]{bootstrap_trendfilter()}}.
}
\examples{
# Example 1: Phase-folded light curve of an eclipsing binary star system
#
# The apparent brightness over time of a star system that has two suns
# that regularly eclipse one another from our vantage point on Earth. Here,
# the time series is stacked according to the orbital period of the binary
# system, with the primary eclipse occuring at `phase = 0` and the input
# domain ranging from -0.5 to 0.5.

data("eclipsing_binary")
head(eclipsing_binary)

x <- eclipsing_binary$phase
y <- eclipsing_binary$flux
weights <- 1 / eclipsing_binary$std_err^2

cv_tf <- cv_trendfilter(
  x = x,
  y = y,
  weights = weights,
  max_iter = 1e4,
  obj_tol = 1e-6
)

boot_tf <- bootstrap_trendfilter(
  obj = cv_tf,
  algorithm = "nonparametric",
  edf = cv_tf$edf_min["MAE"]
)
bands <- vbands(boot_tf)


# Example 2: The "Lyman-alpha forest" in the spectrum of a distant quasar

data("quasar_spectrum")
head(quasar_spectrum)

x <- quasar_spectrum$log10_wavelength
y <- quasar_spectrum$flux
weights <- quasar_spectrum$weights

sure_tf <- sure_trendfilter(x, y, weights)
boot_tf <- bootstrap_trendfilter(
  obj = sure_tf,
  algorithm = "parametric",
  edf = sure_tf$edf_min
)
bands <- vbands(boot_tf)
}
\references{
\enumerate{
\item Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
for time-domain astronomy and astronomical spectroscopy. \emph{MNRAS}, 492(3),
p. 4005-4018.
[\href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{Publisher}]
[\href{https://arxiv.org/abs/1908.07151}{arXiv}].
\item Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical
signals with varying degrees of smoothness. \emph{MNRAS}, 492(3), p. 4019-4032.
[\href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{Publisher}]
[\href{https://arxiv.org/abs/2001.03552}{arXiv}].
}
}
\seealso{
\code{\link[=bootstrap_trendfilter]{bootstrap_trendfilter()}}
}
