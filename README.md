# trendfiltering R package <a href="https://capolitsch.github.io/trendfiltering/"><img src="man/figures/logo.svg" align="right" width="120px"/></a>

### [Package website](https://capolitsch.github.io/trendfiltering/)

A suite of tools for denoising one-dimensional data sets with trend filtering. 
Optimizing the hyparameter via Stein's unbiased risk estimate or $V$-fold cross 
validation with a customizable validation loss functional, and 
one-standard-error rules. Relaxed trend filtering, and various bootstrap 
algorithms for uncertainty quantification.

The `trendfiltering` package can be installed from GitHub using the command 
below.

## Installation
``` r
install.packages("remotes")
remotes::install_github("capolitsch/trendfiltering")
```

## References:

1. Politsch et al. (2020a). Trend Filtering – I. A modern statistical tool for 
time-domain astronomy and Astronomical Spectroscopy. 
*Monthly Notices of the Royal Astronomical Society*, 492(3), p. 4005-4018. 
[[Publisher](https://academic.oup.com/mnras/article/492/3/4005/5704413)] 
[[arXiv](https://arxiv.org/abs/1908.07151)]
[[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].

2. Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical signals
with varying degrees of smoothness. 
*Monthly Notices of the Royal Astronomical Society*, 492(3), p. 4019-4032. 
[[Publisher](https://academic.oup.com/mnras/article/492/3/4019/5704414)] 
[[arXiv](https://arxiv.org/abs/2001.03552)]
[[BibTeX](https://capolitsch.github.io/trendfiltering/authors.html)].

3. Tibshirani (2014). Adaptive piecewise polynomial estimation via trend 
filtering. *The Annals of Statistics*, 42(1), p. 285-323. 
[[Publisher](https://projecteuclid.org/euclid.aos/1395234979)] 
[[arXiv](https://arxiv.org/abs/1304.2986)].
