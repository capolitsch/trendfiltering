# trendfiltering R package

## Installation
```
devtools::install_github("capolitsch/trendfilteringSupp")
```

## Key references:

Politsch et al. (2020a). Trend Filtering – I. A modern statistical tool for time-domain astronomy and Astronomical Spectroscopy. 
*Monthly Notices of the Royal Astronomical Society*, 492(3), p. 4005-4018. [[Link](https://academic.oup.com/mnras/article/492/3/4005/5704413)]

Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical signals with varying degrees of smoothness. 
*Monthly Notices of the Royal Astronomical Society*, 492(3), p. 4019-4032. [[Link](https://academic.oup.com/mnras/article/492/3/4019/5704414)]

Tibshirani (2014). Adaptive piecewise polynomial estimation via trend filtering. 
*The Annals of Statistics*, 42(1), p. 285-323. [[Link](https://projecteuclid.org/euclid.aos/1395234979)]

Ramdas & Tibshirani (2016). Fast and Flexible ADMM Algorithms for Trend Filtering.
*Journal of Computational and Graphical Statistics*, 25(3), p. 839-858.
[[Link](https://amstat.tandfonline.com/doi/abs/10.1080/10618600.2015.1054033#.XfJpNpNKju0)]

Arnold, Sadhanala, and Tibshirani (2014). Fast algorithms for generalized lasso problems. *glmgen* R package,
version 0.0.3. [[Link](https://github.com/glmgen/glmgen)]


## Quasar Lyman-alpha forest example 

A quasar is an extremely luminous galaxy with an active supermassive black 
hole at its center. Absorptions in the spectra of quasars at vast 
cosmological distances from our galaxy reveal the presence of a gaseous 
medium permeating the entirety of intergalactic space -- appropriately 
named the 'intergalactic medium'. These absorptions allow astronomers to 
study the structure of the Universe using the distribution of these 
absorptions in quasar spectra. Particularly important is the 'forest' of 
absorptions that arise from the Lyman-alpha spectral line, which traces 
the presence of electrically neutral hydrogen in the intergalactic medium.

Here, we are interested in denoising the Lyman-alpha forest of a quasar 
spectroscopically measured by the Sloan Digital Sky Survey. SDSS spectra 
are equally spaced in log10 wavelength space, aside from some instances of 
masked pixels.

```{r}
data("quasar_spec")

head(data)
```

```{r}
library(trendfiltering)

SURE.out <- SURE.trendfilter(x = data$log10.wavelength, 
                             y = data$flux, 
                             weights = data$weights)

# Extract the estimated hyperparameter error curve from the `SURE.trendfilter` 
# output and plot it

log.gammas <- log(SURE.out$gammas)
errors <- SURE.out$errors
log.gamma.min <- log(SURE.out$gamma.min)

plot(x = log.gammas, y = errors, main = "SURE error curve", 
     xlab = "log(gamma)", ylab = "SURE error")
abline(v = log.gamma.min, lty = 2, col = "blue3")
text(x = log.gamma.min, y = par("usr")[4], 
     labels = "optimal gamma", pos = 1, col = "blue3")
```

```{r}
# Extract the optimized trend filtering estimate from the `SURE.trendfilter` 
# output, and transform the input grid to wavelength space (in Angstroms).

wavelength <- 10 ^ (SURE.out$x)
wavelength.eval <- 10 ^ (SURE.out$x.eval)
tf.estimate <- SURE.out$tf.estimate

# Run a parametric bootstrap on the optimized trend filtering estimator to 
# obtain uncertainty bands

boot.out <- bootstrap.trendfilter(obj = SURE.out, 
                                  bootstrap.algorithm = "parametric")

# Plot the results

transparency <- function(color, trans){

    num2hex <- function(x){
      hex <- unlist(strsplit("0123456789ABCDEF",split=""))
      return(paste(hex[(x-x%%16)/16+1],hex[x%%16+1],sep=""))
    }
    rgb <- rbind(col2rgb(color),trans)
    res <- paste("#",apply(apply(rgb,2,num2hex),2,paste,collapse=""),sep="")
    return(res)

}

plot(x = wavelength, y = SURE.out$y, type = "l", 
     main = "Quasar Lyman-alpha forest", 
     xlab = "Observed wavelength (Angstroms)", ylab = "Flux")
polygon(c(wavelength.eval, rev(wavelength.eval)), 
        c(boot.out$bootstrap.lower.band, 
        rev(boot.out$bootstrap.upper.band)),
        col = transparency("orange", 90), border = NA)
lines(wavelength.eval, boot.out$bootstrap.lower.band, 
      col = "orange", lwd = 0.5)
lines(wavelength.eval, boot.out$bootstrap.upper.band, 
      col = "orange", lwd = 0.5)
lines(wavelength.eval, tf.estimate, col = "orange", lwd = 2.5)
legend(x = "topleft", lwd = c(1,2,8), lty = 1, cex = 0.75,
       col = c("black","orange", transparency("orange", 90)), 
       legend = c("Noisy quasar spectrum",
                  "Trend filtering estimate",
                  "95% variability band"))
```
