#' Phase-folded light curve of an eclipsing binary star system
#'
#' A binary star system's apparent brightness time series, collected by NASA's
#' [*Kepler* mission](
#' https://www.nasa.gov/mission_pages/kepler/overview/index.html). The
#' gravitationally-bound pair of stars in the system orbit each other with a
#' period of ~1.56 Earth days. The system's *Kepler* identification is KIC
#' 6048106.
#'
#' @format A tibble with column set `c("phase", "flux", "std_err")`.
#' @source <http://keplerebs.villanova.edu/>
"eclipsing_binary"

#' Lyman-alpha forest in the absorption spectrum of a distant quasar
#'
#' @format A tibble with column set `c("log10_wavelength", "flux", "weights")`.
#' @source <http://www.sdss3.org/>
"quasar_spectrum"
