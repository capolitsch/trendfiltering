#' Phase-folded light curve of an eclipsing binary star system
#'
#' A time series of a binary star system's apparent brightness, collected by
#' NASA's [*Kepler* mission](
#' https://www.nasa.gov/mission_pages/kepler/overview/index.html). These
#' "eclipsing binary" stars orbit one other with a period of ~1.56
#' Earth days and eclipse one another from the perspective of Earthly observers.
#' The *Kepler* identification number for this eclipsing binary system is KIC
#' 6048106.
#'
#' @format A tibble with column set `c("phase", "flux", "std_err")`.
#' @source <http://keplerebs.villanova.edu/>
"eclipsing_binary"

#' Spectroscopic measurements of a distant quasar
#'
#' Part of the electromagnetic spectrum of an extremely luminous galaxy. This
#' special type of galaxy -- a quasar -- is the most luminous object class in
#' the Universe. The specific interval of its spectrum that is studied here
#' traces the structure of diffuse gas in intergalactic space, along the quasar
#' light's one-dimensional path to Earth. The intergalactic gas that the light
#' passes through is primarily composed of neutrally-charged hydrogen, which
#' creates a "forest" of absorptions in this specific interval of the quasar's
#' spectrum.
#'
#' @format A tibble with column set `c("log10_wavelength", "flux", "weights")`.
#' @source <http://www.sdss3.org/>
"quasar_spectrum"
