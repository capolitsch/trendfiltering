% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\docType{data}
\name{eclipsing_binary}
\alias{eclipsing_binary}
\title{Phase-folded light curve of an eclipsing binary star system}
\format{
A tibble with column set \code{c("phase", "flux", "std_err")}.
}
\source{
\url{http://keplerebs.villanova.edu/}
}
\usage{
eclipsing_binary
}
\description{
A time series of a binary star system's apparent brightness, collected by
NASA's \href{https://www.nasa.gov/mission_pages/kepler/overview/index.html}{\emph{Kepler} mission}. These
"eclipsing binary" stars orbit one other with a period of ~1.56
Earth days and eclipse one another from the perspective of Earthly observers.
The \emph{Kepler} identification number for this eclipsing binary system is KIC
6048106.
}
\keyword{datasets}
