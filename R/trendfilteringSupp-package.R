#' Optimal one-dimensional data analysis with trend filtering
#'
#' @description This package serves as a software supplement to 
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Politsch et al. (2020a)} and \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' Politsch et al. (2020b)}. We provide a variety of statistical tools for 
#' one-dimensional data analyses with trend filtering
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-42/issue-1/Adaptive-piecewise-polynomial-estimation-via-trend-filtering/10.1214/13-AOS1189.full}{
#' (Tibshirani 2014)}. This package contains user-friendly functionality for optimizing a trend 
# filtering estimator by cross validation or Stein's unbiased risk estimate and 
# various bootstrap algorithms for producing variability bands to quantify the 
# uncertainty in the optimized trend filtering estimate.
#' @name trendfilteringSupp-package
#' @docType package
#' @author Collin A. Politsch \cr \cr 
#' \strong{Maintainer}: Collin A. Politsch <collinpolitsch@@gmail.com>
#' @keywords package
#' @references \cr
#' \strong{Main references} 
#' \enumerate{
#' \item{Politsch et al. (2020a). Trend filtering – I. A modern statistical tool
#' for time-domain astronomy and astronomical spectroscopy. \emph{Monthly 
#' Notices of the Royal Astronomical Society}, 492(3), p. 4005-4018. 
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{[Link]}} \cr
#' \item{Politsch et al. (2020b). Trend Filtering – II. Denoising astronomical 
#' signals with varying degrees of smoothness. \emph{Monthly Notices of the 
#' Royal Astronomical Society}, 492(3), p. 4019-4032.
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{[Link]}} \cr \cr
#' }
#' \strong{Trend filtering theory}
#' \enumerate{
#' \item{Tibshirani (2014). Adaptive piecewise polynomial estimation via trend 
#' filtering. \emph{The Annals of Statistics}. 42(1), p. 285-323.
#' \href{https://projecteuclid.org/euclid.aos/1395234979}{[Link]}} \cr \cr
#' }
#' \strong{Trend filtering convex optimization algorithm}
#' \enumerate{
#' \item{Ramdas and Tibshirani (2016). Fast and Flexible ADMM Algorithms 
#' for Trend Filtering. \emph{Journal of Computational and Graphical 
#' Statistics}, 25(3), p. 839-858.
#' \href{https://amstat.tandfonline.com/doi/abs/10.1080/10618600.2015.1054033#.XfJpNpNKju0}{[Link]}} \cr
#' \item{Arnold, Sadhanala, and Tibshirani (2014). Fast algorithms for 
#' generalized lasso problems. R package \emph{glmgen}. Version 0.0.3. 
#' \href{https://github.com/glmgen/glmgen}{[Link]}} 
#' (Software implementation of Ramdas and Tibshirani algorithm) \cr \cr
#' }
#' \strong{Effect degrees of freedom for trend filtering}
#' \enumerate{
#' \item{Tibshirani and Taylor (2012). Degrees of freedom in lasso problems.
#' \emph{The Annals of Statistics}, 40(2), p. 1198-1232.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full}{[Link]}} 
#' } \cr 
#' \strong{Stein's unbiased risk estimate}
#' \enumerate{
#' \item{Tibshirani and Wasserman (2015). Stein’s Unbiased Risk Estimate.
#' \emph{36-702: Statistical Machine Learning course notes} (Carnegie Mellon).
#' \href{http://www.stat.cmu.edu/~larry/=sml/stein.pdf}{[Link]}} \cr
#' \item{Efron (2014). The Estimation of Prediction Error: Covariance Penalties 
#' and Cross-Validation. \emph{Journal of the American Statistical Association}.
#' 99(467), p. 619-632.
#' \href{https://www.tandfonline.com/doi/abs/10.1198/016214504000000692}{[Link]}} \cr
#' \item{Stein (1981). Estimation of the Mean of a Multivariate Normal 
#' Distribution. \emph{The Annals of Statistics}. 9(6), p. 1135-1151.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-6/Estimation-of-the-Mean-of-a-Multivariate-Normal-Distribution/10.1214/aos/1176345632.full}{[Link]}} \cr
#' } \cr 
#' \strong{The Bootstrap and variations}
#' \enumerate{
#' \item{Efron and Tibshirani (1986). Bootstrap Methods for Standard Errors, 
#' Confidence Intervals, and Other Measures of Statistical Accuracy. Statistical
#' Science, 1(1), p. 54-75.
#' \href{https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full}{[Link]}} \cr
#' \item{Mammen (1993). Bootstrap and Wild Bootstrap for High Dimensional 
#' Linear Models. \emph{The Annals of Statistics}, 21(1), p. 255-285.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-21/issue-1/Bootstrap-and-Wild-Bootstrap-for-High-Dimensional-Linear-Models/10.1214/aos/1176349025.full}{[Link]}} \cr
#' \item{Efron (1979). Bootstrap Methods: Another Look at the Jackknife.
#' \emph{The Annals of Statistics}, 7(1), p. 1-26.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full}{[Link]}} \cr
#' } \cr
#' \strong{Cross validation}
#' \enumerate{
#' \item Hastie, Tibshirani, and Friedman (2009). The Elements of Statistical 
#' Learning: Data Mining, Inference, and Prediction. 2nd edition. Springer 
#' Series in Statistics. 
#' \href{https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf}{
#' [Online print #12]}. (See Sections 7.10 and 7.12) \cr
#' \item James, Witten, Hastie, and Tibshirani (2013). An Introduction to 
#' Statistical Learning : with Applications in R. Springer.
#' \href{https://www.statlearning.com/}{[Most recent online print]} (See 
#' Section 5.1). \emph{Less technical than the above reference.}\cr
#' \item Tibshirani (2013). Model selection and validation 2: Model
#' assessment, more cross-validation. \emph{36-462: Data Mining course notes} 
#' (Carnegie Mellon).
#' \href{https://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf}{[Link]}
#' }
NULL

