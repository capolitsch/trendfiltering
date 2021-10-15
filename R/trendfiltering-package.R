#' @docType package
#' @name trendfiltering-package
#'
#' @references \cr
#' \bold{Closely related to this R package}
#' \enumerate{
#' \item{Politsch et al. (2020a).
#' \href{https://academic.oup.com/mnras/article/492/3/4005/5704413}{
#' Trend filtering – I. A modern statistical tool for time-domain astronomy and
#' astronomical spectroscopy}. \emph{MNRAS}, 492(3), p. 4005-4018.} \cr
#' \item{Politsch et al. (2020b).
#' \href{https://academic.oup.com/mnras/article/492/3/4019/5704414}{
#' Trend Filtering – II. Denoising astronomical signals with varying degrees of
#' smoothness}. \emph{MNRAS}, 492(3), p. 4019-4032.}}
#'
#' \bold{Trend filtering theory}
#' \enumerate{
#' \item{Tibshirani (2014).
#' \href{https://projecteuclid.org/euclid.aos/1395234979}{Adaptive piecewise
#' polynomial estimation via trend filtering}. \emph{The Annals of Statistics}.
#' 42(1), p. 285-323.} \cr
#' \item{Tibshirani (2020). \href{https://arxiv.org/abs/2003.03886}{Divided
#' Differences, Falling Factorials, and Discrete Splines: Another Look at Trend
#' Filtering and Related Problems}. arXiv: 2003.03886.}}
#'
#' \bold{Trend filtering convex optimization algorithm}
#' \enumerate{
#' \item{Ramdas and Tibshirani (2016).
#' \href{https://amstat.tandfonline.com/doi/abs/10.1080/10618600.2015.1054033#.XfJpNpNKju0}{
#' Fast and Flexible ADMM Algorithms for Trend Filtering}. \emph{Journal of
#' Computational and Graphical Statistics}, 25(3), p. 839-858.} \cr
#' \item{Arnold, Sadhanala, and Tibshirani (2014).
#' \href{https://github.com/glmgen/glmgen}{Fast algorithms for generalized
#' lasso problems}. R package \emph{glmgen}. Version 0.0.3.}}
#'
#' \bold{Effective degrees of freedom for trend filtering}
#' \enumerate{
#' \item{Tibshirani and Taylor (2012)}.
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Degrees-of-freedom-in-lasso-problems/10.1214/12-AOS1003.full}{
#' Degrees of freedom in lasso problems}. \emph{The Annals of Statistics},
#' 40(2), p. 1198-1232.}
#'
#' \bold{Stein's unbiased risk estimate}
#' \enumerate{
#' \item{Tibshirani and Wasserman (2015).
#' \href{http://www.stat.cmu.edu/~larry/=sml/stein.pdf}{Stein’s Unbiased Risk
#' Estimate}. \emph{36-702: Statistical Machine Learning course notes}
#' (Carnegie Mellon University).} \cr
#' \item{Efron (2014).
#' \href{https://www.tandfonline.com/doi/abs/10.1198/016214504000000692}{
#' The Estimation of Prediction Error: Covariance Penalties
#' and Cross-Validation}. \emph{Journal of the American Statistical
#' Association}. 99(467), p. 619-632.} \cr
#' \item{Stein (1981).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-6/Estimation-of-the-Mean-of-a-Multivariate-Normal-Distribution/10.1214/aos/1176345632.full}{
#' Estimation of the Mean of a Multivariate Normal Distribution}.
#' \emph{The Annals of Statistics}. 9(6), p. 1135-1151.}}
#'
#' \bold{The Bootstrap and variations}
#' \enumerate{
#' \item{Efron and Tibshirani (1986).
#' \href{https://projecteuclid.org/journals/statistical-science/volume-1/issue-1/Bootstrap-Methods-for-Standard-Errors-Confidence-Intervals-and-Other-Measures/10.1214/ss/1177013815.full}{
#' Bootstrap Methods for Standard Errors, Confidence Intervals, and Other
#' Measures of Statistical Accuracy}.
#' \emph{Statistical Science}, 1(1), p. 54-75.} \cr
#' \item{Mammen (1993).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-21/issue-1/Bootstrap-and-Wild-Bootstrap-for-High-Dimensional-Linear-Models/10.1214/aos/1176349025.full}{
#' Bootstrap and Wild Bootstrap for High Dimensional Linear Models}. \emph{The
#' Annals of Statistics}, 21(1), p. 255-285.} \cr
#' \item{Efron (1979).
#' \href{https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full}{
#' Bootstrap Methods: Another Look at the Jackknife}.
#' \emph{The Annals of Statistics}, 7(1), p. 1-26.}}
#'
#' \bold{Cross validation}
#' \enumerate{
#' \item{Hastie, Tibshirani, and Friedman (2009).
#' \href{https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf}{
#' The Elements of Statistical Learning: Data Mining, Inference, and
#' Prediction}. 2nd edition. Springer Series in Statistics. (See Sections 7.10
#' and 7.12)} \cr
#' \item{Tibshirani (2013).
#' \href{https://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf}{
#' Model selection and validation 2: Model assessment, more cross-validation}.
#' \emph{36-462: Data Mining course notes} (Carnegie Mellon University).}}
