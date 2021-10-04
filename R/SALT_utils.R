#' @importFrom dplyr %>%
#' @importFrom magrittr %$%
#' @noRd
#' @export
trendfilter.interval <- function(x, y, weights, 
                                 B = 100, alpha = 0.05, bootstrap.bands = T,
                                 max_iter = 5000, obj_tol = 1e-12, ...){
  
  SURE.out <- SURE.trendfilter(x, y, weights,
                               optimization.params = list(max_iter = max_iter, 
                                                          obj_tol = obj_tol)
                               )
  if ( !bootstrap.bands ){
    return(SURE.out)
  }else{
    boot.out <- bootstrap.trendfilter(obj = SURE.out, B = B, alpha = alpha, ...)
    return(boot.out)
  }
}


#' @importFrom dplyr %>% filter pull
#' @noRd
#' @export
mask.intervals <- function(df, min.mask.width = 20){
  
  masked <- df %>% filter(mask == 1) %>% pull(wavelength)
  masked.intervals <- list()
  itr <- 1
  diffs <- sapply(X = 2:length(masked), FUN = function(X) masked[X] - masked[X-1])
  while ( max(diffs) >= min.mask.width ){
    ind <- min(which(diffs >= min.mask.width))
    masked.intervals[[itr]] <- masked[1:(ind)]
    masked <- setdiff(masked, masked[1:(ind)])
    itr <- itr + 1
    diffs <- sapply(X = 2:length(masked), FUN = function(X) masked[X] - masked[X-1])
  }
  if ( itr > 1 ){
    inds <- unlist(lapply(X = 1:length(masked.intervals), FUN = function(X) length(masked.intervals[[X]]) > min.mask.width))
    if ( length(masked.intervals[inds]) > 0 ){
      masked.intervals <- lapply(X = 1:length(masked.intervals[inds]), FUN = function(X) masked.intervals[inds][[X]])
    }else{
      masked.intervals <- NULL
    }
  }else{
    masked.intervals <- NULL  
  }
  
  if ( length(masked.intervals) > 0 ){
    inds <- c(df %>% pull(wavelength) %>% min - 1, 
              sapply(X = 1:length(masked.intervals), FUN = function(X) min(masked.intervals[[X]])), 
              df %>% pull(wavelength) %>% max)
  }else{
    inds <- c(df %>% pull(wavelength) %>% min - 1, df %>% pull(wavelength) %>% max)
  }
  
  good.wavelength.intervals <- lapply(X = 1:(length(inds)-1), FUN = function(X) (inds[X] + 1):(inds[X+1]-1))
  
  df$segment <- NA
  for ( itr in 1:length(good.wavelength.intervals) ){
    i <- match(good.wavelength.intervals[[itr]], df$wavelength)
    df$segment[i] <- itr
  }
  
  df %>% filter(mask == 0) %>% select(-mask)
}


#' @noRd
#' @export
transparency <- function(color, trans){
  num2hex <- function(x){
    hex <- unlist(strsplit("0123456789ABCDEF",split=""))
    return(paste(hex[(x-x%%16)/16+1],hex[x%%16+1],sep=""))
  }
  rgb <- rbind(col2rgb(color),trans)
  res <- paste("#",apply(apply(rgb,2,num2hex),2,paste, collapse=""),sep="")
  return(res)
}
