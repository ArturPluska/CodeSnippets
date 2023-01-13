# Transfering from Matrix to List

MatrixOrFrameToList <- function(dataSeries) {
  if(is.matrix(dataSeries) || is.data.frame(dataSeries)) {
    matrixToList <- list()
    for(i in 1:ncol(dataSeries)) {
      matrixToList <- c(matrixToList, list(dataSeries[, i, drop = T]))
    }
    names(matrixToList) <- colnames(dataSeries)
  } else {
    stop('Wrong "dataSeries" input! It should be a matrix.')
  }
return(matrixToList)
}

# Rate of Return

RateReturn <- function(dataSeries, time = 'continuous') {
  if((is.vector(dataSeries) || is.matrix(dataSeries) || is.data.frame(dataSeries))) {
    if((time == 'continuous' || time == 'digital')) {
      if(time == 'continuous') {
        if(is.vector(dataSeries)) {
          changeRate <- log(dataSeries[-1] / dataSeries[-length(dataSeries)])
        } else if(is.matrix(dataSeries)) {
          changeRate <- matrix(NA, nrow = dim(dataSeries)[1] - 1, ncol = dim(dataSeries)[2])
          colnames(changeRate) <- colnames(dataSeries)
          for(i in 1:dim(dataSeries)[2]) {
            for(j in 1:dim(dataSeries)[1] - 1) {
              changeRate[j, i] <- log(dataSeries[j + 1, i]/ dataSeries[j , i])
            }
          }
        } else if(is.data.frame(dataSeries)) {
          changeRate <- matrix(NA, nrow = dim(dataSeries)[1] - 1, ncol = dim(dataSeries)[2])
          colnames(changeRate) <- colnames(dataSeries)
          changeRate <- as.data.frame(changeRate)
          for(i in 1:dim(dataSeries)[2]) {
            for(j in 1:dim(dataSeries)[1] - 1) {
              changeRate[j, i] <- log(dataSeries[j + 1, i]/ dataSeries[j , i])
            }
          }
        }
      } else if(time == 'digital') {
        if(is.vector(dataSeries)) {
          changeRate <- dataSeries[-1] / dataSeries[-length(dataSeries)]
        } else if(is.matrix(dataSeries)) {
          changeRate <- matrix(NA, nrow = dim(dataSeries)[1] - 1, ncol = dim(dataSeries)[2])
          colnames(changeRate) <- colnames(dataSeries)
          for(i in 1:dim(dataSeries)[2]) {
            for(j in 1:dim(dataSeries)[1] - 1) {
              changeRate[j, i] <- dataSeries[j + 1, i]/ dataSeries[j , i]
            }
          }
        } else if(is.data.frame()) {
          changeRate <- matrix(NA, nrow = dim(dataSeries)[1] - 1, ncol = dim(dataSeries)[2])
          colnames(changeRate) <- colnames(dataSeries)
          changeRate <- as.data.frame(changeRate)
          for(i in 1:dim(dataSeries)[2]) {
            for(j in 1:dim(dataSeries)[1] - 1) {
              changeRate[j, i] <- dataSeries[j + 1, i]/ dataSeries[j , i]
            }
          }
        }
      }  
    } else {
      stop('Wrong "time" input! Choose between "continuous" and "digital".')
    }
  } else {
    stop('Wrong "dataSeries" input! It should be as vector or matrix.')
  }
  return(changeRate)
}

# Semi-Variance

SemiVariance <- function(dataSeries, targetReturn = 0, direction = 'upper', method = 'full') {
  semiVariance <- c()
  if(is.vector(dataSeries)) {
    if(is.atomic(targetReturn) && length(targetReturn) == 1L) {
      if(direction == 'upper' || direction == 'lower') {
        if(method == 'full' || method == 'subset') {
          if(method == 'full') {
            if(direction == 'upper') {
              semiVariance <- mean((pmax(dataSeries - targetReturn, 0)) ^ 2)
            } else if(direction == 'lower') {
              semiVariance <- mean((pmax(targetReturn - dataSeries, 0)) ^ 2)
            }
          } else if(method == 'subset') {
            if(direction == 'upper') {
              upsideSeries <- dataSeries[dataSeries > targetReturn]
              semiVariance <- mean((upsideSeries - targetReturn) ^ 2)
            } else if(direction == 'lower') {
              downsideSeries <- dataSeries[dataSeries < targetReturn]
              semiVariance <- mean((targetReturn - downsideSeries) ^ 2)
            }
          }
        } else {
          stop('Wrong "method" input! Choose between "full" and "subset".')
        }
      } else {
        stop('Wrong "direction" input! Choose between "upper" and "lower".')
      }
    } else {
      stop('Wrong "targetReturn" input! It should be a scalar.')
    }
  } else {
    stop('Wrong "dataSeries" input! It should be as vector.')
  }
  return(semiVariance)
}

# Lower Partial Moments

LowerPartialMoments <- function(dataSeries, targetReturn = 0, direction = 'upper', method = 'full', riskAversion = 2) {
  lowerPartialMoments <- c()
  if(is.vector(dataSeries)) {
    if(is.atomic(targetReturn) && length(targetReturn) == 1L) {
      if(direction == 'upper' || direction == 'lower') {
        if(method == 'full' || method == 'subset') {
           if(method == 'full') {
            if(direction == 'upper') {
              n <- length(dataSeries)
              lowerParialMoments <- mean(pmax(dataSeries - targetReturn, 0) ^ riskAversion)
            } else if(direction == 'lower') {
              lowerParialMoments <- mean(pmax(targetReturn - dataSeries, 0) ^ riskAversion)
            }
          } else if(method == 'subset') {
            if(direction == 'upper') {
              upsideSeries <- dataSeries[dataSeries > targetReturn]
              lowerParialMoments <- mean((upsideSeries - targetReturn) ^ riskAversion)
            } else if(direction == 'lower') {
              downsideSeries <- dataSeries[dataSeries < targetReturn]
              lowerParialMoments <- mean((targetReturn - downsideSeries) ^ riskAversion)
            }
          }
        } else {
          stop('Wrong "method" input! Choose between "full" and "subset".')
        }
      } else {
        stop('Wrong "direction" input! Choose between "upper" and "lower".')
      }
    } else {
      stop('Wrong "targetReturn" input! It should be a scalar.')
    }
  } else {
    stop('Wrong "dataSeries" input! It should be as vector.')
  } 
  return(lowerParialMoments)
}

# Value-at-Risk

ValueAtRisk <- function(returnPortfolio, valuePortfolio = 1, prob = 0.95, horizon = 1, method = 'historical') {
  valueAtRisk <- c()
  if(is.vector(returnPortfolio)) {
    if(is.atomic(valuePortfolio) && length(valuePortfolio) == 1) {
      if(is.atomic(prob) && length(prob) == 1L && any(prob > 0 || prob < 1)) {
        if(is.atomic(horizon) && length(horizon) == 1) {
          if(method == 'historical' || method == 'gaussian') {
            if(method == 'historical') {
              valueAtRisk <- quantile(returnPortfolio, 1 - prob) * valuePortfolio
            } else if(method == 'gaussian') {
              meanReturn <- mean(returnPortfolio)
              sdReturn <- sd(returnPortfolio)
              tailQnorm <- qnorm(1 - prob, 0, 1)
              valueAtRisk <- -meanReturn + tailQnorm * sdReturn * sqrt(horizon) * valuePortfolio
            }  
          } else {
            stop('Wrong "method" input! Choose between "historical" and "gaussian".')
          }
        } else {
          stop('Wrong "horizon" input! It should be a scalar.')
        }
      } else {
        stop('Wrong "prob" input! It should be a scalar between 0 and 1.')
      }
    } else {
    stop('Wrong "valuePortfolio" input! It should be a scalar.')
    }
  } else {
    stop('Wrong "returnPortfolio" input! It should be a vector.')
  }
  return(valueAtRisk)
}

# Expected Shortfall (Conditional Value-at-Risk)

ExpectedShortfall <- function(returnPortfolio, valuePortfolio = 1, prob = 0.95, horizon = 1, method = 'historical') {
  expectedShortfall <- c()
  if(is.vector(returnPortfolio)) {
    if(is.atomic(valuePortfolio) && length(valuePortfolio) == 1) {
      if(is.atomic(prob) && length(prob) == 1 && and(prob <= 0 || prob >= 1)) {
        if(is.atomic(horizon) && length(horizon) == 1) {
          if(method == 'historical' || method == 'gaussian') {
            if(method == 'historical') {
              quantileVaR <- quantile(returnPortfolio, 1 - prob)
              expectedShortfall <- mean(returnPortfolio[returnPortfolio <= quantileVaR]) * valuePortfolio
            } else if(method == 'gaussian') {
              meanReturn <- mean(returnPortfolio)
              sdReturn <- sd(returnPortfolio)
              tailQnorm <- qnorm(1 - prob, meanReturn, sdReturn)
              tailExp <- integrate(function(x) {x * dnorm(x, meanReturn, sdReturn)}, -Inf,  tailQnorm)$value / (1 - prob)
              expectedShortfall <- tailExp * valuePortfolio
            }
          } else {
            stop('Wrong "method" input! Choose between "historical" and "gaussian".')
          }
        } else {
          stop('Wrong "horizon" input! It should be a scalar.')
        }
      } else {
        stop('Wrong "prob" input! It should be a scalar and between 0 and 1.')
      }
    } else {
      stop('Wrong "valuePortfolio" input! It should be a scalar.')
    }
  } else {
    stop('Wrong "returnPortfolio" input! It should be a vector.')
  }
  return(expectedShortfall)
}

# Monte Carlo Simulation

MonteCarloSimulationPortfolio <- function(funtionReturn, returnFXSeries, rFree, simulationNumber, steps, horison) {
  monteCarloScenarios <- matrix(NA, nrow = steps, ncol = simulationNumber)
  for(i in 1:simulationNumber) {
    monteCarloScenarios[, i] <- unlist(lapply(runif(steps), funtionReturn, returnFXSeries, rFree, steps , horison))
  }
  return(monteCarloScenarios)
}
