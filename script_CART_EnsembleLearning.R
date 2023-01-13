# czyszczenie Global Environment

rm(list=ls()) #usuniecie obiektow ze srodowiska globalnego
gc(full = TRUE) # garbage collector
.rs.restartR() # restart R

# ustawienie ziarna losowania

set.seed(1)

# wczytanie bibliotek
require(party)
require(rpart)
require(randomForest)
require(RColorBrewer)
require(fastAdaboost)
require(xgboost)

# funkcje zdefiniowane przez uzytkownika

BaggingModelrPart <- function(trainingSet, testingSet, sampleCount) {
  rPartModels <- list()
  rPartPredictClass <- c()
  for(i in 1:sampleCount) {
    bootstrapIndices <- sample(dim(trainingSet)[1],replace=T)
    bootstrapSample <- trainingSet[bootstrapIndices, ]
    bootstrapSampleModel <- rpart(class ~ ., bootstrapSample ,cp = 0.0000001, minsplit = 2, maxsurrogate = 5)
    optComplexity <- bootstrapSampleModel$cptable[which.min(bootstrapSampleModel$cptable[, "xerror"]), "CP"]
    rPartModels[[i]] <- prune(bootstrapSampleModel, cp = optComplexity)
    rPartPredictClass <- cbind(rPartPredictClass, as.numeric(predict(rPartModels[[i]], new = testingSet, type = "class"))-1)
    }
  colnames(rPartPredictClass) <- paste("B", 1:sampleCount, sep = "")
  return(list("Models" = rPartModels, "Class" = as.data.frame(rPartPredictClass)))
}

# wczytanie danych

zrodloWWW <- "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
dataSet <- read.csv(zrodloWWW, header = FALSE)
names(dataSet) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")

# struktura danych i obrobka zbioru

dim(dataSet)
head(dataSet)
summary(dataSet)
dataSet$class <- factor(ifelse(dataSet$class == "unacc", 0, 1))
is.factor(dataSet$doors)
table(dataSet$class)

# podzial zbioru na uczacy i testujacy

trainingFraction <- 0.2
trainingIndex <- (runif(nrow(dataSet)) < trainingFraction)
table(trainingIndex)
trainSet <- dataSet[trainingIndex, ]
testSet <- dataSet[!trainingIndex, ]
nrow(trainSet)
nrow(testSet)

# model cTree

model_cTree <- ctree(factor(class) ~ ., data = trainSet, controls = ctree_control(mincriterion = 0.99, minsplit = 20))
plot(model_cTree, tnex=2 ,type = "extended")
devAskNewPage(ask = TRUE)

# model rPart

model_rPart <- rpart(class ~ ., trainSet ,cp = 0.0000001, minsplit = 2, maxsurrogate = 5)
plotcp(model_rPart)
minError <- which.min(model_rPart$cptable[, "xerror"])
minError
optComplexity <- model_rPart$cptable[minError, "CP"]
optComplexity
points(minError, model_rPart$cptable[minError, "xerror"], col = "red", pch = 19)
model_rPart_opt <- prune(model_rPart, cp = optComplexity)
plot(model_rPart_opt, compress = TRUE, uniform = TRUE, margin = 0.1, branch = 0.3, nspace = 2)
text(model_rPart_opt, use.n = TRUE, pretty = 0)

# model Random Forest

model_RandomForest <- randomForest(class ~ ., data = trainSet, ntree = 300, do.trace = TRUE)
par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))
plot(model_RandomForest, col = "black")
varImpPlot(model_RandomForest , bg = 11)
plot(margin(model_RandomForest, sort = TRUE), ylim = c(-1, 1), ylab = "margin")
abline(h = 0, lty = 2)
devAskNewPage(ask = FALSE)

# model rPart with Bagging

model_rPartBagging <- BaggingModelrPart(trainSet, testSet, 1000)
par(mfrow = c(2, 2))
for(i in 1:4) {
  plot(model_rPartBagging$Models[[i]], compress = TRUE, uniform = TRUE, margin = 0.1, branch = 0.3, nspace = 2)
  text(model_rPartBagging$Models[[i]], use.n = FALSE, pretty = 0)
}
par(mfrow = c(1, 1))

# model AdaBoost

model_AdaBoost <- adaboost(class ~ ., data = trainSet, nIter = 100)
names(model_AdaBoost)
model_AdaBoost$trees # poszczegole drzewa
model_AdaBoost$weights # wagi dla poszczegolnych drzew

# model XGBoost

model_XGBoost <- xgboost(data = data.matrix(trainSet[, -7]), label = data.matrix(trainSet[, 7]), nrounds = 100, objective = "binary:logistic")
names(model_XGBoost)
model_XGBoost$evaluation_log
plot(model_XGBoost$evaluation_log, type = 'l')

# porownanie modeli wg trafnosci i macierzy pomylek na zbiorze testowym

confusion.matrix <- list()

cat("Macierz trafnosci cTree")
print(confusion.matrix[[1]] <- table(predict(model_cTree, new = testSet), testSet$class))
cat("\nMacierz trafnosci rPart\n")
print(confusion.matrix[[2]] <- table(predict(model_rPart_opt, type = "class", newdata = testSet), testSet$class))
cat("\nMacierz trafnosci las losowy\n")
print(confusion.matrix[[3]] <- table(predict(model_RandomForest, newdata = testSet), testSet$class))
cat("\nMacierz trafnosci rPart Bagging\n")
print(confusion.matrix[[4]] <- table(ifelse(margin.table(as.matrix(model_rPartBagging$Class), 1) / 1000 < 0.5, 0, 1), testSet$class))
cat("\nMacierz trafnosci AdaBoost\n")
print(confusion.matrix[[5]] <- table(predict(model_AdaBoost, newdata = testSet)$class, testSet$class))
cat("\nMacierz trafnosci XGBoost\n")
print(confusion.matrix[[6]] <- table(ifelse(predict(model_XGBoost, newdata = data.matrix(testSet[, -7])) < 0.5, 0, 1), testSet$class))

cat("\nPorównanie dokladnooci modeli\n")
CalculateAccuracy<-function(confusion.matrix) {return(sum(diag(confusion.matrix))/sum(confusion.matrix))}
print(data.frame(Model=c("cTree","rPart", "las losowy", "rPart Bagging", "AdaBoost", "XGBoost"), Accuracy = round(sapply(confusion.matrix, CalculateAccuracy),3) * 100), row.names = FALSE)

