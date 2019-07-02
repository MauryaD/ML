dataset =  read.csv("Data.csv")

dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN=function(x) mean(x, rm.na=TRUE)),
                     dataset$Age) 

dataset$Country = factor(dataset$Country, levels = c('France', 'Spain','Germany'),
                         labels = c(1,2,3))


dataset$Purchased = factor(dataset$Purchased, levels = c('No', 'Yes'),
                         labels = c(0,1))