system.time({
rm(list=ls())
library(rpart)
library(nnet)
library(caTools)#logicbag
#library(RSNNS)#mlp
library(e1071)
library(randomForest)
library(ROCR)
library(foreach)
library(doSNOW)
library(glmnet)
set.seed(111)

cluster = makeCluster(4, type = "SOCK")
registerDoSNOW(cluster)


auc<- function(outcome, proba){
 N = length(proba)
 N_pos = sum(outcome)
 df = data.frame(out = outcome, prob = proba)
 df = df[order(-df$prob),]
 df$above = (1:N) - cumsum(df$out)
 return( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
}



TomekLinks<-function(trn,yi,label){
 #yi the index of y
 #trn the set to be processed
 #label of majority examples
 library(FNN)
 trn=na.omit(trn)
 my=knn.index(trn[,-c(yi)],k=1) 
 
 len=nrow(trn)
 tlindex<-NULL
 
   for (i in 1:len){
 
    # real index in trn  
   if (trn[rownames(trn)[i],yi]!=trn[rownames(trn)[my[i]],yi]){
     if (trn[rownames(trn)[i],yi]==label)
	     tlindex=cbind(rownames(trn)[i],tlindex)
	 else
	     tlindex=cbind(rownames(trn)[my[i]],tlindex)
		}
	}
return (trn[-c(as.numeric(tlindex)),])
}


#setwd("/var/www/") #for linux
setwd("e:/creditcard")
# data with 80 predictors
mydata <- read.csv("cs-training.csv")


testresult <-data.frame(row.names=1:50)

auc_le<-c(rep(0,50))
auc_rf<-c(rep(0,50))
auc_glm<-c(rep(0,50))
auc_rtree<-c(rep(0,50))
auc_logicbag<-c(rep(0,50))
auc_mlp<-c(rep(0,50))
auc_c50<-c(rep(0,50))
auc_svm<-c(rep(0,50))


mydata <-na.omit(mydata)
# spit the data into training mytrset and test teset
#for ( kkk in 1:50) {

L=sample(1:nrow(mydata),ceiling(nrow(mydata)*0.8))
trset<-mydata[L,]
teset<-mydata[-L,]
rm(mydata)
obse<-teset[,"SeriousDlqin2yrs"]

#mytrset<-TomekLinks(mytrset,1,0)

#trset<-xform_data(trset)
#teset<-xform_data(teset)

#postive trset
ptrset=trset[trset[,1]==1,]
#negative trset
ntrset=trset[trset[,1]==0,]

#number of k clusters in the negative trset
k=14
myk=kmeans(ntrset[,-c(1)],k)
mycluster=order(myk$cluster)

#partition into k negative sets
ntrsetlist <- vector(mode = "list", length = k)
j=1
jj=0
for (m in 1:k){
    
   ntrsettemp<-NULL  
   ntrsettemp=rep(myk$size[m],0) 
   for ( kk in 1: myk$size[m])
      {ntrsettemp[kk]=names(myk$cluster[mycluster[j]])
      j=j+1	  
	  }
	  
if (myk$size[m]>quantile(myk$size)[2])	 
  {
  jj=jj+1
  ntrsetlist[[jj]]=ntrsettemp  
  }
}


#actual training set: cycle through one cluster of the negative trset and the whole ptrset
#actual OOB data: trset-the actual training set

length_divisor=10



varimportance=c(rep(0,81))

varimpor2=c(rep(0,81))

trlen=jj
oobacc<-NULL
oobac<-NULL
varim1<-NULL

trees <- vector(mode = "list", length = trlen)
lambdas<- vector(mode = "list", length = trlen)
# build the classfier of trees
foreach(i = 1:trlen) %do%{
    varimport1=c(rep(0,80))
    #training_positions <- sample(nrow(trset), size=floor((nrow(trset)/length_divisor)))

	tesetn=trset[ntrsetlist[[i]],]
	
	trsetn=trset[-as.integer(ntrsetlist[[i]]),]
	
	training_positionsp <- sample(nrow(ptrset), size=nrow(ptrset)/3)
	train_posp<-1:nrow(ptrset) %in% training_positionsp
	#positive bag examples
	trsetp=ptrset[!train_posp,]
	#positive oob
	tesetp=ptrset[train_posp,]
	
	#negative bag examples
	#random choose bag examples from one or more clusters
	#uneven distribution in cluster problem
	
	trsetnew=rbind(trsetn,trsetp)
	
	tesetnew=rbind(tesetn,tesetp)
 
    #
    var_pos<-sample(3:ncol(trsetnew)-1,size=ncol(trsetnew)-floor(sqrt(ncol(trsetnew))))
 
    # var_pos<-1:ncol(trset) %in% var_positions
	# var_pos[1]=FALSE
   # vaiable selection for glmnet
 
	#trees[[i]]=glmnet(as.matrix(trset[!train_pos,-c(1)]),as.matrix(trset[!train_pos,1]),family="binomial",exclude=var_pos,alpha=1)
	trees[[i]]=glmnet(as.matrix(trsetnew[,-c(1)]),as.matrix(trsetnew[,1]),family="binomial",alpha=1)
	#outofbag error

	
	oobpredicts<- predict(trees[[i]],as.matrix(tesetnew[,-c(1)]),type="response")
	oobobse<-tesetnew[,"SeriousDlqin2yrs"]
	oobacc<-NULL
  for (lambda in 1:length(trees[[i]]$lambda))
     {
	 oob=auc(oobobse,as.vector(oobpredicts[,lambda]))
     
    # # print(oob)
	oobacc=rbind(oobacc,oob) 
	}
	#best OOB classifier's lambda is chosen
	lambdas[[i]]=trees[[i]]$lambda[which.max(oobacc)]
	
	#record varialbe occurrence
	vari2=coef(trees[[i]], s=lambdas[[i]])
	print(vari2)
	
	varimportance[which (abs(vari2) >1e-04)]=varimportance[which (abs(vari2) >1e-04)]+1
	
	varimpor2=varimpor2+abs(vari2)
	obbac1=max(oobacc)
	oobac=rbind(oobac,obbac1) 
	
	# CAL AUC DECREASE

	#test first variable
	X=cbind(sample(tesetnew[,2]),tesetnew[,3:81])
	oobpredicts2<- predict(trees[[i]],as.matrix(X),type="response",s=lambdas[[i]])
    obbac2=auc(oobobse,as.vector(oobpredicts2))
	varimport1[1]=obbac1-obbac2
	
   #test last column
    X=cbind(tesetnew[,2:80],sample(tesetnew[,81]))
	oobpredicts2<- predict(trees[[i]],as.matrix(X),type="response",s=lambdas[[i]])
    obbac2=auc(oobobse,as.vector(oobpredicts2))
	varimport1[80]=obbac1-obbac2

    #test others
	for (jjj in 3:80){
  	X=cbind(tesetnew[,2:(jjj-1)],sample(tesetnew[,jjj]),tesetnew[,(jjj+1):81])
	oobpredicts2<- predict(trees[[i]],as.matrix(X),type="response",s=lambdas[[i]])
    obbac2=auc(oobobse,as.vector(oobpredicts2))
	varimport1[jjj-1]=obbac1-obbac2
	}
	#print(varimport1)
    varim1=cbind(varim1,varimport1)	
	
 }



# classify the test data   
testpre<-NULL
foreach(i = 1:trlen) %do%{
#if (oobacc[i]<=avroobacc)
{ 
predicts<-predict(trees[[i]],as.matrix(teset[,-c(1)]),s=lambdas[[i]],type="response")
testpre<-cbind(predicts,testpre)
}
}

# oobacc[oobacc<0.5]=0 
# #oobmean<-mean(oobacc)
# oobmin=min(oobacc)
# oobmax=max(oobacc)

  oobweight= 1 / (1 + exp((-oobac))) 

foreach(ii = 1:nrow(teset)) %do%{
testpre[ii,]=testpre[ii,]*oobweight
}
ensemble_predictions<-rowSums(testpre)/sum(oobweight)
#ensemble_predictions<-rowMeans(testpre)

#Caculate AUC
auc2<-auc(obse,ensemble_predictions)
#pre<-prediction(ensemble_predictions,obse)
#perf <- performance(pre, "auc")
#change a list into a vector
#auc2=unlist(perf@y.values)


print(auc2)


print(varimportance)

})
