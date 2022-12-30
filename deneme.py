import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB
trainset = pd.read_csv('trainSet.csv')
trainset['employment'].replace('?','unemployed',inplace=True)
trainset['credit_history'].replace('?','existing paid',inplace=True)
trainset['credit_amount'].replace('?','5000',inplace=True)
trainset['property_magnitude'].replace('?','real estate',inplace=True)
trainset['age'].replace('?','30',inplace=True)#Olmayan değerleri göz kararı değiştirdim
trainset=trainset.apply(lambda col : pd.factorize(col,sort=True)[0])#Daha rahat işlem yapabilmek için faktörize edip kutuladım
trainset[['credit_amount','age']]=preprocessing.minmax_scale(trainset[['credit_amount','age']])
trainset['credit_amount'] = pd.qcut(trainset['credit_amount'],q=[0, .21, .41, .61, .81, 1],labels=['A', 'B', 'C', 'D', 'E'])
trainset['credit_history']=pd.cut(trainset['credit_history'], bins=6, labels=['A','B','C','D','E','F'])
trainset['employment']=pd.cut(trainset['employment'], bins=5, labels=['A','B','C','D','E'])
trainset['age'] = pd.qcut(trainset['age'],q=[0, .25, .5, .75, 1],labels=['A', 'B', 'C', 'D'])
testset=pd.read_csv('testSet.csv')
testset.replace('?',"unemployed",inplace=True)#Olmayan değerleri göz kararı değiştirdim
testset=testset.apply(lambda col : pd.factorize(col,sort=True)[0])#Daha rahat işlem yapabilmek için faktörize edip kutuladım
testset[['credit_amount','age']]=preprocessing.minmax_scale(testset[['credit_amount','age']])
testset['credit_history']=pd.cut(testset['credit_history'], bins=6, labels=['A','B','C','D','E','F'])
testset['employment']=pd.cut(testset['employment'], bins=5, labels=['A','B','C','D','E'])
testset['credit_amount'] = pd.qcut(testset['credit_amount'],q=[0, .21, .41, .61, .81, 1],labels=['A', 'B', 'C', 'D', 'E'])
testset['age'] = pd.qcut(testset['age'],q=[0, .25, .5, .75, 1],labels=['A', 'B', 'C', 'D'])
X_trainset,y_trainset=trainset.drop(['class'],axis=1),trainset.drop(['credit_amount','credit_history','employment','age','property_magnitude'],axis=1)
X_testset,y_testset=testset.drop(['class'],axis=1),testset.drop(['credit_amount','credit_history','employment','age','property_magnitude'],axis=1)
X_trainset=X_trainset.apply(lambda col: pd.factorize(col, sort=True)[0])
X_testset=X_testset.apply(lambda col: pd.factorize(col,sort=True)[0])
y_trainset=y_trainset.apply(lambda col: pd.factorize(col,sort=True)[0])
y_testset=y_testset.apply(lambda col: pd.factorize(col,sort=True)[0])#Kutuladıktan sonra tekrar faktörize edip kategori tipine dönüştürdüm
X_trainset['credit_history']=X_trainset['credit_history'].astype('category')
X_trainset['credit_history']=X_trainset['credit_amount'].astype('category')
X_trainset['credit_history']=X_trainset['employment'].astype('category')
X_trainset['credit_history']=X_trainset['property_magnitude'].astype('category')
X_trainset['credit_history']=X_trainset['age'].astype('category')
X_testset['credit_history']=X_testset['credit_history'].astype('category')
X_testset['credit_history']=X_testset['credit_amount'].astype('category')
X_testset['credit_history']=X_testset['employment'].astype('category')
X_testset['credit_history']=X_testset['property_magnitude'].astype('category')
X_testset['credit_history']=X_testset['age'].astype('category')
y_trainset['class']=y_trainset['class'].astype('category')
y_testset['class']=y_testset['class'].astype('category')
nbmodel=CategoricalNB()
nbmodel.fit(X_trainset,y_trainset)
y_predicted=nbmodel.predict(X_testset)#Hazır algoritmaya train setleri ve testsetimizin gövdesini gönderdim
tp=0
tn=0
fp=0
fn=0
for x in range(250):
    if(y_predicted[x]==0 and testset['class'][x]==0 ):#Sırayla bulduğum ve asıl olanı karşılaştırdım
        tn=tn+1
    if(y_predicted[x]==0 and testset['class'][x]==1):
        fn=fn+1
    if(y_predicted[x]==1 and testset['class'][x]==0 ):
        fp=fp+1
    if(y_predicted[x]==1 and testset['class'][x]==1 ):
        tp=tp+1

accuracy=(tp+tn)/250
truepositiverate= tp/(tp+fn)
truenegativerate=tn/(tn+fp)
print(y_predicted)
print("Accuracy ",accuracy*100)
print("True positive oranı: ",truepositiverate*100)
print("True negative oranı: ",truenegativerate*100)
print("True positive adedi: ",tp)
print("True negative adedi: ",tn)

