
############################################## Imports ###############################################

#import general packages
from req_imports import *

#import defined functions
from functions_defined import *




    

######################################## Variable Initialization ########################################

tf.logging.set_verbosity('ERROR')

#Data start date
start_date='01-01-1980'
#data end date
end_date='14-06-2019'

#stock name
stock_name='MSFT'

#return period
default_return_period=1

#train validation test splits
train_fraction=0.6
val_fraction=0.15

train_fraction1=1
val_fraction1=0.01

#lags
lags = 10
lags_ann=50

#multinomial returns true or false
multinomial_ret=1

########################### Sourcing Data, obtaining features  ############################################

#Getting Historical Data
data = quandl.get('EOD/'+stock_name, start_date=start_date, end_date=end_date)
data=data[['Adj_Open','Adj_Close','Adj_High','Adj_Low','Adj_Volume']]
#data = pd.DataFrame(web.DataReader(stock_name, data_source='yahoo',start=start_date, end=end_date))

#data=data.rename(columns = {'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
data=data.rename(columns = {'Adj_Open':'open','Adj_High':'high','Adj_Low':'low','Adj_Close':'close','Adj_Volume':'volume'})

#Plot data
data[['open','high','low','close']].plot(figsize=(10, 6))

#Obtain features using OHLC data
clfi=[1]
data['r'] = np.log(data['close'] / data['close'].shift(default_return_period))
data['rs'] = (data['r'] - data['r'].mean()) / data['r'].std()

data['d'] = np.where(data['r'] > 0, 1, 0)
if multinomial_ret:
    data['d']=0
data['c-o'] = data['close'] - data['open']
##data['u-d'] = np.where(data['close'] - data['open'] > 0, 1, 0)
data['h-l'] = data['high'] - data['low']
data['h-o'] = data['high'] - data['open']
data['o-l'] = data['open'] - data['low']
data['h-c'] = data['high'] - data['close']
data['c-l'] = data['close'] - data['low']


#call functions to create features
data=MACD(data,3,6)
data=MACD(data,12,26)
data=EMA(data,5)
data=EMA(data,18)
#data=EMA(data,50)
#data=ATR(data,10)
#data=VA(data,5)
#data=MOM(data,1)
#data=MOM(data,5)
##data=RSI(data,1)
data=RSI(data,5)
data=RSI(data,18)
##data=RSI(data,50)
#data=STOK(data)
data=zy_vol(data,5)
data=zy_vol(data,18)
#data=ForceIndex(data,1)
#data=ForceIndex(data,5)


data=data[51:]
data.dropna(inplace=True)
#features = list(data.columns)
#features.remove('Adj Close')
ld = len(data)
ld


#data=data['r']
start = '2014-07-18 12:33:49 +0000'

fb_dm=quandl.get_table('SMA/FBD', date = {'gte': start}, brand_ticker=stock_name, paginate=True)
fb_dm=fb_dm.loc[fb_dm['geography']=='Worldwide']
#fb_dm=fb_dm.loc[fb_dm['sector']!='Non Profits']
#fb_dm=fb_dm.set_index('date');

fb_dm=fb_dm.groupby(['date']).sum()

data=fb_dm.join(data)
data.dropna(inplace=True)

features = list(data.columns)
features.remove('r')

############################################## Preprocess Data & Algorithm run ############################################


#split data into training, validation & training sets
split = int(len(data) * train_fraction)
val_size = int(split * val_fraction)


train = data.iloc[:split]
val = train[-val_size:]
train = train[:-val_size]
test = data.iloc[split:].copy()


#Multinomial returns
if multinomial_ret:
    clfi=[1,2,3,4,5]
    std=train['r'].std()
    train['d']=pd.cut(train.r, [-500*std,-1*std, -0.5*std, 0.5*std, 1*std,500*std], labels=clfi)
    test['d']=pd.cut(test.r, [-500*std,-1*std, -0.5*std, 0.5*std, 1*std,500*std], labels=clfi)
    val['d']=pd.cut(val.r, [-500*std,-1*std, -0.5*std, 0.5*std, 1*std,500*std], labels=clfi)

    

train,val,test,cols=normalize_and_lag(train,val,test,features,lags)
len(cols)
train.head(5)

if len(clfi)<2:    
    y_true=np.where(test['d'] > 0, 1, -1)
else:
    clfi=np.array(clfi)-3 
    y_true=np.array(test['d'])-3
################################# LSTM ###################################



# Keras LSTM ML model & backtesting




model = Sequential()
model.add(LSTM(128, activation='relu',return_sequences=True,
                kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),input_shape=(lags,len(features))
                )
         )
    
model.add(BatchNormalization())
model.add(Dropout(0.3, seed=100))
model.add(BatchNormalization())
model.add(LSTM(128, activation='relu',
                kernel_regularizer=l2(0.01)
               )
         )
model.add(BatchNormalization())
model.add(Dropout(0.3, seed=100))
model.add(BatchNormalization())
model.add(Dense(len(clfi)+1,kernel_regularizer=l2(0.01), 
                activation='sigmoid'
                ))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()
callbacks = [EarlyStopping(monitor='val_acc', patience=25)]


categorical_labels = to_categorical(train['d'])
categorical_labels1 = to_categorical(val['d'])
categorical_labels2 = to_categorical(test['d'])

get_ipython().run_cell_magic(u'time', u'', u"model.fit(np.array(train[cols]).reshape(len(train),lags,len(features)), np.array(categorical_labels),\n          epochs=250, batch_size=32, verbose=False,\n          validation_data=(np.array(val[cols]).reshape(len(val),lags,len(features)), np.array(categorical_labels1)),\n          callbacks=callbacks);")

res = pd.DataFrame(model.history.history)
res.tail(3)
res.plot(figsize=(10, 6), style=['--', '--', '-', '-']);




#Evaluate model
print(model.evaluate(np.array(test[cols]).reshape(len(test),lags,len(features)), np.array(categorical_labels2)))

#Predict returns for test set
test['p'] = model.predict_classes(np.array(test[cols]).reshape(len(test),lags,len(features)))
if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3


plot_confusion_matrix(y_true, test['p'], classes=clfi,title='Confusion matrix LSTM')
#Backtesting    
test['s_lstm'] = test['p'] * test['r']
print(test[['r', 's_lstm']].sum().apply(np.exp))
sum(test['p'].diff() != 0)
test['p'].value_counts()
test[['r', 's_lstm']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="performance of strategy on testing set LSTM")


################################# Neural Nets ###################################


## Keras ML model & backtesting


model = Sequential()
model.add(Dense(128, activation='relu',
                kernel_regularizer=l2(0.01),
                input_shape=(len(cols),)
               )
         )
model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
model.add(Dropout(0.3, seed=100))
model.add(Dense(128, activation='relu',
                kernel_regularizer=l2(0.01)
               )
         )
model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
model.add(Dropout(0.3, seed=100))
model.add(Dense(len(clfi)+1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



model.summary()
callbacks = [EarlyStopping(monitor='val_acc', patience=25)]
get_ipython().run_cell_magic(u'time', u'', u"model.fit(train[cols], np.array(categorical_labels),\n          epochs=250, batch_size=32, verbose=False,\n          validation_data=(val[cols], np.array(categorical_labels1)),\n          callbacks=callbacks);")
res = pd.DataFrame(model.history.history)
res.tail(3)
res.plot(figsize=(10, 6), style=['--', '--', '-', '-']);


#Evaluate model
print(model.evaluate(test[cols], np.array(categorical_labels2)))
#Predict returns for test set
test['p'] = model.predict_classes(test[cols])
if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3
    
plot_confusion_matrix(y_true, test['p'], classes=clfi,title='Confusion matrix KERAS')

#Backtesting 
test['s_ann'] = test['p'] * test['r']
print(test[['r', 's_ann']].sum().apply(np.exp))
sum(test['p'].diff() != 0)
test['p'].value_counts()
test[['r', 's_ann']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="KERAS performance of strategy on testing set");





################################# DECISION TREES ###################################
SVM_SVC = SVC(C=1, probability=True)
naivebayes = BernoulliNB()
## DECISION TREES ML model & backtesting
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
regr_1.fit(train[cols], train['d'])
regr_2.fit(train[cols], train['d'])
#Predict returns for test set
test['p'] = regr_1.predict(test[cols])
test['p_adar'] = regr_2.predict(test[cols])

if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
    test['p_adar'] = np.where(test['p_adar'] > 0, 1, -1)
else:
    test['p']=test['p']-3
    test['p']=round(test['p']);
    test['p_adar']=test['p_adar']-3
    test['p_adar']=round(test['p_adar']);

#Backtesting 
test['s_dtr'] = test['p'] * test['r']
test['s_adar'] = test['p_adar'] * test['r']
print(test[['r', 's_dtr','s_adar']].sum().apply(np.exp))
sum(test['p'].diff() != 0)
test['p'].value_counts()
test[['r', 's_dtr','s_adar']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="Decision trees regressor performance of strategy on testing set");

#dot_data = tree.export_graphviz(regr_1)
#graph = graphviz.Source(dot_data,filename='tempy', format='png')
#graph.render() # tree saved to wine.pdf


## XGBOOST ML model & backtesting

#np.random.seed(100)
#tf.random.set_random_seed(100)
if multinomial_ret==0:
    model = xgb.XGBClassifier()
    model.fit(train[cols], train['d'])
    
    #Predict returns for test set
    test['p'] = model.predict(test[cols])
    test['p'] = np.where(test['p'] > 0, 1, -1)
    
    #Backtesting 
    test['s_xgb'] = test['p'] * test['r']
    print(test[['r', 's_xgb']].sum().apply(np.exp))
    sum(test['p'].diff() != 0)
    test['p'].value_counts()
    test[['r', 's_xgb']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="XGBOOST performance of strategy on testing set");

else:
    test['s_xgb']=0



## ADABOOST ML model & backtesting
# Create adaboost classifer object

model = AdaBoostClassifier(n_estimators=500,learning_rate=1)
# Train Adaboost Classifer
model.fit(train[cols], train['d'])
#Predict the response for test dataset
test['p'] = model.predict(test[cols])

#Evaluate model
print("Accuracy:",accuracy_score(test['d'], test['p']))

if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3

plot_confusion_matrix(y_true, test['p'], classes=clfi,title='Confusion matrix ADABOOST')

#Backtesting 
test['s_ada'] = test['p'] * test['r']
print(test[['r', 's_ada']].sum().apply(np.exp))
sum(test['p'].diff() != 0)
test['p'].value_counts()
test[['r', 's_ada']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="ADABOOST performance of strategy on testing set")

#export the learned decision tree
#dot_data = tree.export_graphviz(model, out_file=None,
#                         feature_names=cols,
#                         class_names=[0,1],
#                         filled=True, rounded=True,
#                         special_characters=True)
#dot_data = tree.export_graphviz(model.estimators_[0])
#graph = graphviz.Source(dot_data,filename='tempy', format='png')
#graph.render() # tree saved to wine.pdf




################################# SVM ###################################

## SVM ML model & backtesting
SVM_SVC = SVC(C=1, probability=True)
SVM_SVC.fit(train[cols], train['d']) ###FITTING DONE HERE
    
print("Prediction Accuracy of traning set is ")
    
print(accuracy_score(SVM_SVC.predict(train[cols]),train['d']))
print("Prediction Accuracy of testing set is ")
print(accuracy_score(SVM_SVC.predict(test[cols]),test['d']))

#Predict returns for test set
test['p'] = SVM_SVC.predict(test[cols])

if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3

#Backtesting 
test['s_svm'] = test['p'] * test['r']
test[['r', 's_svm']].sum().apply(np.exp)
sum(test['p'].diff() != 0)
test['p'].value_counts()

conf_mat = confusion_matrix(test['d'], test['p'])
print(classification_report(test['d'], test['p']))


plot_confusion_matrix(y_true, test['p'], classes=clfi,title='Confusion matrix SVM')

print(conf_mat)

test[['r', 's_svm']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="SVM performance of strategy on testing set")

#test[['r','s_lstm','s_ann','s_ada', 's_svm','s_xgb']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="LSTM, ANN, SVM, XGBoost, ADABoost performance of strategy on testing set")

#test1=test

################################ LOGISTIC & NAIVE BAYEES ###################################


## LOGISTIC ML model & backtesting

logit = linear_model.LogisticRegression(C=1,multi_class='auto',solver='lbfgs',max_iter=10000)
    
kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=False) # RandomState is the seed used by the RNG

print(kfold)

#print(kfold.get_n_splits(X_Features)) # returns the number of splitting iterations in the cross-validator


#Fit &Evaluate model 
crossval = model_selection.cross_val_score(logit, train[cols], train['d'], cv=kfold, scoring='accuracy')
crossval_predict = model_selection.cross_val_predict(logit, test[cols], test['d'], cv=kfold)
print("5-fold crossvalidation accuracy: %.4f" % (crossval.mean())) #average accuracy


conf_mat = confusion_matrix(test['d'], crossval_predict)
print(classification_report(test['d'], crossval_predict))

print(conf_mat)

plot_confusion_matrix(test['d'], crossval_predict, classes=clfi,title='Confusion matrix LOGISTIC')
test['p'] = crossval_predict
if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3

#Backtesting 
test['s_logistic'] = test['p'] * test['r']
test[['r', 's_logistic']].sum().apply(np.exp)
sum(test['p'].diff() != 0)
test['p'].value_counts()
    

test[['r', 's_logistic']].cumsum().apply(np.exp).plot(figsize=(10, 6),title=" LOG performance of strategy on testing set")


## NAIVE BAYEES ML model & backtesting


naivebayes = BernoulliNB()
kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=False) # RandomState is the seed used by the RNG

print(kfold)

#Fit &Evaluate model 
crossval = model_selection.cross_val_score(naivebayes, train[cols], train['d'], cv=kfold, scoring='accuracy')
crossval_predict = model_selection.cross_val_predict(naivebayes, test[cols], test['d'], cv=kfold)
print("5-fold crossvalidation accuracy: %.4f" % (crossval.mean())) #average accuracy

conf_mat = confusion_matrix(test['d'], crossval_predict)
print(classification_report(test['d'], crossval_predict))

print(conf_mat)

plot_confusion_matrix(test['d'], crossval_predict, classes=clfi,title='Confusion matrix NAIVE BAYES')
test['p'] = crossval_predict

if len(clfi)<2:
    test['p'] = np.where(test['p'] > 0, 1, -1)
else:
    test['p']=test['p']-3

#Backtesting 
test['s_nb'] = test['p'] * test['r']
test[['r', 's_nb']].sum().apply(np.exp)
sum(test['p'].diff() != 0)
test['p'].value_counts()

test[['r', 's_nb']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="BAYEES performance of strategy on testing set")

test[['r','s_lstm','s_ann','s_ada','s_dtr', 's_svm','s_xgb', 's_nb','s_logistic']].sum().apply(np.exp)

test[['r','s_lstm','s_ann','s_ada','s_dtr', 's_svm','s_xgb', 's_nb','s_logistic']].cumsum().apply(np.exp).plot(figsize=(10, 6),title="LSTM, ANN, SVM, XGBoost, DT Reg, ADABoost, BAYEES & LOGISTIC performance of strategy on testing set")


