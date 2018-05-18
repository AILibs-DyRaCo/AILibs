import sys
import arffcontainer
import numpy as np
import sklearn.metrics
#import#

trainFile=sys.argv[1]
testFile=sys.argv[2]

trainData = arffcontainer.parse(trainFile)
X_train = np.array(trainData.input_matrix)
y_train = []
for crow in trainData.output_matrix:
    for x in range(0, len(crow)):
	    if crow[x] == 1:
		    y_train.append(x)
y_train = np.array(y_train)

testData = arffcontainer.parse(testFile)
X_test = np.array(testData.input_matrix)
y_test = []
for crow in testData.output_matrix:
    for x in range(0, len(crow)):
	    if crow[x] == 1:
		    y_test.append(x)
y_test = np.array(y_test)
    
mlpipeline = #pipeline#

mlpipeline.fit(X_train, y_train)

y_hat = mlpipeline.predict(X_test)

errorRate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
print(errorRate)