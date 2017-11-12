from sklearn.svm import SVC

class LinearSVM:

    def __init__(self, config):
        self.config = config
            

    def train(self, x, label):

        print 'SVM Linear Training'

        self.train_x = x
        self.train_label = label
        
        self.clf5 = SVC(C=1, kernel='linear')
        self.clf5.fit(self.train_x, self.train_label)
                    
    def measure_accuracy(self, x, label):

        print 'SVM Measure Accuracy'
        
        self.test_x = x
        self.test_label = label

        pred = self.clf5.predict(self.test_x)

        #print accuracy_score(testLabels , pred)

        allTestClasses = sorted(list(set(self.test_label.tolist())))
        dict_correct = {}
        dict_total = {}
        
        for ii in allTestClasses:
            dict_total[ii] = 0 
            dict_correct[ii] = 0

        for ii in range(0, len(self.test_label)):
            if(self.test_label[ii] == pred[ii]):
                dict_correct[pred[ii]] += 1

            dict_total[self.test_label[ii]] += 1 

        avgAcc = 0.0
        for ii in allTestClasses:
            avgAcc = avgAcc + (dict_correct[ii]*1.0)/(dict_total[ii])

        avgAcc = avgAcc/len(allTestClasses) 
        print 'Average Class Accuracy = ' + str(avgAcc)
        return avgAcc
