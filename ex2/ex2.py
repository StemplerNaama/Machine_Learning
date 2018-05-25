import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy

#calcultae the soft max
def softMax(w,x):
    wMultx =  numpy.matmul(w, numpy.matrix(x).T)
    m = numpy.max(wMultx)
    vec = numpy.exp(wMultx - m)
    sumvec = numpy.sum(vec)
    return vec / sumvec

if __name__ == '__main__':
    # create the training set
    numOfPoints = 100
    numClasses = 3
    arr1tag = [1 for x in range(numOfPoints)]
    arr2tag = [2 for x in range(numOfPoints)]
    arr3tag = [3 for x in range(numOfPoints)]
    arr1 = numpy.random.normal(2 * 1, 1, 100)
    arr2 = numpy.random.normal(2 * 2, 1, 100)
    arr3 = numpy.random.normal(2 * 3, 1, 100)
    arr1zipped = zip(arr1, arr1tag)
    arr2zipped = zip(arr2, arr2tag)
    arr3zipped = zip(arr3, arr3tag)

    set = arr1zipped + arr2zipped + arr3zipped

    # initialize w in randomaly
    w = numpy.random.rand(3,2)
    # update w by runing on the training set 200 times
    for i in range(200):
        numpy.random.shuffle(set)
        for j in range(3 * numOfPoints):
            x = set[j][0]
            xPlusBias = [x, 1]
            tag = set[j][1]
            vecTag = numpy.eye(3)[int(tag) - 1]
            softmaxOutput= softMax(w, xPlusBias)

            w = w - 0.01 * numpy.matmul(softmaxOutput - numpy.matrix(vecTag).T , numpy.matrix(xPlusBias))


    arrTest = numpy.linspace(0,10,100)
    probabiltyTest = [1 for x in range(100)]
    for i in range(100):
        vecX = [arrTest[i], 1]
        aftersoftmax = softMax(w, vecX)
        probabiltyTest[i] = aftersoftmax[0,0]

    realTest = []

    #draw real probability and probabilty by learning
    for i in range(100):
        p1 = mlab.normpdf(arrTest[i], 2, 1)
        p2 = mlab.normpdf(arrTest[i], 4, 1)
        p3 = mlab.normpdf(arrTest[i], 6, 1)
        p = p1/(p1+p2+p3)
        realTest.append(p)

    plt.plot(arrTest,realTest, label = "real distrebition")
    plt.plot(arrTest, probabiltyTest, label = "by learning" )
    plt.legend()
    plt.show()


