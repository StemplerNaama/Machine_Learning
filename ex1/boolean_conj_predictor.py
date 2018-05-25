import numpy
import sys

#Places the current example in current hypothesis
def helper(array,if_negative_exsite,if_variable_exsit):
    for x in range(0, numpy.size(array,0)):
        if array[x] == 1 and if_negative_exsite[x] ==1:
            return 0
        if array[x] == 0 and if_variable_exsit[x] ==1:
            return 0
    return 1

#write to file the hypothesis we get
def print_statement(if_negative_exsite,if_variable_exsit):
    retVal = ''
    for i in range(0,numpy.size(if_negative_exsite,0)):
        if if_variable_exsit[i] == 1:
            retVal += ',x' + str(i+1)
        if if_negative_exsite[i] == 1:
            retVal += ',not(x' + str(i+1) + ')'
    with open('output.txt', "w") as file:
        file.write(retVal[1:retVal.__len__()])

def main(argv):
    import numpy as np
    training_examples = np.loadtxt(sys.argv[1])
    lastColumnIndex =  training_examples.shape[1] -1
    num_of_rows = training_examples.shape[0]
    lastColumn =  training_examples[:, lastColumnIndex]
    intances = training_examples[:, 0:(lastColumnIndex)]
    if_negative_exsite = [1 for x in range(lastColumnIndex)]
    if_variable_exsit = [1 for x in range(lastColumnIndex)]
  #  if_negative_exsite = [1] * lastColumnIndex
   # if_variable_exsit = [1] * lastColumnIndex


    #iterate on every instance we get
    for x in range(0, num_of_rows):
        if lastColumn[x] == 1 and helper(intances[x],if_negative_exsite,if_variable_exsit) == 0:
            for n in range(0, numpy.size(intances[x],0)):
                if intances[x][n] == 1:
                    if_negative_exsite[n] = 0
                if  intances[x][n] == 0:
                    if_variable_exsit[n] = 0
    print_statement(if_negative_exsite,if_variable_exsit)

if __name__ == "__main__":
   main(sys.argv[1])

