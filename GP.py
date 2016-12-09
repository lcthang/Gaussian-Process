import GPflow
import numpy as np
import csv

DATA_FILE = 'student-por.csv'
TEST_FILE = 'student-por.csv'
ATTR_FILE = 'attributes.txt'
FEATURES_FILE = 'X.npy'
LABEL_FILE = 'Y.npy'
MEAN_FILE = 'mean.npy'
VAR_FILE = 'var.npy'
WHITE_NOISE_VAR = 0.01

def getRegressionModel(X, Y):
    # Build the GPR object
    k = GPflow.kernels.RBF(input_dim=X.shape[1])
    meanf = GPflow.mean_functions.Zero()
    m = GPflow.gpr.GPR(X, Y, k, meanf)
    m.likelihood.variance = WHITE_NOISE_VAR

    print("Here are the parameters before optimization")
    print(m)
    return m

def optimizeModel(m):
    m.optimize()
    print("Here are the parameters after optimization")
    print(m)

def convertToNum(colIdx, cell):
    # convert binary & nominal to numeric values
    conversion = {
        0: {
            'GP': -1,
            'MS': +1
        },
        2: {
            'F': -1,
            'M': +1
        },
        3: {
            'U': -1,
            'R': +1
        },
        4: {
            'LE3': -1,
            'GT3': +1
        },
        5: {
            'T': -1,
            'A': +1
        },
        8: {
            'teacher': 4,
            'health': 3,
            'services': 2,
            'at_home': 1,
            'other': 0
        },
        10: {
            'home': 3,
            'reputation': 2,
            'course': 1,
            'other': 0
        },
        11: {
            'mother': 2,
            'father': 1,
            'other': 0
        }, 
        15: {
            'yes': 1,
            'no': -1
        }
    }

    if colIdx >= 15 and colIdx <=22:
        return conversion.get(15).get(cell)
    elif colIdx == 8 or colIdx == 9:
        return conversion.get(8).get(cell)
    else:
        try:
            value = conversion.get(colIdx).get(cell)
            return value
        except AttributeError:
            return cell


# extract columns with selected attributes
def transform(data):
    numCol = len(data[0])
    X = np.zeros((len(data), numCol-1))
    Y = np.zeros((len(data), 1))

    rowIdx = 0
    for row in data:
        colIdx = 0
        rowMat = np.zeros(numCol-1)

        for (k, v) in row.items():
            if colIdx < numCol-1 and (colIdx == 29 or colIdx == 2):
                rowMat[colIdx] = convertToNum(k, v)
            elif(colIdx == numCol - 1):
                Y[rowIdx] = convertToNum(k, v)

            colIdx+=1

        X[rowIdx] = rowMat
        rowIdx+=1


    return X, Y
            

# read input file
def readInput(fileName):
    csvfile = open(fileName, 'r')
    reader = csv.reader(csvfile, delimiter = ';')
    attributes = next(reader)

    data = []
    listIdx = []

    i=0
    for i in range(len(attributes)):
        listIdx.append(i)
        i+=1

    for row in reader:
        row = dict(zip(listIdx, row))
        data.append(row)

    csvfile.close()

    return data

def main():
    # Read data
    data = readInput(DATA_FILE)
    X_train, Y_train = transform(data)

    testData = readInput(TEST_FILE)
    X_test, Y_test = transform(testData)
    numTest = len(X_test)

    m = getRegressionModel(X_train, Y_train)
    optimizeModel(m)

    # test on train set
    mean, var = m.predict_y(X_test)

    np.save(open(FEATURES_FILE, "wb"), X_test)
    np.save(open(LABEL_FILE, "wb"), Y_test)
    np.save(open(MEAN_FILE, "wb"), mean)
    np.save(open(VAR_FILE, "wb"), var)
    
    #for i in range(numTest):
    #    print(testData[i])
    #    print("Actual Y: (%s)" % (str(Y_test[i][0])))
    #    print("Predicted f mean: (%s)" % (str(mean[i][0])))
    #    print("Predicted f var: (%s)" % (str(var[i][0])))
    #    print()

    #print(X_test)
    #print(mean)

if __name__ == "__main__":
    main()
