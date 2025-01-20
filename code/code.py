import numpy as np
import csv
from google.colab import drive

# Database Link -> https://www.kaggle.com/datasets/vetrirah/customer
# Data Path (Google Drive)
# !!! FIX DATA PATH TO WORK FROM A FILE IN THE SAME FOLDER INSTEAD OF THROUGH GOOGLE DRIVE !!!
data_path = "/content/drive/My Drive/Customer_Segmentation/data/"

rNum = 0 #number of rows
cNum = 0 #number of columns
num_components = 3 #reducing the dataset dimensions to this many dimensions (PCA)
trainingData = [] #storing the data

# Read data from dataset
# The dataset is in the same folder as the .py file
def ReadData():
    global rNum
    global cNum
    global trainingData

    with open(f"{data_path}Train.csv", "r") as Xfile:
        reader = csv.reader(Xfile)

        for row in reader:
            trainingData.append(row)
            rNum += 1

        cNum = len(trainingData[0])



# Get list of values written in the i-th column
def GetValues(trainingData, i):
    ls = []
    index = i
    for i in range(1, len(trainingData)):
        find_list = False
        for j in range(len(ls)):
            if trainingData[i][index] == ls[j]:
                find_list = True
        if not find_list: #don't store duplicates
            ls.append(trainingData[i][index])
    return ls



# Replace all values with numbers 
# the numbers are centered around 0
def ReplaceWordsWithNumbers(trainingData, index):
    values = GetValues(trainingData, index) #get all unique values written in the data
    if (len(values) == 2):
        cnt = 0
    else:
        cnt = -int(len(values)/2)

    for i in range(len(values)):
        for j in range(1, len(trainingData)):
            if trainingData[j][index] == values[i]:
                trainingData[j][index] = cnt
        cnt+=1
    
    return trainingData



# Replace non-numeric values with numbers (to use in PCA)
def ReplaceNonNumeric(trainingData):
    for i in range(len(trainingData[0])):
        try:
            # Try converting to int first (if fails, try float)
            int(trainingData[1][i])
            continue
        except ValueError:
            try:
                float(trainingData[1][i])
                continue
            except ValueError:
                trainingData = ReplaceWordsWithNumbers(trainingData, i)
    return trainingData



# Remove rows with missing values except for the profession column - empty fields are regarded as not employed
def RemoveInvalidRows(trainingData):
    profession_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Profession"):
            profession_index = i
    trainingData = [row for row in trainingData if all(value != '' for idx, value in enumerate(row) if idx != profession_index)]
    return trainingData



# Remove unnecessary columns - ID, Var_1, Segmentation)
def RemoveColumns(trainingData):
    return [row[1:-2] for row in trainingData]



# Convert numbers in string format to float/int
def GetStringNumbers(trainingData):
    for i in range(1, len(trainingData)):
        for j in range(len(trainingData[0])):
            if '.' in str(trainingData[i][j]):
                trainingData[i][j] = float(trainingData[i][j])
            else:
                trainingData[i][j] = int(trainingData[i][j])
    return trainingData



def ScaleNumbers(trainingData, index):
    values = GetValues(trainingData, index)
    if (len(values) == 2):
        cnt = 0
    else:
        cnt = -int(len(values)/2)

    min = cnt
    max = min+len(values)-1

    for i in range(len(values)):
        for j in range(1, len(trainingData)):
            if trainingData[j][index] == values[i]:
                #scale values to [0, 1]
                # x_scaled = (x-minx)/(xmax-xmin)
                trainingData[j][index] = (cnt-min)/(max-min)
        cnt+=1
    
    return trainingData

# !!! IMPLEMENT MORE EFFICIENTLY/ASTHETICALLY !!!
def ScaleColumnValues(trainingData):
    for i in range(len(trainingData[0])):
        trainingData = ScaleNumbers(trainingData, i)
    return trainingData
    


# Clean the Dataset
def CleanData(trainingData):
    trainingData = RemoveColumns(trainingData)
    trainingData = RemoveInvalidRows(trainingData)
    trainingData = ReplaceNonNumeric(trainingData)
    trainingData = GetStringNumbers(trainingData)
    trainingData = ScaleColumnValues(trainingData)
    return trainingData



#Reduce the dimensions to the given number of components
def PCA(trainingData, num_components):
    #get mean values for all parameters across x axis
    mean_values = [0 for i in range(len(trainingData[0]))]

    print("-------------------------------- PCA --------------------------------")

    for row in trainingData:
        for i in range(len(row)):
            mean_values[i] += row[i]

    for i in range(len(mean_values)):
        mean_values[i] /= len(trainingData)

    print("Mean values")
    print(mean_values)
    print()

    #find the covariance matrix
    #covariance between two points
    #cov(X, Y) = E[XY] - E[X]E[Y]
    covariance_matrix = []
    for i in range(len(trainingData[0])):
        covariance_matrix.append([])
        for j in range(len(trainingData[0])):
            Ex = mean_values[i]
            Ey = mean_values[j]
            Exy = 0

            for k in range(len(trainingData)):
                Exy += trainingData[k][i] * trainingData[k][j]

            Exy /= len(trainingData)
            covariance_matrix[i].append(Exy - Ex * Ey)

    print("Covariance matrix")
    for row in covariance_matrix:
        print(row)
    print()

    #find the eigenvectors and corresponding eigenvalues, sort them and get the first k(=num_components) largest
    print("Eigenvectors and eigenvalues")
    return



# K-Means Clustering



# Testing



# Visualize the results



# Main
ReadData() #number of rows, columns, data titles, training data
print("Number of rows in dataset:")
print(rNum)
print("Number of columns (parameters) in dataset:")
print(cNum)
print("All parameters in datase:")
print(trainingData[0])
print()

cleanedData = CleanData(trainingData)
cleanedTitles = cleanedData[0]

print("Cleaned data titles:")
print(cleanedTitles)
print()

cleanedData = cleanedData[1:]
print("Cleaned data:")
print(cleanedData)
print()

PCA(cleanedData, num_components)