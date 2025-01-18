import numpy as np
import csv
from google.colab import drive

# Database Link -> https://www.kaggle.com/datasets/vetrirah/customer
# Data Path (Google Drive)
data_path = "/content/drive/My Drive/Customer_Segmentation/data/"

rNum = 0
cNum = 0
num_components = 3
trainingData = []

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
        if not find_list:
            ls.append(trainingData[i][index])
    return ls



# Replace non-numeric values with numbers (to use in PCA)
def ReplaceNonNumeric(trainingData):
    #replace gender data
    gender_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Gender"):
            gender_index = i
    for i in range(1, len(trainingData)):
        if trainingData[i][gender_index] == 'Male':
            trainingData[i][gender_index] = 1
        elif trainingData[i][gender_index] == 'Female':
            trainingData[i][gender_index] = 0

    #replace married data
    married_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Ever_Married"):
            married_index = i
    for i in range(1, len(trainingData)):
        if trainingData[i][married_index] == 'Yes':
            trainingData[i][married_index] = 1
        elif trainingData[i][married_index] == 'No':
            trainingData[i][married_index] = 0

    #replace graduated data
    graduated_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Graduated"):
            graduated_index = i
    for i in range(1, len(trainingData)):
        if trainingData[i][graduated_index] == 'Yes':
            trainingData[i][graduated_index] = 1
        elif trainingData[i][graduated_index] == 'No':
            trainingData[i][graduated_index] = 0

    #replace profession data
    #get list of values for profession
    profession_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Profession"):
            profession_index = i

    professions = GetValues(trainingData, profession_index)
    cnt = -int(len(professions)/2)

    for i in range(len(professions)):
        for j in range(1, len(trainingData)):
            if trainingData[j][profession_index] == professions[i]:
                trainingData[j][profession_index] = cnt
        cnt += 1

    #replace spending data
    #get list of values for spending score
    spending_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Spending_Score"):
            spending_index = i

    spending_scores = GetValues(trainingData, spending_index)
    cnt = -int(len(spending_scores)/2)

    for i in range(len(spending_scores)):
        for j in range(1, len(trainingData)):
            if trainingData[j][spending_index] == spending_scores[i]:
                trainingData[j][spending_index] = cnt
        cnt += 1

    return trainingData



# Replace rows with missing values (except for the profession column)
def RemoveInvalidRows(trainingData):
    profession_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Profession"):
            profession_index = i
    # Keep rows where all values are non-empty, except for the 'Profession' column
    trainingData = [row for row in trainingData if all(value != '' for idx, value in enumerate(row) if idx != profession_index)]
    return trainingData



# Remove unnecessary columns (ID, Var_1, Segmentation)
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



def ScaleNumbers(trainingData):
    profession_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Profession"):
            profession_index = i

    professions = GetValues(trainingData, profession_index)
    min = -int(len(professions)/2)
    max = min+len(professions)-1
    cnt = min

    for i in range(len(professions)):
        for j in range(1, len(trainingData)):
            if trainingData[j][profession_index] == professions[i]:
                #scale profession values to [0, 1]
                # x' = (x-minx)/(xmax-xmin)
                trainingData[j][profession_index] = (cnt-min)/(max-min)
        cnt += 1

    #replace spending data
    #get list of values for spending score
    spending_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Spending_Score"):
            spending_index = i

    spending_scores = GetValues(trainingData, spending_index)
    min = -int(len(spending_scores)/2)
    max = min+len(spending_scores)-1
    cnt = min

    for i in range(len(spending_scores)):
        for j in range(1, len(trainingData)):
            if trainingData[j][spending_index] == spending_scores[i]:
                #scale spending score values to [0, 1]
                trainingData[j][spending_index] = (cnt-min)/(max-min)
        cnt += 1

    #scale age values to [0, 1]
    age_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Age"):
            age_index = i

    min = trainingData[1][age_index]
    max = trainingData[1][age_index]
    for i in range(1, len(trainingData)):
        if (trainingData[i][age_index] > max):
            max = trainingData[i][age_index]
        if (trainingData[i][age_index] < min):
            min = trainingData[i][age_index]
            
    for i in range(1, len(trainingData)):
        trainingData[i][age_index] = (trainingData[i][age_index]-min)/(max-min)

    #scale work experience values to [0, 1]
    experience_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Work_Experience"):
            experience_index = i

    min = trainingData[1][experience_index]
    max = trainingData[1][experience_index]
    for i in range(1, len(trainingData)):
        if (trainingData[i][experience_index] > max):
            max = trainingData[i][experience_index]
        if (trainingData[i][experience_index] < min):
            min = trainingData[i][experience_index]
            
    for i in range(1, len(trainingData)):
        trainingData[i][experience_index] = (trainingData[i][experience_index]-min)/(max-min)

    #scale family values to [0, 1]
    family_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Family_Size"):
            family_index = i

    min = trainingData[1][family_index]
    max = trainingData[1][family_index]
    for i in range(1, len(trainingData)):
        if (trainingData[i][family_index] > max):
            max = trainingData[i][family_index]
        if (trainingData[i][family_index] < min):
            min = trainingData[i][family_index]
            
    for i in range(1, len(trainingData)):
        trainingData[i][family_index] = (trainingData[i][family_index]-min)/(max-min)

    return trainingData



# Clean the Dataset
def CleanData(trainingData):
    #Remove rows if something except Profession is missing
    trainingData = RemoveColumns(trainingData)
    trainingData = RemoveInvalidRows(trainingData)
    trainingData = ReplaceNonNumeric(trainingData)
    trainingData = GetStringNumbers(trainingData)
    trainingData = ScaleNumbers(trainingData)
    #scale all data values to a unit interval so that the PCA is not affected incorrectly from the start
    return trainingData



#Reduce the dimensions to only 3 (so it will be easy to visualize)
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

    #find the eigenvectors and corresponding eigenvalues, sort them and get the first k largest


    #eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return

# PCA
# 2. Find the covariance matrix
# 3. Find the first k biggest eigenvectors and their values
# 4. Project the data


# K-Means Clustering



# Main
ReadData() #number of rows, columns, data titles, training data
print("Number of rows in dataset:")
print(rNum)
print("Number of columns (parameters) in dataset:")
print(cNum)
print("All parameters in datase:")
print(trainingData)
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