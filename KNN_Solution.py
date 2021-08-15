import numpy as np
import pandas as pd

np.set_printoptions(precision=4, linewidth=160)

#********************************************** QUESTION 1 **********************************************#

university_train = np.genfromtxt('KNN_Q1_Train.csv', dtype= None, delimiter=',', encoding="utf-8-sig")
university_test= np.genfromtxt('KNN_Q1_Test.csv', dtype= None, delimiter=',', encoding="utf-8-sig")
university_train = np.array(university_train)

def fetcher(list_of_tups, ind):
    return [x[ind] for x in list_of_tups]

Verbal = fetcher(university_test, 3)+  fetcher(university_train, 3)
Math = fetcher(university_test, 4)+  fetcher(university_train, 4)

Min_Verbal = min(Verbal)
Max_Verbal = max(Verbal)
Min_Math = min(Math)
Max_Math = max(Math)

Train_Verbal = np.zeros([20,1])
Train_Math = np.zeros([20,1])

Test_Verbal = np.zeros([5,1])
Test_Math = np.zeros([5,1])

Train_Academics = np.zeros([20,1])
Train_Social= np.zeros([20,1])
Train_QualityOfLife = np.zeros([20,1])
Test_Academics = np.zeros([5,1])
Test_Social = np.zeros([5,1])
Test_QualityOfLife = np.zeros([5,1])

Max_Ordinal = 5
Min_Ordinal = 1



for i in range(20):
    Train_Verbal[i] = (university_train[i][3] - Min_Verbal)/(Max_Verbal - Min_Verbal)
    Train_Math[i] = (university_train[i][4] - Min_Math)/(Max_Math - Min_Math)
    Train_Academics[i] = (university_train[i][5] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)
    Train_Social[i] = (university_train[i][6] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)
    Train_QualityOfLife[i] = (university_train[i][7] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)

for i in range(5):
    Test_Verbal[i] = (university_test[i][3] - Min_Verbal)/(Max_Verbal - Min_Verbal)
    Test_Math[i] = (university_test[i][4] - Min_Math)/(Max_Math - Min_Math)
    Test_Academics[i] = (university_test[i][5] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)
    Test_Social[i] = (university_test[i][6] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)
    Test_QualityOfLife[i] = (university_test[i][7] - Min_Ordinal)/(Max_Ordinal - Min_Ordinal)


# print(Test_Academics)




def university_calculations(university_index):
    university_matrix = np.zeros([20,5])
    for i in range(20):
        university_matrix[i][0] = i + 1
        if(university_train[i][1]==university_test[university_index][1]):
            university_matrix[i][1] = 0
        else:
            university_matrix[i][1] = 1

        university_matrix[i][2] = (abs(Train_Verbal[i]-Test_Verbal[university_index]) + abs(Train_Math[i]-Test_Math[university_index]))/2
        university_matrix[i][3] = (abs(Train_Academics[i] - Test_Academics[university_index]) + abs(Train_Social[i] - Test_Social[university_index]) + abs(Train_QualityOfLife[i] - Test_QualityOfLife[university_index]))/3
        university_matrix[i][4] = (university_matrix[i][1] + (2 * university_matrix[i][2] ) + (3*university_matrix[i][3] ))/6

    university_matrix = sort_matrix(university_matrix)

    return university_matrix

def sort_matrix(university_to_sort):
    university_sorted = university_to_sort[np.argsort(university_to_sort[:, 4])]
    return university_sorted


def find_class(university_name, num_neighbour, university_array, university_index):
    public=0
    private=0
    classification = None
    for i in range(num_neighbour):
        indexer = int(university_array[i][0])-1
        if(university_train[indexer][2] == "private"):
            private = private + 1
        else:
            public = public + 1

    if(public > private):
        classification = "public"
    else:
        classification = "private"
    numerrors = 0
    # print("test data ", university_test[university_index][2])
    # print("classification ", classification)
    # print("university ", university_test[university_index][0])
    # print("index ", university_index)
    if(university_test[university_index][2] != classification):
        numerrors = 1
    else: numerrors = 0
    print("The classification for " + str(university_name) + " with " + str(num_neighbour) + " neighbours is " + str(classification))

    return numerrors



def solve_question(university_name, university_matrix, uni_index):
    print(str(university_name) + " is " + university_test[uni_index][2])
    numerrors =0
    numerrors = numerrors + find_class(university_name, 3, university_matrix, uni_index)
    numerrors = numerrors + find_class(university_name, 5, university_matrix, uni_index)
    numerrors = numerrors + find_class(university_name, 7, university_matrix, uni_index)

    print("There were " + str(numerrors) + " errors")

albany = university_calculations(0)
syracuse = university_calculations(1)
tufts = university_calculations(2)
golden_gate = university_calculations(3)
san_jose = university_calculations(4)

print("\n Question 1 \n")

solve_question("Albany", albany, 0)
solve_question("Syracuse", syracuse, 1)
solve_question("Tufts", tufts, 2)
solve_question("Golden Gate", golden_gate, 3)
solve_question("San Jose", san_jose, 4)

#**********************************************QUESTION 2**********************************************#

patient_train = np.genfromtxt('KNN_Q2_Train.csv', dtype= None, delimiter=',', encoding="utf-8-sig")
patient_test= np.genfromtxt('KNN_Q2_Test.csv', dtype= None, delimiter=',', encoding="utf-8-sig")
patient_train = np.array(patient_train)



def fetcher(list_of_tups, ind):
    return [x[ind] for x in list_of_tups]

Temperature = fetcher(patient_test, 1)+  fetcher(patient_train, 1)
Min_Temp = min(Temperature)
Max_Temp = max(Temperature)


Train_Temp = np.zeros([20,1])
Train_Nausea = np.zeros([20,1])
Train_LumbarP = np.zeros([20,1])
Train_UrineP = np.zeros([20,1])
Train_MictP = np.zeros([20,1])
Train_BurningU = np.zeros([20,1])
Train_BladderI= np.zeros([20,1])
Train_Nephritis = np.zeros([20,1])

Test_Temp= np.zeros([5,1])
Test_Nausea = np.zeros([5,1])
Test_LumbarP = np.zeros([5,1])
Test_UrineP = np.zeros([5,1])
Test_MictP = np.zeros([5,1])
Test_BurningU = np.zeros([5,1])
Test_BladderI= np.zeros([5,1])
Test_Nephritis = np.zeros([5,1])

for i in range(20):
    Train_Temp[i] = (patient_train[i][1] - Min_Temp)/(Max_Temp - Min_Temp)
    Train_Nausea[i] = 1 if patient_train[i][2] == "yes" else 0
    Train_LumbarP[i] = 1 if patient_train[i][3] == "yes" else 0
    Train_UrineP[i] = 1 if patient_train[i][4] == "yes" else 0
    Train_MictP[i] = 1 if patient_train[i][5] == "yes" else 0
    Train_BurningU[i] = 1 if patient_train[i][6] == "yes" else 0
    Train_BladderI[i] = 1 if patient_train[i][7] == "yes" else 0
    Train_Nephritis[i] = 1 if patient_train[i][8] == "yes" else 0

for i in range(5):
    Test_Temp[i] = (patient_test[i][1] - Min_Temp)/(Max_Temp - Min_Temp)
    Test_Nausea[i] = 1 if patient_test[i][2] == "yes" else 0
    Test_LumbarP[i] = 1 if patient_test[i][3] == "yes" else 0
    Test_UrineP[i] = 1 if patient_test[i][4] == "yes" else 0
    Test_MictP[i] = 1 if patient_test[i][5] == "yes" else 0
    Test_BurningU[i] = 1 if patient_test[i][6] == "yes" else 0
    Test_BladderI[i] = 1 if patient_test[i][7] == "yes" else 0
    Test_Nephritis[i] = 1 if patient_test[i][8] == "yes" else 0

def bladder_inflammation_calculations(patient_index):
    patient_matrix = np.zeros([20,4])
    for i in range(20):
        patient_matrix[i][0] = i + 1

        patient_matrix [i][1] = abs(Train_Temp[i]-Test_Temp[patient_index])
        patient_matrix[i][2] = (abs(Train_Nausea[i] - Test_Nausea[patient_index]) + abs(Train_LumbarP[i] - Test_LumbarP[patient_index]) + abs(Train_UrineP[i] - Test_UrineP[patient_index]) +\
                                abs(Train_MictP[i] - Test_MictP[patient_index]) + abs(Train_BurningU[i] - Test_BurningU[patient_index]) + abs(Train_Nephritis[i] - Test_Nephritis[patient_index]))/6

        patient_matrix[i][3] = (patient_matrix[i][1] + (6 * patient_matrix[i][2] )  )/7

    patient_matrix = sort_matrix(patient_matrix, 3)


    return patient_matrix

def nephritis_calculations(patient_index):
    patient_matrix = np.zeros([20,4])
    for i in range(20):
        patient_matrix[i][0] = i + 1

        patient_matrix [i][1] = abs(Train_Temp[i]-Test_Temp[patient_index])
        patient_matrix[i][2] = (abs(Train_Nausea[i] - Test_Nausea[patient_index]) + abs(Train_LumbarP[i] - Test_LumbarP[patient_index]) + abs(Train_UrineP[i] - Test_UrineP[patient_index]) +\
                                abs(Train_MictP[i] - Test_MictP[patient_index]) + abs(Train_BurningU[i] - Test_BurningU[patient_index]) + abs(Train_BladderI[i] - Test_BladderI[patient_index]))/6

        patient_matrix[i][3] = (patient_matrix[i][1] + (6 * patient_matrix[i][2] )  )/7

    patient_matrix = sort_matrix(patient_matrix, 3)


    return patient_matrix


def sort_matrix(patient_to_sort, disease_index):
    patient_sorted = patient_to_sort[np.argsort(patient_to_sort[:, disease_index])]
    return patient_sorted


def find_class(patient_ID, num_neighbour, patient_array, disease_index):
    no=0
    yes=0
    classification = None
    for i in range(num_neighbour):
        indexer = int(patient_array[i][0])-1
        if(patient_train[indexer][disease_index].upper() == "yes".upper()):
            yes = yes + 1
        else:
            no = no + 1

    if(yes > no):
        classification = "yes"
    else:
        classification = "no"
    numerrors = 0
    # print("patient test ", patient_test[patient_ID][disease_index])
    # print("class ", classification)
    if(patient_test[patient_ID-1][disease_index].upper() != classification.upper()):
        numerrors = 1
    else:
        numerrors = 0
    print("The classification for patient " + str(patient_ID) + " with " + str(num_neighbour) + " neighbours is " + str(classification))

    return numerrors



def solve_question(patient_ID, patient_matrix, disease_index):
    print("\nPatient " + str(patient_ID) + " is " + patient_test[patient_ID-1][disease_index])
    numerrors =0
    numerrors = numerrors + find_class(patient_ID, 3, patient_matrix, disease_index)
    numerrors = numerrors + find_class(patient_ID, 5, patient_matrix, disease_index)
    numerrors = numerrors + find_class(patient_ID, 7, patient_matrix, disease_index)

    print("There were " + str(numerrors) + " errors")

bladder_1 = bladder_inflammation_calculations(0)
bladder_2 = bladder_inflammation_calculations(1)
bladder_3 = bladder_inflammation_calculations(2)
bladder_4 = bladder_inflammation_calculations(3)
bladder_5 = bladder_inflammation_calculations(4)

nephritis_1 = nephritis_calculations(0)
nephritis_2 = nephritis_calculations(1)
nephritis_3 = nephritis_calculations(2)
nephritis_4 = nephritis_calculations(3)
nephritis_5 = nephritis_calculations(4)
print(" \n Question 2 \n")

print("**************** Bladder Results ******************* \n")
solve_question(1, bladder_1, 7)
solve_question(2, bladder_2, 7)
solve_question(3, bladder_3, 7)
solve_question(4, bladder_4, 7)
solve_question(5, bladder_5, 7)
print("\n**************** Nephritis Results ******************* \n")
solve_question(1, nephritis_1, 8)
solve_question(2, nephritis_2, 8)
solve_question(3, nephritis_3, 8)
solve_question(4, nephritis_4, 8)
solve_question(5, nephritis_5, 8)


#********************************************** QUESTION 3 **********************************************#

poker_train = np.loadtxt('poker-hand-training.txt', dtype= None, delimiter=',', encoding="utf-8-sig")
poker_test =  np.loadtxt('poker-hand-testing.txt', dtype= None, delimiter=',', encoding="utf-8-sig")


Min = 1
Max_num = 13
Max_suit = 4

poker_train = poker_train[0:505]
poker_test =  poker_test[0:505]



Train_Length = poker_train.shape[0]
Test_Length = poker_test.shape[0]



Hand_Train = np.zeros([Train_Length,10])
Hand_Test = np.zeros([Test_Length,10])
Hand_NumDistance = np.zeros([Train_Length,2])

def Numeric_Solution(Hand_Index, knn_num):
    for i in range(Train_Length):
        Hand_Train[i][0] = (poker_train[i][0] - Min) /(Max_suit - Min)
        Hand_Train[i][1] = (poker_train[i][1] - Min) /(Max_num - Min)
        Hand_Train[i][2] = (poker_train[i][2] - Min) /(Max_suit - Min)
        Hand_Train[i][3] = (poker_train[i][3] - Min) /(Max_num - Min)
        Hand_Train[i][4] = (poker_train[i][4] - Min) /(Max_suit - Min)
        Hand_Train[i][5] = (poker_train[i][5] - Min) /(Max_num - Min)
        Hand_Train[i][6] = (poker_train[i][6] - Min) /(Max_suit - Min)
        Hand_Train[i][7] = (poker_train[i][7] - Min) /(Max_num - Min)
        Hand_Train[i][8] = (poker_train[i][8] - Min) /(Max_suit - Min)
        Hand_Train[i][9] = (poker_train[i][9] - Min) /(Max_num - Min)

    for i in range(Test_Length):
        Hand_Test[i][0] = (poker_test[i][0] - Min) /(Max_suit - Min)
        Hand_Test[i][1] = (poker_test[i][1] - Min) /(Max_num - Min)
        Hand_Test[i][2] = (poker_test[i][2] - Min) /(Max_suit - Min)
        Hand_Test[i][3] = (poker_test[i][3] - Min) /(Max_num - Min)
        Hand_Test[i][4] = (poker_test[i][4] - Min) /(Max_suit - Min)
        Hand_Test[i][5] = (poker_test[i][5] - Min) /(Max_num - Min)
        Hand_Test[i][6] = (poker_test[i][6] - Min) /(Max_suit - Min)
        Hand_Test[i][7] = (poker_test[i][7] - Min) /(Max_num - Min)
        Hand_Test[i][8] = (poker_test[i][8] - Min) /(Max_suit - Min)
        Hand_Test[i][9] = (poker_test[i][9] - Min) /(Max_num - Min)




    for i in range(Train_Length):
        Hand_NumDistance[i][0] = i + 1
        Hand_NumDistance[i][1] = (abs(Hand_Train[i][0] - Hand_Test[Hand_Index][0]) + abs(Hand_Train[i][1] - Hand_Test[Hand_Index][1]) + \
                              abs(Hand_Train[i][2] - Hand_Test[Hand_Index][2]) + abs(Hand_Train[i][3] - Hand_Test[Hand_Index][3]) + \
                              abs(Hand_Train[i][4] - Hand_Test[Hand_Index][4]) + abs(Hand_Train[i][5] - Hand_Test[Hand_Index][5]) + \
                              abs(Hand_Train[i][6] - Hand_Test[Hand_Index][6]) + abs(Hand_Train[i][7] - Hand_Test[Hand_Index][7]) + \
                              abs(Hand_Train[i][8] - Hand_Test[Hand_Index][8]) + abs(Hand_Train[i][9] - Hand_Test[Hand_Index][9]))/10

    # # sort the dataA


    Hand_Sorted = Hand_NumDistance[np.argsort(Hand_NumDistance[:, 1])]
    # print(Hand_Sorted)
    hand_errors = 0
    #classify hand
    def classify_hand():
        hands = np.zeros(10)

        for i in range(knn_num):
            hand_indexer = int(Hand_Sorted[i][0]-1)
            for j in range(10):
                    if(poker_train[hand_indexer][10] == j):
                        hands[j] = hands[j] + 1

        #find maximum = find class
        hand_class = np.argmax(hands, axis =0)

        #find errors

        if(poker_test[Hand_Index][10] == hand_class):
            hand_errors = 0
        else:
            hand_errors = 1
        return hand_errors


    hand_errors_count = classify_hand()

    return hand_errors_count

Hand_NomTrain = np.zeros([Train_Length,10])
Hand_NomDistance = np.zeros([Train_Length,2])

def Nominal_Solution(Hand_Index, knn_num):

    for i in range(Train_Length):
        for j in range(10):
            Hand_NomTrain[i][j] = 1 if poker_train[i][j] == poker_test[Hand_Index][j] else 0

    for i in range(Train_Length):
        Hand_NomDistance[i][0]= i+1
        Hand_NomDistance[i][1] = Hand_NomDistance[i][1] + np.sum(Hand_NomTrain[i], axis =0)
        Hand_NomDistance[i][1] = Hand_NomDistance[i][1] / 10

    #sort it
    Hand_NomSorted = Hand_NomDistance[np.argsort(Hand_NomDistance[:, 1])]

    #classify
    def nom_classify_hand():
            nomhands = np.zeros(10)

            for i in range(knn_num):
                hand_indexer = int(Hand_NomSorted[i][0]-1)
                for j in range(10):
                        if(poker_train[hand_indexer][10] == j):
                            nomhands[j] = nomhands[j] + 1

            #find maximum = find class
            nomhand_class = np.argmax(nomhands, axis =0)

            #find errors

            if(poker_test[Hand_Index][10] == nomhand_class):
                hand_errors = 0
            else:
                hand_errors = 1
            return hand_errors

    nomhand_errors_count = nom_classify_hand()

    return nomhand_errors_count



def SolutionforNominalApproach():
    num_k5 = 0
    num_k25 = 0
    num_k45 = 0
    num_k65 = 0
    num_k85 = 0
    num_k105 = 0
    num_k205 = 0
    num_k305 = 0
    num_k405 = 0
    num_k505 = 0

    for i in range(Test_Length):
        num_k5 = num_k5 + Nominal_Solution(i, 5)

    for i in range(Test_Length):
        num_k25 = num_k25 + Nominal_Solution(i, 25)

    for i in range(Test_Length):
        num_k45 = num_k45 + Nominal_Solution(i, 45)

    for i in range(Test_Length):
        num_k65 = num_k65 + Nominal_Solution(i, 65)

    for i in range(Test_Length):
        num_k85 = num_k85 + Nominal_Solution(i, 85)

    for i in range(Test_Length):
        num_k105 = num_k105 + Nominal_Solution(i, 105)

    for i in range(Test_Length):
        num_k205 = num_k205 + Nominal_Solution(i, 205)

    for i in range(Test_Length):
        num_k305 = num_k305 + Nominal_Solution(i, 305)

    for i in range(Test_Length):
        num_k405 = num_k405 + Nominal_Solution(i, 405)

    for i in range(Test_Length):
        num_k505 = num_k505 + Nominal_Solution(i, 505)

    print("Number of errors for Nominal Approach where k is 5 is ", num_k5)
    print("Number of errors for Nominal Approach where k is 25 is ", num_k25)
    print("Number of errors for NominalApproach where k is 45 is ", num_k45)
    print("Number of errors for Nominal Approach where k is 65 is ", num_k65)
    print("Number of errors for Nominal Approach where k is 85 is ", num_k85)
    print("Number of errors for Nominal Approach where k is 105 is ", num_k105)
    print("Number of errors for Nominal Approach where k is 205 is ", num_k205)
    print("Number of errors for Nominal Approach where k is 305 is ", num_k305)
    print("Number of errors for Nominal Approach where k is 405 is ", num_k405)
    print("Number of errors for Nominal Approach where k is 505 is ", num_k505)



def SolutionforNumericApproach():
    num_k5 = 0
    num_k25 = 0
    num_k45 = 0
    num_k65 = 0
    num_k85 = 0
    num_k105 = 0
    num_k205 = 0
    num_k305 = 0
    num_k405 = 0
    num_k505 = 0


    for i in range(Test_Length):
        num_k5 = num_k5 + Numeric_Solution(i, 5)

    for i in range(Test_Length):
        num_k25 = num_k25 + Numeric_Solution(i, 25)

    for i in range(Test_Length):
        num_k45 = num_k45 + Numeric_Solution(i, 45)

    for i in range(Test_Length):
        num_k65 = num_k65 + Numeric_Solution(i, 65)

    for i in range(Test_Length):
        num_k85 = num_k85 + Numeric_Solution(i, 85)

    for i in range(Test_Length):
        num_k105 = num_k105 + Numeric_Solution(i, 105)

    for i in range(Test_Length):
        num_k205 = num_k205 + Numeric_Solution(i, 205)

    for i in range(Test_Length):
        num_k305 = num_k305 + Numeric_Solution(i, 305)

    for i in range(Test_Length):
        num_k405 = num_k405 + Numeric_Solution(i, 405)

    for i in range(Test_Length):
        num_k505 = num_k505 + Numeric_Solution(i, 505)

    print("Number of errors for Numeric Approach where k is 5 is ", num_k5)
    print("Number of errors for Numeric Approach where k is 25 is ", num_k25)
    print("Number of errors for Numeric Approach where k is 45 is ", num_k45)
    print("Number of errors for Numeric Approach where k is 65 is ", num_k65)
    print("Number of errors for Numeric Approach where k is 85 is ", num_k85)
    print("Number of errors for Numeric Approach where k is 105 is ", num_k105)
    print("Number of errors for Numeric Approach where k is 205 is ", num_k205)
    print("Number of errors for Numeric Approach where k is 305 is ", num_k305)
    print("Number of errors for Numeric Approach where k is 405 is ", num_k405)
    print("Number of errors for Numeric Approach where k is 505 is ", num_k505)

print("\n Question 3 \n")
SolutionforNumericApproach()
print(" ")
SolutionforNominalApproach()
