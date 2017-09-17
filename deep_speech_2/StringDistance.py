def stringDistance(lst1, lst2):
    len1 = len(lst1) + 1
    len2 = len(lst2) + 1
    costM = [[0 for y in range(len2)] for x in range(len1)]


    for i in range(len1):
        costM[i][0] = i

    for j in range(len2):
        costM[0][j] = j

    substitutionCost = 0
    for j in range(1, len2):
        for i in range(1, len1):
            if lst1[i-1] == lst2[j-1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            costM[i][j] = min(costM[i-1][j] + 1,                   # deletion
                             costM[i][j-1] + 1,                   # insertion
                             costM[i-1][j-1] + substitutionCost)  # substitution

    return costM

#real data is lst1, test data is lst 2
def errorTrack(lst1, lst2, costM):
    ind1, ind2 = len(lst1), len(lst2)
    cost = costM[ind1][ind2]
    result = cost
    while(ind1 >= 0 and ind2 >= 0):
        if(costM[ind1 - 1][ind2 - 1] == cost - 1):
            #print("Incorrectly substituted " + lst2[ind2 - 1] + " for " + lst1[ind1 - 1])
            ind1, ind2 = ind1 - 1, ind2 - 1
            cost -= 1
            #Substitution matrix edit
        elif(costM[ind1 - 1][ind2] == cost - 1):
            #print("Deleted " + lst1[ind1 - 1] + " incorrectly")
            ind1 -= 1
            cost -= 1
            #Deletion vector edit
        elif(costM[ind1][ind2 - 1] == cost - 1):
            #print("Inserted " + lst2[ind2 - 1] + " incorrectly")
            ind2 -= 1
            cost -= 1
            #Insertion vector edit
        else:
            ind1, ind2 = ind1 - 1, ind2 - 1
            #No change/ Match
    return result
