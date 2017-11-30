def avpre(data, k):
    sumY = 0.0
    sumAvY = 0.0
    for i in range(k):
        sumY += data[i][-1]
        sumAvY += sumY / (i+1)
    return sumAvY / k

def partial(data, num):
    sumY = 0.0
    for i in range(num):
        sumY += data[i][-1]
    return sumY / num

def transfer(data, hasY = False):
    data_x = data[:]
    data_y = []
    for index in range(len(data_x)):
        if hasY:
            data_y.append(data_x[index][-1])
            del data_x[index][-1]
        # del data_x[index][17] # Bill_ATM_6
        # del data_x[index][16] # Bill_ATM_5
        # del data_x[index][15] # Bill_ATM_4
        # del data_x[index][14] # Bill_ATM_3
        # del data_x[index][5]  # AGE
        # del data_x[index][4]  # Marriage
        # del data_x[index][3]  # Edu
        # del data_x[index][2]  # SEX
        del data_x[index][0]  # ID
    return data_x, data_y
