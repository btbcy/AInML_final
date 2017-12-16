import pandas as pd
import sys

alldata1 = pd.read_csv('credit30000.csv')
alldata2 = pd.read_csv('credit30000.csv')
publictest = pd.read_csv('Test_Public.csv')
privatetest = pd.read_csv('Test_Private.csv')

# alldata = alldata.as_matrix()
# publictest = publictest.as_matrix()
# privatetest = privatetest.as_matrix()

header = list(publictest.head(1))[1:]
publicOutput = publictest.head(0)

for _, rowAAA in publictest.iterrows():
    isfound = True
    for idx, rowBBB in alldata1.iterrows():
        isfound = True
        for name in header:
            isfound = isfound and (rowAAA[name] == rowBBB[name])
            if not isfound:
                break
        if isfound:
            rowAAA['Y'] = rowBBB['Y']
            alldata1.drop(alldata1.index[[idx]])
            break
    if not isfound:
        rowAAA['Y'] = 'NaN'
    publicOutput = publicOutput.append(rowAAA, ignore_index=True)
    print "a ", _, rowAAA['Y']

publicOutput.to_csv(sys.argv[1], index=False)

privateOutput = privatetest.head(0)
for _, rowAAA in privatetest.iterrows():
    isfound = True
    for idx, rowBBB in alldata2.iterrows():
        isfound = True
        for name in header:
            isfound = isfound and (rowAAA[name] == rowBBB[name])
            if not isfound:
                break
        if isfound:
            rowAAA['Y'] = rowBBB['Y']
            alldata2.drop(alldata2.index[[idx]])
            break
    if not isfound:
        rowAAA['Y'] = 'NaN'
    privateOutput = privateOutput.append(rowAAA, ignore_index=True)
    print "b ", _, rowAAA['Y']

privateOutput.to_csv(sys.argv[2], index=False)
