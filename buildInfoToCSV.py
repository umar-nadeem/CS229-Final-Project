import json
import csv
import pandas as pd

# initialize data list with labels
rowList = [["commitID", "CPUTime"]]

def extractData(commitID):
    # read json into panda dataframe
    filepath = "/Users/muhammadumarnadeem/CS229-Final-Project/Build_Info/" + str(commitID) + "_commit.json"
    df = pd.read_json(filepath, lines=True)
    
    # extract cpuTime from dataframe
    buildMetricColumn = df.get('buildMetrics')
    buildMetricRow = buildMetricColumn.dropna()
    actions = buildMetricRow.iloc[0]
    timingInfo = actions['timingMetrics']
    cpuTime = timingInfo['cpuTimeInMs']
    
    # append data to list
    rowList.append([commitID, cpuTime])

if __name__ == '__main__':
    #extract CPU Rumtimes from json files
    for i in range(1,2):
        extractData(i)
    for i in range(3,20):
        extractData(i)
    
    # Write CPU Rumtimes into csv file
    with open('CPUTimes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rowList)

