import os
import csv

rowList = [["ID", "prefix", "type"]]

def getPrefix(gitDiff):
    dict = {}
    dict["bazelci/"] = gitDiff.count("bazelci/")
    dict["examples/"] = gitDiff.count("examples/")
    dict["scripts/"] = gitDiff.count("scripts/")
    dict["site/"] = gitDiff.count("site/")
    dict["src/conditions"] = gitDiff.count("src/conditions")
    dict["src/java_tools"] = gitDiff.count("src/java_tools")
    dict["src/main"] = gitDiff.count("src/main")
    dict["src/test"] = gitDiff.count("src/test")
    dict["src/tools"] = gitDiff.count("src/tools")
    dict["third_party/"] = gitDiff.count("third_party/")
    dict["tools/"] = gitDiff.count("tools/")
    
    # add selection between two equivalent counts
    return max(dict, key=dict.get)

def getType(gitDiff):
    dict = {}
    dict["JAVA"] = gitDiff.count(".java")
    dict["C/C++"] = gitDiff.count(".c")
    dict["C/C++"] += gitDiff.count(".h")
    dict["C/C++"] += gitDiff.count(".cc")
    dict["C/C++"] += gitDiff.count(".cpp")
    dict["Starlark"] = gitDiff.count("BUILD")
    dict["Starlark"] += gitDiff.count(".bazel")
    dict["Starlark"] += gitDiff.count("WORKSPACE")
    dict["Starlark"] += gitDiff.count("*.bzl")
    dict["python"] = gitDiff.count(".py")
    dict["HTML/CSS/JS"] = gitDiff.count(".html")
    dict["HTML/CSS/JS"] += gitDiff.count(".css")
    dict["HTML/CSS/JS"] += gitDiff.count(".js")
    
    # add other category for file types
    return max(dict, key=dict.get)

def parseCommits(name):
    # move to bazel project
    os.chdir("/Users/muhammadumarnadeem/bazel")

    # obtain information about last commit
    os.system(f"git checkout HEAD~1")
    id = os.popen("git rev-parse --short HEAD").read()
    idLength = len(id)
    id = id[1:idLength-1]
    gitDiff = os.popen("git diff HEAD~1..HEAD --name-only").read()
    prefix = getPrefix(gitDiff)
    type = getType(gitDiff)

    # append commit information to list
    rowList.append([id, prefix, type])
    print(rowList)

    # pull JSON files from bazel build info
    os.system(f"bazel build //src:bazel-dev --build_event_json_file=/Users/muhammadumarnadeem/CS229-Final-Project/Build_Info/{name}_commit.json")
    
    # move back to local project
    os.chdir("/Users/muhammadumarnadeem/CS229-Final-Project")


if __name__ == '__main__':
    # extract JSON files (for CPUTimes) and InputData from Commits
    numCommits = 101
    for i in range(1, numCommits):
        parseCommits(i)
    
    # Write InputData into csv file
    with open('InputData.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rowList)
