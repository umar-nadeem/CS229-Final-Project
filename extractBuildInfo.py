import os
import csv
import pandas as pd
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# df = pd.DataFrame({"ID": [],
#                   "prefix": [],
#                   "type": []})
rowList = [["ID", "prefix", "type"]]

def runCommands(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # os.system(f"echo Hello {name}")
    os.chdir("/Users/sarahraza/bazel")
    # os.system("ls")
    # print("'cd /Users/sarahraza/bazel' ran with exit code %d" % bazel_dir)
    os.system(f"git checkout HEAD~1")
    id = os.popen("git rev-parse --short HEAD").read()
    gitDiff = os.popen("git diff HEAD~1..HEAD --name-only").read()
    prefix = getPrefix(gitDiff)
    type = getType(gitDiff)
    # df3 = pd.DataFrame({"ID": [id],
    #                   "prefix": [prefix],
    #                    "type": [type]})
    # global df
    # df.append(df3, ignore_index=True)
    rowList.append([id, prefix, type])
    print(rowList)
    os.system(f"bazel build //src:bazel-dev --build_event_json_file=/Users/sarahraza/Desktop/Build_Info/{name}_commit.json") #only local locations work?


def getPrefix(gitDiff):
    """"
    prefix = ""
    slash = False
    for letter in gitDiff:
        if letter == '/':
            if not slash:
                slash = True
            else:
                break
        prefix += letter
    return prefix
    """
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nCommits = 2
    for i in range(1, nCommits):
        runCommands(i)
    # readJSONtoCSV(nCommits)
    # global df
    # df.to_csv(r'/Users/sarahraza/Desktop/data.csv', index=False, header=True)
    with open('InputData.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rowList)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
