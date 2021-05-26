import os
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def runCommands(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # os.system(f"echo Hello {name}")
    os.chdir("/Users/muhammadumarnadeem/bazel")
    # os.system("ls")
    # print("'cd /Users/muhammadumarnadeem/bazel' ran with exit code %d" % bazel_dir)
    os.system(f"git checkout Head~1")
    os.system(f"bazel build //src:bazel-dev --build_event_json_file=/Users/muhammadumarnadeem/Desktop/Build_Info/{name}_commit.json")

"""
def readJSONtoCSV(nCommits):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # os.system(f"echo Hello {name}")
    os.chdir("/Users/muhammadumarnadeem/Desktop/Build_Info")
    for j in range(1, nCommits):
        f = open(f"{j}_commit.json")
        data = json.load(f)
        for i in data ["id"]:
            print(i)

    f.close()
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nCommits = 2
    for i in range(1, nCommits):
        runCommands(i)
    # readJSONtoCSV(nCommits)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
