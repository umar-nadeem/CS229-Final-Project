import os
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def runCommands(name):
    os.chdir("/Users/muhammadumarnadeem/bazel")


if __name__ == '__main__':
    nCommits = 2
    for i in range(1, nCommits):
        runCommands(i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
