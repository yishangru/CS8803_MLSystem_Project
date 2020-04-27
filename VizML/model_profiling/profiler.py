"""
Input:
dict -> list ->
{"tensor":
     {"mem":
          {"max": 0, "min": float("inf")},
      "grad":
          {"max": 0, "min": float("inf")}
      },
 "layer":
     {"mem":
          {"max": 0, "min": float("inf")},
      "grad":
          {"max": 0, "min": float("inf")}
      }
}

"""
import os
import collections
import matplotlib.pyplot as plt

class BlockProfiler(object):
    def __init__(self, generationPath: str):
        self.generationPath = generationPath

    def generateBlockImage(self, blockMemDict):
        markerCandidate = ["o", "v", "s", "p", "X", "*", "P", "H", "d"]
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')

        for blockMem in blockMemDict.keys(): # just for max and min
            metricDict = collections.defaultdict(lambda: list())
            blockMemList = blockMemDict[blockMem]
            for i, blockMemEpochDict in enumerate(blockMemList):
                for target in blockMemEpochDict.keys():
                    for metric in blockMemEpochDict[target].keys():
                        for meta in blockMemEpochDict[target][metric].keys():
                            metricDict[target + "_" + metric + "_" + meta].append(blockMemEpochDict[target][metric][meta])
            iterationRange = [i + 1 for i in range(len(blockMemList))]
            counter = 0
            for target_metric_meta in metricDict.keys():
                plt.plot(iterationRange, metricDict[target_metric_meta], color=palette((counter + 1)), linestyle='dashed',
                         linewidth=2, marker=markerCandidate[counter], markersize=6, label=target_metric_meta)
                counter += 1

            #plt.xscale('log')
            plt.title(blockMem + " profiling", loc='center', fontsize=15, fontweight=100, color='orange')
            plt.xlabel("Epochs")
            plt.ylabel("KB")
            plt.legend(loc=1, ncol=2, prop={'size': 12})
            plt.savefig(os.path.join(self.generationPath, blockMem + '_profile.png'), dpi=300)

"""
blockMemDict = {"block1": [{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}},
{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}},
{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}},
{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}},
{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}},
{"tensor": {"mem": {"max": 20, "min": 20}, "grad": {"max": 40, "min": 20}},  "layer":{"mem": {"max": 50, "min": 20}, "grad": {"max": 40, "min": 20}}}]}

profiler = BlockProfiler("./")
profiler.generateBlockImage(blockMemDict)
"""