"""
Input:

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

class BlockProfiler(object):
    def __init__(self, generationPath: str):
        self.generationPath = generationPath

    def generateBlockImage(self, blockMemDict):
