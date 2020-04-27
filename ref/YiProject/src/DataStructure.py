class SentenceObject:

    def __init__(self, sentence='', tabNum=0, blank=0, thisType='normal'):
        self.sentence = sentence
        self.tabNum = tabNum
        self.blank = blank
        self.thisType = thisType


class Network:

    def __init__(self, layerType, paraDict, name):
        self.type = 'nn.' + layerType
        self.paraDict = paraDict
        self.name = name

    def addName(self, name):
        self.name = name


class Layer:

    def __init__(self, name):
        self.name = name
        self.sequential = []

    def addNetwork(self, network):
        self.sequential.append(network)

    def toSentenceList(self):
        sentenceList = []
        initSentence = SentenceObject(self.name+' = nn.Sequential()', 2)
        sentenceList.append(initSentence)
        for network in self.sequential:
            sentence = self.name + '.add_module(\'' + network.name + '\', ' + network.type + '('
            for paraName in network.paraDict:
                sentence += paraName + '=' + str(network.paraDict[paraName]) + ', '
            sentence = sentence[:-2]
            sentence += '))'
            networkSentence = SentenceObject(sentence, 2)
            sentenceList.append(networkSentence)
        assignmentSentence = SentenceObject('self.'+self.name+' = '+self.name, 2, 1)
        sentenceList.append(assignmentSentence)

        return sentenceList


class Forward:

    def __init__(self):
        self.forwardRules = []

    def addRule(self, rule):
        self.forwardRules.append(rule)

    def toSentenceList(self):
        sentenceList = []
        initSentence = SentenceObject('def forward(self, x):', 1)
        sentenceList.append(initSentence)
        for rule in self.forwardRules:
            if rule['type'] == 'move':
                sentence = rule['name'] + ' = self.' + rule['layer'] + '(' + rule['target'] + ')'
                sentenceList.append(SentenceObject(sentence, 2))
            elif rule['type'] == 'reshape':
                sentence = rule['name'] + ' = ' + rule['target'] + '.view('
                paraList = rule['paraList']
                for para in paraList:
                    sentence += para + ', '
                sentence = sentence[:-2] + ")"
                sentenceList.append(SentenceObject(sentence, 2))
            elif rule['type'] == 'end':
                sentence = 'return ' + rule['name']
                sentenceList.append(SentenceObject(sentence, 2))
        return sentenceList


class Model:

    def __init__(self):
        self.layerList = []
        self.forward = Forward()

    def addLayer(self, layer):
        self.layerList.append(layer)

    def addForwardRule(self, rule):
        self.forward.addRule(rule)

    def toSentenceList(self):
        sentenceList = []
        for layer in self.layerList:
            sentenceList.extend(layer.toSentenceList())
        sentenceList.extend(self.forward.toSentenceList())
        return sentenceList
