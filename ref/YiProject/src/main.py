from DataStructure import Network, Layer, Model
from utils import writeFile, copyFile


if __name__ == '__main__':
    if __name__ == '__main__':
        copyFile('../template.txt', './CNN.py')
        str1 = {'name': 'layer1',
                'networks': {
                    'conv1': {
                        'type': 'Conv2d',
                        'paraDict': {
                            'in_channels': 3,
                            'out_channels': 32,
                            'kernel_size': 3,
                            'stride': 1,
                            'padding': 1
                            }
                        },
                    'relu1': {
                        'type': 'ReLU',
                        'paraDict': {
                            'inplace': 'True'
                            }
                        },
                    'pool1': {
                        'type': 'MaxPool2d',
                        'paraDict': {
                            'kernel_size': '2',
                            'stride': '2'
                            }
                        }
                    }
                }

        str2 = {'name': 'layer2',
                'networks': {
                    'conv2': {
                        'type': 'Conv2d',
                        'paraDict': {
                            'in_channels': 32,
                            'out_channels': 64,
                            'kernel_size': 3,
                            'stride': 1,
                            'padding': 1
                            }
                        },
                    'relu2': {
                        'type': 'ReLU',
                        'paraDict': {
                            'inplace': 'True'
                            }
                        },
                    'pool2': {
                        'type': 'MaxPool2d',
                        'paraDict': {
                            'kernel_size': '2',
                            'stride': '2'
                            }
                        }
                    }
                }

        str3 = {'name': 'layer3',
                'networks': {
                    'conv3': {
                        'type': 'Conv2d',
                        'paraDict': {
                            'in_channels': 64,
                            'out_channels': 128,
                            'kernel_size': 3,
                            'stride': 1,
                            'padding': 1
                            }
                        },
                    'relu3': {
                        'type': 'ReLU',
                        'paraDict': {
                            'inplace': 'True'
                            }
                        },
                    'pool3': {
                        'type': 'MaxPool2d',
                        'paraDict': {
                            'kernel_size': '2',
                            'stride': '2'
                            }
                        }
                    }
                }

        str4 = {'name': 'layer4',
                'networks': {
                    'fc1': {
                        'type': 'Linear',
                        'paraDict': {
                            'in_features': 2048,
                            'out_features': 512,
                        }
                    },
                    'fc_relu1': {
                        'type': 'ReLU',
                        'paraDict': {
                            'inplace': 'True'
                        }
                    },
                    'fc2': {
                        'type': 'Linear',
                        'paraDict': {
                            'in_features': 512,
                            'out_features': 64,
                        }
                    },
                    'fc_relu2': {
                        'type': 'ReLU',
                        'paraDict': {
                            'inplace': 'True'
                        }
                    },
                    'fc3': {
                        'type': 'Linear',
                        'paraDict': {
                            'in_features': 64,
                            'out_features': 10,
                            }
                        },
                    }
                }

        str5 = {'name': 'forward',
                'rule1': {
                    'name': 'conv1',
                    'target': 'x',
                    'layer': 'layer1',
                    'type': 'move',
                    'paraList': []
                    },
                'rule2': {
                    'name': 'conv2',
                    'target': 'conv1',
                    'layer': 'layer2',
                    'type': 'move',
                    'paraList': []
                    },
                'rule3': {
                    'name': 'conv3',
                    'target': 'conv2',
                    'layer': 'layer3',
                    'type': 'move',
                    'paraList': []
                    },
                'rule4': {
                    'name': 'fc_input',
                    'target': 'conv3',
                    'layer': '',
                    'type': 'reshape',
                    'paraList': ['conv3.size(0)', '-1']
                    },
                'rule5': {
                    'name': 'fc_out',
                    'target': 'fc_input',
                    'layer': 'layer4',
                    'type': 'move',
                    'paraList': []
                    },
                'rule6': {
                    'name': 'fc_out',
                    'target': '',
                    'layer': '',
                    'type': 'end',
                    'paraList': []
                    },
                }

        CNN = Model()
        layer1 = Layer(str1['name'])
        networks = str1['networks']
        for networkName in networks:
            jsonNetWork = networks[networkName]
            network = Network(jsonNetWork['type'], jsonNetWork['paraDict'], networkName)
            layer1.addNetwork(network)
        CNN.addLayer(layer1)

        layer2 = Layer(str2['name'])
        networks = str2['networks']
        for networkName in networks:
            jsonNetWork = networks[networkName]
            network = Network(jsonNetWork['type'], jsonNetWork['paraDict'], networkName)
            layer2.addNetwork(network)
        CNN.addLayer(layer2)

        layer3 = Layer(str3['name'])
        networks = str3['networks']
        for networkName in networks:
            jsonNetWork = networks[networkName]
            network = Network(jsonNetWork['type'], jsonNetWork['paraDict'], networkName)
            layer3.addNetwork(network)
        CNN.addLayer(layer3)

        layer4 = Layer(str4['name'])
        networks = str4['networks']
        for networkName in networks:
            jsonNetWork = networks[networkName]
            network = Network(jsonNetWork['type'], jsonNetWork['paraDict'], networkName)
            layer4.addNetwork(network)
        CNN.addLayer(layer4)

        for rule in str5:
            if rule != 'name':
                CNN.addForwardRule(str5[rule])

        sentenceList = CNN.toSentenceList()
        writeFile('./CNN.py', sentenceList)
