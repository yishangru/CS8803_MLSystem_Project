class GlobalManager(object):
    def __init__(self, name: str = "GlobalManager"):
        super(GlobalManager, self).__init__()
        self.name = name
        self.block_id = 0
        self.node_id = 0

    def get_block_id(self):
        block_id_return = self.block_id
        self.block_id += 1
        return block_id_return

    def get_node_id(self):
        node_id_return = self.node_id
        self.node_id += 1
        return node_id_return