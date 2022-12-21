class BinaryPathTree:
    def __init__(self):
        self.count = 0
        self.active = None
        self.inactive = None
        
    def __len__(self):
        return self.count
    
    def add(self, path):
        if path==None:
            return 
            
        active = path[0]
        
        if len(path)==1:
            path = None
        else:
            path = path[1:]
        
        if active==1:
            if self.active == None:
                self.active = BinaryPathTree()
            
            self.active.add(path)
        else:
            if self.inactive == None:
                self.inactive = BinaryPathTree()
            
            self.inactive.add(path)
            
        self.count += 1

    def get_number_of_leafs(self):
        if self.active==None and self.inactive==None:
            return 1
        else:
            def get_count(node):
                if node == None:
                    return 0
            
                return node.get_number_of_leafs()
                
            left_count = get_count(self.active)
            right_count = get_count(self.inactive)
            
            return left_count + right_count