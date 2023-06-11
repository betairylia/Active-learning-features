import os

class Test():
    
    def __init__(self):
        self.isu = False
    
    def getu(self):
        return self.isu
    
a = Test()
foo = a.getu

a.isu = True

print(foo)
print(foo())
