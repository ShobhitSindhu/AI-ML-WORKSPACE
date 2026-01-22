dict = []

class employee:
    def __init__(self,id,name,salary):
        self.__id = id
        self.__name = name
        self.__salary = salary
    
    def put_in(self):
        dict.append({"sal":self.__salary,"id":self.__id,"name":self.__name})
e1 = employee("1","shobhit","2000")
e2 = employee("2","sindhu","3000")
e1.put_in()
e2.put_in()