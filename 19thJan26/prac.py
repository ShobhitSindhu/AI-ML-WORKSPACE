
def even_odd(num):
    if(num%2==0):
        print("Even")
    elif(num%2!=0):
        print("Odd")

num = int(input("Enter num: "))
even_odd(num)
print("------------------------------------------")
'''____________________________________________________________________________________________________'''


employee_dataset = []
employee_ids = set()  

class Employee:
    def __init__(self, id, name, salary):
        self.id = id
        self.name = name
        self.salary = salary

    def show_info(self):
        if self.id in employee_ids:
            print(f"Employee with id {self.id} already exists")
        else:
            employee_dataset.append({
                "id": self.id,
                "name": self.name,
                "salary": self.salary
            })
            employee_ids.add(self.id)

e1 = Employee("12", "shobhit", "20lpa")
e2 = Employee("13", "shobhit1", "20lpa")
e3 = Employee("14", "shobhit2", "20lpa")
e4 = Employee("14", "shobhit3", "20lpa")

e1.show_info()
e2.show_info()
e3.show_info()
e4.show_info()
print("------------------------------------------")


'''____________________________________________________________________________________________________'''

  





employee_dataset = []
employee_ids = set()

class Employee:
    def __init__(self, id, name, salary):
        self.__id = id
        self.__name = name
        self.__salary = salary

    def show_details(self):
        print(f"ID: {self.__id}")
        print(f"Name: {self.__name}")
        print(f"Salary: {self.__salary}")

class Manager(Employee):
    def __init__(self, id, name, salary, no_of_teams):
        super().__init__(id, name, salary)
        self.__no_of_teams = no_of_teams

    def show_details(self):
        super().show_details()
        print(f"No of Teams: {self.__no_of_teams}")

mgr = Manager(101, "abc", 222, 5)
mgr.show_details()
print("------------------------------------------")

'''__________________________________________________________________________________'''

from abc import ABC, abstractmethod

class Electronics(ABC):
    @abstractmethod
    def play_video():
        pass
class Laptop(Electronics):
    def play_video(self):
        print("Playing the video in Laptop")
class mobile(Electronics):
    def play_video(self):
        print("Playing video in the mobile")
lap = Laptop()
mob = mobile()
lap.play_video()
mob.play_video()
print("------------------------------------------")
'''__________________________________________________________________________________'''
'''Association'''
class Marks:
    def __init__(self, subject, score):
        self.subject = subject
        self.score = score

    def display(self):
        print(f"{self.subject}: {self.score}")

class Student:
    def __init__(self, name):
        self.name = name
        self.marks = []   

    def add_marks(self, marks):
        self.marks.append(marks)

    def show_marks(self):
        print(f"Marks of {self.name}")
        for m in self.marks:
            m.display()

m1 = Marks("Maths", 90)
m2 = Marks("Physics", 85)

s = Student("Shobhit")
s.add_marks(m1)
s.add_marks(m2)

s.show_marks()
print("------------------------------------------")
'''---------------------------------------------------------------------'''

class Student:
    def __init__(self, id, name, total_marks):
        self.__id = id
        self.__name = name
        self.__total_marks = total_marks


class Teacher:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def student_strength(self):
        return len(self.students)

s1 = Student(1, "A", 90)
s2 = Student(2, "R", 85)

t1 = Teacher(101, "Teacher1")

t1.add_student(s1)
t1.add_student(s2)

print("Student strength:", t1.student_strength())

print("------------------------------------------")

'''---------------------------------------------------------------------'''

class Student:
    def __init__(self, id, name, total_marks):
        self.__id = id
        self.__name = name
        self.__total_marks = total_marks

    def get_marks(self):
        return self.__total_marks

    def get_name(self):
        return self.__name


class Teacher:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def student_strength(self):
        return len(self.students)

    def sort_students_by_marks(self):
        n = len(self.students)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.students[j].get_marks() < self.students[j + 1].get_marks():
                    self.students[j], self.students[j + 1] = self.students[j + 1], self.students[j]

        names = []
        for student in self.students:
            names.append(student.get_name())

        return names

s1 = Student(1, "A", 90)
s2 = Student(2, "R", 85)
s3 = Student(3, "K", 55)
s4 = Student(4, "M", 40)
s5 = Student(5,"l",19)

t1 = Teacher(101, "Teacher1")

t1.add_student(s1)
t1.add_student(s2)
t1.add_student(s3)
t1.add_student(s4)
t1.add_student(s5)

print("Student strength:", t1.student_strength())
print("Sorted students by marks:", t1.sort_students_by_marks())

cnt1=0
for student in t1.students:
    marks = student.get_marks()

    if marks >= 60:
        print(f"1st Division : {student.get_name()}")
        cnt1+=1
    elif marks >= 45:
        print(f"2nd Division : {student.get_name()}")
    elif marks >= 30:
        print(f"3rd Division : {student.get_name()}")
    else:
        print(f"Fail : {student.get_name()}")

print(f"Total students who got distinction: {cnt1}")

