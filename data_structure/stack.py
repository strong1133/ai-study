class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.isEmpty():
            return None
        return self.stack.pop()

    def top(self):
        if self.isEmpty():
            return None
        return self.stack[-1]

    def isEmpty(self):
        return len(self.stack) == 0


stack = Stack()
arr = [9, 7, 2, 5, 6, 4, 2]
for i in arr:
    stack.push(i)

while not stack.isEmpty():
    print(stack.pop())
