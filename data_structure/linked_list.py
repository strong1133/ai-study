class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if self.head is None:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)

    def show(self):
        current = self.head
        while current:
            print(current.data, end=" ")
            current = current.next

    def search(self, index):
        node = self.head
        for _ in range(index):
            node = node.next
        return node

    def insert(self, index, data):
        new = Node(data)

        if index == 0:
            new.next = self.head
            self.head = new
            return
        node = self.search(index - 1)
        next = node.next
        node.next = new
        new.next = next

    def delete(self, index):
        if index == 0:
            self.head = self.head.next
            return

        front = self.search(index - 1)
        front.next = front.next.next


linked_list = LinkedList()
data_list = [3, 5, 9, 8, 5, 6, 1, 7]
for data in data_list:
    linked_list.append(data)

print("전체 노드 출력:", end=" ")
linked_list.show()

linked_list.insert(4, 4)
print("\n전체 노드 출력:", end=" ")
linked_list.show()

linked_list.delete(7)
print("\n전체 노드 출력:", end=" ")
linked_list.show()
