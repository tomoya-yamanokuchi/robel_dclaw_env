from pprint import pprint

def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


a = [0, 1, 2, 3, 4, 5, 6, 7]

b = split_list(a, 3)
print(list(b))


