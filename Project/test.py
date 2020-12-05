l = [1, 2, 3]
l2 = [1, 2, 3]

for x, y in zip(l, l2):
    print('x:', x, 'y:', y)

print()

for x in l:
    for y in l2:
        print('x:', x, 'y:', y)
