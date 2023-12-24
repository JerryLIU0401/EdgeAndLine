import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
labels = ['Radius 1', 'Radius 2', 'Radius 3', 'Radius 4', 'Radius 5']
y = [i**2 for i in x]

plt.plot(x, y)
plt.xticks(x, labels, rotation=45)
# plt.margins(x=0.1, y=0.5)
plt.margins(x=2, y=2)
plt.show()
