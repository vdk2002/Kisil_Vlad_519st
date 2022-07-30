import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from scipy import integrate
from scipy.stats import chi2

sample = 256
random_list = []
for i in range(sample):
    random_list.append(-((30 + (900 + 40 * random.random()) ** (1 / 2)) / 4))
print(random_list)

plt.plot(random_list)
plt.show()
plt.plot(random_list, "o")
plt.show()
arrange = np.linspace(-22.34, -22.18, endpoint=True, num=50)
count, bins, ignored = plt.hist(random_list, 10, edgecolor='k', density=True, label='Гистограма шуму')
plt.plot(arrange, (53 / 3) * np.sqrt(arrange - (-22.34)), linewidth=2, color='r',label='Теоретична щільність розподілення шуму')
plt.legend()
plt.show()


print(str(np.mean(random_list)), "- выборочное среднее для выборки")
print(str(np.var(random_list)), "- дисперсия для выборки")


count2, bins2, ignored2 = plt.hist(random_list, 10, edgecolor='k', density=True, )
result = 0
Pk = []
Nk = count2


def f(x):
    return (0.8 * x) + 0.6


for interval in range(len(bins2)):
    if interval + 1 == len(bins2):
        break

    integral, err = scipy.integrate.quad(f, bins2[interval], bins2[interval + 1])
    Pk.append(integral)

for i in range(len(count2)):
    result += (sample * Pk[i] - Nk[i]) ** 2 / sample * Pk[i]

print("Pirson - " + str(result))

i = 7
reference_signal = np.loadtxt('Signal_Values_second_part.csv', usecols=i, delimiter=';')

M = 10
L = 2
sigm = M - L - 1
print("Кол-во степеней свободы: " + str(sigm))
print(1 - chi2.cdf(result, sigm))

c = 0.03
x = []
testList = []
n = random_list
for i in range(len(n)):
    x.append(reference_signal[i] + c * n[i])
    testList.append(reference_signal[i] + c * n[i])

plt.plot(reference_signal)
plt.show()
plt.plot(x)
plt.show()

w = [3, 5, 7, 9, 11]
result_line = [[], [], [], [], []]
result_median = [[], [], [], [], []]
index_append = 0

for i in w:
    window = int((i - 1) / 2)
    windowList = []
    for _ in range(window):
        testList.insert(0, x[0])
        testList.insert(len(testList), x[-1])
    for f in range(len(x)):
        for _ in range(i):
            windowList.append(testList[f])
            f += 1
        result_line[index_append].append(sum(windowList) / len(windowList))
        windowList.sort()
        result_median[index_append].append(windowList[math.floor(len(windowList) / 2)])
        windowList.clear()
    index_append += 1

for i in range(5):
    plt.figure()
    plt.plot(result_line[i])
plt.show()

for i in range(5):
    plt.figure()
    plt.plot(result_median[i])
plt.show()

error = []
for i in range(5):
    result_median_error = 0
    result_line_error = 0
    for f in range(len(x)):
        result_median_error += (x[f] - result_median[i][f])**2
        result_line_error += (x[f] - result_line[i][f]) ** 2
    result_median_error = result_median_error/len(x)
    result_line_error = result_line_error/len(x)
    print("Среднеквадартическая ошибка(Линейный фильтр) при w = " + str(w[i]) + ": " + str(result_line_error))
    print("Среднеквадартическая ошибка(медианный фильтр) при w = " + str(w[i]) + ": " + str(result_median_error))

