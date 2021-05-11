import numpy as np
import matplotlib.pyplot as plt
import os
import help_laba
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def create_graphic(time_1, time_2, time_3, temp_1, temp_2, temp_3, name, n, del_H):
    print(name)
    a1, b1 = help_laba.LSM(time_1, temp_1)
    a3, b3 = help_laba.LSM(time_3, temp_3)
    plt.plot(np.linspace(0, 9, 5), np.linspace(0, 9, 5) * a1 + b1, '-', color='red', linewidth=0.5)
    plt.plot(np.linspace(7, time_3[len(time_3) - 1], 10), np.linspace(7, time_3[len(time_3) - 1], 10) * a3 + b3, '-', color='blue', linewidth=0.5)
    plt.plot(time_1, temp_1, '.', color='red', markersize=2)
    plt.plot(time_2, temp_2, '.', color='green', markersize=2)
    plt.plot(time_3, temp_3, '.', color='blue', markersize=2)
    plt.xlabel(r'Время, мин')
    plt.ylabel(r'Температура, $\degree$C')

    t1 = time_1[len(time_1) - 1]
    t2 = time_3[0]
    t_mean = (t2 - t1) / 2
    delta_t = (t1 + t_mean) * a1 + b1 - ((t1 + t_mean) * a3 + b3)
    print('delta t = ' + str(delta_t))

    plt.plot(np.linspace((t1 + t_mean), (t1 + t_mean), 10),
             np.linspace((t1 + t_mean) * a3 + b3, (t1 + t_mean) * a1 + b1, 10), '--', color='k', linewidth=1)
    plt.savefig('time temp ' + name + '.png')
    plt.show()
    plt.close()

    if name != 'соль':
        Q = n * del_H
        print('Q = ' + str(Q))

        K = Q/delta_t - 200 * 4.18 * 10 ** (-3)
        print('K = ' + str(K))
        return K
    else:
        return delta_t



time_2_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
time_2_2 = np.array([8 + 10/60, 8 + 20/60, 8 + 30/60, 8 + 40/60, 8 + 50/60, 9])
time_2_3 = np.array([9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15.5, 17, 18, 19, 20, 21, 22])

temp_2_1 = np.array([27, 27.1, 27.1, 27.15, 27.2, 27.25, 27.3, 27.35, 27.38])
temp_2_2 = np.array([26.9, 26.95, 26.9, 26.92, 26.93, 26.95])
temp_2_3 = np.array([27, 27, 27.03, 27.05, 27.1, 27.1, 27.15, 27.15, 27.2, 27.2, 27.22, 27.3, 27.38, 27.4, 27.5, 27.52, 27.6, 27.62])

time_4_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
time_4_2 = np.array([8 + 10/60, 8 + 20/60, 8 + 30/60, 8 + 40/60, 8 + 50/60, 9])
time_4_3 = np.array([9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18])

temp_4_1 = np.array([26.8, 26.9, 26.95, 27, 27.02, 27.1, 27.1, 27.1, 27.12])
temp_4_2 = np.array([26.3, 26.15, 26.12, 26.2, 26.2, 26.2])
temp_4_3 = np.array([26.2, 26.2, 26.25, 26.3, 26.35, 26.35, 26.4, 26.4, 26.45, 26.5, 26.55])

time_6_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
time_6_2 = np.array([8 + 10/60, 8 + 20/60, 8 + 30/60, 8 + 40/60, 8 + 50/60, 9])
time_6_3 = np.array([9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 17, 18, 19])

temp_6_1 = np.array([26.9, 27, 27.05, 27.1, 27.1, 27.15, 27.2, 27.2, 27.5])
temp_6_2 = np.array([26.1, 25.8, 25.7, 25.8, 25.8, 25.82])
temp_6_3 = np.array([25.85, 25.9, 25.9, 25.95, 25.98, 26, 26.05, 26.1, 26.12, 26.2, 26.22, 26.27, 26.32])

time_8_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
time_8_2 = np.array([7 + 10/60, 7 + 20/60, 7 + 30/60, 7 + 40/60, 7 + 50/60, 8])
time_8_3 = np.array([8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18])

temp_8_1 = np.array([26.9, 27, 27, 27.05, 27.07, 27.1, 27.12, 27.15])
temp_8_2 = np.array([25.5, 25.2, 25.11, 25.11, 25.15, 25.18])
temp_8_3 = np.array([25.2, 25.21, 25.25, 25.3, 25.32, 25.33, 25.4, 25.42, 25.5, 25.5, 25.55, 25.6, 25.68])

time_10_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
time_10_2 = np.array([7 + 10/60, 7 + 20/60, 7 + 30/60, 7 + 40/60, 7 + 50/60, 8])
time_10_3 = np.array([8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17])

temp_10_1 = np.array([26.9, 26.92, 26.98, 27, 27, 27.02, 27.08, 27.08])
temp_10_2 = np.array([25.2, 24.6, 24.52, 24.55, 24.57, 24.6])
temp_10_3 = np.array([24.61, 24.63, 24.7, 24.7, 24.75, 24.8, 24.85, 24.9, 24.95, 25, 25.2])

K = np.zeros(5)

K[0] = create_graphic(time_2_1, time_2_2, time_2_3, temp_2_1, temp_2_2, temp_2_3, '2г', 2/74.5, 17.55)
K[1] = create_graphic(time_4_1, time_4_2, time_4_3, temp_4_1, temp_4_2, temp_4_3, '4г', 4/74.5, 17.57)
K[2] = create_graphic(time_6_1, time_6_2, time_6_3, temp_6_1, temp_6_2, temp_6_3, '6г', 6/75.5, 17.50)
K[3] = create_graphic(time_8_1, time_8_2, time_8_3, temp_8_1, temp_8_2, temp_8_3, '8г', 8/74.5, 17.43)
K[4] = create_graphic(time_10_1, time_10_2, time_10_3, temp_10_1, temp_10_2, temp_10_3, '10г', 10/74.5, 17.40)

print('K mean = ' + str(np.mean(K)) + '+-' + str(np.std(K)))

n = np.array([2/74.5, 4/74.5, 6/75.5, 8/74.5, 10/74.5])
print('n = ' + str(n))

m_kcl = n * 1000 / 200
print('m kcl = ' + str(m_kcl))

time_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
time_2 = np.array([7 + 10/60, 7 + 20/60, 7 + 30/60, 7 + 40/60, 7 + 50/60, 8])
time_3 = np.array([8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17])

temp_1 = np.array([26.9, 26.9, 26.9, 26.91, 26.93, 26.96, 26.98, 27])
temp_2 = np.array([25.8, 25.78, 25.78, 25.78, 25.8, 25.8])
temp_3 = np.array([25.8, 25.82, 25.85, 25.88, 25.9, 25.9, 25.92, 25.98, 26, 26.01, 26.03, 26.08])

delta_t = create_graphic(time_1, time_2, time_3, temp_1, temp_2, temp_3, 'соль', 0, 0)

Q_x = (np.mean(K) + 200 * 4.18 * 10 ** (-3)) * delta_t
print('Qx = ' + str(Q_x))

q_x = Q_x / 4
print('q_x = ' + str(q_x))

del_H_x = q_x * 53.5  # NH4Cl
print('del_H_x = ' + str(del_H_x))

n_x = 4 / 53.5
print('n_x = ' + str(n_x))

m_x = n_x * 1000 / 200  # по таблице 15.23
print('m_x = ' + str(m_x))

delta = abs(15.27 - del_H_x) / 15.27 * 100
print('error = ' + str(delta) + '%')