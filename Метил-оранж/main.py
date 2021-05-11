import numpy as np
import matplotlib.pyplot as plt
import os
import help_laba

from math import log

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def import_file(file):
    file_str = file.read().splitlines()
    lambda_wave = []
    D = []
    for i in range(len(file_str)):
        lambda_wave.append(float(file_str[i].split()[0]))
        D.append(float(file_str[i].split()[1]))
    return lambda_wave, D

def plot_work_wave(wave1, D1, wave6, D6, wave9, D9, wave10, D10, wave11, D11):  # 1, 6, 9 - 11
    plt.plot(wave1, D1, label='Раствор 1', linewidth=0.8)
    plt.plot(wave6, D6, label='Раствор 6', linewidth=0.8)
    plt.plot(wave9, D9, label='Раствор 9', linewidth=0.8)
    plt.plot(wave10, D10, label='Раствор 10', linewidth=0.8)
    plt.plot(wave11, D11, label='Раствор 11', linewidth=0.8)
    plt.plot(np.linspace(530, 530, 10), np.linspace(0, 1.2, 10), '--', label='$\lambda_1$ = 530 нм', linewidth=0.8)
    plt.plot(np.linspace(438, 438, 10), np.linspace(0, 1.2, 10), '--', label='$\lambda_2$ = 438 нм', linewidth=0.8)
    plt.plot(np.linspace(468, 468, 10), np.linspace(0, 1.2, 10), '--', label='$\lambda^*$ = 468 нм', linewidth=0.8)
    plt.legend()
    plt.xlabel(r'$\lambda$, нм', size=16)
    plt.ylabel(r'$D$', size=16)
    plt.savefig(r'1_9_10_11.png')
    plt.show()
    plt.close()


def booger_lambert_behr(С_0, D_lambda, name):
    D_sour = np.array([D_lambda[11], D_lambda[0], D_lambda[1], D_lambda[2], D_lambda[3]])
    C_sour = np.array([С_0[11], С_0[0], С_0[1], С_0[2], С_0[3]])

    D_lye = np.array([D_lambda[11], D_lambda[4], D_lambda[5], D_lambda[6], D_lambda[7]])
    C_lye = np.array([С_0[11], С_0[4], С_0[5], С_0[6], С_0[7]])

    eps_sour = D_sour / C_sour
    eps_lye = D_lye / C_lye

    print('Коэф. экс. кисл. = ' + str(eps_sour))
    print('Коэф. экс. щел. = ' + str(eps_lye))

    a_sour, b_sour = help_laba.LSM(C_sour, D_sour)
    a_lye, b_lye = help_laba.LSM(C_lye, D_lye)

    print('Коэф. наклона кисл. = ' + str(a_sour))
    print('Коэф. наклона щел. = ' + str(a_lye))

    plt.plot(C_sour, D_sour, '.', label='Протонированная форма', color='steelblue', markersize=5)
    plt.plot(np.linspace(0, 3 * 10 ** (-5), 20), np.linspace(0, 3 * 10 ** (-5), 20) * a_sour + b_sour, '-', color='steelblue', linewidth=0.5)
    plt.plot(C_lye, D_lye, '.', label='Депротонированная форма', color='magenta')
    plt.plot(np.linspace(0, 3.5 * 10 ** (-5), 20), np.linspace(0, 3.5 * 10 ** (-5), 20) * a_lye + b_lye, '-', color='magenta', linewidth=0.5, markersize=5)
    plt.legend()
    plt.xlabel(r'$C_0$, моль/л', size=16)
    plt.ylabel(r'$D_{\lambda}$', size=16)
    plt.savefig(r'D_C'+name+'.png')
    plt.show()


def plot_all(wave, D, num):
    plt.plot(wave, D, '-', label='Раствор ' + str(num), linewidth=0.8)
    plt.legend()
    plt.xlabel(r'$\lambda$, нм', size=16)
    plt.ylabel(r'$D$', size=16)
    plt.savefig(r'спект' + str(num) + '.png')
    # plt.show()
    plt.close()


def const():
    C_hcl = 5 * 0.1 / 50
    C_naoh = 5 * 0.1 / 50

    ph = np.array([- log(C_hcl, 10), 14 + log(C_naoh, 10), 3.93, 3.7, 3.5])
    print('ph = ' + str(ph))

    C_Z2_hcl = 2 * C_hcl + 2 * 1.22324159e-05
    C_Z2_naoh = 2 * C_naoh + 2 * 2.44648318e-05
    C_1 = 2 * 0.004 + 2 * 2.44648318e-05
    C_2 = 2 * 0.004 + 2 * 2.44648318e-05
    C_3 = 2 * 0.004 + 2 * 2.44648318e-05

    I_c = np.array([C_Z2_hcl, C_Z2_naoh, C_1, C_2, C_3]) / 2
    print('Ионная сила = ' + str(I_c))

    Lg_gamma = - 0.509 * 1 * np.sqrt(I_c) / (1 + np.sqrt(I_c))
    print('lg gamma = ' + str(Lg_gamma))

    D_l1 = np.array([0.465088, 0.130412, 0.348706, 0.439664, 0.539034])
    D_l2 = np.array([0.091118, 0.539880, 0.462101, 0.411096, 0.373455])

    C0 = np.array([1.22324159e-05, 2.44648318e-05, 2.44648318e-05, 2.44648318e-05, 2.44648318e-05])

    eps_sour_1 = 41000.813117785176
    eps_lye_1 = 5385.597209266121
    eps_sour_2 = 9570.0310612529
    eps_lye_2 = 21891.129916494418

    D_sour_1 = eps_sour_1 * C0[2:]
    D_sour_2 = eps_sour_2 * C0[2:]
    D_lye_1 = eps_lye_1 * C0[2:]
    D_lye_2 = eps_lye_2 * C0[2:]

    alpha1 = (D_l1[2:] - D_sour_1) / (D_lye_1 - D_sour_1)
    alpha2 = (D_l2[2:] - D_sour_2) / (D_lye_2 - D_sour_2)


    print('alpha1 = ' + str(alpha1))
    print('alpha2 = ' + str(alpha2))

    lg_a1 = np.log10(alpha1/(1-alpha1))
    lg_a2 = np.log10(alpha2/(1 - alpha2))

    print('lg_1 = ' + str(lg_a1))
    print('lg_2 = ' + str(lg_a2))

    lg_Ka1 = lg_a1 - ph[2:]  + Lg_gamma[2:]
    lg_Ka2 = lg_a2 - ph[2:]  + Lg_gamma[2:]

    print('lg Ka1 = ' + str(lg_Ka1))
    print('lg Ka2 = ' + str(lg_Ka2))

    Ka1 = 10 ** (lg_Ka1)
    Ka2 = 10 ** (lg_Ka2)

    print('Ka1 = ' + str(Ka1))
    print('Ka2 = ' + str(Ka2))

    print('Среднее 1 =' + str(np.mean(Ka1)) + '+-' + str(np.std(Ka1)))
    print('Среднее 2 =' + str(np.mean(Ka2)) + '+-' + str(np.std(Ka2)))


file_1 = open(r'Аксенова Гарина/раствор 1, 300-600.txt')
file_2 = open(r'Аксенова Гарина/раствор 2, 300-600.txt')
file_3 = open(r'Аксенова Гарина/раствор 3, 300-600.txt')
file_4 = open(r'Аксенова Гарина/раствор 4, 300-600.txt')
file_5 = open(r'Аксенова Гарина/раствор 5, 300-600.txt')
file_6 = open(r'Аксенова Гарина/раствор 6, 300-600.txt')
file_7 = open(r'Аксенова Гарина/раствор 7, 300-600.txt')
file_8 = open(r'Аксенова Гарина/раствор 8, 300-600.txt')
file_9 = open(r'Аксенова Гарина/раствор 9, 300-600.txt')
file_10 = open(r'Аксенова Гарина/раствор 10, 300-600.txt')
file_11 = open(r'Аксенова Гарина/раствор 11, 300-600.txt')
file_water = open(r'Аксенова Гарина/вода.txt')

wave1, D1 = import_file(file_1)
wave2, D2 = import_file(file_2)
wave3, D3 = import_file(file_3)
wave4, D4 = import_file(file_4)
wave5, D5 = import_file(file_5)
wave6, D6 = import_file(file_6)
wave7, D7 = import_file(file_7)
wave8, D8 = import_file(file_8)
wave9, D9 = import_file(file_9)
wave10, D10 = import_file(file_10)
wave11, D11 = import_file(file_11)
wavew, Dw = import_file(file_water)

plot_work_wave(wave1, D1, wave6, D6, wave9, D9, wave10, D10, wave11, D11)

D_lambda1 = np.array([0.999166, 0.745096, 0.465088, 0.242207, 0.161769, 0.130412,
                      0.093437, 0.065792, 0.348706, 0.439664, 0.539034, -0.003237])

D_lambda2 = np.array([0.234249, 0.160784, 0.091118, 0.046813, 0.668780, 0.539880,
                      0.399550, 0.276869, 0.462101, 0.411096, 0.373455, -0.001427])

C_1 = 0.2 / 327
V_ind = np.array([2, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 2, 2, 2, 0])
V_all = 50

C_0 = C_1 * V_ind / V_all
print('C0 = ' + str(C_0))

print('На рабочей длине волны 1 530 нм')
booger_lambert_behr(C_0, D_lambda1, 'lambda1')
print('На рабочей длине волны 2 438 нм')
booger_lambert_behr(C_0, D_lambda2, 'lambda2')

plot_all(wave1, D1, 1)
plot_all(wave2, D2, 2)
plot_all(wave3, D3, 3)
plot_all(wave4, D4, 4)
plot_all(wave5, D5, 5)
plot_all(wave6, D6, 6)
plot_all(wave7, D7, 7)
plot_all(wave8, D8, 8)
plot_all(wave9, D9, 9)
plot_all(wave10, D10, 10)
plot_all(wave11, D11, 11)
plot_all(wavew, Dw, 12)


const()
