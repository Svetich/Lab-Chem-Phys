import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def LSM(x, y): # y = bx + a
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y)[0]
    return b, a

def calculation(concentration, kappa, lim_lambda_plus, lim_lambda_minus, KD_th, name):
    lambda_ = 1000 * kappa / concentration
    lim_lambda = lim_lambda_minus + lim_lambda_plus
    alpha = lambda_ / lim_lambda
    KD_array = concentration * np.power(alpha, 2) / (1 - alpha)
    x = 1 / lambda_  # Сименс * см^2 / моль
    y = lambda_ * concentration * 10
    b, a = LSM(x, y)
    plt.plot(x, y, '.')
    plt.plot(x, x * b + a, '-', linewidth=1)
    plt.xlabel(r'$\lambda$c, См/см', size=14)
    plt.ylabel(r'$\frac{1}{\lambda}$, моль/(См $\cdot$ см$^{2})$', size=14)
    plt.savefig('{}1.png'.format(name))
    plt.show()
    plt.close()

    KD_gr = 1 / (b * lim_lambda)
    lim_lambda_gr = 1 / a

    file = open((name + 'table.txt'), 'w')
    file.write('c \t\t kappa \t\t lambda \t\t alpha \t\t KD \n')
    for i in range(len(concentration)):
        file.write(str(concentration[i]) + '\t\t' + str(kappa[i]) + '\t\t' +
                   str(lambda_[i]) + '\t\t' + str(alpha[i]) + '\t\t' + str(KD_array[i]) + '\n')

    file.write('KD (справочник) = ' + str(KD_th) + '\n' +
               'KD (среднее) = ' + str(np.mean(KD_array)) + '\n' +
               'KD (из графика) = ' + str(KD_gr) + '\n' +
               'Проводимость при предельном разведении (справочник) - ' + str(lim_lambda) + '\n' +
               'Проводимость при предельном разведении (из графика) - ' + str(lim_lambda_gr))

    b_ckappa, a_ckappa = LSM(concentration, kappa)

    plt.plot(concentration, kappa, '.')
    # plt.plot(concentration, b_ckappa * concentration + a_ckappa, '-', linewidth=1)
    plt.xlabel(r'c, моль/л', size=14)
    plt.ylabel(r'$\varkappa$, См/см', size=14)
    plt.savefig('{} c(kappa).png'.format(name))
    plt.show()
    plt.close()

    plt.plot(concentration, lambda_, '.')
    plt.xlabel(r'c, моль/л', size=14)
    plt.ylabel(r'$\lambda$, См $\cdot$ см$^2$/моль', size=14)
    plt.savefig('{} lambda(kappa).png'.format(name))
    plt.show()
    plt.close()

    plt.plot(concentration, alpha, '.')
    plt.xlabel(r'c, моль/л', size=14)
    plt.ylabel(r'$\alpha$', size=14)
    plt.savefig('{} alpha(kappa).png'.format(name))
    plt.show()
    plt.close()

concentration_vinegar = np.array([3, 1.5, 0.75, 0.375, 0.1875, 0.09375]) # моль/литр
kappa_vinegar = np.array([1.486, 1.340, 1.163, 0.733, 0.602, 0.416]) * 10 ** (-3) # Сименс/см
lim_lambda_plus_vinegar = 349.80
lim_lambda_minus_vinegar = 40.90  # Сименс * см^2 / моль
KD_th_vinegar = 1.74 * 10 ** (-5)
calculation(concentration_vinegar, kappa_vinegar, lim_lambda_plus_vinegar, lim_lambda_minus_vinegar, KD_th_vinegar, 'Уксусная')


concentration_ant = np.array([6.5, 3.25, 1.625, 0.8125]) # моль/литр
kappa_ant = np.array([8.25, 10.01, 6.77, 5.22]) * 10 ** (-3) # Сименс/см
lim_lambda_plus_ant = 349.80
lim_lambda_minus_ant = 54.60  # Сименс * см^2 / моль
KD_th_ant = 1.772 * 10 ** (-4)
calculation(concentration_ant, kappa_ant, lim_lambda_plus_ant, lim_lambda_minus_ant , KD_th_ant, 'Муравьиная')