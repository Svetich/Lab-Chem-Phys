import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr

def LSM(x, y):
    function = lambda x, a, b: a*x  + b
    popt, pcov = curve_fit(function, xdata=x, ydata=y)

    sigma_a = np.sqrt(pcov[0,0])
    sigma_b = np.sqrt(pcov[1, 1])

    return popt[0], popt[1], sigma_a, sigma_b
def chi_sq(x, y, err):
    function = lambda x, a, b: a * x + b
    popt, pcov = curve_fit(function, xdata=x, ydata=y, sigma=err)

    sigma_a = np.sqrt(pcov[0, 0])
    sigma_b = np.sqrt(pcov[1, 1])

    return popt[0], popt[1], sigma_a, sigma_b


function = lambda x, a, b: a*x + b

dens1 = np.array([2.799, 1.945, 1.478, 1.221, 0.597, 0.358, 0])
conc = np.array([0.011, 0.00726, 0.0055, 0.0044, 0.0022, 0.0011, 0])
G = np.array([2.536063854, 1.763298836, 1.340107947, 1.10763145, 0.541859601, 0.325180802])
equilibrium = np.array([0.009329761, 0.00537128, 0.003881258, 0.002605405, 0.000954439, 0.000301117])
nG = 1/G
nC = 1/equilibrium
lnG = np.array([0.404160184, 0.246325921, 0.127139783, 0.044395278, -0.266113227, -0.487875102])
lnC = np.array([-2.030129499,-2.269922191,-2.411027436,-2.58412469,-3.020251886,-3.521264266])


p1, p2, a, b = LSM(lnC, lnG)
x = np.linspace(-3.521264266, -2.030129499, 100)
y = p1*x + p2


#plt.scatter(equilibrium, G, color = 'g', marker = '.')

plt.scatter(lnC, lnG, color = 'k', marker = '.')
plt.plot(x, y, color = 'k', lw = 0.5)
plt.xlabel('lnC')
plt.ylabel('lnГ')
print(p1)
print(p2)
print((pearsonr(lnC, lnG)[0])**2)
print((pearsonr(nC, nG)[0])**2)
print((pearsonr(equilibrium, G)[0])**2)
plt.close()

#plt.show()

a1, b1, sigmaa, sigmab = LSM(equilibrium, G)
x1 = np.linspace(0, 0.01, 500)
y1 = a1 * x1 + b1
plt.scatter(equilibrium, G, color='green', marker='.')
plt.plot(x1, y1, color='green', lw=0.5)
plt.xlabel('Cравн, моль/л')
plt.ylabel('Г, моль/г')
plt.savefig('G(c).png')
plt.show()
plt.close()
