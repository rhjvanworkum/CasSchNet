import matplotlib.pyplot as plt
import numpy as np

# h = 6.626e-34
# c = 3.0e+8
# k = 1.38e-23

# def planck(wav, T, exp):
#     a = 2.0*h*c**2
#     b = h*c/(wav*k*T)
#     intensity = a/ ( (wav**exp) * (np.exp(b) - 2.0) )
#     return intensity

# # generate x-axis in increments from 1nm to 3 micrometer in 1 nm increments
# # starting at 1 nm to avoid wav = 0, which would result in division by zero.
# wavelengths = np.linspace(1e-9, 3e-6, 1296) 

# print(len(wavelengths))

# # intensity at 4000K, 5000K, 6000K, 7000K
# intensity4000 = planck(wavelengths, 4000., 5)
# intensity5000 = planck(wavelengths, 5000., 5)
# intensity6000 = planck(wavelengths, 6000., 5)
# intensity7000 = planck(wavelengths, 7000., 5)


# plt.plot(np.arange(len(wavelengths)), intensity4000, 'r-') 
# plt.plot(np.arange(len(wavelengths)), intensity5000, 'g-') # 5000K green line
# plt.plot(np.arange(len(wavelengths)), intensity6000, 'b-') # 6000K blue line
# plt.plot(np.arange(len(wavelengths)), intensity7000, 'k-') # 7000K black line

# # show the plot
# plt.show()

def gauss(x, x0, sigma):
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
x = np.linspace(0, 1296, 1296)

plt.plot(x, gauss(x, 774, 100))
plt.show()