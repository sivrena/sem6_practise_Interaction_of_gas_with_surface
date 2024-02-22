import matplotlib.pyplot as plt
import numpy as np

with open('isotherms_3d.txt', 'r') as file:
    lines = file.readlines()

isotherm_x = np.array([0. for i in range (len(lines))])
isotherm_y = np.array([0. for i in range (len(lines))])

plt.title('Adsorption isotherm', fontsize=20, fontname='Times New Roman')
plt.xlabel('Pressure (pascals)', color='gray')
plt.ylabel('Adsorption (mole per nanometer)',color='gray')
for i in range (len(lines)):
    line = (lines[i]).split()
    isotherm_x[i] += float(line[0])
    isotherm_y[i] += float(line[1])
    string = (line[3])
    plt.text(isotherm_x[i], isotherm_y[i], string)
plt.grid(True)
plt.plot(isotherm_x, isotherm_y, 'r^')
lgnd = ['Temp. = ' + line[2] + 'K\n' + 'Part.num = ' + line[4] + 'x' + line[5] + 'x' + line[6] + 'nm\n'
                                                                                'See graph for number of particles']
plt.legend(lgnd, loc=4)
plt.show()