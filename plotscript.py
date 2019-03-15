import numpy as np
y=np.array([740,
            1305,
            1575,
            1700,
            1450,
            960,
            1345,
            1245,
            1540,
            510,
            1535,
            1500,
            690,
            1000,
            775,
            1205,
            1090,
            1595,
            1815,
            825,
            2365,
            915,
            540,
            1280,
            685,
            1320,
            1690,
            1495,
            1460,
            2225,
            2340,
            1450,
            2180,
            2700,
            2415,
            1960,
            1465])
t=np.linspace(1,38,37)

promedio = np.mean(y)
sumaTotal = np.sum(y)
print("Promedio = " + str(promedio))
print("suma = " + str(sumaTotal))

import matplotlib.pyplot as plt
plt.plot(t,y)

plt.title('Ventas por dia')
plt.xlabel('Dia')
plt.ylabel('$')
#plt.grid(False,color='k')

plt.show()
