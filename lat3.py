import numpy as np
from matplotlib import pyplot

odswiez = 50

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    szerokosc_x = 300
    wysokosc_y = 100
    tau = .53
    Nt = 4000  # ilosc krokow czasowych

    #lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((wysokosc_y, szerokosc_x, NL)) + .01 * np.random.randn(wysokosc_y, szerokosc_x, NL)

    # Nadanie cząsteczkom kierunku w prawo
    F[:,:,3] = 2.3

    cylinder = np.full((wysokosc_y, szerokosc_x), False)


    # Cylindy (można zmienic położenie i rozmiar) (ostrpożnie bo program jest wrażliwy)
    for y in range(0, wysokosc_y):
        for x in range(0, szerokosc_x):
            if (distance(szerokosc_x // 5, wysokosc_y // 2, x, y) < 15) or (distance(2 * szerokosc_x // 5, wysokosc_y // 2, x, y) < 20):
                cylinder[y][x] = True

    for it in range(Nt):
        print(it)

        F[:,-1, [6, 7, 8]] = F[:,-2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i,cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # kolizje
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3 * (cx * ux + cy * uy) + 9 * (cx*ux + cy * uy)**2 /2 - 3 * (ux**2 + uy**2)/2
             )

        F = F + -(1/tau) * (F - Feq)

        if (it%odswiez == 0):
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfxdy - dfydx

            pyplot.imshow(curl, cmap="RdYlBu")
            pyplot.pause(.01)
            pyplot.cla()

if __name__ == "__main__":
    main()