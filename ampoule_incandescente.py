import numpy as np
import logging as log
import matplotlib.pyplot as plt
import scipy.constants as const


class MathFunction:

    """
    Class MathFunction permet la manipulation de fonction mathématique.
    """

    gaussxqValuesForN = dict()

    def __init__(self, func):
        """
        Constructeur de la class MathFunction.
        :param func: Fontion mathématique. (Function or lambda expression)
        """
        self.function = func

    def __call__(self, *args):
        """
        Call de la fonction mathématique courante.
        :param args: arguments de la fonction courante.
        :return: return de self.function(*args)
        """
        return self.function(*args)

    def gaussxq(self, N: int) -> (float, float):
        """
        A faire

        :param N:
        :return:
        """
        # Approximation initial des racines des polynomes de Legendre
        a = np.linspace(3, 4 * N - 1, N)/(4 * N + 2)
        x = np.cos(np.pi * a + 1 / (8 * (N**2) * np.tan(a)))

        # On trouve les racines avec la méthode de newton
        epsilon = 1e-15
        delta = 1.0
        while delta > epsilon:
            p0 = np.ones(N, float)
            p1 = np.copy(x)
            for k in range(1, N):
                p0, p1 = p1, ((2 * k + 1) * x * p1 - k*p0) /(k + 1)

            dp = (N + 1) * (p0 - x * p1) / (1 - x**2)
            dx = p1 / dp
            x -= dx
            delta = max(abs(dx))

        # Calcul des poids
        w = 2 * ((N + 1) ** 2) / ((N ** 2) * (1 - x ** 2) * (dp ** 2))

        return x, w

    def gaussian_quadrature_integration(self, **kwargs) -> float:
        # P.170 Computational physics
        assert "bounds" in kwargs and "N" in kwargs
        bounds = kwargs["bounds"]
        N = kwargs["N"]
        [a, b] = bounds

        # Calculate the sample points ans weights, then map them
        # to the required integration domain
        if N in self.gaussxqValuesForN:
            x, w = self.gaussxqValuesForN[N]
        else:
            x, w = self.gaussxq(N)
            self.gaussxqValuesForN[N] = (x, w)

        xp = 0.5 * (b - a) * x + 0.5 * (b + a)
        wp = 0.5 * (b - a) * w

        # perform the integration
        s = np.sum(wp * self.function(xp))
        return s

    def gaussian_quad_find_N(self, **kwargs) -> (int, float):
        import warnings
        assert "bounds" in kwargs
        bounds = kwargs["bounds"]
        error_target = kwargs.get("error_target", 1e-9)
        max_iteration = kwargs.get("max_iteration", 1_000)

        memory: dict = {
            "I_i": self.gaussian_quadrature_integration(bounds=bounds, N=1),
            "I_i+1": self.gaussian_quadrature_integration(bounds=bounds, N=2)
        }

        bestN: int = 2

        error: float = np.inf
        itr: int = 1
        for i in range(1, max_iteration + 1):
            error = np.abs(memory["I_i+1"] - memory["I_i"])
            if error <= error_target:
                itr = i
                break

            bestN = 2 ** (i + 1)

            memory["I_i"] = memory["I_i+1"]
            memory["I_i+1"] = self.gaussian_quadrature_integration(bounds=bounds, N=bestN)

        if itr >= max_iteration + 1:
            warnings.warn("Max iteration has been reach for gaussian_quad_find_N")

        return bestN, error

    def adaptative_gaussian_quadrature(self, **kwargs) -> (float, float):
        assert "bounds" in kwargs
        N, error = self.gaussian_quad_find_N(**kwargs)
        kwargs["N"] = N
        return self.gaussian_quadrature_integration(**kwargs), error

    def afficher_fonction(self, debut_de_la_plage: float, fin_de_la_plage: float, titre: str ,
                          axe_x_name: str, axe_y_name: str, nb_de_points_dans_la_plage) -> None:
        """
        Cette methode affiche trois différentes formes de fonction d'onde
        pour un puit de potentiiel rectangulaire pour la plage spécifiée.

        Parameters
        ----------
        debut_de_la_plage :
            Début de la plage pour l'affichage
        fin_de_la_plage :
            Fin de la plage pour l'affichage
        titre :
            Titre du graphique a afficher
        axe_x_name :
            Titre de l'axe des x
        axe_y_name :
            Titre de l'axe des y
        """
        try:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            nb_de_bonds = nb_de_points_dans_la_plage
            valeurs_de_x = np.linspace(debut_de_la_plage, fin_de_la_plage, nb_de_bonds)
            valeurs_de_y =[]
            for element in np.nditer(valeurs_de_x):
                valeurs_de_y.append(self(element))

            valeurs_de_y = np.asarray(valeurs_de_y)
            ax.plot(valeurs_de_x, valeurs_de_y, color='blue', lw=2, label="y_1")
            ax.set_title(titre)
            ax.set_xlabel(axe_x_name)
            ax.set_ylabel(axe_y_name)
            plt.grid()
            plt.show()
        except ZeroDivisionError:
            log.error("L'évaluation de la fonction dans la plage cause une division par 0")


if __name__ == "__main__":
    fonction_a_integrer = MathFunction(lambda x: (15/(np.pi**2))*((x**3)/((np.e**x) - 1)))
    borne_inferieur = (const.h*const.c)/((750e-9)*const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    borne_superieur = (const.h*const.c)/((300e-9)*const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    fonction_a_afficher = MathFunction(lambda x: fonction_a_integrer.gaussian_quadrature_integration(N=100,
                                                                        bounds=[borne_inferieur/x, borne_superieur/x]))
    fonction_a_afficher.afficher_fonction(300, 10000,
                                          "Éfficacité de l'ampoule incandescente en fonction de sa température",
                                          "Température [k]", "Efficacité", 100)
