import numpy as np
import logging as log
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.optimize as sciopt


class MathFunction:

    """
    Class MathFunction permet la manipulation de fonction mathématique.
    """

    gaussxqValuesForN = dict()

    def __init__(self, func):
        """
        constructeur

        Parameters
        ----------
        func :
           fonction mathématique (fonction ou lambda expression)
        """
        self.function = func

    def __call__(self, *args):
        """
        Cette fonction évalue tout simplement la fonction

        Return
        ----------
        Résultat de l'appel de la fonction
        """

        return self.function(*args)

    def gaussxq(self, N: int) -> (float, float):
        """
        Cette methode affiche trois différentes formes de fonction d'onde
        pour un puit de potentiiel rectangulaire pour la plage spécifiée.

        Parameters
        ----------
        N :
           Correspond au nombre de sous intervalle de l'intégrale

        Returns
        ----------
        x:
            les racines
        w:
            le poid de chaque racines

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
        """
        Cette méthode évalue par quadrature l'intégrale bornée de la
        fonction

        Parameters
        ----------
        kwargs:
            bounds : list
                Correespond aux bornes de l'intégrale
            N: int
                Correspond au nombre de sous intervalle de l'intégrale

        Returns
        ----------
        s:
            le résultat de l'intégrale définie
        """
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

    def afficher_fonction(self, debut_de_la_plage: float, fin_de_la_plage: float, titre: str ,
                          axe_x_name: str, axe_y_name: str, nb_de_points_dans_la_plage: int) -> None:
        """
        Cette methode affiche la fonction sur une plage spécifiée.

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
        nb_de_points_dans_la_plage:
            nombre d'évaluation pour générer la courbe sur tout l'intervalle
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

    def trouver_maximum_de_la_fonction_ratio_dore(self, valeur_initiale_gauche: float,
                                                  valeur_initiale_droite: float, erreur_vise: float) -> float:
        """
        Cette methode affiche trois différentes formes de fonction d'onde
        pour un puit de potentiiel rectangulaire pour la plage spécifiée.

        Parameters
        ----------
        valeur_initiale_gauche :
            Valeur initiale à gauche pour débuter l'algorithme
        valeur_initiale_droite :
            Valeur initiale à gauche pour débuter l'algorithme
        erreur_vise :
            erreur qui arr^te l'algorithme lorsqu'elle est atteinte

        Returns
        ---------
        point_maximal:
            Valeur de l'abcisse correspondant au max de la fonction entre les
            deux valeurs initiales.
        """
        assert valeur_initiale_gauche < valeur_initiale_droite
        nombre_d_or = (1 + np.sqrt(5)) / 2
        position_a = valeur_initiale_gauche + (valeur_initiale_droite - valeur_initiale_gauche) / nombre_d_or
        position_b = valeur_initiale_droite - (valeur_initiale_droite - valeur_initiale_gauche) / nombre_d_or
        while (valeur_initiale_droite - valeur_initiale_gauche) >= 2*erreur_vise:
            if self(position_a) > self(position_b):
                position_b, valeur_initiale_gauche = position_a, position_b
                position_a = valeur_initiale_gauche + (valeur_initiale_droite - valeur_initiale_gauche) / nombre_d_or
            else:
                valeur_initiale_droite, position_a = position_a, position_b
                position_b = valeur_initiale_gauche - (valeur_initiale_droite - valeur_initiale_gauche) / nombre_d_or

        point_maximal = (valeur_initiale_droite + valeur_initiale_gauche) / 2
        return point_maximal


if __name__ == "__main__":
    # Code pour le numéro a)
    """
    fonction_a_integrer = MathFunction(lambda x: (15/(np.pi**4))*((x**3)/((np.e**x) - 1)))
    borne_inferieur = (const.h*const.c)/((750e-9)*const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    borne_superieur = (const.h*const.c)/((300e-9)*const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    fonction_a_afficher = MathFunction(lambda x: fonction_a_integrer.gaussian_quadrature_integration(N=100,
                                                                        bounds=[borne_inferieur/x, borne_superieur/x]))
    fonction_a_afficher.afficher_fonction(300, 50000,
                                          "Éfficacité de l'ampoule incandescente en fonction de sa température",
                                          "Température [k]", "Efficacité", 100)
    """
    # Code pour le numéro b)
    """
    fonction_a_integrer = MathFunction(lambda x: (15 / (np.pi ** 4)) * ((x ** 3) / ((np.e ** x) - 1)))
    borne_inferieur = (const.h * const.c) / (
                (750e-9) * const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    borne_superieur = (const.h * const.c) / (
                (300e-9) * const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    fonction_a_optimiser = MathFunction(lambda x: fonction_a_integrer.gaussian_quadrature_integration(N=100,
                                                                  bounds=[borne_inferieur / x, borne_superieur / x]))
    erreur = 1
    plage = [8040, 8060]
    maxi = fonction_a_optimiser.trouver_maximum_de_la_fonction_ratio_dore(plage[0], plage[1], erreur)
    print(f"La température pour laquelle l'efficacité est maximal est de: {maxi}")
    print(f"Pour une plage de {plage[0]} à {plage[1]} et une erreur de {erreur}")
    """
    # Code pour la référence du numéro b)
    """
    fonction_a_integrer = MathFunction(lambda x: (15 / (np.pi ** 4)) * ((x ** 3) / ((np.e ** x) - 1)))
    borne_inferieur = (const.h * const.c) / (
                (750e-9) * const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    borne_superieur = (const.h * const.c) / (
                (300e-9) * const.Boltzmann)  # Le T au dénominateur va apparaitre dans la fonction lambda
    fonction_a_optimiser = lambda x: -fonction_a_integrer.gaussian_quadrature_integration(N=100,
                                                                  bounds=[borne_inferieur / x, borne_superieur / x])
    plage = [6000, 10000]
    maxi = sciopt.golden(fonction_a_optimiser, brack=(plage[0], plage[1]))
    print("Voici les résultats de l'algorithme de scipy")
    print(f"La température pour laquelle l'efficacité est maximal est de: {maxi}")
    print(f"Pour une plage de {plage[0]} à {plage[1]}")
    """
