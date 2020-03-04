import numpy as np
import sympy as sym
import logging as log
import matplotlib.pyplot as plt


def resolution_par_relaxation(fonction, point_initial: float, erreur_visee: float,
                              fonction_derivee=None, x=sym.symbols("x")) -> (float, float, int):
    """
    Cette méthode utilise la méthode de résolution d'équation non-linéaire par relaxation.
    Pour être utilisé, l'équation doit être de la forme x = f(x) afin que l'algorithme ne travail
    qu'à évaluer f(x). La méthode redéfinis x_n+1 = f(x_n) jusqu'à ce que l'erreur sur x_n+1
    soit inférieur à l'erreur visée.

    Parameters
    ----------
    fonction : SymPy object
        f(x) à évaluer pour la méthode
    point_initial :
        Valeur initial pour débuter l'application de la méthode
        par relaxation
    erreur_visee :
        Valeur de l'erreur que nous visons pour le résultats de la résolution
        de l'équation
    fonction_derivee : SymPy object
        f'(x) pourcalculer l'évolution de l'erreur sur le résultat. Si ce paramètre
        n'est pas remplis, la fonction f(x) sera dérivé à l'aide de sympy
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    x_0 = point_initial
    erreur = 9999999999999999999999999999
    precision = 1 - int(np.log10(erreur_visee))
    iterator = 0
    if fonction_derivee is None:
        fonction_derivee = sym.diff(fonction, x)

    while erreur > erreur_visee:
        try:
            erreur = fonction_derivee.evalf(2, subs={x: x_0}) * erreur
            if fonction_derivee.evalf(10, subs={x: x_0}) > 1.00000000000:
                raise ValueError
            x_0 = fonction.evalf(precision, subs={x: x_0})
            iterator += 1

        except ValueError:
            log.error("La méthode ne converge pas puisque l'évaluation de la dérivée > 1")
            return 0, 0, 0

    return x_0, erreur, iterator


def resolution_de_facon_graphique(fonction_1, fonction_2, debut_de_la_plage_d_affichage: int,
                                  fin_de_la_plage_d_affichage: int, x=sym.symbols("x"), ) -> None:
    """
    Cette méthode affiche les deux fonctions des deux côtés de l'équation
    afin d'observer de façon graphique les résultats possibles de l'équation non-linéaire
    pour la plage donnée en argument.

    Parameters
    ----------
    fonction_1 : SymPy object
       fonction du côté gauche de l'équation
    fonction_2 : SymPy object
        fonction du côté droit de l'équation
    debut_de_la_plage_d_affichage :
        Début de la plage
    fin_de_la_plage_d_affichage :
        Fin de la plage
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    try:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        nb_de_bonds = int((fin_de_la_plage_d_affichage - debut_de_la_plage_d_affichage) * 10)
        valeurs_de_x = np.linspace(debut_de_la_plage_d_affichage, fin_de_la_plage_d_affichage, nb_de_bonds)
        f1 = sym.lambdify(x, fonction_1, "numpy")
        f2 = sym.lambdify(x, fonction_2, "numpy")
        ax.plot(valeurs_de_x, f1(valeurs_de_x), color='blue', lw=2)
        ax.plot(valeurs_de_x, f2(valeurs_de_x), color='red', lw=2)
        ax.set_title("Graphique illustrant les résultats de l'équation correspondant\n"
                     " aux intersection entre les fonction de gauche et de droite de\n l'équation.")
        ax.set_xlabel("Valeurs de x")
        ax.set_ylabel("Valeurs de f(x) gauche et droite")

        plt.grid()
        plt.show()
        plt.close(fig)

    except ZeroDivisionError:
        log.error("Le domaine choisi engendre une division par zéro")


if __name__ == "__main__":
    # Code pour le b)
    """
    x = sym.symbols("x")
    fonction_a_resoudre = -5*sym.exp(-x) + 5
    fonction_a_resoudre_derivee = 5*sym.exp(-x)
    resultat, erreur, nombre_iteration = resolution_par_relaxation(fonction_a_resoudre, 40, 0.000001,
                                                                   fonction_a_resoudre_derivee)
    print(f"Le résultat est :{resultat}")
    print(f"L'erreur sur le résultat est :{erreur}")
    print(f"Le nombre d'itération pour atteindre ce résultat :{nombre_iteration}")
    """

    # Code pour le c)
    """
    x = sym.symbols("x")
    fonction_a_droite = -5*sym.exp(-x) + 5
    fonction_a_gauche = x
    resolution_de_facon_graphique(fonction_a_droite, fonction_a_gauche, -1, 6)
    """
