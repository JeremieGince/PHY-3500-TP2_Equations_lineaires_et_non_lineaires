import numpy as n
import sympy as sym
import logging as log
import matplotlib as mp


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
    precision = 1 - int(n.log10(erreur_visee))
    iterator = 0
    if fonction_derivee is None:
        fonction_derivee = sym.diff(fonction, x)

    while erreur > erreur_visee:
        try:
            x_0 = fonction.evalf(precision, {x: x_0})
            erreur = fonction_derivee.evalf(2, {x: x_0})*erreur
            iterator += 1
            if fonction_derivee.evalf(2, {x: x_0}) > 1:
                raise ValueError

        except ValueError:
            log.error("La méthode ne converge pas puisque l'évaluation de la dérivée > 1")
            return 0, 0, iterator

    return x_0, erreur, iterator


def resolution_de_facon_graphique(fonction_1, fonction_2, x=sym.symbols("x")) -> None:
    """
    Cette méthode affiche les deux fonctions des deux côtés de l'équation
    afin d'observer de façon graphique les résultats possibles de l'équation non-linéaire.

    Parameters
    ----------
    fonction_1 : SymPy object
       fonction du côté gauche de l'équation
    fonction_2 :
        fonction du côté droit de l'équation
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    return None


if __name__ == "__main__":

    x = sym.symbols("x")
    fonction_a_resoudre = -5*sym.exp(-x) + 5
    fonction_a_resoudre_derivee = 5*sym.exp(-x)
    resultat, erreur, nombre_iteration = resolution_par_relaxation(fonction_a_resoudre, 2, 0.000001,
                                                                   fonction_a_resoudre_derivee)
    print(f"Le résultat est :{resultat}")
    print(f"L'erreur sur le résultat est :{erreur}")
    print(f"Le nombre d'itération pour atteindre ce résultat :{nombre_iteration}")
