import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import logging as log


def afficher_fonction_d_onde(potentiel: float, largeur_du_puit: float, masse: float, debut_de_la_plage_energie: float,
                             fin_de_la_plage_energie: float) -> None:
    """
    Cette methode affiche trois différentes formes de fonction d'onde
    pour un puit de potentiiel rectangulaire pour la plage spécifiée.

    Parameters
    ----------
    potentiel :
       hauteur du puit en eV
    largeur_du_puit :
        largeur du puit en m
    masse :
        masse de la particule en eV
    debut_de_la_plage_energie :
        Début de la plage en eV
    fin_de_la_plage_energie :
        Fin de la plage
    """
    try:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # les plus ou moins 0.001 sont présent pour éviter les divisions par 0
        nb_de_bonds = int((fin_de_la_plage_energie + 0.001 - debut_de_la_plage_energie - 0.001) * 10)
        valeurs_de_x = np.linspace(debut_de_la_plage_energie + 0.001, fin_de_la_plage_energie - 0.001, nb_de_bonds)
        c = 3e8  # m/s
        h_bar = 6.582e-16  # eVs
        fonction_1 = lambda x: np.tan(np.sqrt(((largeur_du_puit ** 2) * masse * x) / (2 * (c ** 2) * (h_bar ** 2))))
        fonction_2 = lambda x: np.sqrt((potentiel - x) / x)
        fonction_3 = lambda x: -np.sqrt(x / (potentiel - x))
        ax.plot(valeurs_de_x, fonction_1(valeurs_de_x), color='blue', lw=2, label="y_1")
        ax.plot(valeurs_de_x, fonction_2(valeurs_de_x), color='red', lw=2, label="y_2")
        ax.plot(valeurs_de_x, fonction_3(valeurs_de_x), color='black', lw=2, label="y_3")
        ax.set_title("Graphique illustrant les résultats de l'équation correspondant\n"
                     " aux intersection entre les fonction de gauche et de droite de\n l'équation.")
        ax.set_xlabel("Énergie [eV]")
        ax.set_ylabel("Valur associé à la fonction [-]")
        ax.legend()
        plt.grid()
        plt.show()

    except ZeroDivisionError:
        log.error("Le domaine choisi engendre une division par zéro")


def resolution_par_bissection(fonction_gauche, fonction_droite, point_initial_gauche: float, point_initial_droit: float,
                              erreur_visee: float, x=sym.symbols("x")) -> (float, float, int):
    """
    Cette méthode utilise la méthode de résolution d'équation non-linéaire par bissexion.
    Pour être utilisé, l'équation doit être de la forme f(x) = f(x) afin que l'algorithme ne travail
    qu'à évaluer la différence entre les fonctions pour différents points. La méthode itère ne nombre de fois requis
    pour obtenir l'erreur ou un peu inférieur à l'erreur visée (le nombre d'itération calculé n'est pas entier)

    Parameters
    ----------
    fonction_gauche : SymPy object
        fontion du côté gauche de l'équation à résoudre
    fonction_droite : SymPy object
        fontion du côté droite de l'équation à résoudre
    point_initial_gauche :
        Valeur initial à gauche pour débuter l'application de la méthode
        par bissexion
    point_initial_droit :
        Valeur initial à droite pour débuter l'application de la méthode
        par bissexion
    erreur_visee :
        Valeur de l'erreur que nous visons pour le résultats de la résolution
        de l'équation
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    nombre_de_decimal = int(-np.log10(erreur_visee) + 2)
    try:
        f_g_moins_f_d_point_gauche = (fonction_gauche.evalf(nombre_de_decimal, subs={x: point_initial_gauche})
                                      - fonction_droite.evalf(nombre_de_decimal, subs={x: point_initial_gauche}))
        f_g_moins_f_d_point_droit = (fonction_gauche.evalf(nombre_de_decimal, subs={x: point_initial_droit})
                                     - fonction_droite.evalf(nombre_de_decimal, subs={x: point_initial_droit}))
        if f_g_moins_f_d_point_gauche * f_g_moins_f_d_point_droit > 0:
            raise ValueError

        nombre_iteration_pour_erreur = float(np.log2((point_initial_droit - point_initial_gauche) / erreur_visee))
        # Le moins 0.000001 est la pour s'assurer que si le résultat de nombre_iteration_pour_erreur
        # est un nombre entier, que le nombre d'itération soit ce nombre et non ce nombre plus 1
        nombre_iteration = int(float(nombre_iteration_pour_erreur) - 0.0000001) + 1
        erreur_global = (point_initial_droit - point_initial_gauche) / (2 ** nombre_iteration)

        for i in range(nombre_iteration):
            # On vérifie que les deux points sont de part et d'autre d'un intersection
            f_g_moins_f_d_point_gauche = (fonction_gauche.evalf(nombre_de_decimal, subs={x: point_initial_gauche})
                                      - fonction_droite.evalf(nombre_de_decimal, subs={x: point_initial_gauche}))
            f_g_moins_f_d_point_droit = (fonction_gauche.evalf(nombre_de_decimal, subs={x: point_initial_droit})
                                     - fonction_droite.evalf(nombre_de_decimal, subs={x: point_initial_droit}))
            point_centrale = (point_initial_droit + point_initial_gauche) / 2
            if f_g_moins_f_d_point_gauche * f_g_moins_f_d_point_droit > 0:
                point_initial_gauche = point_centrale
            else:
                point_initial_droit = point_centrale

        point_centrale_finale = (point_initial_droit + point_initial_gauche) / 2

        return point_centrale_finale, erreur_global, nombre_iteration

    except (ValueError, ZeroDivisionError) as e:
        if e is ValueError:
            log.error("Les deux points initiales ne sont pas de par et d'autre d'une intersection")
        else:
            log.error("Il y a eu une division par zero lors de l'évaluation de la fonction.")
        return 0, 0, 0


if __name__ == "__main__":
    # Code pour le a)
    """
    masse = 511e3  # eV
    potentiel = 20  # eV
    largeur = 1e-9  # m
    afficher_fonction_d_onde(potentiel, largeur, masse, 0, 20)
    """
    # Code pour le b)
    """
    E = sym.symbols("E")
    c = 3e8  # m/s
    h_bar = 6.582e-16  # eVs
    largeur_du_puit = 1e-9  # nm
    masse = 511e3  # eV
    potentiel = 20  # eV
    y_1 = sym.tan(sym.sqrt(((largeur_du_puit ** 2) * masse * E) / (2 * (c ** 2) * (h_bar ** 2))))
    y_2 = sym.sqrt((potentiel - E) / E)
    y_3 = -sym.sqrt(E / (potentiel - E))
    liste_reslutats = []
    point_1, erreur_1, nb_iteration_1 = resolution_par_bissection(y_1, y_2, 2.5, 3, 0.001, E)
    liste_reslutats.append([point_1, erreur_1, nb_iteration_1])
    point_2, erreur_2, nb_iteration_2 = resolution_par_bissection(y_1, y_2, 7, 9, 0.001, E)
    liste_reslutats.append([point_2, erreur_2, nb_iteration_2])
    point_3, erreur_3, nb_iteration_3 = resolution_par_bissection(y_1, y_2, 10, 17, 0.001, E)
    liste_reslutats.append([point_3, erreur_3, nb_iteration_3])
    point_4, erreur_4, nb_iteration_4 = resolution_par_bissection(y_1, y_3, 0.5, 2.5, 0.001, E)
    liste_reslutats.append([point_4, erreur_4, nb_iteration_4])
    point_5, erreur_5, nb_iteration_5 = resolution_par_bissection(y_1, y_3, 4, 7, 0.001, E)
    liste_reslutats.append([point_5, erreur_5, nb_iteration_5])
    point_6, erreur_6, nb_iteration_6 = resolution_par_bissection(y_1, y_3, 10, 17, 0.001, E)
    liste_reslutats.append([point_6, erreur_6, nb_iteration_6])
    point_7, erreur_7, nb_iteration_7 = resolution_par_bissection(y_1, y_3, 18.8, 19.9, 0.001, E)
    liste_reslutats.append([point_7, erreur_7, nb_iteration_7])
    iterateur = 0
    # les trois premiers points correspondent aux points de y_1 avec y_2
    # tandis que les trois derniers correspondent aux points de y_1 avec y_3
    for resultat in liste_reslutats:
        iterateur += 1
        print("-----------------------------------------")
        print(f"(Valeur du point {iterateur}: {resultat[0]}")
        print(f"(Valeur de l'erreur pour le point {iterateur}: {resultat[1]}")
        print(f"(Nombre d'itération pour obtenir le point {iterateur}: {resultat[2]}")
        """
