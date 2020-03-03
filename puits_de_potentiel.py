import numpy as np
import matplotlib.pyplot as plt


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
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # les plus ou moins 0.001 sont présent pour éviter les divisions par 0
    nb_de_bonds = int((fin_de_la_plage_energie +0.001 - debut_de_la_plage_energie - 0.001)*10)
    valeurs_de_x = np.linspace(debut_de_la_plage_energie + 0.001, fin_de_la_plage_energie - 0.001, nb_de_bonds)
    c = 3e8 #m/s
    h_bar = 6.582e-16 #eVs
    fonction_1 = lambda x: np.tan(np.sqrt(((largeur_du_puit**2)*masse*x)/(2*(c**2)*(h_bar**2))))
    fonction_2 = lambda x: np.sqrt((potentiel-x)/x)
    fonction_3 = lambda x: -np.sqrt(x/(potentiel-x))
    ax.plot(valeurs_de_x, fonction_1(valeurs_de_x), color='blue', lw=2)
    ax.plot(valeurs_de_x, fonction_2(valeurs_de_x), color='red', lw=2)
    ax.plot(valeurs_de_x, fonction_3(valeurs_de_x), color='black', lw=2)
    ax.set_title("Graphique illustrant les résultats de l'équation correspondant\n"
                 " aux intersection entre les fonction de gauche et de droite de\n l'équation.")
    ax.set_xlabel("Énergie [eV]")
    ax.set_ylabel("Valur associé à la fonction [-]")

    plt.grid()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    afficher_fonction_d_onde(20, 1e-9, 511e3, 0, 20)

