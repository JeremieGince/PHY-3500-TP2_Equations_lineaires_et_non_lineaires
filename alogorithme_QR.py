import numpy as np
import logging as log


def decomposition_matriciel_qr(matrice_a_decomposer: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Cette méthode décompose une matrice réel carré en deux matrice carrée, une constitué de N
    vecteurs orthonormaux et l'autre est triangulaire supérieur. Pour ce faire nous devons considérer un
    chaque colonne de la matrce comme des vecteurs que nous changeons dans un autre espace.

    Parameters
    ----------
    matrice_a_decomposer :
       matrice carrée réelle

    Returns
    ----------
    matrice_q:
        La matrice carrée réelle Q qui contient les vecteurs orthonormaux
    matrice_r:
        La matrice carrée réelle R correspondant à la matrice triangulaire supérieur
    """
    dimension = len(matrice_a_decomposer)
    try:
        if dimension != len(matrice_a_decomposer[0]):
            raise ValueError
        if matrice_a_decomposer.dtype == complex:
            raise ValueError

    except ValueError:
        log.error("La matrice donnée à la fonction decomposition_matriciel_qr doit être carrée et réel")
        return np.zeros((dimension, len(matrice_a_decomposer[0]))), np.zeros((dimension, len(matrice_a_decomposer[0])))

    matrice_q = np.empty((dimension, dimension))
    matrice_r = np.zeros((dimension, dimension))
    # Nous allons chercher chaque colonne de la matrice à décomposer
    for i in range(dimension):
        colonne = matrice_a_decomposer[:, i]
        q_i_non_normalise = colonne - somme_de_projections_vectoriels_sur_colonnes_de_matrice(colonne, matrice_q,
                                                                                              list(range(0, i)))
        norme_q_i = np.linalg.norm(q_i_non_normalise)
        colonne_q = q_i_non_normalise / norme_q_i
        matrice_q[:, i] = colonne_q
        # Nous incérons les valeurs de la matrice R pour la colonne correspondante
        for j in range(i + 1):
            if j == i:
                matrice_r.itemset((j, i), norme_q_i)

            else:
                matrice_r.itemset((j, i), float(np.dot(colonne, matrice_q[:, j])))

    return matrice_q, matrice_r


def somme_de_projections_vectoriels_sur_colonnes_de_matrice(vecteur_projete: np.ndarray, matrice: np.array,
                                                            liste_des_colonnes_pour_la_somme: list) -> np.ndarray or int:
    """
    Cette méthode calcule la somme de la projection d'un vecteur sur
    les colonnes d'une matrice. La projection se fait que sur les colonnes
    dont l'indice est dans la liste_des_colonnes_pour_la_somme. Pour que la méthode
    s'exécute, il faut que les colonnes de la matrice et la longueur du vecteur correspondent.

    Parameters
    ----------
    vecteur_projete :
       Vecteur que l'on projette sur les colonnes
    matrice :
       matrice contenant les colonnes pour la projection
    liste_des_colonnes_pour_la_somme:
        liste des indices correspondants aux colonnes
        que nous voulons utiliser pour les projections

    Returns
    ----------
    resultat_de_la_somme:
        retourne le vecteur résultant de la somme.
        Si la somme est nulle, la méthode retourne 0
    """
    dimension_colonne_matrice = len(matrice[0])
    try:
        if dimension_colonne_matrice != len(vecteur_projete):
            raise ValueError

    except ValueError:
        log.error("Les colonnes de la matrice doivent être de la même longueur que le vecteur")
        return 0

    if liste_des_colonnes_pour_la_somme == []:
        return 0
    else:
        resultat_de_la_somme = 0
        for i in liste_des_colonnes_pour_la_somme:
            colonne = matrice[:, i]
            resultat_projection = np.dot(vecteur_projete, colonne) * colonne
            resultat_de_la_somme = resultat_de_la_somme + resultat_projection

        return resultat_de_la_somme


def decomposition_matricielle_valeurs_propres_et_vecteurs_prpres(matrice: np.ndarray,
                                                    valeur_max_tolere_non_diagonale: float) -> (list, list, int, float):
    """
    Cette méthode décompose une matrice réel carré en une matrice diagonale contenant les valeurs propres
    de la matrice et une matrice composé de vecteurs orthonormaux propres à la matrice.
    Pour ce faire, l'algorithme pour obtenir les matrices QR est utiliser jusqu'à ce QR donne une matrice diagonale.

    Parameters
    ----------
    matrice :
       matrice carrée réelle que nous voulons décomposer
    valeur_max_tolere_non_diagonale:
        valeur maximal que nous tolérons sur les éléments non diagonales

    Returns
    ----------
    matrice_diag:
        La matrice diagonalisé
    V:
        Matrice contenant les vecteurs propres associés aux valeurs propres
    iterateur:
        Indice pour savoir le nombre d'itération requise pour produire le résultat
    erreur:
        Différence moyenne des éléments non diagonales par rapport à 0 représente l'erreur sur les résultat données
    """
    matrice_diag = matrice
    V = np.identity(4)
    diag_mask = (np.zeros((len(matrice), len(matrice))) + np.diag(np.ones(len(matrice)))).astype(np.bool)
    iterateur = 0
    while np.amax(np.ma.masked_where(diag_mask, matrice_diag)) >= valeur_max_tolere_non_diagonale:
        iterateur += 1
        Q, R = decomposition_matriciel_qr(matrice_diag)
        matrice_diag = R @ Q
        V = V @ Q
    erreur = np.abs(np.ma.masked_where(diag_mask, matrice_diag)).mean()
    return matrice_diag, V, iterateur, erreur


if __name__ == "__main__":
    # Code pour la question a)
    """
    matriceA = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9], [4, 7, 9, 2]], float)
    print(f"Matrice A:{matriceA}")
    matriceQ, matriceR = decomposition_matriciel_qr(matriceA)
    print(f"Matrice Q:{matriceQ}")
    print(f"Matrice R:{matriceR}")
    produit = np.matmul(matriceQ,matriceR)
    print(f"Matrice QR:{produit}")
    """
    matriceA = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9], [4, 7, 9, 2]], float)
    matrice_diag, V, iterateur, erreur = decomposition_matricielle_valeurs_propres_et_vecteurs_prpres(matriceA,
                                                                                                      0.000001)
    print(matrice_diag)
    print(V)
    print(iterateur)
    print(erreur)
