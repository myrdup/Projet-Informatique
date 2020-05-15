import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

#img = plt.imread (c"#adresse de l'image#")

'''
Appeler :
LPE(img: numpy.ndarray, image à segmenter
    tolérance : int, hauteur d'un étage de gradients
    flou: int, le nombre de fois que l'image initiale est floutée avant traitement
    seuil: int, taille maximale d'un bassin considéré "fusionnable avec un plus grand"
    intervalle: int, valeur maximale de la norme carrée de la différence des composantes RVB moyennes de deux bassins pour qu'ils soient considérés fusionnables ensembles

Paramètres numériques utilisés pour l'image test : 5,8,120,180
'''

## floutage

def flougris (image):
    im = np.copy (image)
    im [1:-1,1:-1] = 0.1*im[:-2,:-2]+0.1*im[1:-1,:-2]+0.1*im[2:,:-2] + 0.1*im[:-2,1:-1]+0.1*im[2:,1:-1] + 0.1*im[:-2,2:]+0.1*im[1:-1,2:]+0.1*im[2:,2:] + 0.2*im[1:-1,1:-1]
    return im

def floumult (im, n):
    for i in range (n):
        im=flougris (im)
    return im

## gradient

def gris(image, r=1, v=1, b=1):
    compoR=np.array(image[:,:,0], dtype=int)
    compoV=np.array(image[:,:,1], dtype=int)
    compoB=np.array(image[:,:,2], dtype=int)
    retour= r*compoR +v*compoV+b*compoB
    return floumult(retour,5)

def dx (im, x, y):
    """
    paramètres =
    image traitée : im de type np.array
    coordonnées x et y du pixel : x,y de type nombre
    sortie = dérivée en x du pixel : dx de type nombre
    """
    return -1*im[x-1,y-1] -2*im[x-1,y] -1*im[x-1,y+1] + im[x+1,y-1] +2*im[x+1,y] + im[x+1,y+1]

def dy (im, x, y):
    """
    paramètres =
    image traitée : im de type np.array
    coordonnées x et y du pixel : x la ligne,y la colonne de type nombre
    sortie = dérivée en y du pixel : dy de type nombre
    """
    return -1*im[x-1,y+1] -2*im[x,y+1] -1*im[x+1,y+1] + im[x-1,y-1] + 2*im[x,y-1] + im[x+1,y-1]

def gradient_pixel (im, x, y):
    """
    paramètres =
    image traitée : im de type np.array
    coordonnées x et y du pixel : x,y de type nombre
    sortie = norme de gradient de im[x,y] = sqrt(dx**2 + dy**2)
    (méthode du filtre de Sobel)
    ramené entre 0 et 255
    """
    return 255/(np.sqrt(2)*1020) * np.sqrt(dx(im,x,y)**2 + dy(im,x,y)**2)

def gradient_imageNG (im):
    """
    paramètres =
    image traitée : im de type np.array en niveaux de gris
    sortie = tableau des normes de gradient pour chaque pixel de l'image
    de type np.array
    /!\ les pixels de bord ont un gradient nul par convention
    """
    lx= len(im)
    ly= len(im[0])
    gr=np.zeros((lx, ly))
    for x in range (1,lx-1):
        for y in range (1,ly-1):
            gr[x,y]= int(gradient_pixel(im,x,y))
    return gr

def aff_gris (im):
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.show()

def GRADIENT (im):
    GR= gradient_imageNG(gris(im,1,1,1))
    #aff_gris(GR)
    return GR

## Travail préparatoire

def InitTableau (image) :
    #On crée un tableau contenant la valeur du gradient du pixel, et le label qui correspond au bassin de ce pixel, -1 si le pixel n’est pas affecté
    n= len(image)
    m=len(image[0])
    tableau = np.ones ( (n, m, 2) )
    tableau[:,:,1]*=-1
    tableau[0,:,1]=0
    tableau [n-1,:,1]=0
    tableau[:,0,1]=0
    tableau[:,m-1,1]=0
    for i in range (n) :
        for j in range (m) :
            tableau[i,j,0] = image[i][j]
    return tableau

def tri (t) :
    paquets = []
    for i in range (256) :
        paquets.append([])
    for x in t :
        paquets[int(x[0])].append (x)
    resultat = []
    for i in range (256) :
         resultat += paquets[i]
    return resultat


def InitListeOrdonnee (image) :
    #On crée une liste, triée par ordre croissant de gradient qui contient la valeur du gradient associée au pixel et sa position i,j dans l'image
    liste = []
    for i in range (1, len(image)-1) :
        for j in range (1, len(image[0])-1) :
            liste.append ([image[i][j],i,j])
    liste = tri(liste)
    return liste

def CreerEtage (t, n):
    # (t,n) = tableau des pixels ordonnés, tolerance de l'étage, renvoie l = liste python de n listes de pixels(tableaux np) par gradient croissant
    num_et=1
    gr_max= n
    etage=[]
    l=[]
    for p in t: #pour tous les pixels du tableau, sachant qu’ils sont triés par ordre croissant
        if p[0]<=gr_max:
            etage.append(p[1:]) #ajoutés à l’étage si leur gradient< gradient limite - on n’ajoute
        else :
            l.append(etage) #sinon l’étage est bouclé, on l’ajoute à la liste des étages
            while p[0]>gr_max :
                num_et+=1 # on crée l’étage suivant
                gr_max=num_et*n # chaque étage est de tolérance n, donc la valeur limité du gradient est n* (numéro de l’étage)
                etage=[] # on réinitialise la liste des pixels de l’étage
            etage.append(p[1:])
    l.append(etage) #après avoir affecté le dernier pixel, on ajoute le dernier étage à la liste des étages
    return l # liste des étages, chaque étage est une liste de pixels


def suppr_doublons (liste):
    res = []
    for e in liste:
        if e not in res:
            res.append(e)
    return res

## fonctions sur les files

def cree_file ():
    return {"entree":("sortie",None),"sortie":(None,"entree")}

def pousse(file, x):
    erreur=False
    try:
        file[x] #si c'est dans la file alors on ne l'ajoute pas
        erreur=True
    except Exception: #sinon on l'ajoute
        (pred,_)=file["entree"]
        file["entree"]=(x,None)
        file[x]=(pred,"entree")
        (pred_pred,_)=file[pred]
        file[pred]=(pred_pred,x)
    return erreur

def est_vide(file):
    (pred,_)=file["entree"]
    (_,succ)=file["sortie"]
    return pred=="sortie" and succ=="entree"

def tire(file):
    if est_vide (file): return "vide"

    (_,cle)=file["sortie"]
    (_,cle_suc)=file.pop(cle)
    file["sortie"]=(None,cle_suc)
    (_,cle_suc_suc)=file[cle_suc]
    file[cle_suc]=("sortie",cle_suc_suc)
    return(cle)

def est_dans(file,x):
    try:
        file[x]
        return True
    except Exception:
        return False

def detruit(file,x):
    if est_vide(file):
        return False

    try:
        (pred,suc)=file.pop(x)
        (pred_pred,_)=file[pred]
        file[pred]=(pred_pred,suc)
        (_,suc_suc)=file[suc]
        file[suc]=(pred,suc_suc)
        return True

    # accès à un élément en O(1) avec pop(element)
    # 2:(1,3), 3:(2,4), 4:(3,5) --> on ôte 3
    # --> son pred est 2 et son successeur est 4
    # le nouveau successeur de 2 n'est plus 3 mais 4
    # le nouveau predecesseur de 4 n'est plus 3 mais 2

    except Exception:
        return False

## fonctions de segmentation

def labellisation (i,j,TablImage,file_etage,file_attente,Labelspar) :

    Labels = Labelspar #M# ajout
    labels_voisins = []
    nb_lpe=0
    voisins = ((i-1,j),(i+1,j),(i,j-1),(i,j+1))
    for pixel in voisins :
        x,y = pixel
        if TablImage[x,y,1] == 0 : nb_lpe+=1
        elif TablImage[x,y,1] == -1:
            if detruit (file_etage, pixel):
                erreur= pousse(file_attente,pixel) #si c'était dans la file déjà
                if erreur: print ("erreur")

        else : labels_voisins.append(TablImage[x,y,1]) # On répertorie les valeurs de labels des voisins s'ils en ont déjà 1 (hors lpe)
    labels_voisins = suppr_doublons(labels_voisins) # On ne garde que les labels différents

   # choix du label
    if len(labels_voisins) == 1 :
        TablImage[i][j][1] = labels_voisins[0] # on lui attribue le même label que son unique voisin labellisé
    elif len(labels_voisins) > 1 :
        TablImage[i][j][1] = 0 # Le label 0 est celui des lpe
    elif nb_lpe==4 : # si tous les voisins sont des lpe
        TablImage[i][j][1] = 0 # Le label 0 est celui des lpe
    else:
        Labels+=1
        TablImage[i][j][1] = Labels # on crée un nouveau bassin
    return Labels


def lpe (TablImage, ListeOrdonnee, ListeEtage) :
    Labels = 0

    for etage in ListeEtage :

        #création d'une file contenant tous les éléments de etage / C= O(len(etage)
        file_etage = cree_file ()
        for pixel in etage:
            [x,y]=pixel
            pousse (file_etage,(x,y))

        while not est_vide (file_etage):
                #print("on ininitialise une nouvelle file d'attente")
                file_attente= cree_file ()
                pousse(file_attente,tire(file_etage))

                while not est_vide (file_attente):
                    x = tire(file_attente)
                    (i,j)=x
                    Labels=labellisation (i,j,TablImage,file_etage,file_attente,Labels)

    return Labels

## Fusion des petits bassins

def creeEns_labels(Labels):
    """renvoie la file des labels"""
    ens= cree_file()
    for i in range(Labels+1):
        pousse(ens, i)
    return ens

def creeBassins (Labels,TablImage,image) :
    """
    renvoie une liste de taille Labels(initial) +1
    ligne = label du bassin
        colonne 0 = son effectif = int
        colonne 1 = son contenu = liste de pixels
        colonne 2 = valeurs moyennes RVB = liste de 3 float (?)
        colonne 3 = false si on a tenté de traiter ce bassin et ça ne marchait pas, true sinon

    Complexité : O(n)
    """
    bassins = [[0,[],[0,0,0],True] for i in range (Labels+1)]
    for x in range (len(TablImage)) :
        for y in range (len(TablImage[0])) :
            label = int(TablImage[x,y,1])
            [r,v,b]=image[x,y]
            #mise à jour de l'effectif du bassin
            bassins[label][0]+=1
            #mise à jour du contenu du bassin
            bassins[label][1].append ((x,y))
            #mise à jour de la moyenne r, v et b du bassin
            bassins[label][2][0]+=r
            bassins[label][2][1]+=v
            bassins[label][2][2]+=b
    #division des compos rvb par le nombre de pixels pour obtenir une moyenne
    for b in range (1,len(bassins)):
        bassins[b][2]=np.vectorize(lambda x: x/bassins[b][0])(bassins[b][2]) #la compo rvb donne un np.array après ce traitement
    return bassins

def bassin_a_traiter (bassins,seuil,TablImage) :
    """renvoie le n° du bassin, -1 s'il n'y en a pas """
    label_traite = -1
    min = seuil + 1
    for b in range(len(bassins)): #on parcourt tous les bassins
        if bassins[b][3]: #en ne regardant que les bassins traitables
            eff= bassins[b][0]
            if eff<min :
                min = eff
                label_traite = b #on mémorise le label de celui d'effectif minimum
    return label_traite

def dans_intervalle (R_compare,V_compare,B_compare,R_comparant,V_comparant,B_comparant, intervalle) :
    """
    renvoie un booléen
    true si la couleur du comparé est suffisemment proche de la couleur du comparant
    (si la difference pour chaque composante est dans l'intervalle)
    """
    return ( abs(R_compare - R_comparant) <= intervalle and abs(V_compare - V_comparant) <= intervalle and abs(B_compare - B_comparant) <= intervalle )

def norme_diff (R_compare,V_compare,B_compare,R_comparant,V_comparant,B_comparant) :
    """
    renvoie la norme de la différence entre la couleur du comparé et du comparant
    """
    return np.sqrt ((R_compare - R_comparant)**2 + (V_compare - V_comparant)**2 + (B_compare - B_comparant)**2)

def plus_proche (bassins,bassins_voisins, label_traite, intervalle) :
    """
    renvoie le label du bassin_voisin dont la couleur moyenne est la plus proche de celle de notre bassin_traité
    OU
    -5 si aucun bassin_voisin n'est de couleur suffisemment proche pour justifier de fusionner notre bassin_traité avec
    """
    R_compare,V_compare,B_compare= bassins[label_traite][2][0],bassins[label_traite][2][1],bassins[label_traite][2][2]

    norme_min  = 30000 #equivalent à l'infini
    label_min = -5 #inexistant
    for voisin in bassins_voisins : #labelsvoisins a été calculé avec BASSINS_VOISINS
        v=int(voisin)
        if v != 0 : # pour tout bassin voisin hors lpe
            R_comparant,V_comparant,B_comparant = bassins[v][2][0],bassins[v][2][1],bassins[v][2][2],
            if dans_intervalle (R_compare,V_compare,B_compare,R_comparant,V_comparant,B_comparant, intervalle):
            # les 2 bassins sont-ils suffisemment proches au sens de chaque couleur?
                norme = norme_diff (R_compare,V_compare,B_compare,R_comparant,V_comparant,B_comparant)
                if norme<norme_min :
                #forcément <30000, min récupère donc la norme minimale entre le bassin comparé et chaque voisin suffisemment proche ! à condition qu'il existe un bassin suffisemment proche
                    norme_min = norme
                    label_min = v
                    #label_min s'actualise avec le label du bassin le plus proche en terme de couleur moyenne
                    #reste à -5 si aucun bassin n'est suffisemment proche pour qu'on fusionne
    if label_min==-5 : bassins[label_traite][3]=False
    return label_min

def fusion_dans_bassins_et_TablImage (lt, lm, liste_lpes, TablImage, bassins, ens, image):
    """
    lt=label_traite
    lm=label_min
    ens= ensemble des labels
    à appeler si label_min !=-5!
    MODIFIE bassins,TablImage,ens_labels après la fusion de label_traité et label_min
    ne renvoie rien
    """
    #effectif?
    eff_lt = bassins[lt][0]
    eff_lm = bassins[lm][0]
    eff_nv = eff_lt + eff_lm

    #couleur moyenne?
    r = (eff_lt*bassins[lt][2][0] + eff_lm*bassins[lm][2][0])/eff_nv
    v = (eff_lt*bassins[lt][2][1] + eff_lm*bassins[lm][2][1])/eff_nv
    b = (eff_lt*bassins[lt][2][2] + eff_lm*bassins[lm][2][2])/eff_nv

    #label?
    label_nv=len(bassins)

    #création du bassin issu de la fusion
    bassins.append([eff_nv,[],[r,v,b],True])

    #composition?
    for pixel in (bassins[lt][1]):
        (x,y)=pixel
        bassins[label_nv][1].append(pixel)
        TablImage[x,y,1]=label_nv

    for pixel in (bassins[lm][1]):
        (x,y)=pixel
        bassins[label_nv][1].append(pixel)
        TablImage[x,y,1]=label_nv

    #on vide les bassins lt et lm
    bassins[lt]=[0,[],[0,0,0],False]
    bassins[lm]=[0,[],[0,0,0],False]

    #traitement des LPE
    n=len(TablImage)
    m=len(TablImage[0])
    for a in liste_lpes : #lpes entourant le bassin = seules susceptibles d'avoir été modifiées
        (x,y) = a
        if y == 0 and (1 <= x < n-1) :
            voisin = ((x-1,y),(x+1,y),(x,y+1))
        elif y == 0 and x == 0 :
            voisin = ((x,y+1),(x+1,y))
        elif x == 0 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x+1,y))
        elif x == 0 and y == m-1 :
            voisin = ((x,y-1),(x+1,y))
        elif y == m-1 and (1 <= x < n-1) :
            voisin = ((x,y-1),(x-1,y),(x+1,y))
        elif y == m-1 and x == n-1 :
            voisin = ((x-1,y),(x,y-1))
        elif x == n-1 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x-1,y))
        elif x == n-1 and y == 0 :
            voisin = ((x-1,y),(x,y+1))
        else :
            voisin = ((x-1,y),(x,y-1),(x,y+1),(x+1,y))
            #voisins d'un pixel lpe
        nb_label = 0
        label_voisin = []
        for b in voisin :
            if (TablImage[b[0],b[1],1] not in label_voisin) and (TablImage[b[0],b[1],1] != 0) :
                label_voisin.append(TablImage[b[0],b[1],1])
                #liste des labels voisins
                nb_label += 1 #nombre de labels voisins
        if nb_label == 1 :
            nouv_label = int(label_voisin[0])
            if nouv_label != label_nv : print ("c'est bizarre")
            # mise à jour de TablImage
            TablImage[x,y,1] = nouv_label
            # on ne modifie pas les infos relatives aux lpe initiales
            # MAIS on modifie celles du bassin auquel on les ajoute
            bassins[nouv_label][1].append((x,y))
            bassins[nouv_label][2][0] = (bassins[nouv_label][0]*bassins[nouv_label][2][0] + image[x,y,0])/(bassins[nouv_label][0]+1)
            bassins[nouv_label][2][1] = (bassins[nouv_label][0]*bassins[nouv_label][2][1] + image[x,y,1])/(bassins[nouv_label][0]+1)
            bassins[nouv_label][2][2] = (bassins[nouv_label][0]*bassins[nouv_label][2][2] + image[x,y,2])/(bassins[nouv_label][0]+1)
            bassins[nouv_label][0]+=1

    #mise à jour de l'ensemble des labels
    detruit(ens,lt)
    detruit(ens, lm)
    pousse(ens,label_nv)

    return label_nv


def lpe_voisines_et_bassins_voisins (label_traite,TablImage, bassins) :
    """
    renvoie la liste des lpe du bassin_traité
    et la liste des labels voisins du bassin_traité

    pas de modification ni de TablImage ni de bassins
    """
    n=len(TablImage)
    m=len(TablImage[0])
    labels_voisins = []
    voisins_lpe=[]
    for a in bassins[label_traite][1] : #pour tous les pixels du bassin_traité
        (x,y) = a
        if y == 0 and (1 <= x < n-1) :
            voisin = ((x-1,y),(x+1,y),(x,y+1))
        elif y == 0 and x == 0 :
            voisin = ((x,y+1),(x+1,y))
        elif x == 0 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x+1,y))
        elif x == 0 and y == m-1 :
            voisin = ((x,y-1),(x+1,y))
        elif y == m-1 and (1 <= x < n-1) :
            voisin = ((x,y-1),(x-1,y),(x+1,y))
        elif y == m-1 and x == n-1 :
            voisin = ((x-1,y),(x,y-1))
        elif x == n-1 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x-1,y))
        elif x == n-1 and y == 0 :
            voisin = ((x-1,y),(x,y+1))
        else :
            voisin = ((x-1,y),(x,y-1),(x,y+1),(x+1,y))
        for b in voisin :
            if TablImage[b[0],b[1],1] == 0 :
                voisins_lpe.append (b) #on récupère les pixels lpe du bassin traité

    for a in voisins_lpe :
        (x,y) = a
        if y == 0 and (1 <= x < n-1) :
            voisin = ((x-1,y),(x+1,y),(x,y+1))
        elif y == 0 and x == 0 :
            voisin = ((x,y+1),(x+1,y))
        elif x == 0 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x+1,y))
        elif x == 0 and y == m-1 :
            voisin = ((x,y-1),(x+1,y))
        elif y == m-1 and (1 <= x < n-1) :
            voisin = ((x,y-1),(x-1,y),(x+1,y))
        elif y == m-1 and x == n-1 :
            voisin = ((x-1,y),(x,y-1))
        elif x == n-1 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x-1,y))
        elif x == n-1 and y == 0 :
            voisin = ((x-1,y),(x,y+1))
        else :
            voisin = ((x-1,y),(x,y-1),(x,y+1),(x+1,y))
        for b in voisin :
            if TablImage[b[0],b[1],1] != 0 and TablImage[b[0],b[1],1] != label_traite :
                labels_voisins.append (TablImage[b[0],b[1],1]) #on récupère les labels voisins, on supprimera les doublons

    return voisins_lpe, suppr_doublons (labels_voisins)


def fusion (img,TablImage,bassins, ens, intervalle,seuil, Labels, image) :

    for i in range (Labels-1) : #on ne peut fusionner plus de x que le nbr de labels initial-1

        label_traite= bassin_a_traiter(bassins,seuil,TablImage)
        if label_traite != -1 :
            lpe_voisines, bassins_voisins = lpe_voisines_et_bassins_voisins(label_traite,TablImage,bassins)

            label_min = plus_proche(bassins,bassins_voisins, label_traite, intervalle)
            if label_min !=-5 :
                label_nv = fusion_dans_bassins_et_TablImage (label_traite, label_min, lpe_voisines, TablImage, bassins, ens,image)

    traitement_LPE(TablImage)
    traitement_LPE(TablImage)


def traitement_LPE (TablImage) :
    n = len(TablImage)
    m = len(TablImage[0])
    lpe = []
    for i in range (n) :
        if TablImage[i,0,1] == 0 :
            lpe.append ((i,0))
        if TablImage[i,m-1,1] == 0 :
            lpe.append((i,m-1))
    for j in range (1,m-1) :
        if TablImage[0,j,1] == 0 :
            lpe.append((0,j))
        if TablImage[n-1,j,1] == 0 :
            lpe.append((n-1,j))

    for a in lpe :
        (x,y) = a
        if y == 0 and (1 <= x < n-1) :
            voisin = ((x-1,y),(x+1,y),(x,y+1))
        elif y == 0 and x == 0 :
            voisin = ((x,y+1),(x+1,y))
        elif x == 0 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x+1,y))
        elif x == 0 and y == m-1 :
            voisin = ((x,y-1),(x+1,y))
        elif y == m-1 and (1 <= x < n-1) :
            voisin = ((x,y-1),(x-1,y),(x+1,y))
        elif y == m-1 and x == n-1 :
            voisin = ((x-1,y),(x,y-1))
        elif x == n-1 and (1 <= y < m-1) :
            voisin = ((x,y-1),(x,y+1),(x-1,y))
        elif x == n-1 and y == 0 :
            voisin = ((x-1,y),(x,y+1))
        nb_label = 0
        label_voisin = []
        for b in voisin :
            if (TablImage[b[0],b[1],1] not in label_voisin) and (TablImage[b[0],b[1],1] != 0) :
                label_voisin.append(TablImage[b[0],b[1],1])
                nb_label += 1
        if nb_label == 1 :
            TablImage[x,y,1] = label_voisin[0]



## fonctions d'affichage

def Coloration(TablImage,Labels) :
    l=[0 for i in range(Labels+1)]
    n = len(TablImage)
    m = len(TablImage[0])
    resultat = np.zeros((n,m,3),dtype=int)
    coloration = [[0,0,0]]*(Labels+1)
    for i in range (1,len(coloration)) :
        coloration[i] = [rd.randint(100,200),rd.randint(100,200),rd.randint(100,200)]
    #print(coloration)
    for i in range (n) :
        for j in range (m) :
            label = int(TablImage[i,j,1])
            resultat[i,j] = coloration[label]
            l[label]+=1
    return resultat

def Coloration_fusion (TablImage,bassins) :

    #on calcule la moyenne des couleurs pour chaque bassin
    bassins[0][2]=[0,0,0] #lpe en noir

    #création de l'image colorée
    n = len(TablImage)
    m = len(TablImage[0])
    resultat = np.zeros((n,m,3),dtype=int)

    for i in range (n) :
        for j in range (m) :
            label = int(TablImage[i,j,1])
            resultat[i,j] = bassins[label][2]
    return resultat

def nbr_labels (ens):
    n=0
    while not est_vide(ens):
        p=tire(ens)
        n+=1
    return n

def afficher_image (img):
    plt.imshow(img)
    plt.show()

def LPE (img, tolerance,flou,seuil,intervalle):
    start = time.perf_counter ()

    #segmentation
    image=GRADIENT(floumult(img,flou))
    TablImage = InitTableau(image)
    print("TablImage réalisé")
    ListeOrdonnee = InitListeOrdonnee (image)
    print("ListeOrdonnee réalisée de longueur : ",len(ListeOrdonnee))
    ListeEtage = CreerEtage(ListeOrdonnee, tolerance)
    print("ListeEtage réalisée de longueur : ", len(ListeEtage))
    Labels = lpe(TablImage, ListeOrdonnee, ListeEtage)
    ImageColoree1= Coloration(TablImage,Labels)
    endseg = time.perf_counter()
    print("nombre de bassins initial=", Labels)
    afficher_image(ImageColoree1)

    #fusion
    debut_fus= time.perf_counter()
    ens=creeEns_labels(Labels)
    bassins=creeBassins(Labels,TablImage,img)
    fusion(img, TablImage,bassins,ens,intervalle,seuil,Labels, img)
    ImageColoree2=Coloration_fusion(TablImage,bassins)
    nbr_bassins=nbr_labels(ens)
    endfus=time.perf_counter ()
    print("nombre de bassins total=",nbr_bassins)

    dureeseg = endseg-start
    dureefus=endfus-debut_fus
    print("duree segmentation =",dureeseg, "\n", "duree fusion=", dureefus)

    #affichage
    afficher_image(ImageColoree2)