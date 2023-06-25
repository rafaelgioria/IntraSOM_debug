import numpy as np
from sklearn.preprocessing import minmax_scale
from numpy import nan, dot, nanmean
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tqdm.auto import tqdm
from scipy import sparse as sp
from PIL import Image
import glob
import os
from textwrap import fill
from numpy import pi, sin, cos
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pkg_resources
import plotly.graph_objs as go
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.metrics.pairwise import nan_euclidean_distances
import statistics



class PlotFactory(object):

    def __init__(self, som_object):
        self.name = som_object.name
        self.codebook = som_object.codebook.matrix
        self.mapsize = som_object.mapsize
        self.bmus = som_object._bmu[0].astype(int)
        self.bmu_matrix = som_object.bmu_matrix
        self.component_names = som_object._component_names
        self.sample_names = som_object._sample_names
        self.unit_names = som_object._unit_names
        self.rep_sample = som_object.rep_sample
        self.data_denorm = som_object.denorm_data(som_object._data)
        self.data_proj_norm = som_object.data_proj_norm
        
        # Load foot image
        image_file = pkg_resources.resource_filename('intrasom', 'images/foot.jpg')
        self.foot = Image.open(image_file)

    def build_umatrix(self, expanded=False, log=False):
        """
        Função para calcular a Matriz U de distâncias unificadas a partir da
        matriz de pesos treinada.

        Args:
            exapanded: valor booleano para indicar se o retorno será da matriz
                de distâncias unificadas resumida (média das distâncias dos 6
                bmus de vizinhança) ou expandida (todos os valores de distância)
        Retorna:
            Matriz de distâncias unificada expandida ou resumida.
        """
        # Função para encontrar distancia rápido
        def fast_norm(x):
            """
            Retorna a norma L2 de um array 1-D.
            """
            return sqrt(dot(x, x.T))

        # Matriz de pesos bmus
        weights = np.reshape(self.codebook, (self.mapsize[1], self.mapsize[0], self.codebook.shape[1]))

        # Busca hexagonal vizinhos
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]

        # Inicializar Matriz U
        um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))

        # Preencher matriz U
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w_2 = weights[y, x]
                e = y % 2 == 0
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
                        w_1 = weights[y+j, x+i]
                        um[y, x, k] = fast_norm(w_2-w_1)
        if expanded:
            # Matriz U expandida
            return np.log(um) if log else um
        else:
            # Matriz U reduzida
            return nanmean(np.log(um), axis=2) if log else nanmean(um, axis=2)
                        
    def plot_umatrix(self,
                     figsize = (10,10),
                     hits = True,
                     title = "Matriz U",
                     title_size = 40,
                     title_pad = 25,
                     legend_title = "Distância",
                     legend_title_size = 25,
                     legend_ticks_size = 20,
                     save = True,
                     watermark_neurons = False,
                     watermark_neurons_alfa = 0.5,
                     neurons_fontsize = 7,
                     file_name = None,
                     file_path = False, 
                     resume = False,
                     label_plot = False, 
                     label_plot_name = None,
                     project_samples_label = None,
                     samples_label = False,
                     samples_label_index = None,
                     samples_label_fontsize = 8,
                     save_labels_rep = False,
                     label_title_xy = (0,0.5),
                     log=False):
        """
        Função para plotar a Matriz U de distâncias unificadas.

        Args:
            figsize: tamanho da tela de plotagem da matriz U. Padrão: (10, 10)

            hits: booleano para indicar a plotagem dos hits dos vetores de entrada 
                nos BMUs (proporcionais à quantidade de vetores por BMU)

            title: título da figura criada. Padrão: "Matriz U"

            title_size: tamanho do título plotado. Padrão: 40

            title_pad: espaçamento entre o título e a parte superior da matriz. Padrão: 25

            legend_title: título da barra de cores de legenda. Padrão: "Distância"

            legend_title_size: tamanho do título da legenda. Padrão: 25

            legend_ticks_size: tamanho dos algarismos da barra de cores da legenda. Padrão: 20

            save: booleano para definir o salvamento da imagem criada. O salvamento será feito 
                no diretório (Plotagens/Matriz_U). Padrão: True

            watermark_neurons: booleano para adicionar marca d'água com os BMUs na imagem. Padrão: False

            file_name: nome que será dado ao arquivo salvo. Se nenhum nome for fornecido, será 
                utilizado o nome do projeto.

            file_path: caminho no sistema em que a imagem deverá ser salva, caso não se opte por 
                utilizar o caminho padrão.

            resume: booleano para plotar apenas a parte superior da matriz U, omitindo a parte 
                inferior espelhada. Padrão: False

            label_plot: booleano para adicionar rótulos aos hexágonos da matriz U. Padrão: False
            
            label_plot_index: indices que mapeiam rótulos específicos para cada hexágono. Padrão: None

            samples_label: booleano para indicar a plotagem de rótulos de amostras selecionadas.
             Padrão: False.
            
            samples_label_index: lista de indices das amostras selecionadas para plotagem de rótulos.
             Padrão: None. "All" para plotagem de todas as amostras.
            
            samples_label_fontsize: tamanho da fonte dos rótulos plotados. Padrão: 8.
            
            save_labels_rep: booleano para indicação do salvamento de um arquivo .txt em Resultados
             com as amostras selecionadas e a sua ordem de representatividade em cada BMU. Padrão: False.
            
            label_title_xy: coordenadas (x, y) para posicionar o título dos rótulos. Padrão: (-0.02, 1.1)
            
            log: booleano para plotar a matriz U em escala logarítmica para melhor visualização das 
                fronteiras de dissimilaridade na presença de outliers.


        Retorna:
            A imagem com a plotagem da Matriz U de distâncias unificadas.
        """
        def select_keys(original_dict, keys_list):
            new_dict = {}
            keys_list = np.array(keys_list)
            if keys_list.shape[0]>1:
                for key in keys_list:
                    if key in original_dict:
                        new_dict[key] = original_dict[key]
            else:
                if keys_list[0] in original_dict:
                    new_dict[keys_list[0]] = original_dict[keys_list[0]]
            return new_dict
        
        def search_strings(search_list, target_list):
            found_index = None
            found_string = None

            for search_string in search_list:
                for index, target_string in enumerate(target_list):
                    if search_string == target_string:
                        if found_index is None or index < found_index:
                            found_index = index
                            found_string = search_string
                        break

            return found_string, found_index

        if file_name is None:
            file_name = f"Matriz_U_{self.name}"

        if hits:
            bmu_dic = self.hits_dictionary

        # Criar coordenadas
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # Matriz U
        um = self.build_umatrix(expanded = True, log=log)
        umat = self.build_umatrix(expanded = False, log=log)
        
        
        if resume:
            # Plotagem
            prop = self.mapsize[1]*0.8660254/self.mapsize[0]
            f = plt.figure(figsize=(5, 5*prop), dpi=300)
            f.patch.set_facecolor('blue')
            ax = f.add_subplot()
            ax.set_aspect('equal')

            # Normalizar as cores para todos os hexagonos
            norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
            counter = 0

            for j in range(self.mapsize[1]):
                for i in range(self.mapsize[0]):
                    # Hexagono Central
                    hex = RegularPolygon((xx[(j, i)]*2,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor= cm.jet(norm(umat[j][i])),
                                         alpha=1)#, edgecolor='black')

                    ax.add_patch(hex)

                    # Hexagono da Direita
                    if not np.isnan(um[j, i, 0]):
                        hex = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,0])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Hexagono Superior Direita
                    if not np.isnan(um[j, i, 1]):
                        hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,1])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Hexagono Superior Esquerdo
                    if not np.isnan(um[j, i, 2]):
                        hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,2])),
                                             alpha=1)
                        ax.add_patch(hex)
                        

                    
                    if j==0:
                        # Hexagono Central
                        hex = RegularPolygon((xx[(j, i)]*2,
                                              yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='black')
                        ax.add_patch(hex)
                        
                        # Direita
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2+1,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Hexagono Inferior Direito
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Hexagono Inferior Esquerdo
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)
                    if i==0:
                        # Hexagono Central
                        hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='red')
                        ax.add_patch(hex)
                        
                        # Direita
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                                                 
                        # Hexagono Superior Direito
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 0.5,
                                                  yy[(j,i)]*2+(np.sqrt(3)/2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Hexagono Inferior Esquerdo
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 1.5,
                                                  yy[(j,i)]*2+(np.sqrt(3)/2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                    if i==0 and j==0:
                        # Hexagono Central
                        hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2-1,
                                              yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='black')
                        ax.add_patch(hex)
                        
                        # Direita
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                                                 
                        # Hexagono Inferior Direito
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2+ self.mapsize[0]*2-0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Hexagono Inferior Esquerdo
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2+ self.mapsize[0]*2-1.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)

                    #Plotar hits
                    if hits:
                        try:
                            hex = RegularPolygon((xx[(j,i)]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                            if j==0:
                                hex = RegularPolygon((xx[(j,i)]*2,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                                
                            if i==0:
                                hex = RegularPolygon((xx[(j,i)]*2 + self.mapsize[0]*2 - 1,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                            
                            if i==0 and j==0:
                                hex = RegularPolygon((xx[(j,i)]*2 + self.mapsize[0]*2 - 1,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                        except:
                            pass
                            
                    counter+=1

            #Parâmetros de plotagem
            
            plt.xlim(0.5, 2*self.mapsize[0]-0.5)
            plt.ylim(0, 2*self.mapsize[1]*0.8660254)

            f.tight_layout()
            ax.set_axis_off()
            plt.gca().invert_yaxis()
            plt.close()

            os.makedirs("Plotagens/U_matrix", exist_ok=True)


            filename = "Plotagens/U_matrix/plain_umat.jpg"
            f.savefig(filename, dpi=300, bbox_inches = "tight")

            # Read the saved image into a NumPy ndarray
            umat_plain = mpl.image.imread(filename)[30:-30,30:-30]
            
            
            return umat_plain
 
        else:
            # Plotagem
            f = plt.figure(figsize=figsize, dpi=300)
            
            gs = gridspec.GridSpec(100, 100)
            ax = f.add_subplot(gs[:95, 0:90])
            ax.set_aspect('equal')

            # Normalizar as cores para todos os hexagonos
            norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
            counter = 0
            
            if watermark_neurons:
                # Para utilização no plot do número de bmus
                nnodes = self.mapsize[0] * self.mapsize[1]
                grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])

            for j in range(self.mapsize[1]):
                for i in range(self.mapsize[0]):
                    # Hexagono Central
                    hex = RegularPolygon((xx[(j, i)]*2,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor= cm.jet(norm(umat[j][i])),
                                         alpha=1)#, edgecolor='black')

                    ax.add_patch(hex)

                    # Hexagono da Direita
                    if not np.isnan(um[j, i, 0]):
                        hex = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,0])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Hexagono Superior Direita
                    if not np.isnan(um[j, i, 1]):
                        hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,1])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Hexagono Superior Esquerdo
                    if not np.isnan(um[j, i, 2]):
                        hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,2])),
                                             alpha=1)
                        ax.add_patch(hex)

                    if watermark_neurons:
                        # Hexagono Central
                        hex = RegularPolygon((xx[(j, i)]*2,
                                              yy[(j,i)]*2),
                                              numVertices=6,
                                              radius=2/np.sqrt(3),
                                              facecolor= "white",
                                              edgecolor='black',
                                              alpha=watermark_neurons_alfa)#, edgecolor='black')
                        ax.add_patch(hex)

                        ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                                s=f"{int(grid_bmus[j,i])}", 
                                size = neurons_fontsize,
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                color='black')

                    #Plotar hits
                    if hits:
                        if label_plot:
                            ind_var = list(self.component_names).index(label_plot_name)
                            bool_val = self.data_denorm[:,ind_var]
                            bool_val = np.array([0 if number < 0.5 else 1 for number in bool_val])

                            if counter in self.bmus:
                                vars_in_hit =  np.array([ind for ind, value in enumerate(self.bmus) if value == counter])
                                bool_hit = bool_val[vars_in_hit]
                                bool_hit = statistics.mode(bool_hit)

                                facecolor_hits = "black" if bool_hit == 0 else 'white'
                            else:
                                pass
                        else:
                            facecolor_hits = 'white'

                        try:
                            hex = RegularPolygon((xx[(j, i)]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=((1/np.sqrt(3))*bmu_dic[counter]),
                                                 facecolor=facecolor_hits,
                                                 edgecolor=facecolor_hits,
                                                 linewidth=1.1,
                                                 alpha=1)
                            ax.add_patch(hex)
                        except:
                            pass
                    counter+=1

            if samples_label:
                if project_samples_label is not None:
                    samples_label_names = project_samples_label.index.tolist()
                    som_bmus = (project_samples_label.BMU.values-1).tolist()
                    rep_samples_dic = select_keys(self.rep_sample(project = project_samples_label), som_bmus)
                else:
                    samples_label_index = self.sample_names if samples_label_index == "all" else np.array(samples_label_index) 
                    samples_label_names = self.sample_names[samples_label_index]

                    som_bmus = self.bmus.astype(int)[samples_label_index]
                    rep_samples_dic = select_keys(self.rep_sample(), som_bmus)

                if save_labels_rep:
                    def filter_dictionary(dictionary, values):
                        filtered_dict = {}
                        for key, val in dictionary.items():
                            if isinstance(val, list):
                                matching_values = [item for item in val if item in values]
                                if matching_values:
                                    filtered_dict[key] = matching_values
                            else:
                                if val in values:
                                    filtered_dict[key] = val
                        return filtered_dict
                    
                    rep_samples_dic_filter = filter_dictionary(rep_samples_dic, samples_label_names)
                    with open('Resultados/Amostras_representativas_umatrix.txt', 'w', encoding='utf-8') as file:
                        for key, value in rep_samples_dic_filter.items():
                            if isinstance(value, list):
                                value = ', '.join(value)
                            file.write(f'BMU {key+1}: {value}\n')
                
                counter = 0
                for j in range(self.mapsize[1]):
                    for i in range(self.mapsize[0]):
                        try:
                            if isinstance(rep_samples_dic[counter], list):
                                total = len(rep_samples_dic[counter])
                                selected_samples = list(set(samples_label_names).intersection(set(rep_samples_dic[counter])))
                                sample_name, idx_sample = search_strings(selected_samples, rep_samples_dic[counter])
                                rep_sample_name = f"{sample_name}({idx_sample+1}/{total})"
                                                                
                            else:
                                rep_sample_name = rep_samples_dic[counter]
                            hex = RegularPolygon((xx[(j, i)]*2,
                                                yy[(j,i)]*2),
                                                numVertices=6,
                                                radius=((1/np.sqrt(3))*0.5),
                                                facecolor='black',
                                                edgecolor='white',
                                                linewidth=1.1,
                                                alpha=1)
                            ax.add_patch(hex)

                            box_props = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}
                            ax.text(xx[(j,i)]*2, yy[(j,i)]*2+1.5*(1/np.sqrt(3)), 
                                    s=f"{rep_sample_name}", 
                                    size = samples_label_fontsize,
                                    horizontalalignment='center', 
                                    verticalalignment='top', 
                                    color='black',
                                    bbox= box_props)

                        except:
                            pass
                    
                        counter+=1
                    

            #Parâmetros de plotagem
            plt.xlim(-1.1, 2*self.mapsize[0]+0.1)
            plt.ylim(-0.5660254-0.6, 2*self.mapsize[1]*0.8660254-2*0.560254+0.6)
            ax.set_axis_off()
            plt.gca().invert_yaxis()

            # Título do mapa
            plt.title(fill(title,30),
                      horizontalalignment='center',
                      verticalalignment='top',
                      size=title_size,
                      pad=title_pad)

            # Legenda
            ax2 = f.add_subplot(gs[30:70, 95:98])
            cmap = mpl.cm.turbo
            norm = mpl.colors.Normalize(vmin=np.nanmin(um),
                                        vmax=np.nanmax(um))

            # Barra de cores
            cb1 = mpl.colorbar.ColorbarBase(ax2,
                                            cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')
            # Parâmetros da legenda
            cb1.ax.tick_params(labelsize=legend_ticks_size)

            
            cb1.set_label(fill(legend_title, 20), 
                          size=legend_title_size, 
                          labelpad=20)
            cb1.ax.yaxis.label.set_position(label_title_xy)
            """
            # Mover o colorbar um pouco pra direita
            pos = cb1.ax.get_position()
            pos.x0 += 0.08 * (pos.x1 - pos.x0)
            pos.x1 += 0.08 * (pos.x1 - pos.x0)
            cb1.ax.set_position(pos)
            
            cb1.ax.yaxis.label.set_position(label_title_xy)"""
            
            #ADD WATERMARK
            # Add white space subplot below the plot
            ax3 = f.add_subplot(gs[95:100, 0:20], zorder=-1)

            # Add the watermark image to the white space subplot
            ax3.imshow(self.foot, aspect='equal', alpha=1)
            ax3.axis('off')
            
            plt.subplots_adjust(bottom=3, top = 5,wspace=0)
            
            plt.show()
        
        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}.jpg",dpi=300, bbox_inches = "tight")
            else:
                # Criar diretórios se não existirem
                path = 'Plotagens/U_matrix'
                os.makedirs(path, exist_ok=True)

                print("Salvando.")
                if hits:
                    if label_plot:
                        f.savefig(f"Plotagens/U_matrix/{file_name}_with_hits_label.jpg",dpi=300, bbox_inches = "tight") 
                    elif watermark_neurons:
                        f.savefig(f"Plotagens/U_matrix/{file_name}_with_hits_watermarkbmus.jpg",dpi=300, bbox_inches = "tight") 
                    else:
                        f.savefig(f"Plotagens/U_matrix/{file_name}_with_hits.jpg",dpi=300, bbox_inches = "tight")
                else:
                    if label_plot:
                        f.savefig(f"Plotagens/U_matrix/{file_name}_with_label.jpg",dpi=300, bbox_inches = "tight") 
                    elif watermark_neurons:
                        f.savefig(f"Plotagens/U_matrix/{file_name}_watermarkbmus.jpg",dpi=300, bbox_inches = "tight") 
                    else:
                        f.savefig(f"Plotagens/U_matrix/{file_name}.jpg",dpi=300, bbox_inches = "tight")


    def component_plot(self,
                       component_name = 0,
                       figsize = (10,10),
                       title = None,
                       full_title = False,
                       title_size = 30,
                       title_pad = 25,
                       legend_title = False,
                       legend_pad = 0,
                       label_title_xy = (0,0.5),
                       legend_title_size = 24,
                       legend_ticks_size = 20,
                       save = False,
                       file_name = None,
                       file_path = False,
                       collage = False):
        """
        Função para realizar a plotagem de um mapa de variável treinada.

            Args:
                component_name: qual o nome da variável que se deseja plotar. É
                    aceito o nome da váriavel na forma de string ou um número
                    inteiro relacionado ao índice dessa variável na lista de
                    variáveis inicial.

                figsize: tamanho da tela de plotagem da variável.
                    Default: (10,10)

                title: título da figura criada. Default: "Matriz U"

                title_size: tamanho do título plotado. Default: 40

                title_pad: espaçamento entre o título e a parte superior da matriz.
                    Default: 25

                legend_title: título da barra de cores de legenda.
                    Default: "Distância"

                legend_pad: espaçamento entre o título da legenda e a matriz U.
                    Default: 0

                y_legend: espaçamento entre o título da legenda e a parte inferior
                    da figura. Default: 1.12

                legend_title_size = tamanho do título da legenda. Default: 25.

                legend_ticks_size = tamanho dos algarismos da barra de cores da
                    legenda. Default: 20.

                save: booleano para definir o salvamento da imagem criada.
                    Esse salvamento será feito no diretório (Plotagens/Matriz_U).
                    Default: True.

                file_name: o nome que será dado ao arquivo salvo. Caso nenhum nome
                    seja dado, será utilizado o nome do projeto.

                file_path: qual o caminho no sistema em que a imagem deverá ser
                    salva caso não se opte por utilizar o caminho default.

            Retorna:
                A imagem com a plotagem da Matriz U de distâncias unificadas.
        """
        def check_path(filename):
            # Check if the file already exists in the directory
            if os.path.isfile(filename):
                # Extract the filename and extension
                name, ext = os.path.splitext(filename)

                # Append a number to the filename to make it unique
                i = 1
                while os.path.isfile(f"{name}_{i}{ext}"):
                    i += 1

                # Change the filename to the unique name
                filename = f"{name}_{i}{ext}"
            
            return filename
        
        if file_name is None:
            file_name = f"Component_plot_{self.name}"

        if isinstance(component_name, int):
            # Pegar uma variavel
            bmu_var = self.bmu_matrix[:,component_name].reshape(self.mapsize[1], self.mapsize[0])
            var_name = self.component_names[component_name]

            # Captar unidade da legenda
            if not legend_title:
                legend_title = self.unit_names[component_name]
                
        elif isinstance(component_name, str):
            index = list(self.component_names).index(component_name)
            bmu_var = self.bmu_matrix[:, index].reshape(self.mapsize[1], self.mapsize[0])
            var_name = self.component_names[index]
            # Captar unidade da legenda
            if not legend_title:
                legend_title = self.unit_names[index]
        else:
            print("Nome errado para componente, aceito somente string com nome da componente ou int com sua posição")

        # Criar coordenadas
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # Plotagem
        f = plt.figure(figsize=figsize,dpi=300)
        gs = gridspec.GridSpec(100, 20)
        ax = f.add_subplot(gs[:95, 0:19])
        ax.set_aspect('equal')

        # Normalizar as cores para todos os hexagonos
        norm = mpl.colors.Normalize(vmin=np.nanmin(bmu_var), vmax=np.nanmax(bmu_var))

        # Preencher o plot com os hexagonos
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                ax.add_patch(RegularPolygon((xx[(j, i)], yy[(j,i)]),
                                     numVertices=6,
                                     radius=1/np.sqrt(3),
                                     facecolor=cm.jet(norm(bmu_var[j,i])),
                                     alpha=1))

        plt.xlim(-0.5, self.mapsize[0]+0.5)
        plt.ylim(-0.5660254, self.mapsize[1]*0.8660254+2*0.560254)
        ax.set_axis_off()
        plt.gca().invert_yaxis()

        # Título do mapa
        if full_title:
            name = title if title is not None else var_name
        else:
            if title is not None:
                name = title
            else:
                name = var_name.split()
                name = f"{name[0]} {name[1]}" if len(name)>1 else f"{name[0]}"
        plt.title(fill(f"{name}",20), horizontalalignment='center',  verticalalignment='top', size=title_size, pad=title_pad)

        # Legenda
        ax2 = f.add_subplot(gs[27:70, 19])
        cmap = mpl.cm.turbo
        norm = mpl.colors.Normalize(vmin=np.nanmin(bmu_var), vmax=np.nanmax(bmu_var))
        cb1 = mpl.colorbar.ColorbarBase(ax2,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')

        cb1.ax.tick_params(labelsize=legend_ticks_size)
        # Colocar titulo da legenda caso tenh nome da variável
        cb1.set_label(fill(legend_title,15),
                      size=legend_title_size,
                      labelpad=legend_pad,
                      horizontalalignment='right',
                      wrap=True)
        
        cb1.ax.yaxis.label.set_position(label_title_xy)
        if collage == False:
            #ADD WATERMARK
            # Add white space subplot below the plot
            ax3 = f.add_subplot(gs[95:100, 0:4], zorder=-1)

            # Add the watermark image to the white space subplot
            ax3.imshow(self.foot, aspect='equal', alpha=1)
            ax3.axis('off')
        
        plt.subplots_adjust(bottom=3, top = 5,wspace=0)
        if full_title:
            if isinstance(var_name, str):
                label_name = var_name.replace("/", "")
            else:
                label_name = var_name
        else:
            if isinstance(var_name, str):
                label_name = var_name[:7].replace(" ", "").replace(":", "")
            else:
                label_name = var_name[:6]

        if collage:
            path = 'Plotagens/Component_plots/Collage/temp'
            os.makedirs(path, exist_ok=True)
            path_name = check_path(f"Plotagens/Component_plots/Collage/temp/{label_name}.jpg")
            f.savefig(path_name,dpi=300, bbox_inches = "tight")

        if save:
            if file_path:
                path_name = check_path(f"{file_path}/{label_name}.jpg")
                f.savefig(path_name,dpi=300, bbox_inches = "tight")
            else:
                path = 'Plotagens/Component_plots'
                os.makedirs(path, exist_ok=True)
                path_name = check_path(f"Plotagens/Component_plots/{label_name}.jpg")
                f.savefig(path_name,dpi=300, bbox_inches = "tight")

    def multiple_component_plots(self,
                                wich = "all",
                                figsize = (10,10),
                                full_title = False,
                                title_size = 30,
                                title_pad = 25,
                                legend_title = "Presence",
                                legend_pad = 0,
                                label_title_xy = (0,0.5),
                                legend_title_size = 24,
                                legend_ticks_size = 20,
                                save = True,
                                file_path = False, 
                                collage = False):
        """
        Função para a plotagem de uma lista ou todas as variáveis treinadas no
        objeto SOM.

        Args:
            wich: lista de variáveis a ser plotada e salva ou "all" para a
            plotagem de todas as variáveis.
        """
        if isinstance(wich, str):
            iterator = self.component_names
        else:
            iterator = wich

        # Iteração sobre a função de plotagem individual
        pbar = tqdm(iterator, mininterval=1)
        for name in pbar:
            pbar.set_description(f"Componente: {name}")
            self.component_plot(component_name = name,
                               figsize = figsize,
                               full_title = full_title,
                               title_size = title_size,
                               title_pad = title_pad,
                               legend_title = legend_title,
                               legend_title_size = legend_title_size,
                               legend_ticks_size = legend_ticks_size,
                               legend_pad = legend_pad,
                               label_title_xy= label_title_xy,
                               save = save,
                               file_path = file_path,
                               collage = collage)
            plt.close()

        print("Finalizado")



    def component_plot_collage(self,
                               page_size = (2480, 3508), # A4 em pixels
                               grid = (4,4),
                               wich = "all",
                               figsize = (10,10),
                               full_title = False,
                               title_size = 30,
                               title_pad = 25,
                               legend_title = "Presença",
                               legend_title_size = 24,
                               legend_ticks_size = 20,
                               legend_pad = 0,
                               label_title_xy = (0,0.5)):
        """
        Função para criar uma colagem de mapas de componentes de treinamento
        (Component Plots).

        Args:
            page_size: tamanho da página de colagem dos plots em pixels.
                Default: A4 (2480, 3508)

            grid: o formato do grid de colagem. Default: (4,4)

            wich: lista de variáveis a ser colada e salva ou "all" para a
                plotagem de todas as variáveis.

            title_size: tamanho do título plotado. Default: 40

            title_pad: espaçamento entre o título e a parte superior da matriz.
                Default: 25

            legend_title: título da barra de cores de legenda.
                Default: "Distância"

            legend_pad: espaçamento entre o título da legenda e a matriz U.
                Default: 0

            y_legend: espaçamento entre o título da legenda e a parte inferior
                da figura. Default: 1.12

            legend_title_size = tamanho do título da legenda. Default: 25.

            legend_ticks_size = tamanho dos algarismos da barra de cores da
                legenda. Default: 20.
        """
        # Função de suporte somente para uso dentro dessa função
        def resize_image(image, page, grid):
            """
            Função para redimensionar imagem na folha de acordo com o grid
            definido.

            Args:
                image: a imagem que deve ser dimensionada no grid.

                page: as dimensões da página para a colagem.

                grid:  o formato do grid de colagem.
            """
            max_hor = page.size[0] / grid[1]
            max_ver = page.size[1] / grid[0]

            image.thumbnail((max_hor, max_ver))

            return image

        print("Gerando mapas...")
        # Listar component plots que ainda não foram feitos
        if isinstance(wich, str):
            if wich == "all":
                list_figs = "all"  
                n_components = len(self.component_names)
            else:
                print("Os parâmetros aceitos são 'all' ou uma lista de variáveis ou indices de variáveis.")
        if isinstance(wich, list):
            list_figs = wich
            n_components = len(wich)

        # Criar component plots que ainda não foram gerados anteriorente
        self.multiple_component_plots(wich = list_figs,
                                      figsize = figsize,
                                      full_title = full_title,
                                      title_size = title_size,
                                      title_pad = title_pad,
                                      save=False,
                                      legend_title = legend_title,
                                      legend_title_size = legend_title_size,
                                      legend_ticks_size = legend_ticks_size,
                                      legend_pad = legend_pad,
                                      label_title_xy = label_title_xy,
                                      collage = True)
        # Adquirir o caminho para todas as imagens
        images_path = glob.glob('Plotagens/Component_plots/Collage/temp/*.jpg')

        # Número de páginas necessárias para preencher as componentes nesse grid
        n_pages = int(n_components/(grid[0]*grid[1]))+1

        # Número de imagens por página
        im_pp = grid[0]*grid[1]

        print("Gerando colagem...")
        for i in range(n_pages):
            # Imagem para fazer a colagem
            page = Image.new("RGB", page_size, "WHITE")  # Fundo branco

            # Fatiamento dos caminhos de imagem para cada pagina
            images_path_page = images_path[i*im_pp:(i+1)*im_pp]
            xx = np.tile(np.arange(grid[1]), grid[0])
            yy = np.repeat(np.arange(grid[0]), grid[1])

            for j, img_path in enumerate(images_path_page):
                img = Image.open(img_path) 
                img = resize_image(img, page, grid)
                x_pos = int(xx[j] * page_size[0] / grid[1])
                y_pos = int(yy[j] * page_size[1] / grid[0])
                page.paste(img, (x_pos, y_pos))

            #Salvar na pasta raiz de plotagens
            path = 'Plotagens/Component_plots/Collage/pages'
            os.makedirs(path, exist_ok=True)
            
            foot = self.foot
            max_height = page_size[1]/40
            width = int((foot.width / foot.height) * max_height)
            foot.thumbnail((width, max_height))
            x = 0  # Left position
            y = page.height - foot.height  # Down position
            page.paste(foot, (x, y))
            
            page.save(f"Plotagens/Component_plots/Collage/pages/Component_plots_collage_page{i+1}.jpg")

        print("Finalizado.")
        print("A pasta 'Plotagens/Component_plots/Collage/temp' pode ser deletada.")
    
    def bmu_template(self, 
                     figsize = (10,10),
                     title_size = 24,
                     fontsize = 10,
                     save = False,
                     file_name = None,
                     file_path = False):
        """
        Gera o Mapa de BMUS para o mapa Kohonen atual.
        
        Args:
                figsize: tamanho da tela de plotagem da variável.
                    Default: (10,10)

                title_size: tamanho do título plotado. Default: 40

                title_pad: espaçamento entre o título e a parte superior da matriz.
                    Default: 25

                save: booleano para definir o salvamento da imagem criada.
                    Esse salvamento será feito no diretório (Plotagens/Matriz_U).
                    Default: True.

                file_name: o nome que será dado ao arquivo salvo. Caso nenhum nome
                    seja dado, será utilizado o nome do projeto.

                file_path: qual o caminho no sistema em que a imagem deverá ser
                    salva caso não se opte por utilizar o caminho default.
        """

        # Criar coordenadas
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # Plotagem
        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(100, 20)
        ax = f.add_subplot(gs[:, 0:20])
        ax.set_aspect('equal')

        # Normalizar as cores para todos os hexagonos
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        
        # Criar numeração
        nnodes = self.mapsize[0] * self.mapsize[1]
        grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])

        # Preencher o plot com os hexagonos
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                hex = RegularPolygon((xx[(j,i)], yy[(j,i)]), 
                                     numVertices=6, 
                                     radius=1/np.sqrt(3), facecolor=[cm.Pastel1(norm(0.2)) if i%2==0 else cm.Pastel1(norm(0.6))][0],
                                     alpha= [0.3 if j%2==0 else 0.8][0], 
                                     edgecolor='gray'
                                    )
                ax.add_patch(hex)
                
                ax.text(xx[(j,i)], yy[(j,i)], 
                        s=f"{int(grid_bmus[j,i])}", 
                        size = fontsize,
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        color='black')
        
        plt.title("Template de Neurônios", size=title_size)
        
        # Limites de Plotagem e eixos
        plt.xlim(-0.5, self.mapsize[0]+0.5)
        plt.ylim(-0.5660254, self.mapsize[1]*0.8660254+2*0.560254)
        ax.set_axis_off()
        plt.gca().invert_yaxis()
        
        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}_bmu_template.jpg",dpi=300, bbox_inches = "tight")
            else:
                path = 'Plotagens/Bmu_template'
                os.makedirs(path, exist_ok=True)

                f.savefig(f"Plotagens/Bmu_template/{file_name}_bmu_template.jpg",dpi=300, bbox_inches = "tight")

    def generate_rec_lattice(self, n_columns, n_rows):
        """
        Gera as coordenadas xy dos BMUs para um grid retangular.

        Args:
            n_rows: Número de linhas do mapa kohonen.
            n_columns: Número de colunas no mapa kohonen.

        Returns:
            Coordenadas do formato [x,y] para os bmus num grid retangular.

        """
        x_coord = []
        y_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    @property
    def hits_dictionary(self):
        """
        Função para criar um dicionário de hits dos vetores de entrada para
        cada um de seus bmus, proporcional ao tamanho da plotagem.
        """
        # Contagem de hits
        unique, counts = np.unique(self.bmus, return_counts=True)

        # Normalizar essa contagem de 0.5 a 2.0 (de um hexagono pequeno até um
        #hexagono que cobre metade dos vizinhos).
        counts = minmax_scale(counts, feature_range = (0.5,2))

        return dict(zip(unique, counts))


    def generate_hex_lattice(self, n_columns, n_rows):
        """
        Gera as coordenadas xy dos BMUs para um grid hexagonal odd-r.
        Args:
            n_rows:Número de linhas do mapa kohonen.
            n_columns:Número de colunas no mapa kohonen.

        Returns:
            Coordenadas do formato [x,y] para os bmus num grid hexagonal.

        """
        ratio = np.sqrt(3) / 2

        coord_x, coord_y = np.meshgrid(np.arange(n_columns),
                                       np.arange(n_rows), 
                                       sparse=False, 
                                       indexing='xy')
        coord_y = coord_y * ratio
        coord_x = coord_x.astype(float)
        coord_x[1::2, :] += 0.5
        coord_x = coord_x.ravel()
        coord_y = coord_y.ravel()

        coordinates = np.column_stack([coord_x, coord_y])

        return coordinates



    def plot_torus(self, inner_out_prop = 0.4, red_factor = 4, hits=False):
        """
        Retorna as coordenadas (x, y, z) para desenhar um toroide.

        Parâmetros:
        - rows (int): Número de linhas da grade.
        - cols (int): Número de colunas da grade.
        - aspect_ratio (float): Proporção entre a largura e altura da imagem.
        - R_scale (float): Escala do raio externo do toroide (padrão: 0.4).

        Retorna:
        Coordenadas (x, y, z) para desenhar o toroide.
        """

        mat_im = self.plot_umatrix(figsize = (10,10), 
                                       hits = True if hits else False, 
                                       save = True, 
                                       file_name = None,
                                       file_path = False, 
                                       resume=True)
            
        # Resolução Toroide
        y_res = int(mat_im.shape[0]/red_factor)
        x_res = int(mat_im.shape[1]/red_factor)

        if mat_im.shape[0] > mat_im.shape[1]:
            mat_im = rotate(mat_im, 90)

        #Reduzir imagem
        mat_res = (resize(mat_im, (y_res, x_res))*256).astype(int)

        def torus(rows, cols, aspect_ratio, R_scale=0.4):
            r_scale = R_scale * aspect_ratio * inner_out_prop
            u, v = np.meshgrid(np.linspace(0, 2*pi, cols), np.linspace(0, 2*pi, rows))
            return (R_scale+r_scale*np.sin(v))*np.cos(u), (R_scale+r_scale*np.sin(v))*np.sin(u), r_scale*np.cos(v)


        r, c, _ = mat_res.shape
        aspect_ratio = mat_im.shape[1]/mat_im.shape[0]
        x, y, z = torus(r, c, aspect_ratio, R_scale=0.4)
        I, J, K, tri_color_intensity, pl_colorscale = self.mesh_data(mat_res, n_colors=32, n_training_pixels=10000) 
        fig5 = go.Figure()
        fig5.add_mesh3d(x=x.flatten(), y=y.flatten(), z=np.flipud(z).flatten(),  
                                    i=I, j=J, k=K, intensity=tri_color_intensity, intensitymode="cell", 
                                    colorscale=pl_colorscale, showscale=False)
        
        scene_style = dict(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   scene_aspectmode="data")
        
        fig5.update_layout(width=700, height=700,
                        margin=dict(t=10, r=10, b=10, l=10),
                        scene_camera_eye=dict(x=-1.75, y=-1.75, z=1), 
                        **scene_style)
        fig5.show()


    def mesh_data(self, img, n_colors=32, n_training_pixels=800):
        rows, cols, _ = img.shape
        z_data, pl_colorscale = self.image2zvals(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
        triangles = self.regular_tri(rows, cols) 
        I, J, K = triangles.T
        zc = z_data.flatten()[triangles] 
        tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
        return I, J, K, tri_color_intensity, pl_colorscale
    
    def regular_tri(self, rows, cols):
        #define triangles for a np.meshgrid(np.linspace(a, b, cols), np.linspace(c,d, rows))
        triangles = []
        for i in range(rows-1):
            for j in range(cols-1):
                k = j+i*cols
                triangles.extend([[k,  k+cols, k+1+cols], [k, k+1+cols, k+1]])
        return np.array(triangles) 
    
    def image2zvals(self, img,  n_colors=64, n_training_pixels=800, rngs = 123): 
        # Image color quantization
        # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
        # n_colors: int,  number of colors for color quantization
        # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
        # returns the array of z_values for the heatmap representation, and a plotly colorscale
    
        if img.ndim != 3:
            raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
        rows, cols, d = img.shape
        if d < 3:
            raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
            
        range0 = img[:, :, 0].max() - img[:, :, 0].min()
        if range0 > 1: #normalize the img values
            img = np.clip(img.astype(float)/255, 0, 1)
            
        observations = img[:, :, :3].reshape(rows*cols, 3)
        training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
        model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
        
        codebook = model.cluster_centers_
        indices = model.predict(observations)
        z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
        z_vals = z_vals.reshape(rows, cols)
        # define the Plotly colorscale with n_colors entries    
        scale = np.linspace(0, 1, n_colors)
        colors = (codebook*255).astype(np.uint8)
        pl_colorscale = [[sv, f'rgb{tuple(color)}'] for sv, color in zip(scale, colors)]
        
        # Reshape z_vals  to  img.shape[:2]
        return z_vals.reshape(rows, cols), pl_colorscale
