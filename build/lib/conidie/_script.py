from this import d
from tkinter import Image
from tkinter.ttk import Progressbar

import cv2

import subprocess
import pathlib
import tempfile
from zipfile import ZipFile
import os
from torch import _fake_quantize_learnable_per_tensor_affine
import conidie.path as paths

import re
import PIL

import h5py
from collections import Counter
from pandas import DataFrame
import shutil
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from fileinput import filename
from glob import glob

from os import listdir,makedirs
from os.path import isfile, join

import numpy as np
import numpy
from skimage import img_as_uint, img_as_ubyte
import skimage.io
import skimage
from skimage import measure
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.measure import label, regionprops_table
from skimage.transform import resize
from skimage.filters import threshold_multiotsu

import napari
from napari import Viewer
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari import layers
from napari.utils import progress
from napari.utils.colormaps import colormap_utils as cu
from napari.types import ImageData, LabelsData, NewType
from napari.utils.notifications import show_info
from magicgui import magic_factory
from magicgui import magicgui
from magicgui.widgets import Table
from magicgui.tqdm import trange
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QGridLayout, QFileDialog, QListWidget, QHBoxLayout, QPushButton, QWidget
from qtpy.QtCore import Qt

# from ilastik.experimental.api import from_project_file


colormap = cu.label_colormap()
colors = colormap.colors
color_dict = {}
color_dict[0] = colors[0]
color_dict[1] = colors[1]

zip_dir = tempfile.TemporaryDirectory()

def function_central(filepath):
    
    path_image = str(filepath).replace('\\','/')
  
    donner = '--raw_data="'+path_image+'"'
    output_dir = tempfile.TemporaryDirectory()
    recevoir = '--output_filename_format="'+os.path.join(output_dir.name,path_image.split('/')[-1][:-4])+'_result_type.jpg"'
    projet_path = '--project="'+os.path.join(paths.get_models_dir(),'TER-data_NEW.ilp')+'"'
    
    subprocess.run(["C:/Program Files/ilastik-1.3.3post3/ilastik.exe",
                    '--headless',
                    projet_path,
                    '--export_source=Simple Segmentation',
                    donner,
                    recevoir])
    
    f = os.path.join(output_dir.name,path_image.split('/')[-1][:-4])+'_result_type.png'
    # imag = np.squeeze(skimage.io.imread(path_image_n))
    
    ##################
    # Traitement Ante
    ##################
    print("Traitement Ante")
    sep=re.compile(r"\\")
    end=re.compile(r'.(jpg|png)$')

    # CUT = []
    # for f in output_image:
    
    lien=sep.split(f)[0]+r"/Otsu_centrage"
    name=sep.split(f)[len(sep.split(f))-1] # recuperation du nom du fichier de base
    name=end.sub(r'',name)
    img=PIL.Image.open(f)
    gray_img = img.convert("L")
    image=np.array(img)

    
    thresholds = threshold_multiotsu(np.array(gray_img),nbins=3)    
    # Using the threshold values, we generate the three regions.
    region = np.digitize(np.array(gray_img), bins=thresholds)
    print("Region")
    n,m=np.shape(region)
    mk_1=np.where(region==1)
    mk_2=np.where(region==2)
    mk_0=np.where(region==0)
    region[mk_1]=0
    region[mk_0]=1
    region[mk_2]=0    

    boucle=True

    while boucle :
        labels_mask = measure.label(region, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True) 
        big_region=regions[0]
        compteur_0x=len(np.where(big_region.coords[:,0]==0)[0])
        compteur_n=len(np.where(big_region.coords[:,0]==n-1)[0])
        compteur_0y=len(np.where(big_region.coords[:,1]==0)[0])
        compteur_m=len(np.where(big_region.coords[:,1]==m-1)[0])
               
        seuil=25
        #size=250
        size=1000
        
        #Plus gros objet detecté en bordure si on rentre dans ce if.
        if compteur_0x >seuil or compteur_n >seuil or compteur_0y>seuil or compteur_m>seuil: 
            region[big_region.coords[:,0],big_region.coords[:,1]]=0
            size=100 # Réduction de taille pour éviter les pbs
            if len(regions)<2:
                boucle=False
        else:
            boucle=False
        

    if len(regions) > 1: #On mets toutes les regions qui ne sont la plus grande en back ground
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1 #Toutes les coordonnées du gros objet sont unifiées à 1
    element=np.where(labels_mask==1)

    xmin=min(element[0])
    ymin=min(element[1])
    xmax=max(element[0])
    ymax=max(element[1])

    
    if xmin-size<0:
        xmin=0
    else: 
        xmin=xmin-size

    
    if ymin-size<0:
        ymin=0
    else: 
        ymin=ymin-size

    
    if xmax+size>=n:
        xmax=n-1
    else: 
        xmax+=size

    
    if ymax+size>=m:
        ymax=m-1
    else: 
        ymax+=size
    print("CUT")
    cut=image[xmin:xmax,ymin:ymax]
    
    ##########################
    # Traitement final
    ##########################
    
    data = np.array(cut)
    tache=np.where(data==255)
    condide=np.where(data==85)
    hyphe=np.where(data==170)
    data[tache]=0
    data[hyphe]=1
    data[condide]=1

    labels_mask = measure.label(data, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True) 
    if len(regions) > 1: #On mets toutes les regions qui ne sont la plus grande en back ground
        for rg in regions[1:]:          
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1 #Toutes les coordonnées du gros objet sont unifiées à 1
    data = labels_mask
    for j in range(len(hyphe[0])): # on remet les hyphes du plus gros element en label 2
        if data[hyphe[0][j],hyphe[1][j]] == 1 : 
            data[hyphe[0][j],hyphe[1][j]] = 2
    
    return data

def table_to_widget(table: dict) -> QWidget:
    """
    Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
    contains a QTableWidget with that data.
    """
    view = Table(value=table)

    copy_button = QPushButton("Copy to clipboard")

    @copy_button.clicked.connect
    def copy_trigger():
        view.to_dataframe().to_clipboard()

    save_button = QPushButton("Save as csv...")

    @save_button.clicked.connect
    def save_trigger():
        filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
        view.to_dataframe().to_csv(filename)
        
    save_images_button = QPushButton("Save Images")

    @save_images_button.clicked.connect
    def save_images_trigger():
        folder_to_zip = zip_dir.name
        for folder_image_result_sample in os.listdir(folder_to_zip):
            un_chemin = os.path.join(folder_to_zip,folder_image_result_sample)
            for ix in os.listdir(un_chemin):
                if ix.endswith('_result.png'):
                    data = imread(os.path.join(un_chemin,ix))
                    
                    data_label1 = np.array(data)
                    tache=np.where(data_label1==0)
                    condide=np.where(data_label1==1)
                    hyphe=np.where(data_label1==2)
                    data_label1[tache]=0
                    data_label1[hyphe]=150
                    data_label1[condide]=255
                    
                    imsave(os.path.join(un_chemin,ix), img_as_ubyte(data_label1))
        
        filename, _ = QFileDialog.getSaveFileName(save_images_button, "Save as csv...", ".", "*.zip")
        shutil.make_archive(filename,format="zip",root_dir=folder_to_zip)
        show_info('Compressed file done')

    widget = QWidget()
    widget.setWindowTitle("region properties")
    widget.setLayout(QGridLayout())
    widget.layout().addWidget(copy_button)
    widget.layout().addWidget(save_button)
    widget.layout().addWidget(view.native)
    widget.layout().addWidget(save_images_button)

    return widget

def get_quantitative_data(image, napari_viewer):
    img=image
    seuil=25

    connidie=np.where(img==1)
    hyphe=np.where(img==2)
    img[connidie]=0

    labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=False) 

    print(">",len(hyphe[0]))
    print(">",len(connidie[0]))
    
    minus=0
    for rg in regions:
        if len(rg.coords[:,0])>seuil:
            print(">",len(regions)-minus)
            break
        else: 
            minus+=1    
    d = {'nombre dhyphes': [len(regions)-minus], 'hyphe': [len(hyphe[0])], 'connidie': [len(connidie[0])]}

    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right')
    
def quantitative_data_for_all(dictionnaire,napari_viewer):
    A = [] #sous dossier
    B = [] #nom image
    C = []
    D = []
    E = []
    for ix in dictionnaire.keys():
        img=dictionnaire[ix]
        seuil=25

        connidie=np.where(img==1)
        hyphe=np.where(img==2)
        img[connidie]=0

        labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=False) 

        print(">",len(hyphe[0]))
        print(">",len(connidie[0]))
        
        minus=0
        for rg in regions:
            if len(rg.coords[:,0])>seuil:
                print(">",len(regions)-minus)
                name_xx = ix.split('xx')
                A.append(name_xx[0])
                B.append(name_xx[1])
                C.append(len(regions)-minus)
                D.append(len(hyphe[0]))
                E.append(len(connidie[0]))
                break
            else: 
                minus+=1    

    d = {'sous dossier':A,'nom image':B,'nombre dhyphes': C, 'hyphe': D, 'connidie': E}
    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right')
    
    
    
def get_quantitative_data_all_for_csv(dossier_des_images,napari_viewer):
    A = [] 
    B = []
    C = []
    D = []
    E = []
    
    dictionnaire = {}
    
    for ix in os.listdir(dossier_des_images):
        chemin_dans_sousdossier = os.path.join(dossier_des_images,ix)
        print(f'path to {ix} :',chemin_dans_sousdossier)
        if len(os.listdir(chemin_dans_sousdossier))!=0:
            for iy in os.listdir(chemin_dans_sousdossier):
                if iy.find("result")!=-1:
                    data_dico=imread(os.path.join(chemin_dans_sousdossier,iy))
                    print("chemin sous dossier",os.path.join(chemin_dans_sousdossier,iy))
                    dictionnaire[iy]=data_dico

    for ix in dictionnaire.keys():
        print('ix',ix)
        img=dictionnaire[ix]
        seuil=25
        
        print('before',Counter(img.flatten()))
        connidie=np.where(img==1)
        hyphe=np.where(img==2)
        img[connidie]=0

        print('after',Counter(img.flatten()))
        labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=False) 
        
        minus=0
        for rg in regions:
            if len(rg.coords[:,0])>seuil:
                name_xx = ix.split('xx')
                A.append(name_xx[0])
                B.append(name_xx[1][:-4])
                C.append(len(regions)-minus)
                D.append(len(hyphe[0]))
                E.append(len(connidie[0]))
                break
            else: 
                minus+=1    

    d = {'sous dossier':A,'nom image':B,'nombre dhyphes': C, 'hyphe': D, 'connidie': E}

    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right',name="Save")
    
    
@magic_factory(call_button="Run segmentation",filename={"label": "Pick a file:"})
def process_function_segmentation(napari_viewer : Viewer,filename=pathlib.Path.cwd()): 
    
    dico = {}
    with ZipFile(filename,'r') as zipObject:
    
        listOfFileNames = zipObject.namelist()
        
        for i in trange(len(listOfFileNames)):
            
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)            
            temp_i = listOfFileNames[i].replace('/','xx').replace(" ","")       
            temp_i_jpg = listOfFileNames[i].replace('/','xx')[:-4].replace(" ","")
            os.mkdir(zip_dir.name+'\\'+temp_i_jpg)
            shutil.move(zip_dir.name+'\\'+listOfFileNames[i].replace('/','\\'),zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i)
            image_segm = function_central(zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i)
            # imsave(zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i_jpg+'_result.png', img_as_uint(image_segm))
            imsave(zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i_jpg+'_result.png', img_as_ubyte(image_segm))
            dico[temp_i_jpg+'_result.png'] = image_segm
            
    print("Extraction done located into",zip_dir.name)
        
    names = []
    for ix in os.listdir(zip_dir.name):
        if len(os.listdir(os.path.join(zip_dir.name,ix)))!=0:
            names.append(ix)

    def open_name(item):
        
        name = item.text()
        name_folder = name[:-4]
        
        print('Loading', name, '...')

        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
        fname = f'{zip_dir.name}\{name}'
        for fname_i in os.listdir(fname):
            if fname_i.find('result')!=-1:
                data_label = imread(f'{fname}\{fname_i}')
                data_label1 = np.array(data_label)
                print("Image_count :",Counter(data_label.flatten()))
                tache=np.where(data_label1==0)
                condide=np.where(data_label1==257)
                hyphe=np.where(data_label1==514)
                data_label1[tache]=0
                data_label1[hyphe]=2
                data_label1[condide]=1                
                napari_viewer.add_labels(data_label1,name=f'{fname_i[:-4]}')
            else:
                napari_viewer.add_image(imread(f'{fname}\{fname_i}'),name=f'{fname_i[:-4]}')

        print('... done.')


    list_widget = QListWidget()
    for n in names:
        list_widget.addItem(n)    
    list_widget.currentItemChanged.connect(open_name)
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)
    
@magic_factory(call_button="save modification", layout="vertical")
def save_modification(image_seg : napari.layers.Labels, image_raw : ImageData, napari_viewer : Viewer):
    data_label = image_seg.data
    sousdossier = image_seg.name.split('_result')[0]
    nom_image = image_seg.name.split('xx')[1]
    os.remove(f'{zip_dir.name}\{sousdossier}\{image_seg}.png')
    # imsave(f'{zip_dir.name}\{sousdossier}\{image_seg}.png', img_as_uint(data_label))
    imsave(f'{zip_dir.name}\{sousdossier}\{image_seg}.png', img_as_ubyte(data_label))
   

@magic_factory(call_button="execute", layout="vertical")
def quantitative_data_for_all(napari_viewer : Viewer):
    """Add, subtracts, multiplies, or divides to image layers with equal shape."""
    print("OK")
    return get_quantitative_data_all_for_csv(zip_dir.name,napari_viewer)