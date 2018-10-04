#!flask/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import send_file
#from flask import request

from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from flask import request



app = Flask(__name__)



import skimage.io as ski
#import matplotlib.pyplot as plt
import skimage.color as skc
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology

from skimage.util import img_as_ubyte
import copy

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

import os
import urllib.request
import shutil

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

import time

# whole_data = pd.read_csv("Hausable_data.csv")

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
      image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def forest_fire(matrix,i,j):
    
    if(matrix[i,j] != 0):
        for x in range(i-3, i+3):
            for y in range(j-3, j+3):
                matrix[x,y] = 0

    queue = []
    queue.append([i,j])

    for node in queue:
        w = copy.copy(node)
        e = copy.copy(node)

        while(matrix[e[0],e[1]] == 0 and e[0] != matrix.shape[0] -1):
            e[0] += 1

        while(matrix[w[0],w[1]] == 0 and w[0] != 0):
            w[0] -= 1
        for point in range(w[0], e[0]):
            matrix[point, node[1]] = 255
            if(matrix[point,node[1]-1] == 0 and node[1]-1 != 0):
                queue.append([point, node[1]-1])
            if(matrix[point,node[1]+1] == 0 and node[1]+1 != matrix.shape[1] -1):
                queue.append([point, node[1]+1])
        





def predict_adu(image, model, label, address, input_height, input_width, input_layer):
    address_concat = address
    file_name = image
    model_file = model
    label_file = label
    # input_height = 299
    # input_width = 299
    input_mean = 128
    input_std = 128
    # input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    predictions = {"adu": "", "nonadu":""}

    for i in top_k:
        print(labels[i], results[i])
        if(labels[i] == "nonadu"):
            predictions["nonadu"] = str(results[i])
        elif(labels[i] == "adu"):
            predictions["adu"] = str(results[i])

    return predictions





def predict_solar(image, model, label, address, input_height, input_width, input_layer):
    print("inside prediction function")
    address_concat = address
    file_name = image
    model_file = model
    label_file = label
    # input_height = 299
    # input_width = 299
    input_mean = 128
    input_std = 128
    # input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    predictions = {"solar": "", "nonsolar":""}

    for i in top_k:
        print(labels[i], results[i])
        if(labels[i] == "nonsolar jpeg"):
            predictions["nonsolar"] = str(results[i])
        elif(labels[i] == "solar jpeg"):
            predictions["solar"] = str(results[i])

    return predictions


def predict_pool(image, model, label, address, input_height, input_width, input_layer):
    address_concat = address
    file_name = image
    model_file = model
    label_file = label
    # input_height = 299
    # input_width = 299
    input_mean = 128
    input_std = 128
    # input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    predictions = {"pool": "", "nonpool":""}

    for i in top_k:
        print(labels[i], results[i])
        if(labels[i] == "no pool"):
            predictions["nonpool"] = str(results[i])
        elif(labels[i] == "pool"):
            predictions["pool"] = str(results[i])

    return predictions




# def predict(image, model, label, address):
#     address_concat = address
#     file_name = image
#     model_file = model
#     label_file = label
#     input_height = 224
#     input_width = 224
#     input_mean = 128
#     input_std = 128
#     input_layer = "input"
#     output_layer = "final_result"

#     graph = load_graph(model_file)
#     t = read_tensor_from_image_file(file_name,
#                                   input_height=input_height,
#                                   input_width=input_width,
#                                   input_mean=input_mean,
#                                   input_std=input_std)

#     input_name = "import/" + input_layer
#     output_name = "import/" + output_layer
#     input_operation = graph.get_operation_by_name(input_name);
#     output_operation = graph.get_operation_by_name(output_name);

#     with tf.Session(graph=graph) as sess:
#         start = time.time()
#         results = sess.run(output_operation.outputs[0],
#                           {input_operation.outputs[0]: t})
#         end=time.time()
#     results = np.squeeze(results)

#     top_k = results.argsort()[-5:][::-1]
#     labels = load_labels(label_file)

#     print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

#     predictions = {"address": address_concat, "solar": "", "nonsolar":"", "map": "", "outline":"", "image":""}

#     for i in top_k:
#         print(labels[i], results[i])
#         if(labels[i] == "nonsolar jpeg"):
#             predictions["nonsolar"] = str(results[i])
#         elif(labels[i] == "solar jpeg"):
#             predictions["solar"] = str(results[i])

#     return predictions


key = "AIzaSyD5wWKXepOXeOfoPtbOi-HgYs2D969iL8k"





PEOPLE_FOLDER = os.path.join('detection', 'maps')
MAP_FOLDER = os.path.join('adus', 'maps_adu')
OUTLINE_FOLDER = os.path.join('adus', 'outlines_adu')
PROP_FOLDER = os.path.join('adus', 'props_adu')


DB_URL = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user='Apoorv',pw='qwerty123',url='adu-analysis.czgomrftwp3r.us-east-1.rds.amazonaws.com:5432',db='parcels')

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)





# @app.route('/detect/model1/method1/<address>', methods=['GET'])
# def detect(address):
#     #print("here")
#     address_concat = "+".join([x.strip(",") for x in address.split("_")])
#     #print("here")
#     url_map = "https://maps.googleapis.com/maps/api/staticmap?size=256x256&zoom=20&center=" + address_concat + "&maptype=satellite&key="+ key
#     #url_outline = "https://maps.googleapis.com/maps/api/staticmap?size=256x256&zoom=20&center="+address_concat+"&style=feature:landscape.man_made%7Celement:geometry.stroke%7Ccolor:0xff0000&key="+ key
#     content_maps = os.listdir("detection/maps")
#     if (address_concat + ".png") not in content_maps:
#         print("here")
#         try:
#             with urllib.request.urlopen(url_map) as response, open("detection/maps/" + address_concat + ".png", 'wb') as out_file:
#                         shutil.copyfileobj(response, out_file)
#         except:
#             abort(404)



#     file_name = "detection/maps/" + address_concat + ".png"
#     model_file = "detection/tf_files2/retrained_graph.pb"
#     label_file = "detection/tf_files2/retrained_labels.txt"

#     result = predict(file_name, model_file, label_file, address_concat)
#     result["map"] = url_map
#     result["image"] = address_concat + ".png"
#     return jsonify(result)


################    DETECTION OF ADU   #########################



@app.route('/detect_adu/model1/<address>', methods = ['GET'])
def detect_adu(address):
    address_concat = "+".join([x.strip(",") for x in address.split("_")])
    url_map = "https://maps.googleapis.com/maps/api/staticmap?size=512x512&zoom=20&center=" + address_concat + "&maptype=satellite&key="+ key
    url_outline = "https://maps.googleapis.com/maps/api/staticmap?size=512x512&zoom=20&center="+address_concat+"&style=feature:administrative.land_parcel%7Celement:geometry.stroke%7Ccolor:0xff0000&key="+ key
    content_maps = os.listdir("adus/maps_adu")
    content_outlines = os.listdir("adus/outlines_adu")
    content_props = os.listdir("adus/props_adu")

    if (address_concat + ".png") not in content_props:
        if (address_concat + ".png") not in content_maps:
            print("here")
            try:
                with urllib.request.urlopen(url_map) as response, open("adus/maps_adu/" + address_concat + ".png", 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
            except:
                abort(404)

        if (address_concat + ".png") not in content_outlines:
            print("here")
            try:
                with urllib.request.urlopen(url_outline) as response, open("adus/outlines_adu/" + address_concat + ".png", 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
            except:
                abort(404)



        # try:
        map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
        outline_im = ski.imread("adus/outlines_adu/" + address_concat + ".png")

        copy_out = np.empty_like(outline_im)
        copy_out[:] = outline_im
        for i in range(512):
            for j in range(512):                          
                if((copy_out[i][j][0] >= 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                    copy_out[i][j][0] = 255
                    copy_out[i][j][1] = 255
                    copy_out[i][j][2] = 255
                else:
                    copy_out[i][j][0] = 0
                    copy_out[i][j][1] = 0
                    copy_out[i][j][2] = 0
        grey_image = skc.rgb2gray(copy_out)

        fgi = (grey_image*255).astype(np.uint8)

        binary_img = img_as_ubyte(fgi)

        selem = disk(7)
        dilated = dilation(binary_img, selem)
        forest_fire(dilated,256,256)
        eroded2 = erosion(dilated, disk(12))
        binary = eroded2 > 127
        image_cleaned = morphology.remove_small_objects(binary, 750)
        dilated2 = dilation(image_cleaned, disk(10))

        copy_main = np.empty_like (map_im)
        copy_main[:] = map_im
        for i in range(512):
            for j in range(512):
                if(not(dilated2[i][j])):
                    copy_main[i][j][0] = 0
                    copy_main[i][j][1] = 0
                    copy_main[i][j][2] = 0

        
        ski.imsave("adus/props_adu/"+address_concat + ".png", copy_main)
                #print(address)
        # except:
        #     abort(404)

    file_name = "adus/props_adu/" + address_concat + ".png"
    model_file = "detection/tf_files3/retrained_graph.pb"
    label_file = "detection/tf_files3/retrained_labels.txt"

    result = predict_adu(file_name, model_file, label_file, address_concat)
    result["map"] = url_map
    result["outline"] = url_outline
    result["image"] = address_concat + ".png"
    result["link_map"] = "https://hausable-analysis.appspot.com/get_map/" + result["image"]
    result["link_outline"] = "https://hausable-analysis.appspot.com/get_outline/" + result["image"]
    result["link_property_boundary"] = "https://hausable-analysis.appspot.com/get_property_boundary/" + result["image"]
    return jsonify(result)



################ SOLAR MODEL2 ################



@app.route('/detect/model2/method1/<address>', methods = ['GET'])
def detect_solar2(address):
    address_concat = "+".join([x.strip(",") for x in address.split("_")])
    url_map = "https://maps.googleapis.com/maps/api/staticmap?size=512x512&zoom=20&center=" + address_concat + "&maptype=satellite&key="+ key
    url_outline = "https://maps.googleapis.com/maps/api/staticmap?size=512x512&zoom=20&center="+address_concat+"&style=feature:administrative.land_parcel%7Celement:geometry.stroke%7Ccolor:0xff0000&key="+ key
    content_maps = os.listdir("adus/maps_adu")
    content_outlines = os.listdir("adus/outlines_adu")
    content_props = os.listdir("adus/props_adu")

    if (address_concat + ".png") not in content_props:
        if (address_concat + ".png") not in content_maps:
            print("here")
            try:
                with urllib.request.urlopen(url_map) as response, open("adus/maps_adu/" + address_concat + ".png", 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
            except:
                abort(404)

        if (address_concat + ".png") not in content_outlines:
            print("here")
            try:
                with urllib.request.urlopen(url_outline) as response, open("adus/outlines_adu/" + address_concat + ".png", 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
            except:
                abort(404)



        # try:
        map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
        outline_im = ski.imread("adus/outlines_adu/" + address_concat + ".png")

        copy_out = np.empty_like(outline_im)
        copy_out[:] = outline_im
        for i in range(512):
            for j in range(512):                          
                if((copy_out[i][j][0] >= 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                    copy_out[i][j][0] = 255
                    copy_out[i][j][1] = 255
                    copy_out[i][j][2] = 255
                else:
                    copy_out[i][j][0] = 0
                    copy_out[i][j][1] = 0
                    copy_out[i][j][2] = 0
        grey_image = skc.rgb2gray(copy_out)

        fgi = (grey_image*255).astype(np.uint8)

        binary_img = img_as_ubyte(fgi)

        selem = disk(7)
        dilated = dilation(binary_img, selem)
        forest_fire(dilated,256,256)
        eroded2 = erosion(dilated, disk(12))
        binary = eroded2 > 127
        image_cleaned = morphology.remove_small_objects(binary, 750)
        dilated2 = dilation(image_cleaned, disk(10))

        copy_main = np.empty_like (map_im)
        copy_main[:] = map_im
        for i in range(512):
            for j in range(512):
                if(not(dilated2[i][j])):
                    copy_main[i][j][0] = 0
                    copy_main[i][j][1] = 0
                    copy_main[i][j][2] = 0

        
        ski.imsave("adus/props_adu/"+address_concat + ".png", copy_main)
                #print(address)
        # except:
        #     abort(404)

    file_name = "adus/props_adu/" + address_concat + ".png"
    model_file = "detection/solar_24_2/retrained_graph.pb"
    label_file = "detection/solar_24_2/retrained_labels.txt"

    result = predict_solar(file_name, model_file, label_file, address_concat)
    result["map"] = url_map
    result["outline"] = url_outline
    result["image"] = address_concat + ".png"
    result["link_map"] = "https://hausable-analysis.appspot.com/get_map/" + result["image"]
    result["link_outline"] = "https://hausable-analysis.appspot.com/get_outline/" + result["image"]
    result["link_property_boundary"] = "https://hausable-analysis.appspot.com/get_property_boundary/" + result["image"]
    

    return jsonify(result)



@app.route('/get_image/<image_link>')
def get_image(image_link=None):
    print(image_link)
    filename = os.path.join(PEOPLE_FOLDER, image_link)
    return send_file(filename, mimetype='image/png')





@app.route('/get_map/<image_link>')
def get_map(image_link=None):
    print(image_link)
    filename = os.path.join(MAP_FOLDER, image_link)
    return send_file(filename, mimetype='image/png')



@app.route('/get_outline/<image_link>')
def get_outline(image_link=None):
    print(image_link)
    filename = os.path.join(OUTLINE_FOLDER, image_link)
    return send_file(filename, mimetype='image/png')



@app.route('/get_property_boundary/<image_link>')
def get_prop(image_link=None):
    print(image_link)
    filename = os.path.join(PROP_FOLDER, image_link)
    return send_file(filename, mimetype='image/png')
####################################################################################################################

# @app.route('/get_predictions/<pid>', methods = ['GET'])
# def predictions(pid):
#     prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
#     coords = prop_metrics.Coordinates.iloc[0]
#     part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
#     if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
#         if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
#             part_2 = ""
#         else:
#             part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
#     else:
#         part_2 = prop_metrics.SITUS_CITY.iloc[0]
#     address = part_1 + "+" + part_2

#     content_maps = os.listdir("images/maps")
#     content_outlines = os.listdir("images/outlines")
#     content_props_ml = os.listdir("images/props_ml")
#     content_props_web = os.listdir("images/props_web")


#     # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
#     # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
#     url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
#     url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

#     if (address + ".png") not in content_props_web:
#         if (address + ".png") not in content_maps:
#             print("here")
#             try:
#                 with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
#                             shutil.copyfileobj(response, out_file)
#             except:
#                 abort(404)

#         if (address + ".png") not in content_outlines:
#             print("here")
#             try:
#                 with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
#                             shutil.copyfileobj(response, out_file)
#             except:
#                 abort(404)

#         # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
#         outline_im = ski.imread("images/outlines/" + address + ".png")


#         copy_out = np.empty_like (outline_im)
#         copy_out[:] = outline_im
#         for i in range(600):
#             for j in range(600):                          
#                 if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
#                     copy_out[i][j][0] = 255
#                     copy_out[i][j][1] = 255
#                     copy_out[i][j][2] = 255
#                 else:
#                     copy_out[i][j][0] = 0
#                     copy_out[i][j][1] = 0
#                     copy_out[i][j][2] = 0


#         grey_image = skc.rgb2gray(copy_out)
#         fgi = (grey_image*255).astype(np.uint8)

#         binary_img = img_as_ubyte(fgi)

#         selem = disk(3)
#         eroded = erosion(binary_img, selem)

#         selem = disk(10)
#         dilated = dilation(eroded, selem)

#         edge = canny(dilated)

#         selem = disk(1)
#         dilated_edge = dilation(edge, selem)

#         img = Image.open("images/maps/" + address + ".png")
#         img = img.convert("RGBA")
#         img_ml = img.convert("RGB")

#         pixdata = img.load()
#         pixdata_ml = img_ml.load()

#         width, height = img.size
#         for y in range(width):
#             for x in range(height):
#                 if(dilated[y,x]==0):
                    
#                     pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

#                     pixdata_ml[x, y] = (0, 0, 0)
#                 if(dilated_edge[y,x]):
#                     pixdata[x, y] = (255, 0, 0, 255)

#         img.save("images/props_web/" + address + ".png", "PNG")
#         img_ml.save("images/props_ml/" + address + ".png", "PNG")

#     file_name = "images/props_ml/" + address_concat + ".png"
#     model_file_adu = "detection/tf_files3/retrained_graph.pb"
#     label_file_adu = "detection/tf_files3/retrained_labels.txt"
#     model_file_solar = "detection/solar_24_2/retrained_graph.pb"
#     label_file_solar = "detection/solar_24_2/retrained_labels.txt"
#     model_file_pool = "detection/tf_files3/retrained_graph.pb"
#     label_file_pool = "detection/tf_files3/retrained_labels.txt"


#     predictions = {"mapfile": "", "outlinefile": "", "props_web": "", "props_ml": ""}

#     result_adu = predict_adu(file_name, model_file_adu, label_file_adu, address, 299, 299)
#     predictions.update(result_adu)


#     result_solar = predict_solar(file_name, model_file, label_file, address_concat, 299, 299)
#     predictions.update(result_solar)



#     result_solar = predict_solar()

#     # result["map"] = url_map
#     # result["outline"] = url_outline
#     # result["image"] = address_concat + ".png"



##########################################################################################################
Folder = os.path.join("images", "props_web")
@app.route('/get_metrics/<pid>', methods = ['GET'])
def get_metrics(pid):
    start = time.clock()
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    print("Reading the csv", time.clock() - start)
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    print("Initial Processing takes in sec:", time.clock() - start)



    
    content_props_web = os.listdir("images/props_web")


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try:

        if (address + ".png") not in content_props_web:
            content_maps = os.listdir("images/maps")
            content_outlines = os.listdir("images/outlines")
            content_props_ml = os.listdir("images/props_ml")
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    image_link = address + ".png"

    filename = os.path.join(Folder, image_link)
    print("Get_metrics took:", time.clock() - start)
    return send_file(filename, mimetype='image/png')



###########################


@app.route('/get_solar_predictions/<pid>', methods = ['GET'])
def solar_predictions(pid):
    start = time.clock()
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")

    print("time_elapsed:", time.clock() - start)


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")

            print("time_elapsed:", time.clock() - start)


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            print("time_elapsed:", time.clock() - start)

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")

            print("ML and web files are done")
            print("time_elapsed:", time.clock() - start)

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)

    print("All the images required are present")
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/solar_24_2/retrained_graph.pb"
    label_file = "detection/solar_24_2/retrained_labels.txt"

    print("Going to predict now")
    print("time_elapsed:", time.clock() - start)

    result = predict_solar(file_name, model_file, label_file, address, 299, 299, "Mul")

    print("Golden point")
    print("time_elapsed:", time.clock() - start)

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"
    print("get_solar_predictions took:", time.clock() - start)

    return(jsonify(result))


#####################################



###########################


@app.route('/get_solar_predictions_mobile/<pid>', methods = ['GET'])
def solar_predictions_mobile(pid):
    start = time.clock()
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")

    print("time_elapsed:", time.clock() - start)


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")

            print("time_elapsed:", time.clock() - start)


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            print("time_elapsed:", time.clock() - start)

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")

            print("ML and web files are done")
            print("time_elapsed:", time.clock() - start)

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)

    print("All the images required are present")
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/solar_mobile/retrained_graph.pb"
    label_file = "detection/solar_mobile/retrained_labels.txt"

    print("Going to predict now")
    print("time_elapsed:", time.clock() - start)

    result = predict_solar(file_name, model_file, label_file, address, 224, 224, "input")

    print("Golden point")
    print("time_elapsed:", time.clock() - start)

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"
    print("get_solar_predictions took:", time.clock() - start)

    return(jsonify(result))


#####################################


# @app.route('/get_solar_predictions_type2/<pid>', methods = ['GET'])
# def solar_predictions_type_2(pid):
#     start = time.clock()
#     prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
#     coords = prop_metrics.Coordinates.iloc[0]
#     part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
#     if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
#         if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
#             part_2 = ""
#         else:
#             part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
#     else:
#         part_2 = prop_metrics.SITUS_CITY.iloc[0]
#     address = part_1 + "+" + part_2

#     content_maps = os.listdir("images2/maps")
#     content_outlines = os.listdir("images2/outlines")
#     content_props_ml = os.listdir("images2/props_ml")
#     content_props_web = os.listdir("images2/props_web")


#     # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
#     # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
#     url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
#     url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 

#     print("time_elapsed:", time.clock() - start)
    

#     try: 
#         if (address + ".png") not in content_props_web:
#             if (address + ".png") not in content_maps:
#                 print("here")
#                 try:
#                     with urllib.request.urlopen(url_map) as response, open("images2/maps/" + address + ".png", 'wb') as out_file:
#                                 shutil.copyfileobj(response, out_file)
#                 except:
#                     abort(404)

#             if (address + ".png") not in content_outlines:
#                 print("here")
#                 try:
#                     with urllib.request.urlopen(url_outline_ml) as response, open("images2/outlines/" + address + ".png", 'wb') as out_file:
#                                 shutil.copyfileobj(response, out_file)
#                 except:
#                     abort(404)

#             # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
#             outline_im = ski.imread("images2/outlines/" + address + ".png")

#             map_im = ski.imread("images2/maps/" + address + ".png")

#             print("time_elapsed:", time.clock() - start)


#             copy_out = np.empty_like (outline_im)
#             copy_out[:] = outline_im
#             for i in range(600):
#                 for j in range(600):                          
#                     if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
#                         copy_out[i][j][0] = 255
#                         copy_out[i][j][1] = 255
#                         copy_out[i][j][2] = 255
#                     else:
#                         copy_out[i][j][0] = 0
#                         copy_out[i][j][1] = 0
#                         copy_out[i][j][2] = 0


#             grey_image = skc.rgb2gray(copy_out)
#             fgi = (grey_image*255).astype(np.uint8)

#             binary_img = img_as_ubyte(fgi)

#             selem = disk(3)
#             eroded = erosion(binary_img, selem)

#             selem = disk(10)
#             dilated = dilation(eroded, selem)

#             edge = canny(dilated)

#             selem = disk(1)
#             dilated_edge = dilation(edge, selem)


#             print("time_elapsed:", time.clock() - start)

#             # img = Image.open("images/maps/" + address + ".png")
#             # img = img.convert("RGBA")
#             # img_ml = img.convert("RGB")

#             # pixdata = img.load()
#             # pixdata_ml = img_ml.load()

#             # width, height = img.size
#             # for y in range(width):
#             #     for x in range(height):
#             #         if(dilated[y,x]==0):
                        
#             #             pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

#             #             pixdata_ml[x, y] = (0, 0, 0)
#             #         if(dilated_edge[y,x]):
#             #             pixdata[x, y] = (255, 0, 0, 255)

#             # img.save("images/props_web/" + address + ".png", "PNG")
#             # img_ml.save("images/props_ml/" + address + ".png", "PNG")



#             copy_main = np.empty_like (map_im)
#             copy_main[:] = map_im
#             copy_ml = np.empty_like (map_im)
#             copy_ml[:] = map_im
#             for i in range(copy_main.shape[0]):
#                 for j in range(copy_main.shape[1]):
#                     if(dilated[i][j] == 0):
#                         copy_main[i][j][0] = int(copy_main[i][j][0] * 0.5)
#                         copy_main[i][j][1] = int(copy_main[i][j][1] * 0.5)
#                         copy_main[i][j][2] = int(copy_main[i][j][2] * 0.5)
#                         copy_ml[i][j][0] = 0
#                         copy_ml[i][j][1] = 0
#                         copy_ml[i][j][2] = 0
#                     if(dilated_edge[i][j]):
#                         copy_main[i][j][0] = 255
#                         copy_main[i][j][1] = 0
#                         copy_main[i][j][2] = 0

            
#             ski.imsave("images2/props_web/"+address + ".png", copy_main)
#             ski.imsave("images2/props_ml/"+address + ".png", copy_main)

#         print("ML and web files are done")

#         print("time_elapsed:", time.clock() - start)

#     except:
#         abort(404)

#     # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
#     # "map" = url_map,
#     # "outline"= url_outline_ml}
#     # image_link = address + ".png"

#     # filename = os.path.join(Folder, image_link)

#     print("All the images required are present")
#     file_name = "images2/props_ml/" + address + ".png"
#     model_file = "detection/solar_24_2/retrained_graph.pb"
#     label_file = "detection/solar_24_2/retrained_labels.txt"

#     print("Going to predict now")
#     print("time_elapsed:", time.clock() - start)

#     result = predict_solar(file_name, model_file, label_file, address, 299, 299, "Mul")

#     print("Golden point")
#     print("time_elapsed:", time.clock() - start)

#     result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"
#     print("get_solar_predictions took:", time.clock() - start)

#     return(jsonify(result))


#####################################



@app.route('/get_adu_predictions/<pid>', methods = ['GET'])
def adu_predictions(pid):
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/tf_files3/retrained_graph.pb"
    label_file = "detection/tf_files3/retrained_labels.txt"

    result = predict_adu(file_name, model_file, label_file, address, 299, 299, "Mul")

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"

    return(jsonify(result))



#########################



@app.route('/get_adu_predictions_mobile/<pid>', methods = ['GET'])
def adu_predictions_mobile(pid):
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/adu_mobile/retrained_graph.pb"
    label_file = "detection/adu_mobile/retrained_labels.txt"

    result = predict_adu(file_name, model_file, label_file, address, 224, 224, "input")

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"

    return(jsonify(result))



#########################





@app.route('/get_pool_predictions/<pid>', methods = ['GET'])
def pool_predictions(pid):
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".jpg", "JPEG")

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/pool_2/retrained_graph.pb"
    label_file = "detection/pool_2/retrained_labels.txt"

    result = predict_pool(file_name, model_file, label_file, address, 299, 299, "Mul")

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"

    return(jsonify(result))




#########################


@app.route('/get_pool_predictions_mobile/<pid>', methods = ['GET'])
def get_pool_predictions_mobile(pid):
    prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
    coords = prop_metrics.Coordinates.iloc[0]
    part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
    if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
            part_2 = ""
        else:
            part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
    else:
        part_2 = prop_metrics.SITUS_CITY.iloc[0]
    address = part_1 + "+" + part_2

    content_maps = os.listdir("images/maps")
    content_outlines = os.listdir("images/outlines")
    content_props_ml = os.listdir("images/props_ml")
    content_props_web = os.listdir("images/props_web")


    # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
    # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
    url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
    url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
    

    try: 
        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except:
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".jpg", "JPEG")

    except:
        abort(404)

    # links = {"property": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png",
    # "map" = url_map,
    # "outline"= url_outline_ml}
    # image_link = address + ".png"

    # filename = os.path.join(Folder, image_link)
    file_name = "images/props_ml/" + address + ".png"
    model_file = "detection/pool_mobile/retrained_graph.pb"
    label_file = "detection/pool_mobile/retrained_labels.txt"

    result = predict_pool(file_name, model_file, label_file, address, 224, 224, "input")

    result["property"] = "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"

    return(jsonify(result))


#################################################################################

####################################################################################################################


@app.route('/get_predictions/<pid>', methods = ['GET'])
def predictions(pid):
    try:
        # prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
        # coords = prop_metrics.Coordinates.iloc[0]
        start = time.time()
        query = "SELECT * FROM parcel_boundaries WHERE PARCEL_ID = " + "'" +pid + "'"
        print(query)
        # prop_metrics = whole_data[whole_data.PARCEL_ID == pid]
        # print("Reading the csv", time.clock() - start)
        # coords = prop_metrics.Coordinates.iloc[0]
        # cursor.execute(query)
        # results = cursor.fetchall()

        results = db.engine.execute(query).fetchall()
        print("*" * 80)
        print(results[0])



        # part_1 = prop_metrics.SITUS.iloc[0].replace(" ", "+")
        # if(pd.isnull(prop_metrics.SITUS_CITY.iloc[0])):
        #     if(pd.isnull(prop_metrics.MAIL_ADDRE.iloc[0])):
        #         part_2 = ""
        #     else:
        #         part_2 = "+".join(prop_metrics.MAIL_ADDRE.iloc[0].split(" ")[:-1])
        # else:
        #     part_2 = prop_metrics.SITUS_CITY.iloc[0]
        # address = part_1 + "+" + part_2



        try:


            print(results[0])
            coords = results[0][-1]
            print(coords)
            add_temp = results[0][2] + ' ' + results[0][8]
            address = add_temp.replace(" ", "+")
            print(address)

        except Exception as e:
            print(e)
            return("Not Found in Database.")






        content_maps = os.listdir("images/maps")
        content_outlines = os.listdir("images/outlines")
        content_props_ml = os.listdir("images/props_ml")
        content_props_web = os.listdir("images/props_web")


        # url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite"
        # url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords
        url_map = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=satellite&key="+key
        url_outline_ml = "http://maps.googleapis.com/maps/api/staticmap?center="+ address+"&zoom=20&size=600x600&maptype=feature:%7Celement:geometry%7Ccolor:0xff0000&sensor=false&path=color%3ared|weight:1|fillcolor%3ared" + coords + "&key="+key 
        print(url_map)
        print(url_outline_ml)
        

        if (address + ".png") not in content_props_web:
            if (address + ".png") not in content_maps:
                print("here")
                try:
                    with urllib.request.urlopen(url_map) as response, open("images/maps/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except Exception as e1:
                    print(e1)
                    abort(404)

            if (address + ".png") not in content_outlines:
                print("here")
                try:
                    with urllib.request.urlopen(url_outline_ml) as response, open("images/outlines/" + address + ".png", 'wb') as out_file:
                                shutil.copyfileobj(response, out_file)
                except Exception as e2:
                    print(e2)
                    abort(404)

            # map_im = ski.imread("adus/maps_adu/" + address_concat + ".png")
            outline_im = ski.imread("images/outlines/" + address + ".png")


            print("there")


            copy_out = np.empty_like (outline_im)
            copy_out[:] = outline_im
            for i in range(600):
                for j in range(600):                          
                    if((copy_out[i][j][0] > 210 and copy_out[i][j][1] <= 211 and copy_out[i][j][2] <= 211)):
                        copy_out[i][j][0] = 255
                        copy_out[i][j][1] = 255
                        copy_out[i][j][2] = 255
                    else:
                        copy_out[i][j][0] = 0
                        copy_out[i][j][1] = 0
                        copy_out[i][j][2] = 0


            grey_image = skc.rgb2gray(copy_out)
            fgi = (grey_image*255).astype(np.uint8)

            binary_img = img_as_ubyte(fgi)

            selem = disk(3)
            eroded = erosion(binary_img, selem)

            selem = disk(10)
            dilated = dilation(eroded, selem)

            edge = canny(dilated)

            selem = disk(1)
            dilated_edge = dilation(edge, selem)

            img = Image.open("images/maps/" + address + ".png")
            img = img.convert("RGBA")
            img_ml = img.convert("RGB")

            pixdata = img.load()
            pixdata_ml = img_ml.load()

            width, height = img.size
            for y in range(width):
                for x in range(height):
                    if(dilated[y,x]==0):
                        
                        pixdata[x, y] = (int(pixdata[x, y][0]), int(pixdata[x, y][1]), int(pixdata[x, y][2]), 127)

                        pixdata_ml[x, y] = (0, 0, 0)
                    if(dilated_edge[y,x]):
                        pixdata[x, y] = (255, 0, 0, 255)

            img.save("images/props_web/" + address + ".png", "PNG")
            img_ml.save("images/props_ml/" + address + ".png", "PNG")
        print("there")

        file_name = "images/props_ml/" + address + ".png"
        model_file_adu = "detection/adu_mobile/retrained_graph.pb"
        label_file_adu = "detection/adu_mobile/retrained_labels.txt"
        model_file_solar = "detection/solar_mobile/retrained_graph.pb"
        label_file_solar = "detection/solar_mobile/retrained_labels.txt"
        model_file_pool = "detection/pool_mobile/retrained_graph.pb"
        label_file_pool = "detection/pool_mobile/retrained_labels.txt"


        predictions = {"mapfile": "", "outlinefile": "", "props_web": "", "props_ml": "https://hausable-analysis.appspot.com/get_temp_image/" + address + ".png"}

        result_adu = predict_adu(file_name, model_file_adu, label_file_adu, address, 224, 224, "input")
        predictions.update(result_adu)


        result_solar = predict_solar(file_name, model_file_solar, label_file_solar, address, 224, 224, "input")
        predictions.update(result_solar)


        result_pool = predict_pool(file_name, model_file_pool, label_file_pool, address, 224, 224, "input")
        predictions.update(result_pool)

        # predictions["map"] = url_map
        # predictions["outline"] = url_outline_ml
        # predictions["image"] = address + ".png"

        # print(type(predictions))
        return jsonify(predictions)
    
    except Exception as e:
        print(e)
        abort(404)


##############################################################################



@app.route('/get_temp_image/<image_link>')
def get_temp_image(image_link=None):
    print(image_link)
    filename = os.path.join(Folder, image_link)
    return send_file(filename, mimetype='image/png')


@app.route('/')
def index():
    return 'Index Page'

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

    