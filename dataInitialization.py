from particleInteraction import Particle

import numpy as np
from math import pi, sin, cos
from pandas import read_csv
import json

def getData ():
    parameters = {
        'AgNumOfAtoms': 0,
        'AgRadius': 0,
        'AgMass': 0,
        'AgEpsilon': 0,
        'AgSigma': 0,

        'AlNumOfAtoms': 0,
        'AlRadius': 0,
        'AlMass': 0,
        'AlEpsilon': 0,
        'AlSigma': 0,
        'AlEps': 0,
        'AlAlpha': 0,

        'TimeStep': 1e-16,
        'Steps': 100,
        'OutputFrequency': 2,
        'Borders': [[], [], []],
        'OutputFileName': 'output.dump'
    }

    dataObj = read_csv('Data.csv', delimiter=';')
    csvList = [tuple(row) for row in dataObj.values]
    for x in csvList:
        if x[0] == 'Borders':
            parameters[x[0]] = json.loads(x[1])
        elif x[0] == 'OutputFileName':
            parameters[x[0]] = x[1]
        else:
            parameters[x[0]] = float(x[1])
    parameters['AgNumOfAtoms'] = int(parameters['AgNumOfAtoms'])
    parameters['Steps'] = int(parameters['Steps'])
    parameters['OutputFrequency'] = int(parameters['OutputFrequency'])

    return parameters

def init_list_Al(N, radius, mass, epsilon, sigma, alpha, borders):
    particle_list = []
    particle_position_x = np.array([])
    particle_position_y = np.array([])
    particle_position_z = np.array([])

    particle_number_bottom = np.array([1]) #6, 9
    for i in range (2):
        particle_number_bottom = np.append(particle_number_bottom, int(round(2 * pi * radius * (i + 2) / (2 * radius))))
    angle_bottom = [2 * pi / particle_number_bottom[i] for i in range(len(particle_number_bottom))]
    radius_bottom = [radius * (i + 1) for i in range(len(particle_number_bottom))]

    particle_number_layers = np.array([]) #13, 16, 19, 22, 25, 28
    for i in range (6):
        particle_number_layers = np.append(particle_number_layers, int(round(2 * pi * radius *\
                                                                (i + 1 + len(particle_number_bottom)) / (2 * radius))))
    angle_layers = [2 * pi / particle_number_layers[i] for i in range(len(particle_number_layers))]
    radius_layers = [radius * (i + len(particle_number_bottom) + 1) for i in range(len(particle_number_layers))]

    center_x = borders[0] / 2
    center_y = borders[1] / 2
    particle_position_x = np.append(particle_position_x, center_x)
    particle_position_y = np.append(particle_position_y, center_y)
    particle_position_z = np.append(particle_position_z, 0.)

    surface = np.array([31, 34, 39, 44, 49, 54]) #
    angle_surface = [2 * pi / surface[i] for i in range(len(surface))]
    radius_surface = [radius * (i + len(particle_number_bottom) + len(particle_number_layers) + 1) for i in
                      range(len(surface))]

    for i in range(1, len(particle_number_bottom)):
        r = radius_bottom[i]
        angle = angle_bottom[i]
        for j in range(particle_number_bottom[i]):
            particle_position_x = np.append(particle_position_x, center_x + r * cos(angle))
            particle_position_y = np.append(particle_position_y, center_y + r * sin(angle))
            particle_position_z = np.append(particle_position_z, 0.)
            angle += angle_bottom[i]

    for i in range(len(particle_number_layers)):
        r = radius_layers[i]
        angle = angle_layers[i]
        for j in range(int(particle_number_layers[i])):
            particle_position_x = np.append(particle_position_x, center_x + r * cos(angle))
            particle_position_y = np.append(particle_position_y, center_y + r * sin(angle))
            particle_position_z = np.append(particle_position_z, (i + 1) * radius)
            angle += angle_layers[i]

    z = particle_position_z[(len(particle_position_z) - 1)]
    k = 1.0
    for i in range(len(surface)):
        r = radius_surface[i]
        angle = angle_surface[i]
        for j in range(int(surface[i])):
            x = center_x + r * k * cos(angle)
            y = center_y + r * k * sin(angle)
            if (x <= borders[0] and y <= borders[1] and z<= borders[2] and x >= 0 and y >= 0 and z >= 0):
                particle_position_x = np.append(particle_position_x, x)
                particle_position_y = np.append(particle_position_y, y)
                particle_position_z = np.append(particle_position_z, z)
            angle += angle_surface[i]
        k += 0.055

    N += len(particle_position_x)
    for i in range(N):
        v = np.append(1e-10, 1e-10)
        v = np.append(v, 1e-10)
        f = np.array([0. for i in range (len(v))])
        a = np.array([0. for i in range(len(v))])
        pos = np.array([particle_position_x[i], particle_position_y[i], particle_position_z[i]])
        newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=0)

        particle_list.append(newparticle)
    return particle_list

def init_list_Ag (N, radius, mass, epsilon, sigma, alpha, borders, z):
    # Случайным образом генерируем массив объектов Particle, число частиц равно N
    # В данной программе рассмотрен трехмерный случай
    particle_list = []

    for i in range(N):
        v_mag = np.random.rand(1) * 6
        v_ang = np.random.rand(1) * 2 * np.pi
        v_ang_z = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))
        v = np.append(v, v_mag * np.cos(v_ang_z))
        f = np.array([0. for i in range (len(v))])
        a = np.array([0. for i in range(len(v))])

        collision = True
        while (collision == True):
            collision = False
            pos = []
            posx = radius + np.random.rand(1) * (borders[0] - 2 * radius)
            posy = radius + np.random.rand(1) * (borders[1] - 2 * radius)
            posz = radius + np.random.rand(1) * (borders[2] - 2 * radius)
            pos = np.append(pos, posx)
            pos = np.append(pos, posy)
            pos = np.append(pos, posz)
            if pos[2] < z:
                pos[2] += z
            newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=1)
            for j in range(len(particle_list)):
                collision = newparticle.check_coll(particle_list[j])
                if collision == True:
                    break

        particle_list.append(newparticle)
    return particle_list