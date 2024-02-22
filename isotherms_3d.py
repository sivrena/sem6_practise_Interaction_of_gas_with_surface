import Dump

import numpy as np
from math import ceil, pi, sin, cos
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

########################################################################################################################
# Создаем класс "Particle" для хранения информации о каждой частице из модели (координаты, скорость, ...)
# Класс "Particle" содержит функцию compute_step для вычисления координат и скорости частицы для следующего шага с
# помощью метода молекулярной динамики. Также содержит функции вычисления скорости после столкновения.
class Particle:
    def __init__(self, mass, radius, epsilon, sigma, position, velocity, force, acceleration, alpha, adsorbate):
        self.mass = mass #масса частицы
        self.radius = radius #радиус частицы
        self.epsilon = epsilon #глубина потенциальной ямы
        self.sigma = sigma #расстояние, на котором энергия взаимодействия становится нулевой
        self.alpha = alpha

        self.adsorbate = adsorbate  #показывает, чем является частица - адсорбент или адсорбтив
        self.free = True  #флаг для адсорбата, а также для занятых активных центров
        self.count = 0
        self.vel_before_ads = 0.

        # позиция частицы, скорость, ускорение, энергия взаимодействия для данной итерации
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.acceleration = np.array(acceleration)

        # все позиции частицы, скорости, модули скорости, полученные в ходе симуляции
        self.solpos = [np.copy(self.position)]
        self.solvel = [np.copy(self.velocity)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))]

    def compute_step(self, step, z):
        # вычисляем позицию и скорость частицы для следующего шага
        self.acceleration = 1 / self.mass * self.force
        self.position += step * self.velocity + 1 / 2 * self.acceleration * step * step
        if self.adsorbate:
            if self.position[2] < z:
                d = self.position[2]
                self.position[2] += z - d
        self.velocity += self.acceleration * step

        self.solpos.append(np.copy(self.position))
        self.solvel.append(np.copy(self.velocity))
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocity)))

    def check_coll(self, particle):
        # проверяем столкновение частиц
        r1, r2 = self.radius, particle.radius
        x1, x2 = self.position, particle.position
        di = x2 - x1
        norm = np.linalg.norm(di)
        if norm - (r1 + r2) * 1.1 < 0:
            return True
        else:
            return False

    def compute_coll(self, particle, step):
        # вычисляем скорость частиц после столкновения
        m1, m2 = self.mass, particle.mass
        r1, r2 = self.radius, particle.radius
        v1, v2 = self.velocity, particle.velocity
        x1, x2 = self.position, particle.position
        ads1, ads2 = self.adsorbate, particle.adsorbate
        di = x2 - x1
        norm = np.linalg.norm(di)
        if norm - (r1 + r2) * 1.1 < step * abs(np.dot(v1 - v2, di)) / norm:
            if ads1 == ads2 and (self.free and particle.free):
                self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)
            else:
                if ads1:
                    self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                elif ads2:
                    particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)
            """elif ads1 == ads2 and self.free:
                self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
            elif ads1 == ads2 and particle.free:
                particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)
            elif ads1 and particle.free:
                    self.count += 1
                    particle.count += 1
                    self.vel_before_ads = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                    self.velocity -= v1
                    self.position = [particle.position[0], particle.position[1], particle.position[2] + r1 + r2]
                    self.free = False
                    particle.free = False"""

    def compute_refl(self, step, size):
        # вычисляем скорость частицы после столкновения с границей
        r, v, x = self.radius, self.velocity, self.position
        projx = step * abs(np.dot(v, np.array([1., 0., 0.])))
        projy = step * abs(np.dot(v, np.array([0., 1., 0.])))
        projz = step * abs(np.dot(v, np.array([0., 0., 1.])))

        if (abs(x[0]) - r < projx) or abs(size[0] - x[0]) - r < projx:
            self.velocity[0] *= -1
        if abs(x[1]) - r < projy or abs(size[1] - x[1]) - r < projy:
            self.velocity[1] *= -1
        if abs(x[2]) - r < projz or abs(size[2] - x[2]) - r < projz:
            self.velocity[2] *= -1

    def adsorption (self):
        if self.adsorbate and not(self.free) and self.count > 10:
            self.free = True
            self.count = 0
            self.velocity = np.array([self.vel_before_ads[0], self.vel_before_ads[1], 4.])
            self.vel_before_ads = 0.
        elif not(self.adsorbate) and not(self.free) and self.count > 10:
            self.free = True
            self.count = 0
        elif not(self.free):
            self.count += 1

# Вычисляем энергию парного взаимодействия с помощью потенциала Леннарда-Джонса
def LennardJones (particle_list, num):
    force: float
    for i in range(num):
        for j in range(i + 1, len(particle_list)):
            if (particle_list[i].adsorbate and particle_list[j].adsorbate and (
                    particle_list[i].free or particle_list[j].free)) or not (
                    particle_list[i].adsorbate and particle_list[j].adsorbate):

                r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))
                if r < 2.5 * particle_list[i].sigma:
                    force = 48 * particle_list[i].epsilon * ((particle_list[i].sigma ** 12) / (r ** 13) - 0.5 * (particle_list[i].sigma ** 6) / (r ** 7))
                else: force = 0

                vel = -(particle_list[i].position - particle_list[j].position) * force / r

                if (all(particle_list[i].force)):
                    np.add(particle_list[i].force, vel, out=particle_list[i].force, casting="unsafe")
                if (all(particle_list[j].force)):
                    np.add(particle_list[j].force, -vel, out=particle_list[i].force, casting="unsafe")

# Вычисляем энергию парного взаимодействия с помощью потенциала Morze
def Morze (particle_list):
    force: float
    for i in range(len(particle_list)):
        for j in range(i + 1, len(particle_list)):
            r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))
            force = particle_list[i].epsilon * particle_list[i].alpha * np.exp(particle_list[i].alpha*(particle_list[i].sigma - r))

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            if (all(particle_list[i].force)):
                np.add(particle_list[i].force, vel, out=particle_list[i].force, casting="unsafe")
            if (all(particle_list[j].force)):
                np.add(particle_list[j].force, -vel, out=particle_list[i].force, casting="unsafe")

########################################################################################################################
# Вычисляем позиции и скорости частиц для следующего шага
def solve_step(particle_list, Ag_num, Al_num, step, size, z):
    # 1. Проверяем столкновение с границей или другой частицей для каждой частицы
    for i in range(len(particle_list)):
        particle_list[i].compute_refl(step, size)
        for j in range(i + 1, len(particle_list)):
            particle_list[i].compute_coll(particle_list[j], step)

    # 2. С помощью метода молекулярной динамики вычисляем позицию и скорость
    Morze(particle_list[Ag_num:])
    LennardJones(particle_list, Ag_num)
    for particle in particle_list:
        #particle.adsorption()
        particle.compute_step(step, z)

########################################################################################################################
def init_list_Al(N, radius, mass, epsilon, sigma, alpha, borders):
    particle_list = []
    particle_position_x = np.array([])
    particle_position_y = np.array([])
    particle_position_z = np.array([])

    particle_number_bottom = [1, 6, 9]
    angle_bottom = [2 * pi / particle_number_bottom[i] for i in range(len(particle_number_bottom))]
    radius_bottom = [radius * (i + 1) for i in range(len(particle_number_bottom))]

    particle_number_layers = [13, 16, 19, 22, 25, 28]
    angle_layers = [2 * pi / particle_number_layers[i] for i in range(len(particle_number_layers))]
    radius_layers = [radius * (i + len(particle_number_bottom) + 1) for i in range(len(particle_number_layers))]

    center_x = borders[0] / 2
    center_y = borders[1] / 2
    particle_position_x = np.append(particle_position_x, center_x)
    particle_position_y = np.append(particle_position_y, center_y)
    particle_position_z = np.append(particle_position_z, 0.)

    surface = [31, 34, 39, 44, 49, 54]
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
        for j in range(particle_number_layers[i]):
            particle_position_x = np.append(particle_position_x, center_x + r * cos(angle))
            particle_position_y = np.append(particle_position_y, center_y + r * sin(angle))
            particle_position_z = np.append(particle_position_z, (i + 1) * radius)
            angle += angle_layers[i]

    z = particle_position_z[(len(particle_position_z) - 1)]
    k = 1.0
    for i in range(len(surface)):
        r = radius_surface[i]
        angle = angle_surface[i]
        for j in range(surface[i]):
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
        f = np.array([0 for i in range (len(v))])
        a = np.array([0 for i in range(len(v))])
        pos = np.array([particle_position_x[i], particle_position_y[i], particle_position_z[i]])
        newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=0)
        #coeff = np.sqrt(N)
        #collision = True
        #while (collision == True):
            #collision = False
            #pos = np.array([particle_position_x[i], particle_position_y[i], particle_position_z[i]])
            #newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a, alpha, adsorbate=0)
            #for j in range(len(particle_list)):
                #collision = newparticle.check_coll(particle_list[j])
                #if collision == True:
                    #break

        particle_list.append(newparticle)
    return particle_list

def init_list_random_Ag (N, radius, mass, epsilon, sigma, alpha, borders, z):
    # Случайным образом генерируем массив объектов Particle, число частиц равно N
    # В данной программе рассмотрен двумерный случай
    particle_list = []

    for i in range(N):
        v_mag = np.random.rand(1) * 6
        v_ang = np.random.rand(1) * 2 * np.pi
        v_ang_z = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))
        v = np.append(v, v_mag * np.cos(v_ang_z))
        f = np.array([0 for i in range (len(v))])
        a = np.array([0 for i in range(len(v))])

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

#boxsize = 4
Borders = [4, 4, 4] # границы (в нанометрах)
tfin = 10 # время симуляции
stepnumber = 200 # число шагов
timestep = tfin/stepnumber #временной шаг

BoltsmanConstant = 1.38 * 10e-23
temperature = 300 # Кельвины

particle_number_Al = 0 # число частиц
radius_Al = 1.21e-01 # данные рассматриваемой частицы
mass_Al = 4.4803831e-26
epsilon_Al = 0.2703 #0.03917
sigma_Al = 3.253
alpha_Al = 1.1646

particle_number_Ag = 75 # число частиц
radius_Ag = 1.06e-1 # данные рассматриваемой частицы
mass_Ag = 0.17911901e-26
epsilon_Ag = 0.00801
sigma_Ag = 3.54
alpha_Ag = 0

particle_list_Al = init_list_Al(particle_number_Al, radius_Al, mass_Al, epsilon_Al, sigma_Al, alpha_Al, Borders)
z = particle_list_Al[len(particle_list_Al) - 1].position[2]
center_x = Borders[0] / 2
center_y = Borders[1] / 2
rr = (particle_list_Al[len(particle_list_Al) - 1].position[0] - center_x) ** 2 + \
    (particle_list_Al[len(particle_list_Al) - 1].position[1] - center_y) ** 2
volume = Borders[0] * Borders[1] * (Borders[2] - z) + pi * z * z * (np.sqrt(rr) - 1 / 3 * z)

particle_list_Ag = init_list_random_Ag(particle_number_Ag, radius_Ag, mass_Ag, epsilon_Ag, sigma_Ag, alpha_Ag, Borders, z)

particle_number_Al = len(particle_list_Al)
particle_number = particle_number_Ag + particle_number_Al
particle_list = np.concatenate([particle_list_Ag, particle_list_Al])

pressure_moment = np.array([0. for i in range (stepnumber)])
adsorption_moment = np.array([0. for i in range (stepnumber)])
adsorbent_number = 0
for i in range (particle_number_Al):
    if particle_list_Al[i].position[2] < z:
        adsorbent_number += 1
adsorbent_number += round(2*pi*np.sqrt(rr)/2*radius_Al)

# Вычислительный эксперимент
OutputFileName = "output.dump"
if os.path.exists(OutputFileName):
    os.remove(OutputFileName)

for i in range(stepnumber):
    adsorbed_number = 0
    solve_step(particle_list, particle_number_Ag, particle_number_Al, timestep, Borders, z)

    Radius = np.array([particle_list[j].radius for j in range (len(particle_list))])
    Positions = np.array(([particle_list[j].position for j in range (len(particle_list))]))
    Velocities = np.array([particle_list[j].velocity for j in range (len(particle_list))])
    Dump.writeOutput(OutputFileName, particle_number, i, Borders,
                     radius=Radius, pos=Positions, v=Velocities)

    for particle in particle_list:
        if particle.adsorbate and particle.position[2] <= z and \
                (particle.position[0] - center_x)**2 + (particle.position[1] - center_y)**2 < rr:
            adsorbed_number += 1

    pressure_moment[i] += (particle_number_Ag - adsorbed_number) / (volume * 1e-27) * BoltsmanConstant * temperature
    adsorption_moment[i] += (adsorbed_number) / (adsorbent_number)

result = str(sum(pressure_moment)/stepnumber) + ' ' + str(sum(adsorption_moment)/stepnumber) + ' ' + str(temperature) \
         + ' ' + str(particle_number_Ag) + ' ' + str(Borders[0]) + ' ' + str(Borders[1]) + ' ' + str(Borders[2])

with open('isotherms_3d.txt', 'a') as f:
    f.write(result + '\n')

########################################################################################################################\
# Визуализация распределения по скоростям с помощью библиотеки matplotlib
fig = plt.figure(figsize=(6, 6))

hist = fig.add_subplot(111)
hist.set_title('Velocity distribution')

plt.subplots_adjust(bottom=0.2, left=0.15)

# Строим диаграмму распределение молекул по скоростям, исходя из результатов эксперимента
vel_mod = []
for i in range (len(particle_list[:particle_number_Ag])):
    if particle_list[i].free:
        vel_mod.append(particle_list[i].solvel_mag[0])

string = 'Temp. = ' + str(temperature) + '\nPart.numb. = ' + str(particle_number_Ag) + '\nArea = ' + str(Borders[0]) \
             + 'x' + str(Borders[1]) + 'x' + str(Borders[2])
hist.hist(vel_mod, bins=30, density=True, label=string)
hist.set_xlabel("Speed")
hist.set_ylabel("Frecuency Density")
hist.legend(loc="upper right")

# Используем Slider (ползунок) для отображения изменений с течением времени
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider = Slider(slider_ax,  't', 0, tfin, valinit=0, color='#5c05ff')

# функция обновления отображаемой информации
def update(time):
    i = int(np.rint(time / timestep))

    hist.clear()

    # Распределение молекул по скоростям, исходя из результатов эксперимента
    vel_mod = []
    for j in range (len(particle_list[:particle_number_Ag])):
        if particle_list[j].free:
            vel_mod.append(particle_list[j].solvel_mag[i])
            
    hist.hist(vel_mod, bins=30, density=True, label=string)
    hist.set_title('Velocity distribution')
    hist.set_xlabel("Speed")
    hist.set_ylabel("Frecuency Density")
    hist.legend(loc="upper right")

slider.on_changed(update)
plt.show()