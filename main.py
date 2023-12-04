import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

########################################################################################################################
# Создаем класс "Particle" для хранения информации о каждой частице из модели (координаты, скорость, ...)
# Класс "Particle" содержит функцию compute_step для вычисления координат и скорости частицы для следующего шага с
# помощью метода молекулярной динамики. Также содержит функции вычисления скорости после столкновения.
class Particle:
    def __init__(self, mass, radius, epsilon, sigma, position, velocity, force, acceleration):
        self.mass = mass #масса частицы
        self.radius = radius #радиус частицы
        self.epsilon = epsilon #глубина потенциальной ямы
        self.sigma = sigma #расстояние, на котором энергия взаимодействия становится нулевой

        # позиция частицы, скорость, ускорение, энергия взаимодействия для данной итерации
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.acceleration = np.array(acceleration)

        # все позиции частицы, скорости, модули скорости, полученные в ходе симуляции
        self.solpos = [np.copy(self.position)]
        self.solvel = [np.copy(self.velocity)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))]

    def compute_step(self, step):
        # вычисляем позицию и скорость частицы для следующего шага
        self.acceleration = 1 / self.mass * self.force
        self.position += step * self.velocity + 1 / 2 * self.acceleration * step * step
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
        di = x2 - x1
        norm = np.linalg.norm(di)
        if norm - (r1 + r2) * 1.1 < step * abs(np.dot(v1 - v2, di)) / norm:
            self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
            particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)

    def compute_refl(self, step, size):
        # вычисляем скорость частицы после столкновения с границей
        r, v, x = self.radius, self.velocity, self.position
        projx = step * abs(np.dot(v, np.array([1., 0.])))
        projy = step * abs(np.dot(v, np.array([0., 1.])))
        if abs(x[0]) - r < projx or abs(size - x[0]) - r < projx:
            self.velocity[0] *= -1
        if abs(x[1]) - r < projy or abs(size - x[1]) - r < projy:
            self.velocity[1] *= -1.

# Вычисляем энергию парного взаимодействия с помощью потенциала Леннарда-Джонса
def LennardJones (particle_list):
    force: float
    for i in range(len(particle_list)):
        for j in range(i + 1, len(particle_list)):
            r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))
            if r < 2.5 * particle_list[i].sigma:
                force = 48 * particle_list[i].epsilon * ((particle_list[i].sigma ** 12) / (r ** 13) - 0.5 * (particle_list[i].sigma ** 6) / (r ** 7))
            else: force = 0

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            if (all(particle_list[i].force)):
                np.add(particle_list[i].force, vel, out=particle_list[i].force, casting="unsafe")
            if (all(particle_list[j].force)):
                np.add(particle_list[j].force, -vel, out=particle_list[i].force, casting="unsafe")

########################################################################################################################
# Вычисляем позиции и скорости частиц для следующего шага
def solve_step(particle_list, step, size):
    # 1. Проверяем столкновение с границей или другой частицей для каждой частицы
    for i in range(len(particle_list)):
        particle_list[i].compute_refl(step, size)
        for j in range(i + 1, len(particle_list)):
            particle_list[i].compute_coll(particle_list[j], step)

    # 2. С помощью метода молекулярной динамики вычисляем позицию и скорость
    LennardJones(particle_list)
    for particle in particle_list:
        particle.compute_step(step)

########################################################################################################################

def init_list_random(N, radius, mass, epsilon, sigma, boxsize):
    # Случайным образом генерируем массив объектов Particle, число частиц равно N
    # В данной программе рассмотрен двумерный случай
    particle_list = []

    for i in range(N):
        v_mag = np.random.rand(1) * 6
        v_ang = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))
        f = np.array([0 for i in range (len(v))])
        a = np.array([0 for i in range(len(v))])

        collision = True
        while (collision == True):
            collision = False
            pos = radius + np.random.rand(2) * (boxsize - 2 * radius)
            newparticle = Particle(mass, radius, epsilon, sigma, pos, v, f, a)
            for j in range(len(particle_list)):
                collision = newparticle.check_coll(particle_list[j])
                if collision == True:
                    break

        particle_list.append(newparticle)
    return particle_list

particle_number = 150 # число частиц
boxsize = 10 # границы
tfin = 10 # время симуляции
stepnumber = 200 # число шагов
radius = 1.06e-1 # данные рассматриваемой частицы
mass = 6.63352599e-26
epsilon = 0.00801
sigma=3.54

timestep = tfin/stepnumber #временной шаг
particle_list = init_list_random(particle_number, radius, mass, epsilon, sigma, boxsize)

# Вычислительный эксперимент
for i in range(stepnumber):
    solve_step(particle_list, timestep, boxsize)

########################################################################################################################

# Визуализация решения с помощью библиотеки matplotlib

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)

hist = fig.add_subplot(1, 2, 2)

plt.subplots_adjust(bottom=0.2, left=0.15)

ax.axis('equal')
ax.axis([-1, 30, -1, 30])

ax.set_xlim([0, boxsize])
ax.set_ylim([0, boxsize])

# Отображаем движение частиц
circle = [None] * particle_number
for i in range(particle_number):
    circle[i] = plt.Circle((particle_list[i].solpos[0][0], particle_list[i].solpos[0][1]), particle_list[i].radius,
                           ec="black", lw=0.5)
    ax.add_patch(circle[i])

# Строим диаграмму распределение молекул по скоростям, исходя из результатов эксперимента
vel_mod = [particle_list[i].solvel_mag[0] for i in range(len(particle_list))]
hist.hist(vel_mod, bins=30, density=True, label="Simulation Data")
hist.set_xlabel("Speed")
hist.set_ylabel("Frecuency Density")

# Также вычислим распределение Максвелла

# Полная энергия системы должна быть постоянной
def total_Energy(particle_list, index):
    return sum([particle_list[i].mass / 2. * particle_list[i].solvel_mag[index] ** 2 for i in range(len(particle_list))])

E = total_Energy(particle_list, 0) # полная энергия
Average_E = E / len(particle_list) # энергия одной частицы
k = 1.38064852e-23 # постоянная больцмана
T = 2 * Average_E / (2 * k) # температура системы
m = particle_list[0].mass # масса частицы
v = np.linspace(0, 10, 100) # диапазон рассматриваемых скоростей
fv = 4 * np.pi * v ** 2 * np.power(m / (2 * np.pi * T * k), 1.5) * np.exp(-m * v ** 2 / (2 * T * k))
hist.plot(v, fv, label="Maxwell distribution")
hist.legend(loc="upper right")

# Используем Slider (ползунок) для отображения изменений с течением времени
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider = Slider(slider_ax,  't', 0, tfin, valinit=0, color='#5c05ff')

# функция обновления отображаемой информации
def update(time):
    i = int(np.rint(time / timestep))

    # Отображаем движение частиц
    for j in range(particle_number):
        circle[j].center = particle_list[j].solpos[i][0], particle_list[j].solpos[i][1]
    hist.clear()

    # Распределение молекул по скоростям, исходя из результатов эксперимента
    vel_mod = [particle_list[j].solvel_mag[i] for j in range(len(particle_list))]
    hist.hist(vel_mod, bins=30, density=True, label="Simulation Data")
    hist.set_xlabel("Speed")
    hist.set_ylabel("Frecuency Density")

    # Распределение Максвелла
    E = total_Energy(particle_list, i)
    Average_E = E / len(particle_list)
    k = 1.38064852e-23
    T = 2 * Average_E / (2 * k)
    m = particle_list[0].mass
    v = np.linspace(0, 10, 100)
    fv = 4 * np.pi * v ** 2 * np.power(m / (2 * np.pi * T * k), 1.5) * np.exp(-m * v ** 2 / (2 * T * k))
    hist.plot(v, fv, label="Maxwell distribution")
    hist.legend(loc="upper right")

slider.on_changed(update)
plt.show()