import numpy as np

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

        # позиция частицы, скорость, ускорение, энергия взаимодействия для данной итерации
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.force = np.array(force)
        self.acceleration = np.array(acceleration)

        # все позиции частицы, скорости, модули скорости, полученные в ходе симуляции
        self.solpos = [np.copy(self.position)]
        self.solvel = [np.copy(self.velocity)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))]

    def compute_step(self, step, f):
        # вычисляем позицию и скорость частицы для следующего шага
        self.force = f
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

    def compute_coll(self, particle, step, center, R, Z):
        # вычисляем скорость частиц после столкновения
        m1, m2 = self.mass, particle.mass
        r1, r2 = self.radius, particle.radius
        v1, v2 = self.velocity, particle.velocity
        x1, x2 = self.position, particle.position
        ads1, ads2 = self.adsorbate, particle.adsorbate
        if ads1 == 2:
            ads1 = 1
        if ads2 == 2:
            ads2 = 1
        di = x2 - x1
        norm = np.linalg.norm(di)
        if norm - (r1 + r2) * 1.1 < step * abs(np.dot(v1 - v2, di)) / norm:
            if ads1 == ads2:
                self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)
            else:
                if ads1:
                    self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                    rght = center[0] + R
                    lft = center[0] - R
                    up = center[1] + R
                    dwn = center[1] - R
                    if (particle.position[0] < rght and particle.position[0] > lft and \
                            particle.position[1] < up and particle.position[1] > dwn and particle.position[2] < Z):
                        self.adsorbate = 2

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

# Вычисляем энергию парного взаимодействия с помощью потенциала Леннарда-Джонса
def LennardJones (particle_list, num):
    force: float
    force_LJ = np.array([particle.force for particle in particle_list])

    for i in range(num):
        for j in range(i + 1, len(particle_list)):

            r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))
            if r < 2.5 * particle_list[i].sigma:
                force = 48 * particle_list[i].epsilon * ((particle_list[i].sigma ** 12) / ((r * 1e9) ** 13) - 0.5 * (particle_list[i].sigma ** 6) / ((r*1e9) ** 7))
            else: force = 0

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            force_LJ[i][0] += vel[0]
            force_LJ[i][1] += vel[1]
            force_LJ[i][2] += vel[2]
            force_LJ[j][0] -= vel[0]
            force_LJ[j][1] -= vel[1]
            force_LJ[j][2] -= vel[2]

    return force_LJ

# Вычисляем энергию парного взаимодействия с помощью потенциала Morze
def Morze (particle_list):
    force: float
    force_M = np.array([[0., 0., 0.] for i in range (len(particle_list))])

    for i in range(len(particle_list)):
        for j in range(i + 1, len(particle_list)):
            r = np.sqrt(np.sum(np.square(particle_list[i].position - particle_list[j].position)))
            force = particle_list[i].epsilon * particle_list[i].alpha * np.exp(particle_list[i].alpha*(particle_list[i].sigma - (r*1e9)))

            vel = -(particle_list[i].position - particle_list[j].position) * force / r

            force_M[i][0] += vel[0]
            force_M[i][1] += vel[1]
            force_M[i][2] += vel[2]
            force_M[j][0] -= vel[0]
            force_M[j][1] -= vel[1]
            force_M[j][2] -= vel[2]

    return force_M

########################################################################################################################
# Вычисляем позиции и скорости частиц для следующего шага
def solve_step(particle_list, Ag_num, Al_num, step, size, center, R, Z):
    # 1. Проверяем столкновение с границей или другой частицей для каждой частицы
    for i in range(len(particle_list)):
        particle_list[i].compute_refl(step, size)
        for j in range(i + 1, len(particle_list)):
            particle_list[i].compute_coll(particle_list[j], step, center, R, Z)

    # 2. С помощью метода молекулярной динамики вычисляем позицию и скорость
    force_M = Morze(particle_list[Ag_num:])
    force_LJ = LennardJones(particle_list, Ag_num)
    forces = np.concatenate([force_LJ, force_M])

    for i in range (Ag_num + Al_num):
        particle_list[i].compute_step(step, forces[i])