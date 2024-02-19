import numpy as np


def writeOutput(filename, natoms, timestep, box, **data):
    axis = ('x', 'y', 'z')

    with open(filename, 'a') as fp:
        """ Заголовок .dump файла """

        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))

        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))

        """ Преобразование и работа с данными """

        keys = list(data.keys())

        for key in keys:
            isMatrix = len(data[key].shape) > 1
            if isMatrix:
                _, nCols = data[key].shape

                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:, i]
                    else:
                        data['{}_{}'.format(key, axis[i])] = data[key][:, i]

                del data[key]

        keys = data.keys()

        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')

        output = []
        for key in keys:
            output = np.hstack((output, data[key]))

        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))