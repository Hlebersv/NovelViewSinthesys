import numpy as np
import matplotlib.pyplot as plt


def plot_3d_face():

    model_3_d = np.array([[-7.308957, 0.913869, 0.000000],
                        [-6.775290, -0.730814, -0.012799],
                        [-5.665918, -3.286078, 1.022951],
                        [-5.011779, -4.876396, 1.047961],
                        [-4.056931, -5.947019, 1.636229],
                        [-1.833492, -7.056977, 4.061275],
                        [0.000000, -7.415691, 4.070434],
                        [1.833492, -7.056977, 4.061275],
                        [4.056931, -5.947019, 1.636229],
                        [5.011779, -4.876396, 1.047961],
                        [5.665918, -3.286078, 1.022951],
                        [6.775290, -0.730814, -0.012799],
                        [7.308957, 0.913869, 0.000000],
                        [5.311432, 5.485328, 3.987654],
                        [4.461908, 6.189018, 5.594410],
                        [3.550622, 6.185143, 5.712299],
                        [2.542231, 5.862829, 4.687939],
                        [1.789930, 5.393625, 4.413414],
                        [2.693583, 5.018237, 5.072837],
                        [3.530191, 4.981603, 4.937805],
                        [4.490323, 5.186498, 4.694397],
                        [-5.311432, 5.485328, 3.987654],
                        [-4.461908, 6.189018, 5.594410],
                        [-3.550622, 6.185143, 5.712299],
                        [-2.542231, 5.862829, 4.687939],
                        [-1.789930, 5.393625, 4.413414],
                        [-2.693583, 5.018237, 5.072837],
                        [-3.530191, 4.981603, 4.937805],
                        [-4.490323, 5.186498, 4.694397],
                        [1.330353, 7.122144, 6.903745],
                        [2.533424, 7.878085, 7.451034],
                        [4.861131, 7.878672, 6.601275],
                        [6.137002, 7.271266, 5.200823],
                        [6.825897, 6.760612, 4.402142],
                        [-1.330353, 7.122144, 6.903745],
                        [-2.533424, 7.878085, 7.451034],
                        [-4.861131, 7.878672, 6.601275],
                        [-6.137002, 7.271266, 5.200823],
                        [-6.825897, 6.760612, 4.402142],
                        [-2.774015, -2.080775, 5.048531],
                        [-0.509714, -1.571179, 6.566167],
                        [0.000000, -1.646444, 6.704956],
                        [0.509714, -1.571179, 6.566167],
                        [2.774015, -2.080775, 5.048531],
                        [0.589441, -2.958597, 6.109526],
                        [0.000000, -3.116408, 6.097667],
                        [-0.589441, -2.958597, 6.109526],
                        [-0.981972, 4.554081, 6.301271],
                        [-0.973987, 1.916389, 7.654050],
                        [-2.005628, 1.409845, 6.165652],
                        [-1.930245, 0.424351, 5.914376],
                        [-0.746313, 0.348381, 6.263227],
                        [0.000000, 1.400000, 8.063430],
                        [0.746313, 0.348381, 6.263227],
                        [1.930245, 0.424351, 5.914376],
                        [2.005628, 1.409845, 6.165652],
                        [0.973987, 1.916389, 7.654050],
                        [0.981972, 4.554081, 6.301271]])

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x, y, z = model_3_d[:, 0], model_3_d[:, 1], model_3_d[:, 2]

    ax.scatter3D(x, y, z, color='b')
    ax.plot_trisurf(x, y, z, color='black', linewidth=0.2, alpha=0.3, edgecolors='b')
    plt.savefig('./model_3d.png')
    plt.show()


def ref_3d_model():
    model_points = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(model_points, dtype=np.float64)


def ref2d_image_points(shape):
    image_points = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(image_points, dtype=np.float64)


def camera_matrix(fl, center):
    mat = [[fl, 1, center[0]],
           [0, fl, center[1]],
           [0, 0, 1]]
    return np.array(mat, dtype=np.float)