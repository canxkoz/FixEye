import imports
import settings

def check_classes(Y):
    if (settings.checked == False):
        print("-- CLASSES --")
        print(imports.Counter(Y).keys())
        print(imports.Counter(Y).values())
        settings.checked = True
        print("-------------")

def rgb_equalization(image):
    channels = imports.cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(imports.cv2.equalizeHist(ch))

    eq_image = imports.cv2.merge(eq_channels)
    eq_image = imports.cv2.cvtColor(eq_image, imports.cv2.COLOR_BGR2RGB)
    return eq_image

def hsv_equalization(image):
    H, S, V = imports.cv2.split(imports.cv2.cvtColor(image, imports.cv2.COLOR_BGR2HSV))
    eq_V = imports.cv2.equalizeHist(V)
    eq_image = imports.cv2.cvtColor(imports.cv2.merge([H, S, eq_V]), imports.cv2.COLOR_HSV2RGB)
    return eq_image
  
def yuv_equalization(image):
    img_yuv = imports.cv2.cvtColor(image, imports.cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = imports.cv2.equalizeHist(img_yuv[:,:,0])
    eq_image = imports.cv2.cvtColor(img_yuv, imports.cv2.COLOR_YUV2RGB)
    return eq_image

def load_data_eq():
    meta_data = imports.pd.read_csv('data/messidor/train/messidor_annotation.csv')
    Y = meta_data['Retinopathy grade'].values
    # Transform into binary classificaiton
    if settings.nb_classes == 2:
        Y[Y > 0] = 1

    n_samples = Y.shape[0]
    X = imports.np.empty((n_samples, settings.img_rows, settings.img_cols, 3))
  
    for i in range(n_samples):
        filename = './data/messidor/train/{}'.format(meta_data['Image name'][i])
        img_cv = imports.cv2.resize(imports.cv2.imread(filename), (settings.img_rows, settings.img_cols))
        x = imports.img_to_array(img_cv) / 255.0
        X[i] = x.astype('float32')

    input_shape_l = (settings.img_rows, settings.img_cols, 3)
    return X, Y, input_shape_l

def equalize_images(images_in, e_type):
    images_out = imports.deepcopy(images_in)
    for i in range(images_out.shape[0]):
        arr_image = images_out[i]*255
        image_to_eq = imports.array_to_img(arr_image)
        if (e_type == 0):
            images_out[i] = (imports.img_to_array(hsv_equalization(imports.np.asarray(image_to_eq))) / 255).astype('float32')
        if (e_type == 1):
            images_out[i] = (imports.img_to_array(rgb_equalization(imports.np.asarray(image_to_eq))) / 255).astype('float32')
        if (e_type == 2):
            images_out[i] = (imports.img_to_array(yuv_equalization(imports.np.asarray(image_to_eq))) / 255).astype('float32')

    return images_out
    
def adaptive_equalize_images(images_in):
    images_out = imports.deepcopy(images_in)

    for i in range(images_out.shape[0]):
        arr_image = images_out[i]
        imports.plt.rcParams['font.size'] = 8
        img_adapteq = imports.exposure.equalize_adapthist(arr_image, clip_limit=0.03)
        images_out[i] = img_adapteq

    return images_out
    
def load_data():
    meta_data = imports.pd.read_csv('data/messidor/train/messidor_annotation.csv')
    Y = meta_data['Retinopathy grade'].values
    # Transform into binary classificaiton
    if settings.nb_classes == 2:
        Y[Y > 0] = 1
    check_classes(Y)

    n_samples = Y.shape[0]
    X = imports.np.empty((n_samples, settings.img_rows, settings.img_cols, 3))

    for i in range(n_samples):
        filename = 'data/messidor/{}'.format(meta_data['Image name'][i])
        img = imports.load_img(filename, target_size=[settings.img_rows, settings.img_cols])
        x = imports.img_to_array(img) / 255.0
        X[i] = x.astype('float32')

    input_shape_l = (settings.img_rows, settings.img_cols, 3)
    return X, Y, input_shape_l