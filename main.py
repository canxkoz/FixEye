import imports
import functions
import neural_networks
import settings

names = []
metrics_acc = []
metrics_loss = []
count = 1

def print_metrics(opt, classes):
    global names
    global metrics_acc
    global metrics_loss
    global count

    print("--------------------------------------------------------------------------")
    print("---- " + opt + " ----")
    print("---- " + classes + " ----")
    print("--------------------------------------------------------------------------")
    print(" Accuracy | Loss     | Name")
    print("--------------------------------------------------------------------------")
    for i in range(len(names)):
        print(" " + str(format(metrics_acc[i], '.6f')) + " | " + str(format(metrics_loss[i], '.6f')) + " | " + names[i])
    print("--------------------------------------------------------------------------")
    
    names = []
    metrics_acc = []
    metrics_loss = []
    count = 1

def train_module(load_type, adapt, eq_type, opt):
    name = ""

    #load_type: 0 default, 1 equalized
    if (load_type == 0):
        x, y, input_shape = functions.load_data()
        name = "DEFAULT + "
    if (load_type == 1):
        x, y, input_shape = functions.load_data_eq()
        name = "EQUALIZED + "
        
    if (adapt == 1):
        x = functions.adaptive_equalize_images(x)
        name = name + "ADAPTIVE + "

    class_weights = imports.class_weight.compute_class_weight('balanced', imports.np.unique(y), y)
    y = imports.keras.utils.to_categorical(y)
    x_train, x_test, y_train, y_test = imports.train_test_split(x, y, test_size=0.3, random_state=42)

    #eq_type: 0 HSV, 1 RGB, 2 YUV, -1 default
    if (eq_type == -1):
        name = name + "NO-EQUALIZATION"
    if (eq_type == 0):
        x_train = functions.equalize_images(x_train, 0)
        x_test = functions.equalize_images(x_test, 0)
        name = name + "HSV"
    if (eq_type == 1):
        x_train = functions.equalize_images(x_train, 1)
        x_test = functions.equalize_images(x_test, 1)
        name = name + "RGB"
    if (eq_type == 2):
        x_train = functions.equalize_images(x_train, 2)
        x_test = functions.equalize_images(x_test, 2)
        name = name + "YUV"

    global names
    global metrics_acc
    global metrics_loss
    global count
    
    print("-- (" + str(count) + "/16) " + name + " --")

    #Train CNN
    model = neural_networks.cnn_model(input_shape)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=80, verbose=1, shuffle=True, class_weight=class_weights)

    #Test CNN
    metric_loss, metric_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test set loss: ', metric_loss)
    print('Test set accuracy: ', metric_acc)
    print("------------------")

    names.append(name)
    metrics_acc.append(metric_acc)
    metrics_loss.append(metric_loss)
    count = count + 1

def experimentation_module():
    # 2 classes
    sgd = imports.SGD(lr=0.01, nesterov=True)
    train_module(0, 0, -1, sgd)
    train_module(0, 0, 0, sgd)
    train_module(0, 0, 1, sgd)
    train_module(0, 0, 2, sgd)
    train_module(0, 1, -1, sgd)
    train_module(0, 1, 0, sgd)
    train_module(0, 1, 1, sgd)
    train_module(0, 1, 2, sgd)
    train_module(1, 0, -1, sgd)
    train_module(1, 0, 0, sgd)
    train_module(1, 0, 1, sgd)
    train_module(1, 0, 2, sgd)
    train_module(1, 1, -1, sgd)
    train_module(1, 1, 0, sgd)
    train_module(1, 1, 1, sgd)
    train_module(1, 1, 2, sgd)
    print_metrics("SGD", "2")

    rms = imports.keras.optimizers.RMSprop(lr=0.01)
    train_module(0, 0, -1, rms)
    train_module(0, 0, 0, rms)
    train_module(0, 0, 1, rms)
    train_module(0, 0, 2, rms)
    train_module(0, 1, -1, rms)
    train_module(0, 1, 0, rms)
    train_module(0, 1, 1, rms)
    train_module(0, 1, 2, rms)
    train_module(1, 0, -1, rms)
    train_module(1, 0, 0, rms)
    train_module(1, 0, 1, rms)
    train_module(1, 0, 2, rms)
    train_module(1, 1, -1, rms)
    train_module(1, 1, 0, rms)
    train_module(1, 1, 1, rms)
    train_module(1, 1, 2, rms)
    print_metrics("RMSprop", "2")

    adadelta = imports.keras.optimizers.Adadelta(lr=0.01)
    train_module(0, 0, -1, adadelta)
    train_module(0, 0, 0, adadelta)
    train_module(0, 0, 1, adadelta)
    train_module(0, 0, 2, adadelta)
    train_module(0, 1, -1, adadelta)
    train_module(0, 1, 0, adadelta)
    train_module(0, 1, 1, adadelta)
    train_module(0, 1, 2, adadelta)
    train_module(1, 0, -1, adadelta)
    train_module(1, 0, 0, adadelta)
    train_module(1, 0, 1, adadelta)
    train_module(1, 0, 2, adadelta)
    train_module(1, 1, -1, adadelta)
    train_module(1, 1, 0, adadelta)
    train_module(1, 1, 1, adadelta)
    train_module(1, 1, 2, adadelta)
    print_metrics("Adadelta", "2")

    adam = imports.keras.optimizers.Adam(lr=0.01)
    train_module(0, 0, -1, adam)
    train_module(0, 0, 0, adam)
    train_module(0, 0, 1, adam)
    train_module(0, 0, 2, adam)
    train_module(0, 1, -1, adam)
    train_module(0, 1, 0, adam)
    train_module(0, 1, 1, adam)
    train_module(0, 1, 2, adam)
    train_module(1, 0, -1, adam)
    train_module(1, 0, 0, adam)
    train_module(1, 0, 1, adam)
    train_module(1, 0, 2, adam)
    train_module(1, 1, -1, adam)
    train_module(1, 1, 0, adam)
    train_module(1, 1, 1, adam)
    train_module(1, 1, 2, adam)
    print_metrics("Adam", "2")

    
    # 4 classes
    settings.nb_classes = 4
    settings.checked = False
    sgd = imports.SGD(lr=0.01, nesterov=True)
    train_module(0, 0, -1, sgd)
    train_module(0, 0, 0, sgd)
    train_module(0, 0, 1, sgd)
    train_module(0, 0, 2, sgd)
    train_module(0, 1, -1, sgd)
    train_module(0, 1, 0, sgd)
    train_module(0, 1, 1, sgd)
    train_module(0, 1, 2, sgd)
    train_module(1, 0, -1, sgd)
    train_module(1, 0, 0, sgd)
    train_module(1, 0, 1, sgd)
    train_module(1, 0, 2, sgd)
    train_module(1, 1, -1, sgd)
    train_module(1, 1, 0, sgd)
    train_module(1, 1, 1, sgd)
    train_module(1, 1, 2, sgd)
    print_metrics("SGD", "4")

    rms = imports.keras.optimizers.RMSprop(lr=0.01)
    train_module(0, 0, -1, rms)
    train_module(0, 0, 0, rms)
    train_module(0, 0, 1, rms)
    train_module(0, 0, 2, rms)
    train_module(0, 1, -1, rms)
    train_module(0, 1, 0, rms)
    train_module(0, 1, 1, rms)
    train_module(0, 1, 2, rms)
    train_module(1, 0, -1, rms)
    train_module(1, 0, 0, rms)
    train_module(1, 0, 1, rms)
    train_module(1, 0, 2, rms)
    train_module(1, 1, -1, rms)
    train_module(1, 1, 0, rms)
    train_module(1, 1, 1, rms)
    train_module(1, 1, 2, rms)
    print_metrics("RMSprop", "4")

    adadelta = imports.keras.optimizers.Adadelta(lr=0.01)
    train_module(0, 0, -1, adadelta)
    train_module(0, 0, 0, adadelta)
    train_module(0, 0, 1, adadelta)
    train_module(0, 0, 2, adadelta)
    train_module(0, 1, -1, adadelta)
    train_module(0, 1, 0, adadelta)
    train_module(0, 1, 1, adadelta)
    train_module(0, 1, 2, adadelta)
    train_module(1, 0, -1, adadelta)
    train_module(1, 0, 0, adadelta)
    train_module(1, 0, 1, adadelta)
    train_module(1, 0, 2, adadelta)
    train_module(1, 1, -1, adadelta)
    train_module(1, 1, 0, adadelta)
    train_module(1, 1, 1, adadelta)
    train_module(1, 1, 2, adadelta)
    print_metrics("Adadelta", "4")

    adam = imports.keras.optimizers.Adam(lr=0.01)
    train_module(0, 0, -1, adam)
    train_module(0, 0, 0, adam)
    train_module(0, 0, 1, adam)
    train_module(0, 0, 2, adam)
    train_module(0, 1, -1, adam)
    train_module(0, 1, 0, adam)
    train_module(0, 1, 1, adam)
    train_module(0, 1, 2, adam)
    train_module(1, 0, -1, adam)
    train_module(1, 0, 0, adam)
    train_module(1, 0, 1, adam)
    train_module(1, 0, 2, adam)
    train_module(1, 1, -1, adam)
    train_module(1, 1, 0, adam)
    train_module(1, 1, 1, adam)
    train_module(1, 1, 2, adam)
    print_metrics("Adam", "4")

############################################################

startTime = imports.time.time()
experimentation_module()
print("-- Time duration:")
print("%s"%((imports.time.time()-startTime)))
print("-----------------")
