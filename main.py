'''Code for the kaggle.com digit classification competition.
'''

__author__      = "Emanuel Strauss"
__email__       = "kinnskogr@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm, metrics, grid_search
from sklearn.ensemble import RandomForestClassifier
import scipy.ndimage as ndimage
import time
import math

Debug = False

def scat(img):
    '''Calculate the scatting matrix and mean position of the pixel distribution'''

    x_coord = np.array(range(img.shape[0]), dtype=float)
    y_coord = np.array(range(img.shape[1]), dtype=float).reshape(-1, 1)

    x_mean = np.sum(img * x_coord) / np.sum(img)
    y_mean = np.sum(img * y_coord) / np.sum(img)
    
    x_var  = np.sum(np.power(img, 1) * np.square(x_coord - x_mean))
    y_var  = np.sum(np.power(img, 1) * np.square(y_coord - y_mean))
    xy_var = np.sum(np.power(img, 1) * (x_coord - x_mean) * (y_coord - y_mean))

    scattering_matrix = np.array([[x_var, xy_var], [xy_var, y_var]])
    mean_vector = np.array([x_mean, y_mean])
    
    return scattering_matrix, mean_vector

def paxis(scattering_matrix):
    '''Calculate the principal axis values from the scattering matrix'''

    if scattering_matrix.shape != (2, 2):
        print "ERROR: getPrincipleAxes(scattering_matrix), scattering_matrix size is not 2x2"
        return

    #(a + d - sqrt((a+d)**2 - 4(a*d-c*b)))/2                                                                                                                                    
    #a*x*x + b*x + c = 0 --> -b/2 +- sqrt(b*b-4*a*c)                                                                                                                            
    #-(a+d)x + x*x - det(ABCD) = 0                                                                                                                                              
    sol1 = 0.5*(scattering_matrix[0, 0]+scattering_matrix[1, 1]) + 0.5*math.sqrt( math.pow((scattering_matrix[0, 0]+scattering_matrix[1, 1]), 2) - 4*(scattering_matrix[0, 0]*scattering_matrix[1, 1] - scattering_matrix[0, 1]*scattering_matrix[1, 0]) )
    sol2 = 0.5*(scattering_matrix[0, 0]+scattering_matrix[1, 1]) - 0.5*math.sqrt( math.pow((scattering_matrix[0, 0]+scattering_matrix[1, 1]), 2) - 4*(scattering_matrix[0, 0]*scattering_matrix[1, 1] - scattering_matrix[0, 1]*scattering_matrix[1, 0]) )


    den1 = math.sqrt( 1 + math.pow( ((sol1 - scattering_matrix[0, 0])/scattering_matrix[0, 1]), 2) )
    vec1 = np.array([1.0/den1, ((sol1 - scattering_matrix[0, 0])/scattering_matrix[0, 1]) / den1])

    den2 = math.sqrt( 1 + math.pow( ((sol2 - scattering_matrix[0, 0])/scattering_matrix[0, 1]), 2) )
    vec2 = np.array([1.0/den2, ((sol2 - scattering_matrix[0, 0])/scattering_matrix[0, 1]) / den2])

    V = np.zeros( (2, 2) )
    S = np.array([0, 0])

    if sol1 > sol2:
        S = np.array([sol1, sol2])
        V[0, :] = vec1
        V[1, :] = vec2
    else:
        S = np.array([sol2, sol1])
        V[0, :] = vec2
        V[1, :] = vec1

    return V, S

def rotate_image(img):
    '''Rotate an interpolated version of the image to align the
    principal axis with the veritcal axis'''

    if len(img.shape) == 1:
        dim = np.sqrt(img.shape[0])
    else:
        dim = img.shape[0]

    scattering_matrix, mean_vector = scat(img.reshape(dim, dim))

    V, S = paxis(scattering_matrix)

    stheta = V[0, 0]

    rot_img =  ndimage.interpolation.rotate(img.reshape(dim, dim), math.asin(stheta) * 360 / (2*math.pi), reshape = False)

    return rot_img.reshape(-1)
    

def construct_averages(images, labels):
    '''Construct the average image for each digit. Optionally,
    normalize by total cell intensity'''

    output = []
    if images.shape[0] == 0: 
        return output
    
    for num in xrange(10):
        collection = images[labels == num]
        if collection.shape[0] == 0:
            output.append(np.array([]))

        try:
            average = np.sum(collection, axis = 0) / len(collection)
        except: ## Bad form, but there are lots of ways this can fail and I don't care about any of them
            average = np.array([])
        
        output.append(average)
    return output

def edge_laplacian(images):
    '''Do edge detection'''

    from scipy import signal

    output = []
    dim = np.sqrt(images.shape[1])
    for image in images.reshape(len(images), dim, dim):
        ck_spline = signal.cspline2d(image, 8.0, 1.0)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
        deriv2 = signal.convolve2d(ck_spline, laplacian, mode = 'same', boundary = 'symm')
        output.append(deriv2)
        
    return np.array(output, dtype = float).reshape(len(images), -1)


def plot_figure(image, ax = None):
    '''Draw a 2D representation of an array'''

    elements = image

    if len(elements.shape) == 1:
        dim = int(np.sqrt(elements.shape[0]))
        # Image must be a square
        if dim != np.sqrt(elements.shape[0]):
            dim += 1
            delta = pow(dim, 2) - elements.shape[0]
            elements = np.concatenate([elements, [0]*delta])
        elements = elements.reshape(dim, dim)
    else:
        dim = elements.shape[0]

    if ax == None:
        ax = plt.figure()

    try:
        output = plt.imshow(elements, interpolation='nearest', )
        output.set_cmap('hot')
        plt.colorbar()
    except ValueError:
        output = None
    return output
        

def plot_array(images, plot_name = ""):
    '''Take a list of images and plot them on a grid'''

    if Debug:
        print "Plot array of %d categories" % len(images)

    dim_y = int(np.sqrt(len(images)))
    dim_x = dim_y + (dim_y != np.sqrt(len(images)))
    ax = plt.figure(plot_name)

    if plot_name != "":
        ax.set_label(plot_name)

    for img_idx in xrange(len(images)):
        ax.add_subplot(dim_y, dim_x, img_idx+1)
        plot_figure(images[img_idx], ax = ax)

    plt.savefig(plot_name + ".png")
        

def run_pca(images, n_components = 0):
    '''Seed the PCA class'''
    if n_components == 0:
        n_components = images.shape[1]

    pca = PCA(n_components = n_components, whiten = False)
    pca.fit(images)
    return pca

def explore_pca(images):
    '''Work out the variance explained by the PCA components, return
    the PCA object and the results for four operating points (90%,
    95%, 99%, and 99.9%)'''

    pca = run_pca(images, images.shape[1])
    frac_var_explained = [sum(pca.explained_variance_ratio_[:comp_idx]) for comp_idx in range(len(pca.explained_variance_ratio_))]
    plt.figure()
    plt.plot(range(len(frac_var_explained)), frac_var_explained)

    import bisect
    output = {}
    for var in (0.9, 0.95, 0.99, 0.999):
        n_components = bisect.bisect(frac_var_explained, var)
        print "%f variance explained by %d components" % (var, n_components)
        output[var] = n_components

    return pca, output
    

def check_mistakes(data):
    '''Look at the average values for mis-labeled predictions'''

    for i in range(10):
        f_bad = float(sum(data.predicted[data.labels == i] == i)) / len(data.predicted[data.labels == i])
        print "%f mis-classified for %d" % (f_bad, i)

    f_bad = sorted([(i, float(sum(data.predicted[data.labels == i] == i)) / len(data.predicted[data.labels == i])) for i in range(10)], key = lambda x:x[1])
    print "In order of performance:"
    for i, frac in f_bad:
        print "%d %f" % (i, frac)

    badimages = data.orig[data.labels != data.predicted]
    badlabels  = data.labels[data.labels != data.predicted]

    if badimages.shape[0] == 0:
        return

    bad_averages = construct_averages(badimages, badlabels)
    plot_array(bad_averages, plot_name = data.name + "_bad_averages")

    bad_prob = np.max(data.prob[data.labels != data.predicted], axis = 1)
    good_prob = np.max(data.prob[data.labels == data.predicted], axis = 1)
    
    h_good, bin_edges = np.histogram(good_prob, bins = 20)
    h_bad, bin_edges = np.histogram(bad_prob, bins = bin_edges)

    bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(data.name + "_sums")
    ax.set_yscale('log')
    plt.plot(bins, h_good, 'bo')
    plt.plot(bins, h_bad, 'ro')
    plt.savefig(data.name + '_sums.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(data.name + "_fractions")
    ax.set_yscale('log')
    answers = data.predicted == data.labels
    probs = np.max(data.prob, axis = 1)
    xaxis = np.array(range(0, 101, 1), dtype = float) / 100
    sums = np.array([len(answers[(probs <= i)]) for i in xaxis], dtype = float) / len(answers)
    good = np.array([len(answers[(probs <= i) & (answers == True)]) for i in xaxis], dtype = float) / len(answers)
    bad = np.array([len(answers[(probs <= i) & (answers == False)]) for i in xaxis], dtype = float) / len(answers)
    plt.plot(xaxis, sums, 'b')
    plt.plot(xaxis, good, 'g')
    plt.plot(xaxis, bad, 'r')
    plt.savefig(data.name + '_fractions.png')

def make_xy_scatter_matrix(image):
    '''Calculate the scatter matrix (i.e. the X,Y covariance matrix)'''

    cell_values = np.array([val for val in image])
    cell_x      = np.array([range(np.sqrt(len(image)))])
    cell_y      = np.array([range(np.sqrt(len(image)))])

    itot = np.sum(cell_values)
    if itot == 0:
        raise ValueError('This image is blank?!')

    x_mean  = np.sum(cell_values * cell_x)/itot
    y_mean  = np.sum(cell_values * cell_y)/itot
    x_var  = np.sum(cell_values * np.square(cell_x - x_mean))
    y_var  = np.sum(cell_values * np.square(cell_y - y_mean))
    xy_var   = np.sum(cell_values * (cell_x - x_mean)*(cell_y -y_mean))

    scattering_matrix = np.array([[x_var, xy_var], [xy_var, y_var]])
    mean_vector = np.array([x_mean, y_mean])

    return scattering_matrix, mean_vector

class Classifers:
    '''Handle the details of multiple classifier types, expose a single interface. TOD: Basically, everything'''
    def __init__(self):
        self.clf = {}
        self.config = {}

    def train_svm(self, data, **opts):
        t = time.time()
        clf = svm.SVC(probability = True, **opts)
        clf.fit(data.images, data.labels)
        
        self.clf["svm"] = clf
        self.config["svm"] = data.config

        if Debug:
            print "Trained svm in %d seconds" % (time.time() - t)
    
    def train_rf(self, data, **opts):
        t = time.time()
        clf = RandomForestClassifier(verbose = 1, **opts)
        clf.fit(data.images, data.labels)
        
        self.clf["rf"] = clf
        self.config["rf"] = data.config

        if Debug:
            print "Trained rf in %d seconds" % (time.time() - t)    
            
    

class ImgData:
    '''Ship all the image components in a pretty little box. Keep
    track of metadata. Handle the common image processing operations.
    '''

    def __init__(self, name, config = ""):
        self.orig      = np.array([])
        self.images    = np.array([])
        self.labels    = np.array([])
        self.predicted = np.array([])
        self.name      = name
        self.config    = config

    def __len__(self):
        return len(self.orig)

    def preprocess(self, **opts):
        '''Transform the original list of images and store as a new
        list of images. Available options are:
        
        counts -- If True will add a sum of cells above 0, once across the columns and once across the rows.
        rotate -- If True will interpolate and rotate the images to align the principal axis along the vertical.
        edges  -- If 'edges' will apply edge detection. If 'merged' will append the transformed image, rather than replacing it.
        '''

        self.images = self.orig
        if self.images.shape[0] == 0:
            return

        if opts.get('counts', False) == True:
            t = time.time()
            dim = np.sqrt(self.orig.shape[1]) if len(self.orig.shape) == 2 else self.orig.shape[2]
            counts = np.column_stack([np.sum(self.orig.reshape(len(self.images), dim, dim) != 0, axis = 1), np.sum(self.orig.reshape(len(self.images), dim, dim) != 0, axis = 2)])
            if Debug:
                print "Adding counts took %d seconds" % (time.time() - t)
            

        if opts.get('rotate', False) == True:
            t = time.time()
            self.images = np.array([rotate_image(img) for img in self.orig])
            if Debug:
                print "Rotating images took %d seconds" % (time.time() - t)
        else:
            self.images = self.images.reshape(len(self.images), -1)
        dim = np.sqrt(self.images.shape[1])

        if opts.get('edges', 'orig') != 'orig':
            t = time.time()
            edges = edge_laplacian(self.images)
            if opts['edges'] == 'edges':
                self.images = edges
            elif opts['edges'] == 'merged':
                self.images = np.column_stack([self.images, edges])
            if Debug:
                print "Calculating edges took %d seconds" % (time.time() - t)

        if opts.get('counts', False) == True:
            if Debug:
                print "Adding a row of counts"
            self.images = np.column_stack([self.images, counts.reshape(len(counts), -1)])

    def scale(self, scaler = None):
        '''Apply a pre-computed scalling transformation if scaler is
        passed. Otherwise, center each image to the mean with unit
        variance'''
        
        shape = self.images.shape
        if shape[0] == 0:
            return

        if scaler:
            self.images = scaler.transform(self.images.reshape(shape[0], -1))
        else:
            from sklearn import preprocessing
            self.images = preprocessing.scale(self.images.reshape(shape[0], -1), axis = 1)
        self.images = self.images.reshape(shape)

    def apply_pca(self, pca):
        '''Transform the images along the principal components'''
        if self.images.shape[0] == 0:
            return
        
        self.images = pca.transform(self.images)

    def read_csv(self, fname, max_entries = -1):
        '''read a csv file containing pixel intensities and (optionally) labels'''
        if Debug:
            print "Reading csv from %s" % fname

        images = []
        labels = []
        for line in open(fname).readlines():
            line = line.split(",")
            if line[0] == 'label':
                idx_start = 1
                continue
            if line[0] == 'pixel0':
                idx_start = 0
                continue
            line = [float(val) for val in line]
            
            image = line[idx_start:]
            label = -1 if idx_start == 0 else line[0]

            images.append(image)
            labels.append(label)

            if max_entries > 0 and len(images) > max_entries:
                break

            
        self.orig = np.array(images, dtype = float)
        self.labels = np.array(labels, dtype = int)
        self.config += "csv = %s;" % fname
    
    def read_mnist(self, fname):
        '''Read the MNIST ubyte data-format and return a numpy array'''
    
        import struct
        import operator
    
        dtypes = {
            0x08: ('ubyte', 'B', 1), 
            0x09: ('byte', 'b', 1), 
            0x0B: ('int16', 'h', 2), 
            0x0C: ('int32', 'i', 4), 
            0x0D: ('float', 'f', 4), 
            0x0E: ('double', 'd', 8)
        }
        
        f = open(fname)
        magic_number = struct.unpack('>BBBB', f.read(4))
        
        dtype, dtype_s, el_size = dtypes[magic_number[2]]
        dims = int(magic_number[3])
        dims_sizes = struct.unpack('>'+'I'*dims, f.read(4*dims))
        full_length = reduce(operator.mul, dims_sizes, 1)
        result_array = np.ndarray(shape = [full_length], dtype = dtype)
        unpack_str = '>' + dtype_s*full_length
        result_array[0:full_length] = struct.unpack(unpack_str, f.read(full_length * el_size))
        result = np.array(np.reshape(result_array, dims_sizes), dtype = 'float32')

        if len(result.shape) == 3:
            #Images are stored as a triplett
            self.orig = np.array(result, dtype = float)
        elif len(result.shape) == 2:
            #Labels are stored as tuples
            self.labels = np.array(result, dtype = int)
        self.config += ", mnist = %s" % fname
    

    def plot_image(self, idx):
        '''Visualize the image at position idx'''
        plot_figure(self.images[idx])

    def plot_avgimages(self, status = 'processed'):
        '''Average the images by label and visualize on a grid'''
        dim_x = np.sqrt(self.orig.shape[1])
        dim_y = np.sqrt(self.orig.shape[1])
        if status == 'processed':
            averages = construct_averages(self.images, self.labels)
            dim_x = self.images.shape[1] / dim_y
        elif status == 'orig':
            averages = construct_averages(self.orig, self.labels)

        l = len(averages)

        averages = np.array(averages, dtype = float)

        plot_array(averages.reshape(l, dim_x, dim_y), plot_name = self.name + "_averages_" + status)

    def accuracy(self):
        '''Calculate the prediction accuracy (n predicted right / total)'''
        return sum(self.labels == self.predicted) / len(self.labels)

    def bad_indices(self):
        '''Return the a vector for selecting mis-classified images'''
        return self.labels != self.predicted

    def badimages(self):
        '''Return a list of mis-classified images and their correct labels'''
        ids = self.bad_indices()
        return self.images[ids], self.labels[ids]

    def subset(self, **opts):
        '''Return an ImgData object containing a subset of the data,
        valid arguments are:
        start - the 1st id in a range
        end   - the last id in a range (exclusive)
        ids   - a list of indices
        vals  - a list of true/false values
        '''

        output = ImgData(self.name, self.config + str(opts) + ";")
        if 'start' in opts and 'end' not in opts:
            opts['end'] = len(self.labels)
        if 'end' in opts and 'start' not in opts:
            opts['start'] = 0

        if 'start' in opts and 'end' in opts:
            output.orig = self.orig[opts['start']:opts['end']]
            output.labels = self.labels[opts['start']:opts['end']]
            if len(self.images) > 0:
                output.images = self.images[opts['start']:opts['end']]
            if len(self.predicted) > 0:
                output.predicted = self.predicted[opts['start']:opts['end']]
        
        if 'ids' in opts:
            ids = opts['ids']
            ids = np.array([i in ids for i in range(len(self.orig))])
            output.orig = self.orig[ids]
            output.labels = self.labels[ids]
            if len(self.images) > 0:
                output.images = self.images[ids]
            if len(self.predicted) > 0: 
                output.predicted = self.predicted[ids]


        if 'vals' in opts:
            vals = opts['vals']
            output.orig = self.orig[vals]
            output.labels = self.labels[vals]
            if len(self.images) > 0:
                output.images = self.images[vals]
            if len(self.predicted) > 0: 
                output.predicted = self.predicted[vals]

        return output

class Scaler:
    '''Helper class to hide the details of the scaling'''

    def __init__(self):
        self.transform = None

    def train_minmax(self, data, min_val = -1, max_val = 1):
        '''Figure out the range of the samples'''
        cur_min = np.min(data.images)
        cur_max = np.max(data.images)
        self.transform = lambda x: (x - cur_min) * (max_val - min_val) / (cur_max - cur_min)

    def train_scaler(self, data):
        '''Work out the mean and variance of the samples'''
        from sklearn import preprocessing
        scaler = preprocessing.Scaler().fit(data.images)
        self.transform = scaler.transform        


if __name__ == "__main__":
    import sys
    from optparse import OptionParser

    print "Called with:"
    print " ".join(sys.argv)

    parser = OptionParser()
    parser.add_option("--train_csv"   , dest = "train_csv"  , default = None    , help = "read training data from a csv file")
    parser.add_option("--final_csv"   , dest = "final_csv"  , default = None    , help = "read final data to classify from a csv file")
    parser.add_option("--train_mnist" , dest = "train_mnist", default = None    , help = "read the training data from a pair of ubyte files (comma seperated)")
    parser.add_option("--n_train"     , dest = "n_train"    , default = 15000   , type = 'int'  , help = "number of training events to use for training")
    parser.add_option("--n_test"      , dest = "n_test"     , default = 15000   , type = 'int'  , help = "number of training events to use for testing (bounded by n_train)")
    parser.add_option("--scale"       , dest = "scale"      , default = "scale" , help = "What scaling to apply: scale centers the mean and variance per image, sample uses scikit learn scaler, minmax enforces a range from 0 to 1")    
    parser.add_option("--pca"         , dest = "pca"        , default = -1      , type = 'float', help = "Use PCA to reduce the dimensionality, value sets the minimum fraction of variance explained by the components used")
    parser.add_option("--edges"       , dest = "edges"      , default = "orig"  , help = "Use original image (orig), edge detection (edges), or a combination of both (merged)")
    parser.add_option("--counts"      , dest = "counts"     , default = False   , action = 'store_true', help = "Add a set of features to the image, listing the number of non-zero pixels")
    parser.add_option("--rotate"      , dest = "rotate"     , default = False   , action = 'store_true', help = "Interpolate and then rotate the images along their principal axis")    
    parser.add_option("--gridsearch"  , dest = "gridsearch" , default = False   , action = 'store_true', help = "Use grid-search to optimize the svm parameters")
    parser.add_option("--gamma"       , dest = "gamma"      , default = 0.002   , type = 'float',  help = "Gamma value passed to the svm")
    parser.add_option("--C"           , dest = "C"          , default = 5       , type = 'float',  help = "C value passed to the svm")
#    parser.add_option("--cout"        , dest = "cout"       , default = None    , help = "Specify a pickle file to store the classifier in")
#    parser.add_option("--cin"         , dest = "cin"        , default = None    , help = "Specify a pickle file to read the classifier from (ignore training data)")
    parser.add_option("--two_pass"    , dest = "two_pass"   , default = False   , action = 'store_true', help = "Turn on two-pass classification")
    parser.add_option("--plots"       , dest = "plots"      , default = False   , action = 'store_true', help = "Turn on additional plotting")
    parser.add_option("--debug"       , dest = "debug"      , default = False   , action = 'store_true', help = "Turn on debugging printouts")

    (opts, args) = parser.parse_args()
    
    print "Option settings:"
    print opts

    Debug = opts.debug

    all_training_data = ImgData("all_training_data")
    t = time.time()
    if opts.train_csv:
        all_training_data.read_csv(opts.train_csv)
    elif opts.train_mnist:
        for f in opts.train_mnist.split(","):
            all_training_data.read_mnist(f)
    else:
        print "No training data specified"
        exit(1)

    if Debug:
        print "Reading training data took %d seconds" % (time.time() - t)
        print "%d samples loaded" % len(all_training_data)

    t = time.time()
    final_data = ImgData("Final Data")
    if opts.final_csv:
        final_data.read_csv(opts.final_csv)
    if Debug:
        print "Reading final data took %d seconds" % (time.time() - t)

    print "before: ", opts.n_train, opts.n_test
    print len(all_training_data)

    opts.n_train = min(opts.n_train, len(all_training_data))
    opts.n_test = min(opts.n_test, len(all_training_data) - opts.n_train)

    print "after: ", opts.n_train, opts.n_test

    t = time.time()
    
    idxs = range(len(all_training_data))
    np.random.shuffle(idxs)

    training_data = all_training_data.subset(ids = idxs[:opts.n_train])
    training_data.name = "Training Data"
    testing_data  = all_training_data.subset(ids = idxs[opts.n_train:opts.n_train+opts.n_test])
    testing_data.name = "Testing Data"
    if Debug:
        print "Subsetting training and testing data took %d seconds" % (time.time() - t)
        print "There are %d training samples" % len(training_data)
        print "There are %d testing samples" % len(testing_data)

    preprocess_opts = {'edges'  : opts.edges,
                       'counts' : opts.counts,
                       'rotate' : opts.rotate,
                       }

    for data in (training_data, testing_data, final_data):
        data.preprocess(**preprocess_opts)

    scaler = Scaler()
    if opts.scale == "sample":
        scaler.train_scaler(training_data)
    elif opts.scale == "minmax":
        scaler.train_minmax(training_data, 0, 1)
    else:
        scaler = None

    for data in (training_data, testing_data, final_data):
        data.scale(scaler)

    if opts.plots:
        training_data.plot_avgimages('orig')
        training_data.plot_avgimages('processed')

    if (opts.pca > 0):
        pca, pca_op = explore_pca(training_data.images)
        print "%d%% variance explained by %d components (use %d)" % (opts.pca, int(pca_op[opts.pca]), pow(int(np.sqrt(pca_op[opts.pca]))+1, 2))
        pca.n_components = pow(int(np.sqrt(pca_op[opts.pca]))+1, 2)
        for data in (training_data, testing_data, final_data):
            data.apply_pca(pca)

    t = time.time()
    classifier = svm.SVC(probability = True, gamma = opts.gamma, C = opts.C, degree = 5)
    if opts.gridsearch:
        if Debug:
            print "Start a grid search for the best parameters"
        #gammas = np.logspace(-3, -2, 4)
        #gammas = np.logspace(-2.7, -2.5, 3)
        # gammas = [opts.gamma]
        #Cs = np.logspace(-1, 1, 3)
        gammas = np.logspace(-3.3, -2, 5)
        Cs = np.logspace(-1, 1.7, 9)
        print "Search gammas = %s" % str(gammas)
        print "Search Cs = %s" % str(Cs)
        grid = grid_search.GridSearchCV(estimator = classifier, param_grid = dict(gamma = gammas, C = Cs), n_jobs = 3, verbose = 3)
        grid.fit(training_data.images, training_data.labels)
        print "Best score: %f" % grid.best_score
        print "Best gamma: %f" % grid.best_estimator_.gamma
        print "Best C: %f" % grid.best_estimator_.C
        classifier = grid.best_estimator_
        print "Full Results:"
        for A in grid.grid_scores_:
            print A
    else:
        classifier.fit(training_data.images, training_data.labels)

    if Debug:
        print "Fitting SVM took %d seconds" % (time.time() - t)



    if opts.two_pass:
        # training_data.predicted = classifier.predict(training_data.images)
        # training_data.prob = classifier.predict_proba(training_data.images)
    
        # unsure_data = training_data.subset(vals = np.array(np.max(training_data.prob, axis = 1) < 0.9))
        t = time.time()
        if Debug:
            print "Training two-pass classifier"
        try:
            #classifier2 = svm.SVC(probability = True)
            #classifier2.fit(unsure_data.images, unsure_data.labels)
            # classifier2 = RandomForestClassifier(n_estimators = 200)
            # classifier2.fit(unsure_data.images, unsure_data.labels)            
            classifier2 = RandomForestClassifier(n_estimators = 500, verbose = 1, n_jobs = -1)
            classifier2.fit(training_data.images, training_data.labels)            

        except:
            print "SVM2 fit failed"
            exit(1)
        if Debug:
            print "Fitting RF took %d seconds" % (time.time() - t)
            
        # predicted2 = classifier2.predict(unsure_data.images)            
        # print "Accuracy in unsure sample %f" % (sum(predicted2 == unsure_data.labels) / float(len(unsure_data.labels)))
        

    
    for name, data in (
        ("Training", training_data) ,
        ("Testing", testing_data) ,
        ("Final", final_data) ,
        ):
        print "+-"*20
        print "Running on %s" % data.name
        t = time.time()
        if len(data.images) == 0:
            continue
        data.predicted = classifier.predict(data.images)
        data.prob = classifier.predict_proba(data.images)

        if Debug:
            print "Accuracy: %f" % (sum(data.predicted == data.labels) / float(len(data.labels)))

        if np.max(data.labels) and "Training" not in data.name >= 0:
            check_mistakes(data)
            print "Classification report for classifier %s:\n%s\n" % (
                classifier, metrics.classification_report(data.labels, data.predicted))
            print "Confusion matrix:\n%s" % metrics.confusion_matrix(data.labels, data.predicted)

        if opts.two_pass:
            if Debug:
                print "Applying two-pass classifier"
            predicted2 = classifier2.predict(data.images)
            prob2 = classifier2.predict_proba(data.images)

            data.prob1 = data.prob
            data.predicted1 = data.predicted
            data.prob2 = prob2
            data.predicted2 = predicted2
            

            if Debug:
                print "Two-pass Accuracy on full sample: %f" % (sum(predicted2 == data.labels) / float(len(data.labels)))
                print "Two-pass max prob: %f" % np.max(np.max(prob2, axis = 1))


            nreplacements = 0
            for i in xrange(len(data)):
                if np.max(data.prob[i]) < 0.9 and np.max(prob2[i]) > np.max(data.prob[i]):
                    data.predicted[i] = predicted2[i]
                    data.prob[i] = prob2[i]
                    nreplacements += 1

            if Debug:
                print "Replaced %d predictions" % nreplacements
                print "Two-pass Accuracy: %f" % (sum(data.predicted == data.labels) / float(len(data.labels)))
                
            if np.max(data.labels) >= 0:
                check_mistakes(data)

        
        if Debug:
            print "Applying predictions took %d seconds" % (time.time() - t)
        
        if final_data.predicted.shape[0] > 0:
            s = "\n".join(["%d" % f for f in final_data.predicted])
            open('final.csv', 'w').write(s)
                 
    plt.show()

