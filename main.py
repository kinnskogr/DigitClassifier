'''Code for the kaggle.com digit classification competition.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cluster import KMeans
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

    scatM = np.array([[x_var, xy_var], [xy_var, y_var]])
    meanV = np.array([x_mean, y_mean])
    
    return scatM, meanV

def paxis(scatM):
    '''Calculate the principal axis values from the scattering matrix'''

    if scatM.shape != (2, 2):
        print "ERROR: getPrincipleAxes(scatM), scatM size is not 2x2"
        return

    #(a + d - sqrt((a+d)**2 - 4(a*d-c*b)))/2                                                                                                                                    
    #a*x*x + b*x + c = 0 --> -b/2 +- sqrt(b*b-4*a*c)                                                                                                                            
    #-(a+d)x + x*x - det(ABCD) = 0                                                                                                                                              
    sol1 = 0.5*(scatM[0, 0]+scatM[1, 1]) + 0.5*math.sqrt( math.pow((scatM[0, 0]+scatM[1, 1]), 2) - 4*(scatM[0, 0]*scatM[1, 1] - scatM[0, 1]*scatM[1, 0]) )
    sol2 = 0.5*(scatM[0, 0]+scatM[1, 1]) - 0.5*math.sqrt( math.pow((scatM[0, 0]+scatM[1, 1]), 2) - 4*(scatM[0, 0]*scatM[1, 1] - scatM[0, 1]*scatM[1, 0]) )


    den1 = math.sqrt( 1 + math.pow( ((sol1 - scatM[0, 0])/scatM[0, 1]), 2) )
    vec1 = np.array([1.0/den1, ((sol1 - scatM[0, 0])/scatM[0, 1]) / den1])

    den2 = math.sqrt( 1 + math.pow( ((sol2 - scatM[0, 0])/scatM[0, 1]), 2) )
    vec2 = np.array([1.0/den2, ((sol2 - scatM[0, 0])/scatM[0, 1]) / den2])

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

    scatM, meanV = scat(img.reshape(dim, dim))

    V, S = paxis(scatM)

    stheta = V[0, 0]
    ctheta = V[0, 1]

    rot_img =  ndimage.interpolation.rotate(img, math.asin(stheta) * 360 / (2*math.pi), reshape = False)

    return rot_img
    

def construct_averages(images, labels):
    '''Construct the average image for each digit. Optionally,
    normalize by total cell intensity'''

    output = []
    if images.shape[0] == 0: return output
    
    for i in xrange(10):
        collection = images[labels == i]
        if collection.shape[0] == 0:
            output.append(np.array([]))

        try:
            average = np.sum(collection, axis = 0) / len(collection)
        except:
            average = np.array([])
        
        output.append(average)
    return output

def edge_laplacian(images):
    from scipy import signal, misc

    output = []
    dim = np.sqrt(images.shape[1])
    for image in images.reshape(len(images), dim, dim):
        derfilt = np.array([1.0, -2, 1.0], float)
        ck = signal.cspline2d(image, 8.0, 1.0)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
        deriv2 = signal.convolve2d(ck, laplacian, mode = 'same', boundary = 'symm')
        output.append(deriv2)
        
    return np.array(output, dtype = float).reshape(len(images), -1)


def plot_figure(image, ax = None, scaler = None):
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

    if scaler:
        elements = scaler.transform(elements)

    if ax == None:
        ax = plt.figure()

    try:
        output = plt.imshow(elements, interpolation='nearest', )
        output.set_cmap('hot')
        plt.colorbar()
    except ValueError:
        output = None
    return output
        

def plot_array(images, scaler = None, name = ""):
    if Debug:
        print "Plot array of %d categories" % len(images)
    dim_y = int(np.sqrt(len(images)))
    dim_x = dim_y + (dim_y != np.sqrt(len(images)))
    ax = plt.figure(name)
    if name != "":
        ax.set_label(name)
    for i in xrange(len(images)):
        ax.add_subplot(dim_y, dim_x, i+1)
        fig = plot_figure(images[i], ax = ax, scaler = scaler)

    plt.savefig(name + ".png")
        

def run_pca(images, n = 0):
    if n == 0:
        n = images.shape[1]

    pca = PCA(n_components = n, whiten = False)
    pca.fit(images)
    return pca

def explore_pca(images):
    pca = run_pca(images, images.shape[1])
    frac_var_explained = [sum(pca.explained_variance_ratio_[:i]) for i in range(len(pca.explained_variance_ratio_))]
    plt.figure()
    plt.plot(range(len(frac_var_explained)), frac_var_explained)

    import bisect
    output = {}
    for var in (0.9, 0.95, 0.99, 0.999):
        n_components = bisect.bisect(frac_var_explained, var)
        print "%f variance explained by %d components" % (var, n_components)
        output[var] = n_components

    return pca, output
    

def explore_lda(images):
    lda = LDA(n_components == images.shape[1])
    lda.fit(images)
    frac_var_explained = [sum(lda.explained_variance_ratio_[:i]) for i in range(len(lda.explained_variance_ratio_))]
    plt.figure()
    plt.plot(range(len(frac_var_explained)), frac_var_explained)

    import bisect
    output = {}
    for var in (0.9, 0.95, 0.99, 0.999):
        n_components = bisect.bisect(frac_var_explained, var)
        print "%f variance explained by %d components" % (var, n_components)
        output[var] = n_components

    return lda, output


def check_mistakes(data):
    '''Look at the average values for mis-labeled predictions'''

    for i in range(10):
        f_bad = float(sum(data._predicted[data._labels == i] == i)) / len(data._predicted[data._labels == i])
        print "%f mis-classified for %d" % (f_bad, i)

    f_bad = sorted([(i, float(sum(data._predicted[data._labels == i] == i)) / len(data._predicted[data._labels == i])) for i in range(10)], key = lambda x:x[1])
    print "In order of performance:"
    for i, f in f_bad:
        print "%d %f" % (i, f)

    bad_images = data._orig[data._labels != data._predicted]
    bad_labels  = data._labels[data._labels != data._predicted]

    if bad_images.shape[0] == 0:
        return

    bad_averages = construct_averages(bad_images, bad_labels)
    plot_array(bad_averages, name = data._name + "_bad_averages")

    bad_prob = np.max(data._prob[data._labels != data._predicted], axis = 1)
    good_prob = np.max(data._prob[data._labels == data._predicted], axis = 1)
    
    h_good, bin_edges = np.histogram(good_prob, bins = 20)
    h_bad, bin_Edges = np.histogram(bad_prob, bins = bin_edges)

    bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(data._name + "_sums")
    ax.set_yscale('log')
    plt.plot(bins, h_good, 'bo')
    plt.plot(bins, h_bad, 'ro')
    plt.savefig(data._name + '_sums.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(data._name + "_fractions")
    ax.set_yscale('log')
    answers = data._predicted == data._labels
    probs = np.max(data._prob, axis = 1)
    xaxis = np.array(range(0, 101, 1), dtype = float) / 100
    sums = np.array([len(answers[(probs <= i)]) for i in xaxis], dtype = float) / len(answers)
    good = np.array([len(answers[(probs <= i) & (answers == True)]) for i in xaxis], dtype = float) / len(answers)
    bad = np.array([len(answers[(probs <= i) & (answers == False)]) for i in xaxis], dtype = float) / len(answers)
    plt.plot(xaxis, sums, 'b')
    plt.plot(xaxis, good, 'g')
    plt.plot(xaxis, bad, 'r')
    plt.savefig(data._name + '_fractions.png')

def makeXYScatterMatrix(image, pow_forScat = 1, pow_forMean = 1):
    cell_values = np.array([i for i in image])
    cell_x      = np.array([range(np.sqrt(len(image)))])
    cell_y      = np.array([range(np.sqrt(len(image)))])

    x_off = (np.max(cell_x) + np.min(cell_x))/2
    y_off = (np.max(cell_y) + np.min(cell_y))/2

    etot = np.sum((cell_values>0) * np.power(cell_values, pow_forMean))
    if etot == 0:
        print 'Found a jet with no energy.  DYING!'
        sys.exit(1)

    x_1  = np.sum((cell_values>0) * np.power(cell_values, pow_forMean) * cell_x)/etot
    y_1  = np.sum((cell_values>0) * np.power(cell_values, pow_forMean) * cell_y)/etot
    x_2  = np.sum((cell_values>0) * np.power(cell_values, pow_forScat) * np.square(cell_x -x_1))
    y_2  = np.sum((cell_values>0) * np.power(cell_values, pow_forScat) * np.square(cell_y -y_1))
    xy   = np.sum((cell_values>0) * np.power(cell_values, pow_forScat) * (cell_x - x_1)*(cell_y -y_1))

    scatM = np.array([[x_2, xy], [xy, y_2]])
    meanV = np.array([x_1, y_1])

    return scatM, meanV

class Classifers:
    def __init__():
        self.clf = {}
        self.config = {}

    def train_svm(data, **args):
        t = time.time()
        clf = svm.SVC(probability = True, **args)
        clf.fit(data._images, data._labels)
        
        self.clf["svm"] = clf
        self.config["svm"] = data._config

        if Debug:
            print "Trained svm in %d seconds" % (time.time() - t)
    
    def train_rf(data, **args):
        t = time.time()
        clf = RandomForestClassifier(verbose = 1, **args)
        clf.fit(data._images, data._labels)
        
        self.clf["rf"] = clf
        self.config["rf"] = data._config

        if Debug:
            print "Trained rf in %d seconds" % (time.time() - t)    
            
    

class ImgData:
    def __init__(self, name, config = ""):
        self._orig      = np.array([])
        self._images    = np.array([])
        self._labels    = np.array([])
        self._predicted = np.array([])
        self._name      = name
        self._config    = config

    def __len__(self):
        return len(self._orig)

    def preprocess(self, **opts):
        self._images = self._orig
        if self._images.shape[0] == 0:
            return

        if opts.get('counts', False) == True:
            t = time.time()
            dim = np.sqrt(self._orig.shape[1]) if len(self._orig.shape) == 2 else self._orig.shape[2]
            counts = np.column_stack([np.sum(self._orig.reshape(len(self._images), dim, dim) != 0, axis = 1), np.sum(self._orig.reshape(len(self._images), dim, dim) != 0, axis = 2)])
            cmax = np.max(counts)
            cmin = np.min(counts)
            if Debug:
                print "Adding counts took %d seconds" % (time.time() - t)
            

        if opts.get('rotate', False) == True:
            t = time.time()
            self._images = np.array([rotate_image(img) for img in self._orig])
            if Debug:
                print "Rotating images took %d seconds" % (time.time() - t)
        else:
            self._images = self._images.reshape(len(self._images), -1)
        dim = np.sqrt(self._images.shape[1])

        if opts.get('edges', 'orig') != 'orig':
            t = time.time()
            edges = edge_laplacian(self._images)
            if opts['edges'] == 'edges':
                self._images = edges
            elif opts['edges'] == 'merged':
                self._images = np.column_stack([self._images, edges])
            if Debug:
                print "Calculating edges took %d seconds" % (time.time() - t)

        if opts.get('counts', False) == True:
            if Debug:
                print "Adding a row of counts"
            #feature scaling
            #counts = (counts - cmin) * (np.max(self._images) - np.min(self._images)) / (cmax - cmin)
            self._images = np.column_stack([self._images, counts.reshape(len(counts), -1)])

        pass

    def scale(self, scaler):
        shape = self._images.shape
        if shape[0] == 0:
            return

        #self._images = scaler.transform(self._images.reshape(shape[0], -1), axis = 1)
        from sklearn import preprocessing
        self._images = preprocessing.scale(self._images.reshape(shape[0], -1), axis = 1)
        self._images = self._images.reshape(shape)

    def apply_pca(self, pca):
        if self._images.shape[0] == 0:
            return
        
        self._images = pca.transform(self._images)

    def read_csv(self, fname, max_entries = -1):
        '''read a csv file containing pixel intensities and (optionally) labels'''
        if Debug:
            print "Reading csv from %s" % fname

        images = []
        labels = []
        for line in open(fname).readlines():
            l = line.split(",")
            if l[0] == 'label':
                idx_start = 1
                continue
            if l[0] == 'pixel0':
                idx_start = 0
                continue
            l = [float(i) for i in l]
            
            image = l[idx_start:]
            label = -1 if idx_start == 0 else l[0]

            images.append(image)
            labels.append(label)

            if max_entries > 0 and len(images) > max_entries:
                break

            
        self._orig = np.array(images, dtype = float)
        self._labels = np.array(labels, dtype = int)
        self._config += "csv = %s;" % fname
        pass
    
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
            self._orig = np.array(result, dtype = float)
        elif len(result.shape) == 2:
            #Labels are stored as tuples
            self._labels = np.array(result, dtype = int)
        pass
        self._config += ", mnist = %s" % fname
    

    def plot_image(self, idx):
        plot_figure(self._images[idx])
        pass

    def plot_avg_images(self, status = 'processed'):
        dim_x = np.sqrt(self._orig.shape[1])
        dim_y = np.sqrt(self._orig.shape[1])
        if status == 'processed':
            averages = construct_averages(self._images, self._labels)
            dim_x = self._images.shape[1] / dim_y
        elif status == 'orig':
            averages = construct_averages(self._orig, self._labels)

        l = len(averages)

        averages = np.array(averages, dtype = float)

        plot_array(averages.reshape(l, dim_x, dim_y), name = self._name + "_averages_" + status)

    def accuracy(self):
        return sum(self._labels == self._predicted) / len(self._labels)

    def bad_indices(self):
        return self._labels != self._predicted

    def bad_images(self):
        ids = self.bad_indices()
        return self._images[ids], self._labels[ids]

    def subset(self, **args):
        '''Return an ImgData object containing a subset of the data,
        valid arguments are:
        start - the 1st id in a range
        end   - the last id in a range (exclusive)
        ids   - a list of indices
        vals  - a list of true/false values
        '''

        output = ImgData(self._name, self._config + str(args) + ";")
        if 'start' in args and 'end' not in args:
            args['end'] = len(self._labels)
        if 'end' in args and 'start' not in args:
            args['start'] = 0

        if 'start' in args and 'end' in args:
            output._orig = self._orig[args['start']:args['end']]
            output._labels = self._labels[args['start']:args['end']]
            if len(self._images) > 0:
                output._images = self._images[args['start']:args['end']]
            if len(self._predicted) > 0:
                output._predicted = self._predicted[args['start']:args['end']]
        
        if 'ids' in args:
            ids = args['ids']
            ids = np.array([i in ids for i in range(len(self._orig))])
            output._orig = self._orig[ids]
            output._labels = self._labels[ids]
            if len(self._images) > 0:
                output._images = self._images[ids]
            if len(self._predicted) > 0: 
                output._predicted = self._predicted[ids]


        if 'vals' in args:
            vals = args['vals']
            output._orig = self._orig[vals]
            output._labels = self._labels[vals]
            if len(self._images) > 0:
                output._images = self._images[vals]
            if len(self._predicted) > 0: 
                output._predicted = self._predicted[vals]

        return output

class Scaler:
    def __init__(self):
        self.transform = None

    def train_minmax(self, data, minVal = -1, maxVal = 1):
        curMin = np.min(data._images)
        curMax = np.max(data._images)
        self.transform = lambda x: (x - curMin) * (maxVal - minVal) / (curMax - curMin)

    def train_scaler(self, data):
        from sklearn import preprocessing
        scaler = preprocessing.Scaler().fit(data._images)
        self.transform = scaler.transform        


if __name__ == "__main__":
    import sys
    from optparse import OptionParser

    print "Called with:"
    print " ".join(sys.argv)

    parser = OptionParser()
    parser.add_option("--train_csv"   , dest = "train_csv"  , default = None   , help = "read training data from a csv file")
    parser.add_option("--final_csv"   , dest = "final_csv"  , default = None   , help = "read final data to classify from a csv file")
    parser.add_option("--train_mnist" , dest = "train_mnist", default = None   , help = "read the training data from a pair of ubyte files (comma seperated)")
    parser.add_option("--n_train"     , dest = "n_train"    , default = 15000  , type = 'int'  , help = "number of training events to use for training")
    parser.add_option("--n_test"      , dest = "n_test"     , default = 15000  , type = 'int'  , help = "number of training events to use for testing (bounded by n_train)")
    parser.add_option("--scale"       , dest = "scale"      , default = 'scale' , help = "What scaling to apply: scale uses scikit learn scaler, minmax enforces a range from 0 to 1")    
    parser.add_option("--pca"         , dest = "pca"        , default = -1     , type = 'float', help = "Use PCA to reduce the dimensionality, value sets the minimum fraction of variance explained by the components used")
    parser.add_option("--lda"         , dest = "lda"        , default = -1     , type = 'float', help = "Use LDA to reduce the dimensionality, value sets the minimum fraction of variance explained by the components used")
    parser.add_option("--edges"       , dest = "edges"      , default = "orig" , help = "Use original image (orig), edge detection (edges), or a combination of both (merged)")
    parser.add_option("--counts"      , dest = "counts"     , default = False  , action = 'store_true', help = "Add a set of features to the image, listing the number of non-zero pixels")
    parser.add_option("--rotate"      , dest = "rotate"     , default = False  , action = 'store_true', help = "Interpolate and then rotate the images along their principal axis")    
    parser.add_option("--gridsearch"  , dest = "gridsearch" , default = False  , action = 'store_true', help = "Use grid-search to optimize the svm parameters")
    parser.add_option("--gamma"       , dest = "gamma"      , default = 0.002  , type = 'float',  help = "Gamma value passed to the svm")
    parser.add_option("--C"           , dest = "C"          , default = 5      , type = 'float',  help = "C value passed to the svm")
#    parser.add_option("--cout"        , dest = "cout"       , default = None  , help = "Specify a pickle file to store the classifier in")
#    parser.add_option("--cin"         , dest = "cin"        , default = None  , help = "Specify a pickle file to read the classifier from (ignore training data)")
    parser.add_option("--two_pass"    , dest = "two_pass"   , default = False  , action = 'store_true', help = "Turn on two-pass classification")
    parser.add_option("--plots"       , dest = "plots"      , default = False  , action = 'store_true', help = "Turn on additional plotting")
    parser.add_option("--debug"       , dest = "debug"      , default = False  , action = 'store_true', help = "Turn on debugging printouts")

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
    training_data._name = "Training Data"
    testing_data  = all_training_data.subset(ids = idxs[opts.n_train:opts.n_train+opts.n_test])
    testing_data._name = "Testing Data"
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
    if opts.scale == "scale":
        scaler.train_scaler(training_data)
    elif opts.scale == "minmax":
        scaler.train_minmax(training_data, 0, 1)
    else:
        scaler = lambda x: x

    for data in (training_data, testing_data, final_data):
        data.scale(scaler)

    if opts.plots:
        training_data.plot_avg_images('orig')
        training_data.plot_avg_images('processed')

    if (opts.pca > 0):
        pca, pca_op = explore_pca(training_data._images)
        print "%d%% variance explained by %d components (use %d)" % (opts.pca, int(pca_op[opts.pca]), pow(int(np.sqrt(pca_op[opts.pca]))+1, 2))
        pca.n_components = pow(int(np.sqrt(pca_op[opts.pca]))+1, 2)
        for data in (training_data, testing_data, final_data):
            data.apply_pca(pca)

    if (opts.lda > 0):
        lda = LDA(n_components = 9)
        lda.fit(training_data._images)
        for data in (training_data, testing_data):
            data.apply_pca(lda)

                
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
        grid.fit(training_data._images, training_data._labels)
        print "Best score: %f" % grid.best_score
        print "Best gamma: %f" % grid.best_estimator_.gamma
        print "Best C: %f" % grid.best_estimator_.C
        classifier = grid.best_estimator_
        print "Full Results:"
        for A in grid.grid_scores_:
            print A
    else:
        classifier.fit(training_data._images, training_data._labels)

    if Debug:
        print "Fitting SVM took %d seconds" % (time.time() - t)



    if opts.two_pass:
        # training_data._predicted = classifier.predict(training_data._images)
        # training_data._prob = classifier.predict_proba(training_data._images)
    
        # unsure_data = training_data.subset(vals = np.array(np.max(training_data._prob, axis = 1) < 0.9))
        t = time.time()
        if Debug:
            print "Training two-pass classifier"
        try:
            #classifier2 = svm.SVC(probability = True)
            #classifier2.fit(unsure_data._images, unsure_data._labels)
            # classifier2 = RandomForestClassifier(n_estimators = 200)
            # classifier2.fit(unsure_data._images, unsure_data._labels)            
            classifier2 = RandomForestClassifier(n_estimators = 500, verbose = 1, n_jobs = -1)
            classifier2.fit(training_data._images, training_data._labels)            

        except:
            print "SVM2 fit failed"
            exit(1)
        if Debug:
            print "Fitting RF took %d seconds" % (time.time() - t)
            
        predicted2 = classifier2.predict(unsure_data._images)            
        print "Accuracy in unsure sample %f" % (sum(predicted2 == unsure_data._labels) / float(len(unsure_data._labels)))
        

    
    for name, data in (
        ("Training", training_data) ,
        ("Testing", testing_data) ,
        ("Final", final_data) ,
        ):
        print "+-"*20
        print "Running on %s" % data._name
        t = time.time()
        if len(data._images) == 0:
            continue
        data._predicted = classifier.predict(data._images)
        data._prob = classifier.predict_proba(data._images)

        if Debug:
            print "Accuracy: %f" % (sum(data._predicted == data._labels) / float(len(data._labels)))

        if np.max(data._labels) and "Training" not in data._name >= 0:
            check_mistakes(data)
            print "Classification report for classifier %s:\n%s\n" % (
                classifier, metrics.classification_report(data._labels, data._predicted))
            print "Confusion matrix:\n%s" % metrics.confusion_matrix(data._labels, data._predicted)

        if opts.two_pass:
            if Debug:
                print "Applying two-pass classifier"
            predicted2 = classifier2.predict(data._images)
            prob2 = classifier2.predict_proba(data._images)

            data._prob1 = data._prob
            data._predicted1 = data._predicted
            data._prob2 = prob2
            data._predicted2 = predicted2
            

            if Debug:
                print "Two-pass Accuracy on full sample: %f" % (sum(predicted2 == data._labels) / float(len(data._labels)))
                print "Two-pass max prob: %f" % np.max(np.max(prob2, axis = 1))


            nreplacements = 0
            for i in xrange(len(data)):
                if np.max(data._prob[i]) < 0.9 and np.max(prob2[i]) > np.max(data._prob[i]):
                    data._predicted[i] = predicted2[i]
                    data._prob[i] = prob2[i]
                    nreplacements += 1

            if Debug:
                print "Replaced %d predictions" % nreplacements
                print "Two-pass Accuracy: %f" % (sum(data._predicted == data._labels) / float(len(data._labels)))
                
            if np.max(data._labels) >= 0:
                check_mistakes(data)

        
        if Debug:
            print "Applying predictions took %d seconds" % (time.time() - t)
        
        if final_data._predicted.shape[0] > 0:
            s = "\n".join(["%d" % f for f in final_data._predicted])
            open('final.csv', 'w').write(s)
                 
    plt.show()

