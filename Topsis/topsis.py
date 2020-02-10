import numpy as np


class topsis:
    """ Define a TOPSIS decision making process
    TOPSIS (Technique for Order Preference by Similarity to an Ideal Solution)
    chooses an alternative of shortest distance from the 
    """
    C = None
    optimum_choice = None

    def __init__(self, a, w, I):
        """
        :param np.ndarray a: A 2D array of shape (J,n)
        :param np.ndarray w: A 1D array of shape (J)
        :param np.ndarray I: A 1D array of shape (n)
        """
        # Decision Matrix
        self.a = np.array(a, dtype=np.float).T
        assert len(self.a.shape) == 2, "Decision matrix a must be 2D"

        # Number of alternatives, aspects
        (self.n, self.J) = self.a.shape

        # Weight matrix
        self.w = np.array(w, dtype=np.float)
        assert len(self.w.shape) == 1, "Weights array must be 1D"
        assert self.w.size == self.n, "Weights array wrong length, " + \
                                      "should be of length {}".format(self.n)

        # Normalise weights to 1
        self.w = self.w/sum(self.w)

        # Benefit (True) or Cost (False) criteria?
        self.I = np.array(I, dtype=np.int8)
        assert len(self.I.shape) == 1, "Criterion array must be 1D"
        assert len(self.I) == self.n, "Criterion array wrong length, " + \
                                      "should be of length {}".format(self.n)

        # Initialise best/worst alternatives lists
        ab, aw = np.zeros(self.n), np.zeros(self.n)
   
    
    def __repr__(self):
        """ What to print when the object is called?
        """
        print('\n')
        if self.optimum_choice == None:
            self.calc()
            print('Alternatives ranking C:\n{}'.format(self.C))
        return 'Best alternative\na[{}]: {}\n'.format(
                self.optimum_choice, self.a[:,self.optimum_choice])
    
    
    def step1(self):
        """ TOPSIS Step 1
        Calculate the normalised decision matrix
        for j in range(self.J):
            self.r[j] = self.a[j]/np.linalg.norm(self.a[j,:], axis=1)
        """
        self.r = self.a/np.array(np.linalg.norm(self.a, axis=1)[:,np.newaxis])
        return
    
    
    def step2(self):
        """ TOPSIS Step 2
        Calculate the weighted normalised decision matrix
        Two tranposes required so that indicies are multiplied correctly:
        for i in range(w.size):
            v[j,i] = w[j] * r[j,i]
        """
        self.v = (self.w * self.r.T).T
        return
    
    
    def step3(self):
        """ TOPSIS Step 3
        Determine the ideal and negative-ideal solutions
        I[i] defines i as a member of the benefit criteria (True) or the cost
        criteria (False)
        """
        # Calcualte ideal/negative ideals
        self.ab = np.max(self.v, axis=1) * self.I + \
                  np.min(self.v, axis=1) * (1 - self.I)
        self.aw = np.max(self.v, axis=1) * (1 - self.I) +  \
                  np.min(self.v, axis=1) * self.I
        return
   
    
    def step4(self):
        """ TOPSIS Step 4
        Calculate the separation measures, n-dimensional Euclidean distance
        db, dw = [], []
        for j in range(self.J):
            db.append(np.linalg.norm(self.v[:,j] - self.ab))
            dw.append(np.linalg.norm(self.v[:,j] - self.ab))
        """
        # Create two n long arrays containing Eculidean distances
        # Save the ideal and negative-ideal solutions
        self.db = np.linalg.norm(self.v - self.ab[:,np.newaxis], axis=0)
        self.dw = np.linalg.norm(self.v - self.aw[:,np.newaxis], axis=0)
        return
    

    def step5(self):
        """ TOPSIS Step 5 & 6
        Calculate the relative closeness to the ideal solution, then rank the
        preference order
        """
        # Ignore division by zero errors
        #np.seterr(all='ignore')
        # Find relative closeness
        self.C = self.dw / (self.dw + self.db)
        self.optimum_choice = self.C.argsort()[-1]
        return
   
    
    def calc(self):
        """ TOPSIS Calculations
        This can be called once the object is initialised, and is
        automatically called when a representation of topsis is
        needed (eg. print(topsis(matrix, weights, I)). This calls each step in
        TOPSIS algorithm and stores calcultions in self.

        The optimum alternatives index (starting at 0) is saved in
        self.optimum_choice
        """
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return
