import threading, time, random
from args import init_arguments

args = init_arguments().parse_args()
if args.gpu:
    import cupy as np
else:
    import numpy as np

np.random.seed(args.seed)    
random.seed(args.seed)

def save_txt(out:list, PATH:str):
    with open(PATH+'.txt', 'w') as f:
        for row in out:
            f.write(str(row) + '\n')

class BaseModule:
    def __init__(self, theta:list):
        self.theta = np.array(theta)

    def functionF(self, x, result, n): 
        e1 = x * self.theta[2] * np.exp((-1) * self.theta[0] * x)
        e2 = (-1) * x * self.theta[2] * np.exp((-1) * self.theta[1] * x)
        e3 = np.exp((-1) * self.theta[1] * x) - np.exp((-1) * self.theta[0] * x)
        outputs = np.array([[e1, e2, e3]])
        result[n] = np.expand_dims(outputs.T.dot(outputs), axis=0)

    def functionFP(self, x): # x: 1 * 3
        n = x.shape[1]
        thread = [None] * n
        result = [None] * n
        for i in range(n):
            thread[i] = threading.Thread(target=self.functionF, args=(x[0, i], result, i))
            thread[i].start()

        for i in range(n):
            thread[i].join()

        return np.concatenate(result, axis=0)

    def getValue(self, inputs):   # inputs: 2 * 3
        n = inputs.shape[1]
        x = inputs[0:1, :]
        w = inputs[1:2, :]
        ft = self.functionFP(x)
        outputs = ft.reshape((n, -1)).T
        outputs = (ft * w.T).reshape((n, 3, 3))
        outputs = np.sum(outputs, axis=0)
        outputs = np.linalg.det(outputs)
        return outputs * (-1)


class DE(BaseModule):
    ''' Algorithm of Differential Evolution '''

    def __init__(self, theta, F, N):
        super().__init__(theta=theta)
        self.F = F
        self.N = N
        self.vectexs = None
        self.values = None
        self.globalm = None    # the lowest point at the moment

    def train(self, t):
        n = self.N
        self.time_cost = []
        self.lst_globalv = []
        for interation in range(t):
            t0 = time.time()
            thread = [None] * n
            for i in range(1, n+1):
                thread[i-1] = threading.Thread(target=self.run, args=(i,))
                thread[i-1].start()

            for i in range(n):
                thread[i].join()
            
            self.checkGlobal()
            self.globalv = self.values[self.globalm]
            try:
                v = self.globalv.get()
            except:
                v = self.globalv
            print(f'Global good value {interation}: {v:.6f}')
            self.lst_globalv.append(v)
            self.time_cost.append(time.time() - t0)

    def initW(self):
        vectexs = {}
        values = {}
        for i in range(1, self.N + 1):
            vectexs[i] = self.randoms()
            values[i] = self.getValue(vectexs[i])
            if i == 1:
                self.globalm = i

            elif values[i] < values[self.globalm]:
                self.globalm = i
        
        self.vectexs = vectexs
        self.values = values

    def run(self, n):
        x = self.vectexs[n]
        val = self.values[n]
        v = self.randomV()
        u = self.getU(v, x)
        self.adjX(u)
        newx, valueX = self.updateX(x, u)
        self.vectexs[n] = newx
        self.values[n] = valueX

    def adjX(self, x):
        x[0, :] = np.clip(x[0, :], 0, 30)
        x[1, :] = self.sigmoid(x[1, :])

    def updateX(self, x, u):
        valueX = self.getValue(x)
        valueU = self.getValue(u)
        if valueX <= valueU:
            return x, valueX
        else:
            return u, valueU

    def checkGlobal(self):
        k = list(self.vectexs.keys())
        v = list(self.values.values())
        try:
            m = np.argmin(v)
        except:
            m = np.array(v).argmin()
        try:
            self.globalm = k[m]
        except:
            self.globalm = k[int(m)]

    def getU(self, v, x):
        a, b = x.shape[0], x.shape[1]
        n = a * b
        k = random.randint(1, n)
        l = n - k + 1
        l = random.randint(1, l)    # range [k, k + l - 1]
        u = np.zeros(n)
        v0 = v.reshape(-1)
        x0 = x.reshape(-1)
        u[:] = x0[:]
        u[k-1: k+l-1] = v0[k-1: k+l-1]
        return u.reshape(2, -1)
    
    def randomV(self):
        random.seed(args.seed)
        p = random.randint(1, self.N)
        p = self.vectexs[p]
        q = random.randint(1, self.N)
        q = self.vectexs[q]
        r = random.randint(1, self.N)
        r = self.vectexs[r]
        v = p + self.F * (q - r)
        return v

    def randoms(self):
        w = np.zeros((2, 3))
        w[0, :] = np.random.rand(3) * 30
        w[1, :] = np.random.rand(3)
        w[1, :] = self.sigmoid(w[1, :])
        return w

    def sigmoid(self, x):
        outputs = np.exp(x)
        outputs = outputs / np.sum(outputs)
        return outputs


class PSO(BaseModule):
    ''' Algorithm of Particle Swarm Optimization '''

    def __init__(self, theta, velocityP, velocityPT, alpha, beta, gamma, N):
        super().__init__(theta=theta)
        self.velocityP = velocityP    # hyper-paramter
        self.velocityPT = velocityPT  # hyper-paramter
        self.alpha = alpha            # hyper-paramter
        self.beta = beta              # hyper-paramter 
        self.gamma = gamma
        self.N = N                    # no. of points

        self.vectexs = None    # dict of all points
        self.values = None     # dict of values of all current points
        self.localx = None     # dict of points with the local minimum
        self.localv = None     # dict of the local minimun values
        self.velocity = None   # dict of all current velocities

        self.globalx = None    # array of points with the global minimum
        self.globalv = None    # float of the global minimum value

    def initW(self):
        self.vectexs = {}
        self.values = {}
        self.localx = {}
        self.localv = {}
        self.velocity = {}
        for i in range(1, self.N + 1):
            x = self.randoms()
            v = self.getValue(x)
            self.vectexs[i] = x
            self.values[i] = v
            self.localx[i] = x
            self.localv[i] = v
            self.velocity[i] = 0.

            if i == 1:
                self.globalx = x
                self.globalv = v

            elif v < self.globalv:
                self.globalx = x
                self.globalv = v

    def train(self, t):
        n = self.N
        self.time_cost = []
        self.lst_globalv = []
        for interation in range(t):
            t0 = time.time()
            thread = [None] * n
            for i in range(1, n+1):
                thread[i-1] = threading.Thread(target=self.run, args=(i,))
                thread[i-1].start()

            for i in range(n):
                thread[i].join()
            self.checkGlobal()
            self.updateParameters()
            try:
                v = self.globalv.get()
            except:
                v = self.globalv
            print(f'Global good value {interation}: {v:.6f}')
            self.lst_globalv.append(v)
            self.time_cost.append(time.time() - t0)

    def run(self, n):
        x = self.vectexs[n]
        localvalue = self.localv[n]
        v = self.getV(n)
        newx, value = self.updateX(x, v)
        if value < localvalue:
            self.localx[n] = newx
            self.localv[n] = value
        self.vectexs[n] = newx
        self.values[n] = value

    def updateParameters(self):
        self.alpha = self.alpha * np.exp((-1) * self.gamma)
        self.velocityP = self.velocityP * np.exp((-1) * self.velocityPT)

    def checkGlobal(self):
        k = list(self.localv.keys())
        v = list(self.localv.values())
        try:
            m = np.argmin(v)
            index = k[m]
        except:
            m = np.array(v).argmin()
            index = k[int(m)]
        self.globalx = self.localx[index]
        self.globalv = self.localv[index]

    def updateX(self, x, v):
        newx = x + v
        self.adjX(newx)
        value = self.getValue(newx)
        return newx, value

    def getV(self, n):
        x = self.vectexs[n]
        locx = self.localx[n]
        gbx = self.globalx
        vt = self.velocity[n]
        e1 = self.randomE()
        e2 = self.randomE()
        v = self.velocityP * vt + self.alpha * e1 * (gbx - x) + self.beta * e2 * (locx - x)
        return v

    def randomE(self):
        e = np.random.rand(2, 3)
        return e

    def adjX(self, x):
        x[0, :] = np.clip(x[0, :], 0, 30)
        x[1, :] = self.sigmoid(x[1, :])

    def randoms(self):
        w = np.zeros((2, 3))
        w[0, :] = np.random.rand(3) * 30
        w[1, :] = np.random.rand(3)
        w[1, :] = self.sigmoid(w[1, :])
        return w

    def sigmoid(self, x):
        outputs = np.exp(x)
        outputs = outputs / np.sum(outputs)
        return outputs
    

class FA(BaseModule):
    ''' The Firefly Algorithm  '''

    def __init__(self, theta, beta0, gamma, alpha, alpha2, alphaP, lamb, N):
        super().__init__(theta=theta)
        self.beta0 = beta0     # hyper: intension item
        self.gamma = gamma     # hyper: adj beta0
        self.alpha = alpha     # hyper: random item
        self.alpha2 = alpha2
        self.alphaP = alphaP   # 超參數      adj alpha
        self.lamb = lamb       # 超參數        global
        self.N = N             # 要撒幾個點

        self.vectexs = None    # 宣告的所有的點集合  dict
        self.values = None     # 所有的點當前的值集合  dict
        self.localx = None     # 局部最小的點集合  dict
        self.localv = None     # 局部最小的值集合  dict
        self.globalx = None    # 全域最小的點  array
        self.globalv = None    # 全域最小的點的值  float

    def initW(self):
        self.vectexs = {}
        self.values = {}
        self.localx = {}
        self.localv = {}
        for i in range(1, self.N + 1):
            x = self.randoms()
            v = self.getValue(x)
            self.vectexs[i] = x
            self.values[i] = v
            self.localx[i] = x
            self.localv[i] = v

            if i == 1:
                self.globalx = x
                self.globalv = v

            elif v < self.globalv:
                self.globalx = x
                self.globalv = v

    def train(self, t):
        n = self.N
        self.time_cost = []
        self.lst_globalv = []
        for interation in range(t):
            t0 = time.time()
            thread = [None] * n
            resultx = [None] * n
            resultv = [None] * n
            for i in range(1, n+1):
                thread[i-1] = threading.Thread(target=self.run, args=(i, resultx, resultv))
                thread[i-1].start()

            for i in range(n):
                thread[i].join()
            
            for i in range(1, n+1):
                self.vectexs[i] = resultx[i-1]
                self.values[i] = resultv[i-1]

            self.checkGlobal()
            self.updateParameters()
            try:
                v = self.globalv.get()
            except:
                v = self.globalv
            print(f'Global good value {interation}: {v:.6f}')
            self.lst_globalv.append(v)
            self.time_cost.append(time.time() - t0)

    def run(self, n, resultx, resultv):
        x = self.vectexs[n]
        v = self.values[n]
        localvalue = self.localv[n]
        for i in range(1, self.N+1):
            v0 = self.getV(x, v, i)
            newx, newvalue = self.updateX(x, v0)
            if newvalue < localvalue:
                self.localx[n] = newx
                self.localv[n] = newvalue
            x = newx
            v = newvalue

        resultx[n-1] = x
        resultv[n-1] = v

    def updateParameters(self):
        self.alpha = self.alpha * self.alphaP

    def checkGlobal(self):
        k = list(self.localv.keys())
        v = list(self.localv.values())
        try:
            m = np.argmin(v)
        except:
            m = np.array(v).argmin()
        index = k[int(m)]
        self.globalx = self.localx[index]
        self.globalv = self.localv[index]

    def updateX(self, x, v):
        newx = x + v
        self.adjX(newx)
        value = self.getValue(newx)
        return newx, value

    def getV(self, x, v, nj):  # values: np.array(self.values.values()), vectexs: np.array(self.vectexs.values)
        xi = x
        vi = v
        xj = self.vectexs[nj]
        vj = self.values[nj]
        g = self.globalx
        e1 = self.randomE()
        e2 = self.randomE()
        if vj >= vi:
            return self.alpha2 * e1

        distance = xj - xi
        rij = np.sum(distance * distance)
        term1 = self.beta0 * np.exp((-1) * rij * self.gamma) * (distance)
        term2 = self.alpha * e1
        term3 = self.lamb * e2 * (g - xi)
        return  (term1 + term2 + term3)

    def randomE(self):
        np.random.seed(args.seed)
        e = np.random.rand(2, 3)
        return e

    def adjX(self, x):
        x[0, :] = np.clip(x[0, :], 0, 30)
        x[1, :] = self.sigmoid(x[1, :])

    def randoms(self):
        w = np.zeros((2, 3))
        w[0, :] = np.random.rand(3) * 30
        w[1, :] = np.random.rand(3)
        w[1, :] = self.sigmoid(w[1, :])
        return w

    def sigmoid(self, x):
        outputs = np.exp(x)
        outputs = outputs / np.sum(outputs)
        return outputs
