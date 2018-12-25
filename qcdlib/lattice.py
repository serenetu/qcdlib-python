import numpy as np
from numpy import linalg as nplin
import linalg
import os
import re
import collections
from os.path import normpath
import random
import scipy.integrate as integrate
import math

class Vector:

    def __init__(self, vec):
        assert isinstance(vec, (list, tuple, np.ndarray, Vector)), self.__class__.__name__ + ':: Error:: The Type of Input Should be list, tuple, numpy.ndarray or Vector'
        self.vec = []
        for i in range(len(vec)):
            try:
                self.vec.append(float(vec[i]))
            except ValueError:
                print self.__class__.__name__ + ':: Error:: Input Should Be Numbers'
                exit(1)
        self.dim = len(vec)
        self.vec = np.array(self.vec)
        return

    def __getitem__(self, key):
        return self.vec[key]

    def __str__(self):
        string = self.__class__.__name__ + '('
        for i in range(self.dim):
            string += str(self.vec[i])+', '
        string = string[:-2] + ')'
        return string

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        if self.dim != other.dim:
            return False
        for i in range(self.dim):
            if self.vec[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        if not isinstance(other, Vector):
            return True
        if self.dim != other.dim:
            return True
        for i in range(self.dim):
            if self.vec[i] != other[i]:
                return True
        return False
    
    def __add__(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Addition Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        if isinstance(other, Coordinate):
            res = Coordinate(self.vec+other.vec)
        else:
            res = Vector(self.vec+other.vec)
        return res

    def __sub__(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Addition Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        if isinstance(other, Coordinate):
            res = Coordinate(self.vec-other.vec)
        else:
            res = Vector(self.vec-other.vec)
        return res

    def __neg__(self):
        res = Vector(-self.vec)
        return res

    def __mod__(self, a):
        assert isinstance(a, Vector), self.__class__.__name__ + ':: Error:: Addition Should Be Vector'
        assert self.dim == a.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        res = Vector(np.remainder(self.vec, a.vec))
        return res

    def dot(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Dot Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        dot = np.dot(self.vec, other.vec)
        return dot

    def is_parallel_with(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Addition Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        return linalg.if_parallel(self.vec, other.vec)

    def is_parallel_need_norm(self, other):
        assert hasattr(self , 'norm'), self.__class__.__name__ + ':: Error:: self Need Do get_norm() First'
        assert hasattr(other, 'norm'), self.__class__.__name__ + ':: Error:: other Need Do get_norm() First'
        assert self.dim == other.dim, self.__class__.__name__ + ':: Error:: Dim Do not Match'
        norm_mul = self.norm * other.norm
        dot = self.dot(other)
        if np.absolute(np.absolute(dot)-norm_mul) < norm_mul * 0.01:
            if dot > 0:
                return 1
            elif dot < 0:
                return -1
        else:
            return 0

    def parallel_err(self, other):
        assert hasattr(self , 'norm'), self.__class__.__name__ + ':: Error:: self Need Do get_norm() First'
        assert hasattr(other, 'norm'), self.__class__.__name__ + ':: Error:: other Need Do get_norm() First'
        assert self.dim == other.dim, self.__class__.__name__ + ':: Error:: Dim Do not Match'
        norm_mul = self.norm * other.norm
        dot = self.dot(other)
        if dot >= 0:
            return np.absolute(np.absolute(dot)-norm_mul) / norm_mul
        else:
            return -np.absolute(np.absolute(dot)-norm_mul) / norm_mul


    def get_norm(self):
        self.norm = nplin.norm(self.vec)
        return self.norm

    def to_array(self):
        return self.vec


class Coordinate(Vector):

    def __add__(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Addition Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        if isinstance(other, Coordinate):
            res = Vector(self.vec+other.vec)
        else:
            res = Coordinate(self.vec+other.vec)
        return res

    def __sub__(self, other):
        assert isinstance(other, Vector), self.__class__.__name__ + ':: Error:: Addition Should Between Type Vector'
        assert self.dim==other.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        if isinstance(other, Coordinate):
            res = Vector(self.vec-other.vec)
        else:
            res = Coordinate(self.vec-other.vec)
        return res

    def __neg__(self):
        res = Coordinate(-self.vec)
        return res

    def __mod__(self, a):
        assert isinstance(a, Vector), self.__class__.__name__ + ':: Error:: Addition Should Be Vector'
        assert self.dim == a.dim, self.__class__.__name__ + ':: Error:: Different Dimensions'
        res = Coordinate(np.remainder(self.vec, a.vec))
        return res


def mod(x, size):
    assert 0 < size, 'mod():: Error:: The Lenth of Tatal Size in Each Dimension Should Be Large Than 0'
    m = x - int(x / size) * size
    if (0 <= m) :
        return m
    else:
        return m + size


def smod(x, size):
    assert 0 < size, 'smod():: Error:: The Lenth of Tatal Size in Each Dimension Should Be Large Than 0'
    m = mod(x, size)
    if (m * 2 < size):
        return m
    else:
        return m - size


def relative_coordinate(x, size):
    assert isinstance(x, Vector) and isinstance(size, Vector), 'relative_coordinate():: Error:: x or size Should Be Vector'
    return Coordinate([smod(x[0], size[0]), smod(x[1], size[1]), smod(x[2], size[2]), smod(x[3], size[3])])


class Coor_P:

    def __init__(self, coor = None, p = None):
        self.coor = coor
        self.p = float(p)
        return
    
    def __str__(self):
        res = "Coordinate and Probability:\n"
        res += (self.coor).__str__() + '\n'
        res += str(self.p)
        return res

class Gen_Coor_Under_Dist:

    def __init__(self, func_dist, func_sphere_dist):
        self.func_dist = func_dist
        self.func_sphere_dist = func_sphere_dist
        self.icoor = Coordinate([0., 0., 0., 0.])
        self.jump = 5
        self.sleep = 200
        self.total_lat = Coordinate([0, 0, 0, 0]);
        return

    def setup(self, icoor = Coordinate([0, 0, 0, 0]), jump = 5, sleep = 100, total_lat = Coordinate([30, 30, 30, 30])):
        self.icoor = icoor
        self.jump = jump
        self.sleep = sleep
        self.total_lat = total_lat
        return

    def update(self, old):
        rand = Vector(np.random.randint(-self.jump, self.jump + 1, 4))
        new = old + rand
        return new

    def if_jump(self, old, new):
        pnew_pold = self.func_dist(new.get_norm()) / self.func_dist(old.get_norm())
        if pnew_pold >= 1:
            return True
        else:
            rand = random.random()
            if rand <= pnew_pold:
                return True
            else:
                return False

    def heat(self, start_coor, heat_iter):
        old = start_coor
        for i in range(heat_iter):
            new = self.update(old)
            if new == Coordinate([0,0,0,0]):
                continue
            if self.if_jump(old, new):
                old = new
        return old

    def do(self, icoor, num_coor):
        self.num_coor = num_coor
        coor_p_list = []
        self.stats_dic = {}
        start_coor = self.heat(icoor, self.sleep * 50)
        for i in range(num_coor):
            start_coor = self.heat(start_coor, self.sleep)
            r = start_coor.get_norm()
            p = self.func_dist(r) / self.get_sphere_dist_norm()
            coor_p_list.append(Coor_P(start_coor, p))
            r = int(start_coor.get_norm())
            self.stats_dic[r] = self.stats_dic.get(r, 0) + 1
        return coor_p_list

    def do_simple(self, num_coor, total_lat):
        self.total_lat = total_lat
        self.num_coor = num_coor
        coor_p_list = []
        self.stats_dic = {}
        num = 0
        while num < num_coor:
            coor_p = self.get_simple_coor()
            coor_p_list.append(coor_p)
            r = int(coor_p.coor.get_norm())
            self.stats_dic[r] = self.stats_dic.get(r, 0) + 1
            num += 1
        return coor_p_list

    def show_stats(self, low = 0, high = 50):
        line_r = range(low, high+1)
        line_num = []
        line_p   = []
        line_exp = []
        for r in line_r:
            line_num.append(self.stats_dic.get(r, 0))
            line_p.append(float(line_num[-1]) / float(self.num_coor))
            line_exp.append((integrate.quad(self.func_sphere_dist, r, r+1)[0])/(self.get_sphere_dist_norm()))
        res = "Coor Stats:\n"

        res += "r:".ljust(15)
        for r in line_r:
            res += "%6d" % (r) + "  "
        res += "\n"

        res += "num of coor:".ljust(15)
        for i in range(len(line_r)):
            res += "%6d" % (line_num[i]) + "  "
        res += "\n"

        res += "proportion:".ljust(15)
        for i in range(len(line_r)):
            res += "%6.4f" % (line_p[i]) + "  "
        res += "\n"

        res += "dist:".ljust(15)
        for i in range(len(line_r)):
            res += "%6.4f" % (line_exp[i]) + "  "
        res += "\n"
        return res

    def stats_to_file(self, fpath, low = 0, high = 50):
        res = self.show_stats(low, high)
        f = open(fpath, 'w')
        f.write(res)
        f.close()
        return

    def random_coor(self):
        rx = np.random.random_integers(self.total_lat[0])
        ry = np.random.random_integers(self.total_lat[1])
        rz = np.random.random_integers(self.total_lat[2])
        rt = np.random.random_integers(self.total_lat[3])
        return Coordinate([rx, ry, rz, rt])

    def get_simple_coor(self):
        while True:
            coor = self.random_coor()
            p = self.func_dist(coor.get_norm()) / self.get_sphere_dist_norm()
            rand = random.random()
            if rand < p:
                coor_p = Coor_P(coor, p)
                break
        return coor_p

    def get_sphere_dist_norm(self):
        try:
            return self.sphere_dist_norm
        except AttributeError:
            self.sphere_dist_norm = integrate.quad(self.func_sphere_dist, 0, np.inf, limit = 1000)[0]
            print 'sphere_dist_norm:', self.sphere_dist_norm
            return self.sphere_dist_norm


def rotate_thetas_from_0001_to(coor):
    assert isinstance(coor, Vector), 'rotate_thetas_from_0001_to():: Error:: coor Should Be Vector'
    vec = coor.vec / np.linalg.norm(coor.vec)
    theta_xy = math.acos(vec[3])
    if theta_xy == 0.:
        return 0., 0., 0.
    theta_xt = math.acos(-vec[2] / math.sin(theta_xy))
        
    '''
    if abs(vec[1]) > 10. ** (-10.):
        cos_theta_zt = vec[1] / math.sin(theta_xy) / math.sin(theta_xt)
    else:
        cos_theta_zt = 0.0
    '''
    cos_theta_zt = vec[1] / math.sin(theta_xy) / math.sin(theta_xt)

    '''
    if abs(vec[0]) > 10. ** (-10.):
        sin_theta_zt = -vec[0] / math.sin(theta_xy) / math.sin(theta_xt)
    else:
        sin_theta_zt = 0.0
    '''
    sin_theta_zt = -vec[0] / math.sin(theta_xy) / math.sin(theta_xt)

    if sin_theta_zt > 0:
        theta_zt = math.acos(cos_theta_zt)
    elif sin_theta_zt < 0:
        theta_zt = -math.acos(cos_theta_zt)
    else:
        theta_zt = 0.0
    return theta_xy, theta_xt, theta_zt


def rotate_from_0001_to(coor):
    theta_xy, theta_xt, theta_zt = rotate_thetas_from_0001_to(coor)
    res = Coordinate(
            np.dot(rotate_4dmatrix_zt(theta_zt),
                np.dot(rotate_4dmatrix_xt(theta_xt),
                    np.dot(rotate_4dmatrix_xy(theta_xy),
                        np.array([0,0,0,1],dtype=float)))))
    return res


def rotate_coor_with_theta(coor, theta_xy, theta_xt, theta_zt):
    res = Coordinate(
            np.dot(rotate_4dmatrix_zt(theta_zt),
                np.dot(rotate_4dmatrix_xt(theta_xt),
                    np.dot(rotate_4dmatrix_xy(theta_xy),
                        np.array(coor.vec,dtype=float)))))
    return res


def rotate_4dmatrix_xy(theta_xy):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,math.cos(theta_xy),-math.sin(theta_xy)],
        [0,0,math.sin(theta_xy),math.cos(theta_xy)]
        ],dtype=float)


def rotate_4dmatrix_xt(theta_xt):
    return np.array([
        [1,0,0,0],
        [0,math.cos(theta_xt),-math.sin(theta_xt),0],
        [0,math.sin(theta_xt),math.cos(theta_xt),0],
        [0,0,0,1]
        ],dtype=float)


def rotate_4dmatrix_zt(theta_zt):
    return np.array([
        [math.cos(theta_zt),-math.sin(theta_zt),0,0],
        [math.sin(theta_zt),math.cos(theta_zt),0,0],
        [0,0,1,0],
        [0,0,0,1]
        ], dtype=float)


def func_dist(m, a, b = 0):
    return lambda x: np.exp(-float(m) * float(x)) / float(x) ** float(a) * heaviside(x - b)

def func_sphere_dist(m, a, b = 0):
    return lambda x: func_dist(m, a, b)(x) * x ** 3. * 2. * np.pi ** 2.

def heaviside(x):
    if x < 0:
        return 0.
    else:
        return 1.


if __name__ == '__main__':
    coor = Coordinate([89, 45, 52, 30])

    theta_xy, theta_xt, theta_zt = rotate_thetas_from_0001_to(coor)
    print theta_xy, theta_xt, theta_zt
    res = rotate_coor_with_theta(Coordinate([0,0,0,30]), theta_xy, theta_xt, theta_zt)
    print res
    res = rotate_coor_with_theta(Coordinate([1,3,2,0]), theta_xy, theta_xt, theta_zt)
    print res
    '''
    coor = Coordinate([4, 3, 0, 5])
    total_lat = Coordinate([30, 30, 30, 30])
    dist = func_dist(0.2, 3, b = 5)
    sphere_dist = func_sphere_dist(0.2, 3, b = 5)
    gen_coor = Gen_Coor_Under_Dist(dist, sphere_dist)
    coor_p_list = gen_coor.do(coor, 4096)
    gen_coor.show_stats(5, 35)
    coor_p_list = gen_coor.do_simple(4096, total_lat)
    gen_coor.show_stats(5, 35)
    #for i in range(len(coor_list)):
        #print coor_list[i]
    #for i in range(len(coor_p_list)):
        #print coor_p_list[i]
    #gen_coor.total_lat = Coordinate([40, 40, 40, 40])
    '''

    '''
    coor = Coordinate([1, 2, '4'])
    coor2 = Vector([-2, -4, '-8'])
    print coor
    print 'norm: ' + str(coor.norm())
    print -coor
    print coor==coor2
    print coor2+coor2
    coor2.norm()
    print coor-coor
    print 'parallel'
    print (coor2).is_parallel_need_norm(coor)
    '''
