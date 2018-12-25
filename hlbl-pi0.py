import sys
import numpy as np
import qcdlib.op_folder as op_folder
import qcdlib.lattice as lat


def log(func):
    def wrapper(*args, **kw):
        printinfo('Begin %s():' % func.__name__)
        res = func(*args, **kw)
        printinfo('End %s():' % func.__name__)
        return res
    return wrapper

class Two_Folders_One_Direction:

    def __init__(self, folder1, folder2, direction):
        self.folder1   = folder1
        self.folder2   = folder2
        self.direction = direction
        return

    def __str__(self):
        res = 'Two Folders One Direction Info:\n'
        res += 'Folder1: '+str(self.folder1)+'\n'
        res += 'Folder2: '+str(self.folder2)+'\n'
        res += 'Direction:'+str(self.direction)
        return res

    def swap(self, other):
        other.folder1 = self.folder2
        other.folder2 = self.folder1
        other.direction = -self.direction
        return other

class Vec_Probability_Same_Direction_In_Two_Configs:

    def __init__(self, direc, probability, folder1_config1, folder2_config1, folder1_config2, folder2_config2):
        self.direction = direc
        self.probability = probability
        self.folder1_config1 = folder1_config1
        self.folder2_config1 = folder2_config1
        self.folder1_config2 = folder1_config2
        self.folder2_config2 = folder2_config2
        return

    def __str__(self):
        res = 'Vec Probability Same Direction In Two Configs:\n'
        res += 'Direction: '+str(self.direction)+'\n'
        res += 'Probability: '+str(self.probability)+'\n'
        res += 'Config1:\n'
        res += self.folder1_config1+'\n'
        res += self.folder2_config1+'\n'
        res += 'Config2:\n'
        res += self.folder1_config2+'\n'
        res += self.folder2_config2+'\n'
        return res

    def write_to_file(self, fpath):
        f = open(fpath, 'a')
        f.write(str(self.direction)+'\n')
        f.write(str(self.probability)+'\n')
        f.write(self.folder1_config1+'\n')
        f.write(self.folder2_config1+'\n')
        f.write(self.folder1_config2+'\n')
        f.write(self.folder2_config2+'\n')
        f.close()
        #print 'Write To File: ' + fpath
        #print self.__str__()
        return

class Same_Direction_In_Two_Configs:

    def __init__(self, direc, folder1_config1, folder2_config1, folder1_config2, folder2_config2):
        self.direction = direc
        self.folder1_config1 = folder1_config1
        self.folder2_config1 = folder2_config1
        self.folder1_config2 = folder1_config2
        self.folder2_config2 = folder2_config2
        return

    def __str__(self):
        res = 'Same Direction In Two Configs:\n'
        res += 'Direction: '+str(self.direction)+'\n'
        res += 'Config1:\n'
        res += self.folder1_config1+'\n'
        res += self.folder2_config1+'\n'
        res += 'Config2:\n'
        res += self.folder1_config2+'\n'
        res += self.folder2_config2+'\n'
        return res

    def write_to_file(self, fpath):
        f = open(fpath, 'a')
        f.write(str(self.direction)+'\n')
        f.write(self.folder1_config1+'\n')
        f.write(self.folder2_config1+'\n')
        f.write(self.folder1_config2+'\n')
        f.write(self.folder2_config2+'\n')
        f.close()
        print 'Write To File: ' + fpath
        print self.__str__()
        return

class One_Folder_One_Direction:

    def __init__(self, folder, direction):
        self.folder   = folder
        self.direction = direction
        return

def directions_list_in_one_config(prop_top_folder, total_site,
        conditions={}, minimum_norm = 0, maximum_norm = sys.maxint):
    '''
    return: a list of Two_Folders_One_Direction decoded from one configuration
    '''
    direc_list = []
    prop_info_list = op_folder.decode_all_luchang_prop_in_folder(prop_top_folder, conditions)
    num_prop = len(prop_info_list)

    for prop1_i in range(num_prop):
        prop_folder1 = prop_info_list[prop1_i].apath
        coor1 = prop_info_list[prop1_i].xg

        for prop2_i in range(prop1_i+1, num_prop):
            prop_folder2 = prop_info_list[prop2_i].apath
            coor2 = prop_info_list[prop2_i].xg
            vec = lat.relative_coordinate(coor2-coor1, lat.Coordinate(total_site))
            vecnorm = vec.get_norm()
            if vecnorm >= minimum_norm and vecnorm <= maximum_norm:
                info = Two_Folders_One_Direction(prop_folder1, prop_folder2, vec)
                direc_list.append(info)

    printinfo(str(len(direc_list)) + ' Directions in Folder:')
    printinfo(prop_top_folder)
    printinfo('With Conditions: ' + str(conditions))
    printinfo('Minimun Norm: ' + str(minimum_norm))
    printinfo('Maximun Norm: ' + str(maximum_norm))
    return direc_list

@log
def find_same_directions_in_two_configs(direc_list1, direc_list2):
    '''
    return: a list of Same_Direction_In_Two_Configs
    '''
    info = []
    len1 = len(direc_list1)
    len2 = len(direc_list2)

    persent = len1/100
    
    printinfo(str(len1) + ' Directions in Config 1')
    printinfo(str(len2) + ' Directions in Config 2')
    printinfo(str(len1*len2) + ' Pairs Need to Be Compared')

    # Find Norm
    for i in range(len1):
        if not hasattr(direc_list1[i].direction, 'norm'):
            direc_list1[i].direction.get_norm()
    for i in range(len2):
        if not hasattr(direc_list2[i].direction, 'norm'):
            direc_list2[i].direction.get_norm()

    for i1 in range(len1):
        # print progess
        if i1 % persent == 0:
            printinfo('Found Same Directions: {0}, Progress: {1} / 100'.format(len(info), int(i1/persent)))

        direc1 = direc_list1[i1].direction
        folder1_config1 = direc_list1[i1].folder1
        folder2_config1 = direc_list1[i1].folder2

        for i2 in range(len2):
            direc2 = direc_list2[i2].direction
            folder1_config2 = direc_list2[i2].folder1
            folder2_config2 = direc_list2[i2].folder2

            is_parallel = direc1.is_parallel_need_norm(direc2)
            if is_parallel == 1:
                info.append(Same_Direction_In_Two_Configs(direc1,
                    folder1_config1, folder2_config1,
                    folder1_config2, folder2_config2))
            elif is_parallel == -1:
                info.append(Same_Direction_In_Two_Configs(direc1,
                    folder1_config1, folder2_config1,
                    folder2_config2, folder1_config2))
    printinfo('Found All Same Directions: %d' % len(info))
    return info

@log
def match_veclist_in_two_direclist(vec_p_list, direc_list1, direc_list2):
    len1 = len(direc_list1)
    len2 = len(direc_list2)
    lenv = len(vec_p_list)
    ten = lenv/10
    err = 0.005

    def add_parallel(parallel_list, vec, direc_list, err = 0.005):
        for direc in direc_list:
            parallel_err = direc.direction.parallel_err(vec)
            if parallel_err > 0 and parallel_err < err:
                parallel_list.append(direc)
            elif parallel_err < 0 and parallel_err > -err:
                swap = Two_Folders_One_Direction(direc.folder2, direc.folder1, -direc.direction)
                parallel_list.append(swap)
        return

    # Find Norm
    for i in range(len1):
        if not hasattr(direc_list1[i].direction, 'norm'):
            direc_list1[i].direction.get_norm()
    for i in range(len2):
        if not hasattr(direc_list2[i].direction, 'norm'):
            direc_list2[i].direction.get_norm()
    for i in range(lenv):
        if not hasattr(vec_p_list[i].coor, 'norm'):
            vec_p_list[i].coor.get_norm()
    
    info = []
    for i in range(lenv):
        if i % ten == 0:
            printinfo('Progress: {0} / 100'.format(i / ten * 10))
        vec = vec_p_list[i].coor
        probability = vec_p_list[i].p
        # folder 1
        parallel1_list = []
        add_parallel(parallel1_list, vec, direc_list1, err)
        tryerr = err
        while parallel1_list == []:
            tryerr *= 2
            printinfo("Warning:: No Find Match In Folder1")
            printinfo("Try parallel err:", tryerr)
            add_parallel(parallel1_list, vec, direc_list1, err= tryerr)

        # folder 2
        parallel2_list = []
        add_parallel(parallel2_list, vec, direc_list2, err)
        tryerr = err
        while parallel2_list == []:
            tryerr *= 2
            printinfo("Warning:: No Find Match In Folder2")
            printinfo("Try parallel err:", tryerr)
            add_parallel(parallel2_list, vec, direc_list2, err = tryerr)

        if parallel2_list != [] and parallel1_list != []:
            # choice
            direc1 = np.random.choice(parallel1_list)
            direc2 = np.random.choice(parallel2_list)
            info.append(Vec_Probability_Same_Direction_In_Two_Configs(vec, probability, direc1.folder1, direc1.folder2, direc2.folder1, direc2.folder2))
    return info

def exp_dist(m, a, r_left = 0, r_right = np.inf):
    return lambda x: heaviside(x - r_left) * np.exp(-float(m) * float(x)) / float(x) ** float(a) * heaviside_(x - r_right)

def uni_then_exp_dist(m, a, r_mid, r_left = 0, r_right = np.inf):
    func = exp_dist(m, a, r_left, r_right)
    h = func(r_mid)
    return lambda x: heaviside(x - r_left) * h * heaviside_(x - r_mid) + heaviside(x - r_mid) * func(x) * heaviside_(x - r_right)

def from_distx_to_distr(f):
    return lambda x: f(x) * x ** 3. * 2. * np.pi ** 2.

def from_distr_to_distx(f):
    return lambda x: f(x) / (x ** 3. * 2. * np.pi ** 2.)

def heaviside(x):
    if x <= 0:
        return 0
    else:
        return 1

def heaviside_(x):
    if x > 0:
        return 0
    else:
        return 1
    
def find_pairs_in_two_configs(
        direc_list1, direc_list2, 
        dist_of_x, dist_of_r, 
        num_pairs, 
        pairs_fname,
        show_stats_min = 5,
        show_stats_max = 60
        ):

    assert len(direc_list1) > 1000 and len(direc_list2) > 1000, 'Not Enough Directions'

    # get random vector
    i_coor = lat.Coordinate(np.random.choice(range(5, 15) + range(-5, -15), 4))
    gen_coor = lat.Gen_Coor_Under_Dist(dist_of_x, dist_of_r)
    coor_p_list = gen_coor.do(i_coor, num_pairs)
    printinfo(gen_coor.show_stats(show_stats_min, show_stats_max))

    # find same directions with vec
    same_directions_in_two_configs_list = match_veclist_in_two_direclist(coor_p_list, direc_list1, direc_list2)

    f = open(pairs_fname, 'w')
    f.close()
    for same_direction in same_directions_in_two_configs_list:
        same_direction.write_to_file(pairs_fname)
    return


def find_pairs_in_lattice(
        num_pairs, 
        dist_of_x,
        dist_of_r,
        pairs_fname,
        r_pion_to_gamma_range,
        show_stats_min = 30,
        show_stats_max = 120
        ):

    # get random vector
    i_coor = lat.Coordinate(np.random.choice(range(5, 15) + range(-5, -15), 4))
    gen_coor = lat.Gen_Coor_Under_Dist(dist_of_x, dist_of_r)
    coor_p_list = gen_coor.do(i_coor, num_pairs)
    printinfo(gen_coor.show_stats(show_stats_min, show_stats_max))
    f = open(pairs_fname, 'w')
    x = lat.Coordinate([0,0,0,0])
    zp = lat.Coordinate([0,0,0,0])
    for coor_p in coor_p_list:
        r_y_large = coor_p.coor.get_norm()
        y_large = coor_p.coor
        y_large_array = y_large.to_array()

        # z in small lattice
        r_pion_to_gamma = np.random.uniform(r_pion_to_gamma_range[0], r_pion_to_gamma_range[1])
        z_array = np.around(y_large_array * r_pion_to_gamma / r_y_large)
        z = lat.Coordinate(z_array)

        # y in small lattice
        r_pion_to_gamma = np.random.uniform(r_pion_to_gamma_range[0], r_pion_to_gamma_range[1])
        y_array = np.around(y_large_array * r_pion_to_gamma / r_y_large)
        y = lat.Coordinate(y_array)

        f.write(str(y_large) + '\n')
        f.write(str(coor_p.p) + '\n')
        f.write(str(x) + '\n')
        f.write(str(z) + '\n')
        f.write(str(zp) + '\n')
        f.write(str(y) + '\n')
    f.close()
    return


def find_y_and_rotation_angles(
        num_pairs, 
        dist_of_x,
        dist_of_r,
        pairs_fname,
        show_stats_min = 30,
        show_stats_max = 120
        ):

    # get random vector
    i_coor = lat.Coordinate(np.random.choice(range(5, 15) + range(-5, -15), 4))
    gen_coor = lat.Gen_Coor_Under_Dist(dist_of_x, dist_of_r)
    coor_p_list = gen_coor.do(i_coor, num_pairs)
    printinfo(gen_coor.show_stats(show_stats_min, show_stats_max))
    f = open(pairs_fname, 'w')
    for coor_p in coor_p_list:
        y = coor_p.coor
        theta_xy, theta_xt, theta_zt = lat.rotate_thetas_from_0001_to(y)

        f.write(str(y) + '\n')
        f.write(str(coor_p.p) + '\n')
        f.write(str(theta_xy) + '\n')
        f.write(str(theta_xt) + '\n')
        f.write(str(theta_zt) + '\n')
    f.close()
    return


class PrintInfo:
    def __init__(self):
        self.info = ''
        return

    def __call__(self, *string):
        res = ''
        for i in string:
            res += str(i) + ' '
        res = res[:-1]
        print res
        self.info += res + '\n'
        return

    def save_to_file(self, fpath):
        f = open(fpath, 'w')
        f.write(self.info)
        f.close()
        self.info = ''
        return
        
if __name__ == '__main__':
    printinfo = PrintInfo()

    '''
    # setup distribution
    num_pairs = 512
    m = 0
    a = 2
    left = 5
    mid = 40
    right = 60
    distr = uni_then_exp_dist(m, a, mid, left, right)
    distx = from_distr_to_distx(distr)
    printinfo('distr: m = {0}, a = {1}, r_left = {2}, r_mid = {3}, r_right = {4}'.format(m, a, left, mid, right))

    out_fname = 'y_and_rotation_angles_distr:m={0},a={1},r_left={2},r_mid={3},r_right={4},npairs:{5}'.format(m,a,left,mid,right,num_pairs)
    out_path = './hlbl-pi0.out/' + out_fname
    find_y_and_rotation_angles(
            num_pairs, 
            distx, distr, 
            out_path, 
            show_stats_min=left,
            show_stats_max=right)

    printinfo.save_to_file(out_path + '.info')
    '''

    printinfo = PrintInfo()

    # setup distribution
    r_pion_to_gamma_range = [30, 31]
    num_pairs = 1024
    m = 0
    a = 2
    left = 5
    mid = 40
    right = 60
    distr = uni_then_exp_dist(m, a, mid, left, right)
    distx = from_distr_to_distx(distr)
    printinfo('distr: m = {0}, a = {1}, r_left = {2}, r_mid = {3}, r_right = {4}'.format(m, a, left, mid, right))

    out_fname = 'distr:m={0},a={1},r_left={2},r_mid={3},r_right={4},npairs:{5},r_pion_to_gamma:{6}'.format(m,a,left,mid,right,num_pairs,r_pion_to_gamma_range[0])
    out_path = './hlbl-pi0.out/' + out_fname
    find_pairs_in_lattice(
            num_pairs, 
            distx, distr, 
            out_path, 
            r_pion_to_gamma_range,
            show_stats_min=left,
            show_stats_max=right)

    printinfo.save_to_file(out_path + '.info')

    '''
    printinfo = PrintInfo()
    # setup distribution
    small_lat_total = [24, 24, 24, 64]
    m = 0
    a = 2
    left = 5
    mid = 40
    right = 60
    distr = uni_then_exp_dist(m, a, mid, left, right)
    distx = from_distr_to_distx(distr)
    printinfo('distr: m = {0}, a = {1}, r_left = {2}, r_mid = {3}, r_right = {4}'.format(m, a, left, mid, right))

    conditions = {'type': 0, 'accuracy': 0}
    min_norm = 20
    max_norm = 21
    num_pairs = 512
    printinfo('conditions:', conditions)
    printinfo('minimum_norm', min_norm)
    printinfo('maximum_norm', max_norm)
    printinfo('num_pairs', num_pairs)
    printinfo()

    PROP_TOP_FOLDER1 = "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp ; results=2280/huge-data/prop-point-src"
    PROP_TOP_FOLDER2 = "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp ; results=2240/huge-data/prop-point-src"

    # First Config
    direc_list1 = directions_list_in_one_config(
            PROP_TOP_FOLDER1, 
            small_lat_total,
            conditions, 
            min_norm, 
            max_norm
            )

    # Second Config
    direc_list2 = directions_list_in_one_config(
            PROP_TOP_FOLDER2, 
            small_lat_total,
            conditions, 
            min_norm, 
            max_norm
            )
    
    out_fname = '{0}D_config={1}_config={2}_distr:m={3},a={4},r_left={5},r_mid={6},r_right={7},npairs:{8},r_pion_to_gamma:{9}'.format(24, 2280, 2240, m, a, left, mid, right, num_pairs, min_norm)
    out_path = './hlbl-pi0.out/' + out_fname

    find_pairs_in_two_configs(
            direc_list1, 
            direc_list2, 
            distx, distr,
            num_pairs, 
            out_path,
            show_stats_min = left,
            show_stats_max = right
            )

    printinfo.save_to_file(out_path + '.info')
    '''

