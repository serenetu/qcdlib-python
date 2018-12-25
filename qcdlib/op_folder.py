import random
import numpy as np
import lattice as lat
import os
import re
import collections
from os.path import normpath

class Luchang_Prop_Info:

    def __init__(self, fname=''):
        if fname == '':
            self.fname = None
        else:
            self.fname = fname
        self.xg       = None
        self.ptype    = None
        self.accuracy = None
        return

    def get_info(self, ele):
        if   ele == 'fname'   :
            return self.fname
        elif ele == 'xg'      :
            return self.xg
        elif ele == 'type'    :
            return self.ptype
        elif ele == 'accuracy':
            return self.accuracy
        else:
            return None

    def set_info(self, ele, val):
        if   ele == 'xg'      :
            self.xg       = val
        elif ele == 'type'    :
            self.ptype    = val
        elif ele == 'accuracy':
            self.accuracy = val
        return

    def add_apath_info(self, uppath):
        assert self.fname is not None, self.__class__.__name__+':: Error:: No Folder Name Info (fname)'
        self.apath = os.path.normpath(uppath+'/'+self.fname)
        return

    def __str__(self):
        if self.fname is None and self.xg is None and self.ptype is None and self.accuracy is None:
            return 'No Luchang Prop Info'
        else:
            res = 'Luchang Prop Info:\n'
            res += 'fname: ' + str(self.get_info('fname')) + '\n'
            res += 'xg: ' + str(self.get_info('xg')) + '\n'
            res += 'type: ' + str(self.get_info('type')) + '\n'
            res += 'accuracy: ' + str(self.get_info('accuracy')) + '\n'
            return res


def list_folders_in_folder(folder_path):
    subfolder_list = []
    f_all = os.listdir(folder_path)
    for f in f_all:
        if os.path.isdir(folder_path+'/'+f):
            subfolder_list.append(f)
    return subfolder_list

def decode_luchang_prop_fname(fname):
    info = Luchang_Prop_Info(fname)
    fn = fname.replace(" ", "")
    fn = fn.split(";")
    for fn_ele in fn:
        fn_ele_s = fn_ele.split("=")
        if fn_ele_s[0] == 'xg':
            v = re.sub(r'\(|\)', '', fn_ele_s[1])
            v = v.split(",")
            info.xg = lat.Coordinate(tuple(map(int, v)))
        else:
            info.set_info(fn_ele_s[0], int(fn_ele_s[1]))
    return info

def decode_all_luchang_prop_in_folder(prop_path, conditions={}):
    decode_list = []
    subfolder_list = list_folders_in_folder(prop_path)
    for folder_name in subfolder_list:
        f_info = decode_luchang_prop_fname(folder_name)
        if luchang_prop_match_conditions(f_info, conditions):
            apath = os.path.normpath(prop_path+'/'+folder_name)
            f_info.add_apath_info(prop_path)
            decode_list.append(f_info)
    return decode_list

def luchang_prop_match_conditions(prop_info, conditions):
    for key in conditions:
        if prop_info.get_info(key) != conditions[key]:
            return False
    return True


if __name__ == '__main__':
    #print list_folders_in_folder('/home/tucheng')[0]
    #print list_folders_in_folder('/home/tucheng')[1]
    print decode_luchang_prop_fname("xg=(13,17,9,19) ; type=0 ; accuracy=0")
    prop_info = decode_luchang_prop_fname("xg=(13,17,9,19) ; type=0 ; accuracy=0")

    print luchang_prop_match_conditions(prop_info, {'xg': lat.Coordinate([13, 17, 9, 19])})
    print luchang_prop_match_conditions(prop_info, {'type': 0})
