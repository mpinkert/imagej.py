"""
wrapper for imagej and python integration using Pyjnius

"""

__version__ = '0.1.0'
__author__ = 'Yang Liu'

import subprocess
import os
import sys
import scandir
import time


def setenv(k, v):
    """set up an general environment variable

    Args:
        k(string): Environment name
        v(string): Environment value
    """

    os.environ[k] = v


def getenv(k):
    """print the enviroment 

    Args:
        k(string): Enviroment name
    """
    print(os.getenv(k))


def set_conda_env(conda_env_path):
    """set up an conda environment

    Args:
        conda_env_path(string): System path for conda
    """

    setenv('CONDA_PREFIX', conda_env_path)


def set_java_env(java_home_path):
    """set up an Java environment

    Args:
        java_home_path(string): System path for java
    """

    setenv('JAVA_HOME', java_home_path)


def set_pyjnius_env(pyjnius_dir):
    """set up an pyjnius environment

    Args:
        conda_env(string): System path for conda

    return: None if conda_env is not set
    """

    if pyjnius_dir is None:
        print("pyjnius directory is not correct")
        return
    else:
        setenv('PYJNIUS_JAR', pyjnius_dir)


def set_ij_env(ij_dir):
    """make a list of all the required jar file

    Args:
        ij_dir(string): System path for Fiji.app

    return:
        classpath(string): list of required jars
        num_jar(int): number of jars added
    """

    jars = []
    for root, dirs, files in os.walk(ij_dir + '/jars'):
        for each_file in files:
            if each_file.endswith('.jar') and \
                    'imagej-legacy' not in each_file and \
                    'ij1-patcher' not in each_file and \
                    'ij-1' not in each_file:
                jars.append(root + '/' + each_file)
    classpath = ":".join(jars)
    num_jars = len(jars)
    return classpath, num_jars


def set_imglyb_env(imglyb_dir):
    """set up the variable path for imglyb

    Args:
        classpath(string): local path to the imglyb jar
    """

    if imglyb_dir is None:
        print("classpath entered is not correct")
        return
    else:
        setenv('IMGLYB_JAR', imglyb_dir)
        return


def verify_java_env():
    """make sure the java env is correct

    """

    if os.getenv('JAVA_HOME') is None:
        print('Java Environment is not set correctly, \
                please set Java Environment by using set_java_env(your_local_path_to_java)')
        return
    else:
        java_home = os.getenv('JAVA_HOME')
        if os.path.isfile(java_home + '/bin/java'):
            print('Java environment: ' + os.getenv('JAVA_HOME'))
            return
        else:
            print('Java Environment is not set correctly, \
                            please set Java Environment by execute the top block')
            return


# TODO: make this work with pypi and more
def verify_conda_env():
    """make sure the conda env is correct

    return: conda_env(string): if correct, return conda environment variable
    """

    conda_env = os.getenv('CONDA_PREFIX')
    if conda_env is None:
        print('Conda environment is not set, \
                please execute the top block')
    else:
        try:
            subprocess.check_output([conda_env + '/bin/conda', '--version'])
        except OSError:
            print('Conda Environment is not set correctly,\
                    please set Conda Environment by execute the top block')
            return None
        print('Conda environment: ' + conda_env)
        return conda_env


def quiet_init(ij_dir):
    """quietly setup the whole environment and run checks

    Args: ij_dir(String): System path for Fiji.app

    """

    verify_java_env()
    configure_path()
    # ImageJ
    classpath, num_jars = set_ij_env(ij_dir)

    print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")


def help():
    """print the instruction for using imagej module

    """

    print(("Please set the environment variables first:\n" 
           "1. Java:       set_java_env('your local java path')\n"
           "2. Fiji.app:   ij_dir = 'your local fiji.app path'\n"
           "Then call quiet_init(ij_dir)"))

def error_message(error):
    print ( error + "can not be found, it might not be correctly installed.")
    print ("if you believe it is correctly install, you can set up the path manually by calling")
    print ("'set_"+ error + "_env(your_path_to_" + error + ".jar)'")


def conda_path_check(p, checked):
    split_list = p.split("/")
    index = 0
    for level in split_list:
        index += 1
        if level == "envs":
            break

    basedir = "/".join(split_list[0:index + 1])
    if basedir  in checked:
        return None, None

    test_path_imglyb = basedir + "/share/imglyb/"
    test_path_pyjnius = basedir + "/share/pyjnius/"

    if os.path.isdir(test_path_imglyb):
        for file in os.listdir(test_path_imglyb):
            if ".jar" in file:
                imglyb_jar = test_path_imglyb + file

    if os.path.isdir(test_path_pyjnius):
        for file in os.listdir(test_path_pyjnius):
            if ".jar" in file:
                pyjnius_jar = test_path_pyjnius + file

    checked.append(basedir)
    return imglyb_jar, pyjnius_jar

def pypi_path_check(p, checked):
    split_list = p.split("/")
    index = 0
    for level in split_list:
        index += 1
        if level == "site_packages":
            break

    basedir = "/".join(split_list[0:index + 1])
    if basedir in checked:
        return None, None

    test_path_imglyb = basedir  + "/imglyb/"
    test_path_pyjnius = basedir  + "pyjnius/"

    if os.path.isdir(test_path_imglyb):
        for file in os.listdir(test_path_imglyb):
            if ".jar" in file:
                imglyb_jar = test_path_imglyb + file

    if os.path.isdir(test_path_pyjnius):
        for file in os.listdir(test_path_pyjnius):
            if ".jar" in file:
                pyjnius_jar = test_path_pyjnius + file

    checked.append(basedir)
    return imglyb_jar, pyjnius_jar


def configure_path():
    paths = sys.path

    imglyb_path = None
    pyjnius_path = None

    checked = []
    index = 0

    while index < len(paths) and (imglyb_path is None or pyjnius_path is None):
        p = paths[index]
        if "envs" in p:
            imglyb_path, pyjnius_path = conda_path_check(p, checked)
        elif "site_packages" in p:
            pypi_path_check(p, checked)
        index += 1

    print("imglyb: " + imglyb_path)
    print("pyjnius: " + pyjnius_path)

    if imglyb_path is None:
        error_message("imglyb")
    if pyjnius_path is None:
        error_message("pyjnius")
    return