#!/bin/bash -e

prompt () {
    # prints a yes / no question to the user and returns the answer
    # first argument is the prompt question
    # second argument is the default answer - Y / N
    local default_answer

    # set the default value
    case "${2}" in
        y|Y ) default_answer=1; options="[Y/n]";;
        n|N ) default_answer=0; options="[y/N]";;
        "" ) default_answer=; options="[y/n]";;
        * ) echo "invalid default value"; exit;;
    esac

    while true; do
        # read the user choice
        read -p "${1} ${options} " choice

        # return the choice or the default value if an enter was pressed
        case "${choice}" in
            y|Y ) retval=1; return;;
            n|N ) retval=0; return;;
            "" ) if [ ! -z "${default_answer}" ]; then retval=${default_answer}; return; fi;;
        esac
    done
}

add_to_bashrc () {
    # adds an env variable to the bashrc
    # first argument is the variable name
    # second argument is the variable value

    EXISTS_IN_BASHRC=`awk  '/${2}/{print $1}' ~/.bashrc`
    if [ "${EXISTS_IN_BASHRC}" == "" ]; then
        echo "export ${1}=${2}" >> ~/.bashrc
    fi
}

GET_PREFERENCES_MANUALLY=1

INSTALL_COACH=0
INSTALL_DASHBOARD=0
INSTALL_GYM=0
INSTALL_NEON=0
INSTALL_VIRTUAL_ENVIRONMENT=1

# Get user preferences
TEMP=`getopt -o cpgvrmeNndh \
             --long coach,dashboard,gym,no_virtual_environment,neon,debug,help \
             -- "$@"`
eval set -- "$TEMP"
while true; do
#for i in "$@"
    case ${1} in
        -c|--coach)
            INSTALL_COACH=1
            GET_PREFERENCES_MANUALLY=0
            shift;;
        -p|--dashboard)
            INSTALL_DASHBOARD=1
            GET_PREFERENCES_MANUALLY=0;
            shift;;
        -g|--gym)
            INSTALL_GYM=1
            GET_PREFERENCES_MANUALLY=0;
            shift;;
        -N|--no_virtual_environment)
            INSTALL_VIRTUAL_ENVIRONMENT=0
            GET_PREFERENCES_MANUALLY=0;
            shift;;
        -ne|--neon)
            INSTALL_NEON=1
            GET_PREFERENCES_MANUALLY=0;
            shift;;
        -d|--debug) set -x; shift;;
        -h|--help)
            echo "Available command line arguments:"
            echo ""
            echo "   -c | --coach                  - Install Coach requirements"
            echo "   -p | --dashboard              - Install Dashboard requirements"
            echo "   -g | --gym                    - Install Gym support"
            echo "   -N | --no_virtual_environment - Do not install inside of a virtual environment"
            echo "   -d | --debug                  - Run in debug mode"
            echo "   -h | --help                   - Display this help message"
            echo ""
            exit;;
        --) shift; break;;
        *) break;; # unknown option;;
    esac
done

if [ ${GET_PREFERENCES_MANUALLY} -eq 1 ]; then
    prompt "Install Coach requirements?" Y
    INSTALL_COACH=${retval}

    prompt "Install Dashboard requirements?" Y
    INSTALL_DASHBOARD=${retval}

    prompt "Install Gym support?" Y
    INSTALL_GYM=${retval}

    prompt "Install neon support?" Y
    INSTALL_NEON=${retval}
fi

IN_VIRTUAL_ENV=`python3 -c 'import sys; print("%i" % hasattr(sys, "real_prefix"))'`

# basic installations
sudo -E apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y
#pip3 install --upgrade pip

# if we are not in a virtual environment, we will create one with the appropriate python version and then activate it
# if we are already in a virtual environment,

if [ ${INSTALL_VIRTUAL_ENVIRONMENT} -eq 1 ]; then
    if [ ${IN_VIRTUAL_ENV} -eq 0 ]; then
        sudo -E pip3 install virtualenv
        virtualenv -p python3 coach_env
        . coach_env/bin/activate
    fi
fi

#------------------------------------------------
# From now on we are in a virtual environment
#------------------------------------------------

# get python local and global paths
python_version=python$(python -c "import sys; print (str(sys.version_info[0])+'.'+str(sys.version_info[1]))")
var=( $(which -a $python_version) )
get_python_lib_cmd="from distutils.sysconfig import get_python_lib; print (get_python_lib())"
lib_virtualenv_path=$(python -c "$get_python_lib_cmd")
lib_system_path=$(${var[-1]} -c "$get_python_lib_cmd")

# Boost libraries
sudo -E apt-get install libboost-all-dev -y

# Coach
if [ ${INSTALL_COACH} -eq 1 ]; then
    echo "Installing Coach requirements"
    pip3 install -r ./requirements_coach.txt
fi

# Dashboard
if [ ${INSTALL_DASHBOARD} -eq 1 ]; then
    echo "Installing Dashboard requirements"
    pip3 install -r ./requirements_dashboard.txt
    sudo -E apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev \
    freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev -y

    sudo -E -H pip3 install -U --pre -f \
    https://wxpython.org/Phoenix/snapshot-builds/linux/gtk3/ubuntu-16.04/wxPython-4.0.0a3.dev3059+4a5c5d9-cp35-cp35m-linux_x86_64.whl  wxPython

    # link wxPython Phoenix library into the virtualenv since it is installed with apt-get and not accessible
    libs=( wx )
    for lib in ${libs[@]}
    do
        ln -sf $lib_system_path/$lib $lib_virtualenv_path/$lib
    done
fi

# Gym
if [ ${INSTALL_GYM} -eq 1 ]; then
    echo "Installing Gym support"
    sudo -E apt-get install libav-tools libsdl2-dev swig cmake -y
    pip3 install box2d # for bipedal walker etc.
    pip3 install gym[all]==0.9.4
fi

# NGraph and Neon
if [ ${INSTALL_NEON} -eq 1 ]; then
    echo "Installing neon requirements"

    # MKL
    git clone https://github.com/01org/mkl-dnn.git
    cd mkl-dnn
    cd scripts && ./prepare_mkl.sh && cd ..
    mkdir -p build && cd build && cmake .. && make -j
    sudo make install -j
    cd ../..
    export MKLDNN_ROOT=/usr/local/
    add_to_bashrc MKLDNN_ROOT ${MKLDNN_ROOT}
    export LD_LIBRARY_PATH=$MKLDNN_ROOT/lib:$LD_LIBRARY_PATH
    add_to_bashrc LD_LIBRARY_PATH ${MKLDNN_ROOT}/lib:$LD_LIBRARY_PATH

    # NGraph
    git clone https://github.com/NervanaSystems/ngraph.git
    cd ngraph
    make install -j
    cd ..

    # Neon
    sudo -E apt-get install libhdf5-dev libyaml-dev pkg-config clang virtualenv libcurl4-openssl-dev libopencv-dev libsox-dev -y
    pip3 install nervananeon
fi

if ! [ -x "$(command -v nvidia-smi)" ]; then
    # Intel Optimized TensorFlow
    #pip3 install https://anaconda.org/intel/tensorflow/1.3.0/download/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl
    pip3 install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
else
    # GPU supported TensorFlow
    pip3 install tensorflow-gpu==1.4.1
fi
