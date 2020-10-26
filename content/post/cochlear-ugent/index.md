+++
title = "Tensorflow-GPU in multi-user environment"
subtitle = ""
date = 2019-01-27
draft = false

authors = ["admin"]

tags = ["linux", "UGent"]
categories = []

math = true

[image]
  caption = ""
  focal_point = "Smart"

+++

This post is intended for setting up tensorflow-gpu setup in a multi-user setting. This is written as a guide for GPU users at the WAVES research group, Ghent University, Belgium. But these are also applicable to any linux multi-user environment with GPU-based jobs.

  * [Installing Tensorflow in conda](#installing-tensorflow-in-conda)
    * [conda installation](#conda-installation)
    * [conda tensorflow](#conda-tensorflow)
    * [Testing Tensorflow Installation](#testing-tensorflow-installation)
  * [Admin only](#admin-only)
    * [Installing the cuda compiler and nvidia drivers](#installing-the-cuda-compiler-and-nvidia-drivers)

***

## Installing Tensorflow in conda
### conda installation
Anaconda is a popular python environment among the AI/ML community. The anaconda distribution can be downloaded from [here](https://www.anaconda.com/download/). Follow the instructions [here](https://docs.anaconda.com/anaconda/install/linux/) to properly install it to your user account.

Once you have installed anaconda into your user account, you can create a conda environment using  
  ```
  $ conda create -n <name-of-your-environment>
  ```

Then you can activate that environment using:    
  ```
  $ conda activate <name-of-your-environment>
  ```

Once you are in the environment, you can install whatever python packages you want. Anaconda already comes with numpy,scipy and many other useful python libraries. If you need a specific library, google for *conda install <the-library-you-need>* and find how to install it. But always make sure that you are in the right conda environment before installing the new libraries. You can exit the environment by deactivating it     
  ```
  $ conda deactivate
  ```

### conda tensorflow
Anaconda also offers tensorflow and keras installations among many many other libraries. In order to install it to your environment, follow the steps below:

  1. Activate your conda environment      
  2. Install keras            
     ```
     $ conda install -c conda-forge keras
     ```
  3. Install tensorflow GPU version          
     ```
     $ conda install tensorflow-gpu
     ```    

This should install other libraries that are required by keras and tensorflow. I found that it is better to install keras before installing tensorflow since keras also installs a tensorflow that may not be comaptible with the GPU (I am not 100\% sure about this).

{{% callout note %}}
Instead of using ``` conda install ```, we can also use ``` pip install <the-library-you-need>``` in the same environment for installing libraries. But I recommend using ``` conda```.             
{{% /callout %}}

You can use ``` conda list``` to see all the installed libraries in your environment. ``` conda env list``` will list all the conda environments in your system.

{{% callout warning %}}
A popular python editor called sypder also comes pre-installed with anaconda. But it does not work over remote desktop as the keyboard does not work properly. If somebody finds a workaround, kindly update this.        
{{% /callout %}}

### Testing Tensorflow Installation
You can test whether the tensorflow installation is using the GPU using the following options.
  ```
  $ python -c "import tensorflow as tf; sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))"    
  ```
OR
  ```
  $ python -c "import tensorflow as tf; tf.test.is_gpu_available()"
  ```
If it gives message like ``` Adding visible gpu devices:```, then it means that tensorflow indeed uses the GPU. If it only mentions CPU, then you will need to correct the installation. Often, it is better to install keras first and then install tensorflow, or use ``` conda install tensorflow-gpu``` instead of ``` conda install -c conda-forge tensorflow-gpu```.

# 
***

## Admin only
{{% callout warning %}}
**This section details the GPU installation guidelines. Only an admin should take care of this. Notice that this should be done only when there is a kernel update or when ``` nvidia-smi``` command does not list any GPUs (meaning the system does not see the GPUs anymore).**
{{% /callout %}}

### Installing the cuda compiler and nvidia drivers
These steps are adapted from [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Ignore the ```$``` sign in the beginning of the commands.

  1. Install the kernel headers for the current Ubuntu installation.       
     ```
     $ sudo apt-get install linux-headers-$(uname -r)
     ```            
     For other linux flavors, this step is different (Refer [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions) for other linux distributions).
  2. Download the runfile from the [cuda downloads page](https://developer.nvidia.com/cuda-downloads).
  3. Disable the nouveau driver. The instructions are given in this page. For ubuntu, create a new file ``` /etc/modprobe.d/blacklist-nouveau.conf``` with the following contents:          
     
     ```
     blacklist nouveau  <br>                                        
     options nouveau modeset=0              
     ```
  4. Regenerate the kernel initramfs:  
     ```
     $ sudo update-initramfs -u
     ```
  5. Disable the lightdm service to kill the X server from running.
     ```
     $ sudo service lightdm stop
     ```
     Also kill vncserver sessions (if they exist) (e.g., ``` vncserver -kill :1``` to kill the first vncserver and so on.). Also remove the ``` .X0.lock``` or other ``` .lock``` files present in the ``` /tmp``` folder.
  6. Go to the Downloads folder where the downloaded runfile is stored. Make the file executable:   
     ```
     $ chmod +x cuda<version>.linux.run      
     ```
  7. Install the driver and compiler      
     ```
     $ sudo ./cuda<version>.linux.run --no-opengl-libs
     ```
     **The option  ``` --no-opengl-libs``` is important to avoid the login problems.** You will then be asked the following and the requried responses are provided in bold font.
     1. Accept license agreement? **yes**             
        You can press ``` Ctrl+C``` to skip to the end of the license.   
     2. Install NVIDIA driver? **yes**
     3. Should NVIDIA modify the x-config ? **no**
     4. Install CUDA? **yes**
     5. Path where cuda installations should be put: **choose default or provide a path of your choice**
     6. Install symbolic link? **yes**
     7. Install samples? **yes**
     8. Choose samples location: **choose default or enter your choice**      
     This should install both the cuda compiler and nvidia drivers to the machine.


  8. Perform the post installation actions such as adding the cuda installation to your ``` PATH ``` and ``` LD_LIBRARY_PATH ```. Follow the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions). You can also edit the ``` ~/.bashrc``` file to add modify these variables.      
  9. Try ``` nvcc -V``` to check the nvidia compiler version and ``` nvidia-smi``` to see the GPUs’ status in your machine.
  10. Finally, restart the ``` lightdm``` service.
      ```
      $ sudo service lightdm restart      
      ```
The machine will have the GUI after the ```lightdm``` service is restarted. You will need to launch new vnc sessions in order to use remote desktop.
