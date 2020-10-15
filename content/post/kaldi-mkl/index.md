+++
title = "Installing Kaldi with MKL support without root access"
date = 2019-05-21T13:51:12+02:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["admin"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["kaldi", "mkl"]
categories = []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

Kaldi has recently switched to Intel Math Kernel Libraries (MKL) for linear algebra operations (as of April 2019). However, installing MKL (by running `tools/extras/install_mkl.sh`) requires root access. This post details how kaldi (with MKL) can be installed without root access.

1. Download [Kaldi](https://kaldi-asr.org/doc/install.html)
1. Download the MKL standalone installer from [here](https://software.intel.com/en-us/mkl/choose-download/linux).
    * Extract the contents and launch the installer by running `install.sh`.
    * When asked for the path to install, specify a location where you have write access (e.g., `/home/<username>/intel`)
    * Complete the installation of MKL libraries 
1. Navigate to the kaldi folder `kaldi/tools`
1. Typically the first step is to run `extras/check_dependencies.sh`. This will complain about the missing MKL libraries. This is because the script expects the MKL libraries to be located under `/opt/intel` directory. As of now (May 2019), there is no option to pass the `mkl-root` directory to this script. Therefore we will edit the `extras/check_dependencies.sh` script by changing `/opt/intel/mkl/include/mkl.h` to `/home/<username>/intel/mkl/include/mkl.h`. Then running `extras/check_dependencies.sh` should work fine without any MKL related warnings.
1. Then run `make -j <numcpu>` to install the tools required by kaldi
1. Navigate to the `kaldi/src` folder.
1. Run `./configure` with the `--mkl-root` option.
   ```
   ./configure --shared --mkl-root=/home/<username>/intel/mkl
   ```
1. Then install kaldi using the usual steps
   ```
   make depend -j <numcpu>       
   make -j <numcpu>
   ```

This will install Kaldi with MKL support without requiring any root privileges. 
