+++
title = "A quick start guide for Linux users at Intec"
subtitle = ""
date = 2017-06-06
draft = false

authors = ["admin"]

tags = ["linux", "UGent"]
categories = []

math = true

[image]
  caption = ""
  focal_point = "Smart"

+++

* [What does this guide offer?](#what-does-this-guide-offer?)
* [Software packages and OS](#software-packages-and-os)
  * [Install a linux OS](#install-a-linux-os)
  * [Openvpn](#openvpn)
  * [Matlab](#matlab)
* [Miscellaneous](#miscellaneous)
  * [Configuring printer](#configuring-the-printer)
  * [Before using apollo webpage](#before-using-apollo-webpage)
  * [Mounting Intec file share](#mounting-intec-file-share)

***

## What does this guide offer?

This report aims as a quick guide for setting up linux machines in Intec, UGent. It may not contain help on everything, but the goal is to keep adding things as we encounter and solve them. The instructions are for fedora 25, unless otherwise mentioned. For other linux distributions, you might need to appropriately change the command-line instructions.

You might encounter missing libraries while installing various packages. This report may not give solutions to such errors. It is suggested to find the right library for the OS you use and solve the dependencies yourself. Some sections might contain a subsubsection on possible issues and fixes.

## Software packages and OS

### Install a linux OS
The Intec machines typically come with pre-installed windows 10 (or latest release). Installing linux with dual boot option is preferred on these machines. Choose your linux OS and follow the install instructions provided by the respective OS manuals to install with dual boot. This report does not include the installation instructions for dual boot linux with windows.

Also do not forget to enable GRUB bootloader or edit the windows bootloader [easybcd](http://www.techspot.com/downloads/3112-easybcd.html) is a good software to do this, but should be done from windows) so that we can see the dual boot menu on bootup.

### Openvpn
We also need to set-up an VPN connection to be a part of the Intec network. For this, you will need an intec account and password (it is not the ugent account). You will receive an email about this or you might need to request the Tech people at Intec. Once you have the account, you can set up the VPN. You will receive an email with instructions for setting up VPN for windows. The mail will also have a ```.rar``` file which contains a ```.ovpn``` and a ```.crt``` file.

For linux, download the ```.tar.gz``` file from this [page](https://openvpn.net/index.php/open-source/downloads.html). Open a terminal and go to the folder where openvpn is downloaded. You might need to use these commands with ```sudo```.

  ```
  $ tar xzvf openvpn<version>.tar.gz        
  $ cd openvpn<version>      
  $ ./configure      
  $ make     
  $ make install     
  ```    

You might need to install ```make``` or ```cmake``` for your linux OS. Also there could be some missing libraries. Install them and do ```./configure``` until there are no errors, then do ```make``` followed by ```make install```. This post may be updated with instructions for installing the missing libraries. 

Once the openvpn is installed, you can set up the VPN using the ``` .ovpn``` and a ``` .crt``` files. From the terminal, go to the folder where these files are located. Then execute :    
  ```
  $ sudo openvpn --config <.ovpn file> --ca <.crt file>
  ```
This will ask your Intec username and password for authentication (Don't give ugent credentials !). This should give an output similar to:   
  ```
  [sudo] password for <user>:      
  OpenVPN 2.4.0 x86_64-redhat-linux-gnu [SSL (OpenSSL)] [LZO] [LZ4] [EPOLL] [MH/PKTINFO] [AEAD] built on <date>           
  library versions: OpenSSL 1.0.2k-fips  26 Jan 2017, LZO 2.08            
  Enter Auth Username:<Enter Intec username>           
  Enter Auth Password:<Enter Intec password>            
  TCP/UDP: Preserving recently used remote address: [AF_INET]157.193.214.20:443          
  Attempting to establish TCP connection with [AF_INET]157.193.214.20:443 [nonblock]           
  TCP connection established with [AF_INET]157.193.214.20:443              
  TCP_CLIENT link local: (not bound)             
  TCP_CLIENT link remote: [AF_INET]157.193.214.20:443           
  TUN/TAP device tun0 opened           
  do_ifconfig, tt->did_ifconfig_ipv6_setup=0                       
  /usr/sbin/ifconfig tun0 192.168.126.4 netmask 255.255.255.0 mtu 1500 broadcast 192.168.126.255            
  Initialization Sequence Completed           
  ```

You will need to leave that terminal open to maintain the VPN connection. Also, check the terminal if the VPN is active if you encounter internet or printing problems.

> You can ignore warnings related to the cipher length.


### Matlab
The intec file share (ref. [this]()  Section) provides some useful software packages. However, the matlab folder does not contain the complete installation files for linux. You can obtain a copy of linux installation files from me or Sarah (we have matlab 2016a). Copy the contents to ```/home/<username>/matlab2016b/linux```. Copying it to ```/usr/local``` is not recommended as it may not work. And even if you run as ``` su```, it may not work. I don't know why! So copy to ``` /home/<username>/matlab2016b/linux``` and don't run as ``` sudo``` or ``` su```.

The fileshare contains the key and the license files for matlab. Copy those two from the fileshare to some folder. Find the key for matlab 2016a from the file ``` INTEC read key```. From the terminal (Do not run as ``` sudo```, it did not work for me!):    
  ```
  $ cd /home/<username>/matlab2016b
  $ sudo chmod -R 777 linux
  $ cd linux 
  $ ./install
  ```
This should launch a GUI and follow the instructions. Choose install without internet option and paste the key for matlab2016a. Then for the installation folder choose a folder in your home directory, say ``` /home/<username>/matlab2016b\_installation ```. It also asks for the license file, browse for the ``` license.lic``` obtained from the Intec fileshare. Choose which packages you need and then install. Then wait for the installation to finish. 

Add the matlab installation bin folder to the path in your ``` bashrc```. You can open the ``` bashrc``` file using

  ```
  $ gedit ~/.bashrc &
  ```   
This will launch the gedit GUI and you can add the following line to the ``` bashrc``` file and save it.       

  ```
  export PATH=/home/<username>/matlab2016b_installation/bin/:$PATH
  ```

Open a new terminal and Matlab can be launched as:         
  
  ```
  $ matlab &
  ```

## Miscellaneous
### Configuring the printer
Follow the instructions in this [page](https://ibcnintra.intec.ugent.be/index.php/IGent_printers).

> If you encounter errors with printing (eg., printer not responding, connecting to printer, printer may not be connected, etc.), check the openvpn terminal and see if the VPN is still active. If not, relaunch openvpn (refer [openvpn](#openvpn) Section).

### Before using apollo webpage
The [athena](https://athena.ugent.be/) webpage gives access to several online services including apollo, MS office and matlab. But these services are hosted on a Citrix server and you need to install the same to be able to use these services. The instructions can be found in the [citrix information page](http://helpdesk.ugent.be/athena/en/ica.php).

### Mounting Intec file share or Robspear
To download the software packages provided by Intec, the installation files are available from a Samba server. In order to view and download the files, this server should be mounted on the linux machine first. There is also a robspear server for saving huge data files. The following example is to mount the robspear server to your linux folder.

  * create a mount directory. This is where the server will be mounted to. Here the server will be mounted to ``` /media/robspear``` folder. You can choose any other location for mounting the folder.        
    
    ```
    $ sudo mkdir -p /media/robspear
    ```
  * Mount the server to the created location             
    
    ```
    $ sudo mount -t cifs //acoustserv.intec.ugent.be/robspear /media/robspear -o username=<your-intec-username>
    ```          
    This will ask for your intec password and the mounting is done!
