# causalPipeline
A collection of causal resources for psychologists and social scientists. 

Includes:

- 'ML_structural_interactions' PC algo with MI estimators and GCIT (needs tensorflow)
- 


# for GPU stuff:

nvidia-docker run -it --init --ipc=host -v /home/matthewvowels/GitHub/Psych_ML/causalPipeline:/tmp/causalPipeline -p 8889:8888 35f8070e9363 /bin/bash

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
localhost:8889/tree‌  <- access this via host



## Instructions for SAM and Docker

This analysis uses SAM[1] which is included in the Causal Discovery Toolbox https://fentechsolutions.github.io/

The toolbox includes about 20 methods which are both python and R based, and which may use GPU.

Therefore, I recommend using a docker container, in particular with the nvidia-docker.

Once docker and nvidia-docker have been successfully installed, and sufficient permissions have been granted, run ```docker pull xxxx``` for the most recent
version of CDT.


You can list the available docker images by running ```docker images```

The (non nvidia-) docker image can be run interactively via

```
docker run -ti -v /folder/to/mount/:/tmp/folder -p 8889:8888 e5c643806d03
```



where the -p flag will enable you to run:

```jupyter notebook --ip 0.0.0.0 --no-browser --allow-root```

whilst accessing the associated jupyter server on the local host at:
```
localhost:8889/ 
```

When you ```exit``` the interactive docker image, you can save the image in its current state by checking the container ID:

```docker ps``` 

and then commit it:

```docker commit <containerID> <new_name>```

If you want to remove/delete an image:
```docker rmi -f <image id> ```


## For GPU stuff:

You can run:
```
nvidia-docker run -it --init --ipc=host -v /folder/to/mount/:/tmp/folder -p 8889:8888 35f8070e9363 /bin/bash
```

## For most up to date methods

Once in the interactive docker image, run:

$ git clone https://github.com/FenTechSolutions/CausalDiscoveryToolbox.git  # Download the package 
$ cd CausalDiscoveryToolbox
$ pip install -r requirements.txt  # Install the requirements
$ python setup.py install develop --user

Note that the causaleffect package is not available throught pip for python 3.7, therefore install manually:
1. git clone the repo from https://github.com/pedemonte96/causaleffect
2. cd to the repo
3. run python3 setup.py install
4. commit changes to docker image

otherwise, an older version of SAM will be installed, which won't have the capacity to learn graphs from mixed cont/disc. data.

Don't forget to save/commit the docker image (following the instructions above)


### REFERENCES

[26] Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle & Lopez-Paz, Dav