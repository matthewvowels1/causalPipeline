# causalPipeline
A collection of causal resources for psychologists and social scientists accompanying the paper:

'A Causal Research Pipeline and Tutorial for Psychologists and Social Scientists' (blinded for peer review).

# Tutorial File

Includes a tutorial ipynb file  ```causal_pipeline_tutorial.ipynb``` which steps through the analysis.

Utilises data acquired with permission from https://www.icpsr.umich.edu/web/ICPSR/studies/37404/summary


# Installation

## 1. Installing the Docker Image (recommended to use SAM causal discovery)

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

After you ```exit``` the interactive docker image, you can save the image in its current state by first checking the container ID:

```docker ps``` 

and then commit it:

```docker commit <containerID> <new_name>```

If you want to remove/delete an image:
```docker rmi -f <image id> ```


#### For GPU stuff:
```
nvidia-docker run -it --init --ipc=host -v /GitHub/Psych_ML/causalPipeline:/tmp/causalPipeline -p 8889:8888 35f8070e9363 /bin/bash

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
Then, access this via host with a browser: ```localhost:8889/tree‌```  

## 2. Install Packages


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

### Installing R packages

Pyhal uses rpy2==3.2.1 to call functions in R (rpy2 can be installed with ```python3 -mpip install rpy2==3.2.1```).

There is a list of packages needed in the rpacks.txt file. These can be installed by running:

```commandline
while IFS=" " read -r package version; 
do 
  R -e "install.packages('"$package"', lib='/usr/local/lib/R/site-library')"; 
done < "rpacks.txt"
```


and 

```commandline
while IFS=" " read -r package version; 
do 
  R -e "install.packages('"$package"', lib='/usr/lib/R/site-library')"; 
done < "rpacks.txt"
```



# (Extra) PC-Algorithm Evaluation Results

In the figures below we show extra results which are not included in the accompanying paper for the well-known PC algorithm Spirtes et al. 2000
The results illustrate the differences in performance (across a range of sample sizes) between a linear, correlation-based conditional independence test, and a nonparametric, k-nearest neighbours conditional mutual information based conditional independence test (Runge et al., 2018). 

We show below the true, 9-node, 9-edge graph for the underlying data generating structure. All exogenous (not shown) variables are Gaussian. 

![alt text](test_graph.png)


In the next figure we show the Structural Hamming Distance, which is a measure of how successful the algorithm was at correctly inferring the graph (smaller the better). 

![alt text](pearsonr_vs_mi_ci_tests_SHD.png)

Finally, below we show the algorithm runtime (in seconds). Firstly, it can be seen that the correlation based measure is notably better at inferring the structure than the mutual information based approach, particularly for small sample sizes.
It can also be seen that the time taken to run the nonparametric version increases linearly with the sample size (over 50 minutes for a sample size of 2000) vastly exceeding the runtime of the correlation approach (which has a run of 0.26 seconds for a sample size of 2000). 
Thus, the price paid for not having to make parametric assumptions is one of both computation time and, for a parametric data generating process, accuracy. 
Of course, one expects that in cases where the parametric assumption does not hold, the advantages of non-parametric conditional independence tests become self-evident.

![alt text](pearsonr_vs_mi_ci_tests_time.png)

### REFERENCES

[1] Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle & Lopez-Paz, Dav