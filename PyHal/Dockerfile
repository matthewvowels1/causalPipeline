FROM arm64v8/ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CRAN_MIRROR=https://cloud.r-project.org

RUN \
  apt-get update -qq && \
  apt-get install -y \
                     apt-utils \
                     apt-transport-https \
		     dirmngr \
                     gnupg \
		     libcurl4-openssl-dev \
		     libnlopt-dev \
                     lsb-release && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
  apt-get update -qq && \
  apt-get install -y \
                     aptdaemon \
                     ed \
                     git \
		     mercurial \
		     libcairo-dev \
		     libedit-dev \
		     libxml2-dev \
		     python3 \
		     python3-pip \
		     r-base \
		     r-base-dev \
		     sudo \
             cmake \
		     wget &&\
  rm -rf /var/lib/apt/lists/*

RUN \
  echo "broom\n\
        DBI\n\
        dbplyr\n\
        dplyr\n\
        hexbin\n\
        ggplot2\n\
        lme4\n\
        RSQLite\n\
        hal9001\n\
        tidyr" > rpacks.txt && \
  R -e 'install.packages(sub("(.+)\\\\n","\\1", scan("rpacks.txt", "character")), repos="'"${CRAN_MIRROR}"'", \
     lib="/usr/local/lib/R/site-library")' && \
  R -e 'install.packages(sub("(.+)\\\\n","\\1", scan("rpacks.txt", "character")), repos="'"${CRAN_MIRROR}"'", \
     lib="/usr/lib/R/site-library")' && \
  rm rpacks.txt

RUN \
  python3 -m pip install --no-cache-dir --upgrade pip && \
  python3 -m pip install --no-cache-dir --upgrade setuptools && \
  python3 -m pip install --no-cache-dir --upgrade wheel && \
  python3 -m pip install --no-cache-dir --upgrade numpy pandas sphinx jinja2 tzlocal rpy2 matplotlib && \
  rm -rf /root/.cache

COPY . /root