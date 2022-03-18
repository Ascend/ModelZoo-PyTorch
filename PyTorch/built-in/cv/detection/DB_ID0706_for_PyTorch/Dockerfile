ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN apt-get update && \
    apt-get install -y wget
RUN wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
RUN bunzip2 geos-3.8.1.tar.bz2
RUN tar xvf geos-3.8.1.tar
WORKDIR ./geos-3.8.1
RUN ./configure && make && make install
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt