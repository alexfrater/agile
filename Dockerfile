FROM ubuntu:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    xz-utils \
    build-essential

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    g++ \
    git \
    libfl2 \
    libfl-dev \
    bison \
    flex \
    make \
    perl \
    ccache \
    libgoogle-perftools-dev \
    numactl \
    perl-doc \
    wget \
    libevent-dev \
    python3 \
    python3-pip

RUN git clone https://github.com/verilator/verilator.git && \
    cd verilator && \
    git checkout stable && \
    autoconf && \
    ./configure && \
    make && \
    make install


RUN apt-get update && apt-get upgrade -y \
    && wget https://github.com/chipsalliance/verible/releases/download/v0.0-3724-gdec56671/verible-v0.0-3724-gdec56671-linux-static-x86_64.tar.gz \
    &&   tar -xzf verible-v0.0-3724-gdec56671-linux-static-x86_64.tar.gz \
    && mv verible-v0.0-3724-gdec56671/bin/* /usr/local/bin/ 

CMD ["bash"]





