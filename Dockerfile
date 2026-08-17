FROM kspacekelvin/fire-python

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility


# Julia System Dependencies (Merged)
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    g++ \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Julia Installation & Path Configurations
RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz \
    && tar -xzf julia-1.11.5-linux-x86_64.tar.gz \
    && mv julia-1.11.5 /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && rm julia-1.11.5-linux-x86_64.tar.gz

ENV PATH="/opt/julia/bin:${PATH}"

# Global Environment Setup (From Uncompiled)
RUN julia -e 'using Pkg; Pkg.add(["MriResearchTools", "ArgParse","Dates"])'

# Python Pip Dependencies (Merged)
RUN pip3 install nibabel juliacall

# Custom Python-Julia Project Environment Setup (From Compiled)
ENV PYTHON_JULIAPKG_PROJECT="/opt/juliapkg_env"

RUN python3 -c "import juliacall"

RUN python3 -c "import juliapkg; juliapkg.add('MriResearchTools'); juliapkg.add('PackageCompiler'); juliapkg.add('ArgParse'); juliapkg.resolve()"

# Precompiling the mcpc3ds function into a Sysimage
RUN echo 'using MriResearchTools; phase = rand(Float32, 10,10,10,6); mag = rand(Float32, 10,10,10,6); mcpc3ds(phase, mag, TEs=[3.0, 5.5, 8.0, 10.5, 13.0, 15.5]);' > /tmp/dummy.jl

RUN python3 -c 'from juliacall import Main as jl; jl.seval("using PackageCompiler"); jl.seval("create_sysimage([\"MriResearchTools\"], sysimage_path=\"/opt/julia/sys_mcpc.so\", precompile_execution_file=\"/tmp/dummy.jl\", include_transitive_dependencies=false)")'

ENV PYTHON_JULIACALL_SYSIMAGE=/opt/julia/sys_mcpc.so

RUN python3 -c "import juliacall"

# Lock down environments for offline use
ENV PYTHON_JULIAPKG_OFFLINE="yes"
ENV JULIA_PKG_OFFLINE="true"

# Application Modules and Server Scripts
COPY python-modules/MCPC_c_opt.py /opt/code/python-ismrmrd-server/MCPC_c_opt.py
COPY python-modules/MCPCLogger.py /opt/code/python-ismrmrd-server/MCPCLogger.py
# OLD VERSIONS
#COPY python-modules/MCPC_compiled.py /opt/code/python-ismrmrd-server/MCPC_compiled.py # 
#COPY python-modules/MCPC_uncompiled.py /opt/code/python-ismrmrd-server/MCPC_uncompiled.py


# custom dependencies
RUN mkdir -p /opt/code/python-ismrmrd-server/custom_scripts
COPY python-modules/custom_scripts /opt/code/python-ismrmrd-server/custom_scripts

# For testing purposes
#COPY python-modules/test_scripts /opt/code/python-ismrmrd-server/


# RUN mkdir -p /opt/code/python-ismrmrd-server/data

# COPY data/dcm2niix_converts.tar.gz /opt/code/python-ismrmrd-server/data/dcm2niix_converts.tar.gz
# RUN tar -xzf /opt/code/python-ismrmrd-server/data/dcm2niix_converts.tar.gz \
#    -C /opt/code/python-ismrmrd-server/data/

CMD [ "/bin/bash", "-c", "/usr/sbin/ldconfig && exec python3 /opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=MCPC_compiled" ]



# Testing and build command, if container name is changed, MUST BE UPDATED IN tooling>CreateORDockerImage.py
#sudo docker build -t mcpc-dual -f Dockerfile . 

#python3 main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=MCPC_uncompiled_test
#python3 client.py -c MCPC_uncompiled_test -o data/output_u.h5 data/dcm2niix_converts/input.h5

#python3 main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=MCPC_compiled_test
#python3 client.py -c MCPC_compiled_test -o data/output_c.h5 data/dcm2niix_converts/input.h5
#python3 dicom2mrd.py -o input.h5 data