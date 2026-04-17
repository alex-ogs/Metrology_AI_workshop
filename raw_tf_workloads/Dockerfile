FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
LABEL authors="greg"

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_AUTOGRAPH_VERBOSITY=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /app

RUN cat /etc/os-release

RUN apt-get update && apt-get install -y python3.10-venv

# Create virtual environment with system site packages to inherit TensorFlow
RUN python3.10 -m venv /app/venv --system-site-packages

# Add venv to PATH
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt configure.sh /app/

COPY spinnaker-4.0.0.116-amd64/ /app/FLIR

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN chmod +x configure.sh

RUN ./configure.sh

RUN rm configure.sh

# 1. Add Spinnaker binaries to PATH
ENV PATH="/opt/spinnaker/bin:${PATH}"

# 2. Add Spinnaker libraries to LD_LIBRARY_PATH (Critical for finding shared libs)
ENV LD_LIBRARY_PATH="/opt/spinnaker/lib:${LD_LIBRARY_PATH}"

# 3. Configure GenTL Producer Paths (Fixes "Could not load producer")
ENV CTI_PATH="/opt/spinnaker/lib/spinnaker-gentl"
ENV SPINNAKER_GENTL64_CTI="${CTI_PATH}/Spinnaker_GenTL.cti"
ENV GENICAM_GENTL64_PATH="${CTI_PATH}"

#CMD ["sleep", "infinity"]
