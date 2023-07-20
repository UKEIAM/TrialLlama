ARG PYTORCH_VERSION="22.06-py3"
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

ARG USERNAME=dev

# Install the (passwordless) SSH server
# Additionally, allow the user to call python directly
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/* \
    mkdir /var/run/sshd && mkdir -p /run/sshd \
    echo 'root:root' | chpasswd && \
    useradd -m ${USERNAME} && passwd -d ${USERNAME} && \
    sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
        && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config && \
     echo 'export PATH="/opt/conda/bin:$PATH"' >> /home/${USERNAME}/.bashrc ; \

RUN apt-get update && apt-get -y install make

# Install all the requirements in the requirements.txt
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

# Expose SSH Server and install graphxplore module in editable mode
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
