FROM alpine
LABEL Description="MQTT forwarder"

# Install packages
RUN apk --no-cache add mosquitto mosquitto-clients

ENV PYTHONUNBUFFERED=1

RUN echo "**** install Python ****" && \
    apk add --no-cache python3 && \
    if [ ! -e /usr/bin/python ]; then ln -sf python3 /usr/bin/python ; fi && \
    \
    echo "**** install pip ****" && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --no-cache --upgrade pip setuptools wheel && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

RUN pip3 install --no-cache paho-mqtt

# Expose MQTT port
EXPOSE 1883

ENV PATH /usr/sbin:$PATH

WORkDIR /root
COPY forwarder.py .

CMD [ "python3", "./forwarder.py" ]
