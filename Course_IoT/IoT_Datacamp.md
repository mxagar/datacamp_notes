# Datacamp Course: Analyzing IoT Data in Python

These are my personal notes of the Datacamp course [Analyzing IoT Data in Python](https://app.datacamp.com/learn/courses/analyzing-iot-data-in-python).

Note that:

- Images are in [`pics`](pics).
- The code/exercises are in [`lab`](lab).
- The data is in [`data`](data).

Mikel Sagardia, 2023.  
No guarantees.

## Table of Contents

- [Datacamp Course: Analyzing IoT Data in Python](#datacamp-course-analyzing-iot-data-in-python)
  - [Table of Contents](#table-of-contents)
  - [0. Setup](#0-setup)
  - [1. Introduction to IOT Data: Accessing IoT Data](#1-introduction-to-iot-data-accessing-iot-data)
    - [1.1 Data Acquisition](#11-data-acquisition)
    - [1.2 Understanding the Data](#12-understanding-the-data)
    - [1.3 Introduction to Data Streams with MQTT](#13-introduction-to-data-streams-with-mqtt)
      - [Example: Publisher \& Subscriber via Test Mosquitto Broker](#example-publisher--subscriber-via-test-mosquitto-broker)

## 0. Setup

You should create an environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the necessary libraries/packages. A brief recipe for that:

```bash
# Create and activate e
conda create --name ds pip python=3.7
conda activate ds

# Install pip dependencies
pip install -r requirements.txt

# Track any changes and versions you have
pip list --format=freeze > requirements.txt
```

## 1. Introduction to IOT Data: Accessing IoT Data

IoT = Internet of Things: a network of connected devices sharing data from the environment; devices that collect data everywhere.

Common formats:

- JSON
- Plain text
- XML
- Binary
- Closed protocols

Data acquisition:

- Often IoT data is collected in data streams;
- Collected from devices;
- API endpoints.

### 1.1 Data Acquisition

Typical data acquisition with `requests` and `pandas`:

```python
import requests
import pandas as pd

# Option 1: Requests
url = "https://demo.datacamp.com/api/temp?count=3"
r = requests.get(url)
# Extract JSON
r.json()

# Convert JSON to pandas dataframe
df = pd.DataFrame(r.json()).head()

# Option 2: Handle download + conversion with pandas
url = "https://demo.datacamp.com/api/temp?count=3"
df_env = pd.read_json(url)
df_env.head()

# Pandas often takes care of data types, e.g., timestamps
print(df_env.dtypes)
```

To store the data:

```python
# JSON
df_env.to_json("data.json", orient="records")
# CSV
df_temp.to_csv("temperature.csv", index=False)
```

### 1.2 Understanding the Data

```python
import requests
import pandas as pd

DATA_PATH = "../data/"
filename = "environ_MS83200MS_nowind_3m-10min.json"

df = pd.read_json(DATA_PATH+filename)
df.head()

df.info()
df.describe()
```

### 1.3 Introduction to Data Streams with MQTT

Data streams are constant streams of data; e.g.:

- Twitter
- Video
- Sensor IoT data
- Market orders

The MQTT protocol can bee used to deal with them. MQTT = Message Queueing Telemetry Transport. It is used for machine-to-machine communication. Advantages:

- It has a nice **publisher/subscriber** architecture.
- It has a small footprint, it's lightweight.
- It's robust in environments with high latency and low bandwidth.

Concepts:

- There is a server, which is the **broker**; the broker **defines topics**, and any device can **publish to those topics**. Examples of topics: `temperature`, `position`.
- Any device, client, can **subscribe to a topic**.
- Also: **publisher = producer**, **subscriber = consumer**.

Installation of Paho-MQTT, the python library which implements the MQTT protocol:

```bash
python -m pip install paho-mqtt
```

Note for usage: in order to make use of MQTT, we need to set up a broker; we can either install one (e.g., [Eclipse Mosquitto](https://mosquitto.org)), or use available internet brokers created for test purposes:

- [mqtt-dashboard.com](http://www.mqtt-dashboard.com)
- [test.mosquitto.org](https://test.mosquitto.org)
- [iot.eclipse.org](https://iot.eclipse.org)

Interesting links: 

- [MQTT Beginners Guide](https://medium.com/python-point/mqtt-basics-with-python-examples-7c758e605d4).
- [Eclipse Mosquitto: An open source MQTT broker](https://mosquitto.org)

#### Example: Publisher & Subscriber via Test Mosquitto Broker

Source: [MQTT Beginners Guide](https://medium.com/python-point/mqtt-basics-with-python-examples-7c758e605d4); I modified the code.

In this example, 2 publisher scripts publish to a topic on a public broker; then, a subscriber reads from that topic. We need to run each script in a separate shell.

File [`mqtt_publisher_1.py`](./lab/mqtt_publisher_1.py):

```python
import time
import json
from random import uniform
import paho.mqtt.client as mqtt 

# Public broker: remove https://www.
#mqttBroker = "test.mosquitto.org"
mqttBroker = "mqtt.eclipseprojects.io"

# Create a client with a name
client = mqtt.Client("Temperature_Inside")
client.connect(mqttBroker) 

# Topic name: we can use any name we want, as long as it is free.
topic_name = "/mqtt/test/temperature"

while True:
    # Measure the value (or generate)
    rand_temp = uniform(20.0, 21.0)
    # Pack it
    packet = {"temperature": rand_temp, "location": "inside"}
    # PUBLISH to broker topic /mqtt/test/temperature
    # The broker creates the topic if not available
    client.publish(topic_name, json.dumps(packet))
    print(f"Just published {str(packet)} to topic {topic_name}")
    time.sleep(1) # 1 sec

```

File [`mqtt_publisher_2.py`](./lab/mqtt_publisher_2.py):

```python
import time
import json
from random import randrange
import paho.mqtt.client as mqtt

# Public broker: remove https://www.
#mqttBroker = "test.mosquitto.org"
mqttBroker = "mqtt.eclipseprojects.io"

# Create a client with a name
client = mqtt.Client("Temperature_Outside")
client.connect(mqttBroker)

# Topic name: we can use any name we want, as long as it is free.
topic_name = "/mqtt/test/temperature"

while True:
    # Measure the value (or generate)
    rand_temp = randrange(10)
    # Pack it
    packet = {"temperature": rand_temp, "location": "outside"}
    # PUBLISH to broker topic /mqtt/test/temperature
    # The broker creates the topic if not available
    client.publish(topic_name, json.dumps(packet))
    print(f"Just published {str(packet)} to topic {topic_name}")
    time.sleep(1) # 1 sec

```

File [`mqtt_subscribe.py`](./lab/mqtt_subscribe.py): Note that we can either (1) create a client which runs in a `loop` or (2) create a `callback`. For both cases, a function `on_message()` needs to be defined. This script reads the messages sent by the other two to the topic `/mqtt/test/temperature` hosted in the specified public broker.

```python

import time
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe

# We always need on_message with these arguments
# even if they are not used!
def on_message(client, userdata, message):
    # We simple print the message content here = message.payload
    print(f"Received message: {str(message.payload.decode('utf-8'))}")

# Public broker: remove https://www.
#mqttBroker = "test.mosquitto.org"
mqttBroker = "mqtt.eclipseprojects.io"

# Topic name: the name should be the one used by the publishers
topic_name = "/mqtt/test/temperature"

## Option 1: Use a client and a loop
client = False
if client:
    client = mqtt.Client("Smartphone")
    client.connect(mqttBroker)

    # Loop
    client.loop_start()
    client.subscribe(topic_name)
    client.on_message = on_message
    # It means the loop stops after 30 sec!
    # Not that it waits 30 sec after reading once!
    time.sleep(30)
    client.loop_stop()

## Option 2: Use a callback
if not client:
    subscribe.callback(on_message,
                       topics=topic_name,
                       hostname=mqttBroker)

```