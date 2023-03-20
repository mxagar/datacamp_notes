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
