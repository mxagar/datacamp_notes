
import time
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe

# We always need on_message with these arguments
# even if they are not used!
def on_message(client, userdata, message):
    # We simple print the message content here = message.payload
    # We can also access the topic name via message.topic
    print(f"Received message: {str(message.payload.decode('utf-8'))}")
    # To parse a JSON: data = json.loads(message.payload)
    # Then we would store it: store.append(data)
    # And finally as a dataframe outside from on_message:
    # df = pd.DataFrame(store)
    # df.to_csv("datastream.csv", index=False)

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
