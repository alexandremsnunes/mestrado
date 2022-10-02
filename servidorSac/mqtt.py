import paho.mqtt.client as paho
import time
import os#, urlparse

def sendMQTT(msg):
    broker = "soldier.cloudmqtt.com" # Add Broker Link 
    port = 17608 # Set Port
    username = "dehqtywh" # Set Username of Your Broker Link
    password = "nqQ-okDAOovb" # Set Password of Your Broker Link

    publisher = paho.Client("MQTT PYTHON PUBLISHER")
    publisher.username_pw_set(username, password)
    publisher.connect(broker,port)

    publisher.publish("house/Room_Temp",msg)
    print("Please check data on your Subscriber Code \n")
    