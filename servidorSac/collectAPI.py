#!/usr/bin/env python
from flask import Flask, request, Response
import socket


serverAddressPort = ("192.168.1.102", 20001) #ip m5
#serverAddressPort     = ("192.168.1.7",20001) #ip m5
bufferSize = 1024

# Create a UDP socket at client side
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

app = Flask(__name__)

@app.route('/control', methods=['POST','GET'])
def control():
    #print(request.data)                                         # Printa os dados contidos no pacote enviado
    msgFromClient = request.data.decode("utf-8")
    
    bytesToSend = str.encode(msgFromClient)
    # Send to server using created UDP socket
    UDPClientSocket.sendto(bytesToSend, serverAddressPort)
    return Response(status=201)                                 # Se recebeu uma requisição POST do client, responde com status de sucesso



app.run(host='192.168.1.100', port= 5000, debug=False)
#app.run(host="192.168.1.5", port= 5000, debug=False)
#localIP     = 
# Cria rota que a ESP8266 realizará o http.POST com os resultados do SAC-DM





    