__author__ = 'DheerendraArc'

import socket
import time


class SensorStreamingTest(object):
    def __init__(self):

        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.43.20', 8002))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.streaming()

    def streaming(self):

        try:
            print ("Connection from: ", self.client_address)
            start = time.time()

            while True:
                sensor_data = float(self.connection.recv(1024))
                print ("Distance: %0.1f cm" % sensor_data)

                # testing for 10 seconds
                if time.time() - start > 60:
                    break
        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    SensorStreamingTest()
