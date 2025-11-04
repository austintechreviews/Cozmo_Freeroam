#!/bin/bash
# Reconnect Raspberry Pi to Cozmo Wi-Fi

sudo ip addr flush dev wlan0
sudo ip addr add 172.31.1.2/24 dev wlan0
ping -c 2 172.31.1.1
