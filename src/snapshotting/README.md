# Recording system

## Summary

Raspberry pi with webcam, two USB drives, one to keep an encryption key and one
for storing collected 

## Setup instructions

1. Format the key drive and create an encryption key

`openssl rand -out /path/to/key/drive/key 4096`

2. Set up encrypted data drive  

```bash
sudo cryptsetup luksFormat --key-file /path/to/key/drive/key /dev/disk/by-uuid/data-usb-uuid
sudo cryptsetup luksOpen cryptsetup luksOpen /dev/disk/by-uuid/data-usb-uuid recordings
sudo mkfs.ext4 /dev/mapper/recordings
sudo mount /dev/mapper/recordings /data
mkdir /data/recordings 
```

4. Edit `mount_data.sh` to include correct UUIDs for key and data drives

3. Copy and enable scripts and services 

```bash
cp mount-data.service /etc/systemd/system/
cp recording.service /etc/systemd/system/
cp mount_data.sh /usr/local/bin/bash/
systemctl enable mount-data.service
systemctl enable recording.service
```

## Operating instructions
Ensure both the key and data USBs are inserted, boot the raspberry pi, and wait for
a minute before removing the key drive. The systemd services should enable

## TODO
Needs testing, ensure that camera lives on `/dev/video0`, should check required
packages come by default on Raspbian or if it needs proper installation (e.g. for LUKS, ffmpeg). 
