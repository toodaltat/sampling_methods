#!/usr/bin/bash
KEY_UUID="eb351f27-e077-4c89-8786-5d8f55fc9553"
DATA_UUID="a5e8e457-ffe3-48f7-ae33-ebfae756e682"

KEY_DEV=/dev/disk/by-uuid/$KEY_UUID
DATA_DEV=/dev/disk/by-uuid/$DATA_UUID

sleep 5

if [ -b "$KEY_DEV" ] && [ -b "$DATA_DEV" ]; then
  mount $KEY_DEV /mnt/key
  cryptsetup luksOpen --key-file /mnt/key/keyfile $DATA_DEV recordings
  mount /dev/mapper/recordings /data
  umount /mnt/key
else
  echo "USB drives not found, not mounting" | systemd-cat
fi
