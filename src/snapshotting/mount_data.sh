KEY_UUID="insert_key_uuid"
DATA_UUID="insert_data_uuid"

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
