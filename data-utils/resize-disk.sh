echo "Displaying current block devices and disk sizes." 
lsblk
read -p "Does sda have the size you inputted on the GCE console? (y/n)" -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Growing partition for /dev/sda1. (If disk is full, this command will fail.)"
    sudo growpart /dev/sda 1
    echo "Resizing filesystem to size of partititon."
    sudo resize2fs /dev/sda1
    df -h
    echo "If everything worked, /dev/sda1 should show the new size in the table above."
else
    echo "Aborting program. Check the GCE console to confirm increase in disk size."
fi

