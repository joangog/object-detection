# ./yolov4_train.ps1 D:\Media\Documents\GitHub\object-detection C:\Users\ioann\darknet

$PROJECT_PATH = $args[0]
$DARKNET_PATH = $args[1]

cd $PROJECT_PATH
./gen_list.py

cd $DARKNET_PATH
mkdir "masks"
Copy-Item `
`-Path "$PROJECT_PATH/dataset/masks.data" `
`-Path "$PROJECT_PATH/dataset/masks.names" `
`-Path "$PROJECT_PATH/dataset/train.txt" `
`-Path "$PROJECT_PATH/dataset/test.txt" `
`-Path "$PROJECT_PATH/dataset/val.txt" `
`-Destination "$DARKNET_PATH/masks" `
./darknet detector train "$PROJECT_PATH/dataset/masks.data" "cfg/yolo-custom.cfg" "csdarknet53-omega.conv.105"