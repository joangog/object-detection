# ./yolov4_train.ps1 C:\Users\<username>\darknet

$PROJECT_PATH = (get-Item $PWD).parent.FullName
$DARKNET_PATH = $args[0]

cd $PROJECT_PATH
./yolov4/scripts/yolov4_gen_data.py

cd $DARKNET_PATH
mkdir -p "masks"
Copy-Item -Path "$PROJECT_PATH/dataset/masks.data" -Destination $DARKNET_PATH/masks
./darknet detector train "$PROJECT_PATH/dataset/masks.data" "cfg/yolo-custom.cfg" "csdarknet53-omega.conv.105"

cd $PROJECT_PATH/yolov4