# Download the dataset : "bash download.sh"

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
rm ModelNet10.zip
mkdir data
mv ModelNet10 data
cd -
