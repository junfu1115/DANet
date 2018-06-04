mkdir -p "$HOME"/data
cd "$HOME"/data

# augmented PASCAL VOC
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar -zxvf benchmark.tgz
mv benchmark_RELEASE VOCaug
<<<<<<< HEAD
# generate trainval txt
cd VOCaug/dataset/
cp train.txt trainval.txt
cat val.txt >> trainval.txt
cd -

=======

# generate trainval.txt
cd VOCaug/dataset/
cp train.txt trainval.txt
cat val.txt >>  trainval.txt

cd "$HOME"/data
>>>>>>> d7e511b09ee6127be11f03497b7d83327e9f7b1b
# original PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
tar -xvf VOCtrainval_11-May-2012.tar

<<<<<<< HEAD
# for PASCAL VOC testset, you need to login and manually download from (http://host.robots.ox.ac.uk:8080/)
=======
cd -
>>>>>>> d7e511b09ee6127be11f03497b7d83327e9f7b1b
