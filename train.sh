ls_date=`date +%Y%m%d%H`.log
LOG=./log/$ls_date
nohup python classifier.py >$LOG 2>&1 &

