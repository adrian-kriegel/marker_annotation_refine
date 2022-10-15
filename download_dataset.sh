mkdir dataset
cd dataset

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=adkr81&password=z5LXsfu6XR7YNbB*&submit=Login' https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P 

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=adkr81&password=z5LXsfu6XR7YNbB*&submit=Login' https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P 
