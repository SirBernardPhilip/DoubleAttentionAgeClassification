# importing the "tarfile" module
import tarfile

# open file
file = tarfile.open("/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/es/cv-corpus-11.0-2022-09-21-es.tar.gz")

# extracting file
file.extractall("/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/es/clips/")

file.close()
